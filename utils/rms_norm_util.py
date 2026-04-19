import jax.numpy as jnp
from jax import jit

from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

@jit
def rms_normalize(x, gamma, eps=1e-6):
    """
    RMS Normalization forward pass.

    Formula:
        rms(x)  =  sqrt( mean(x^2) + eps )        shape: (batch, 1)
        y       =  (x / rms(x)) * gamma            shape: (batch, n_embed)

    Args:
        x     : input tensor,          shape (batch, n_embed)
        gamma : learnable scale,        shape (n_embed,)
        eps   : numerical stability constant

    Returns:
        out   : normalised output,     shape (batch, n_embed)
        rms   : root mean square,      shape (batch, 1)   — saved for backward
    """
    x_float  = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    rms      = jnp.sqrt(variance + eps).astype(x.dtype)   # (batch, 1)
    out      = x * (1.0 / rms) * gamma.astype(x.dtype)
    return out, rms


# ─────────────────────────────────────────────────────────────────────────────
# Backward pass  —  RMSNorm Jacobian-vector product
# ─────────────────────────────────────────────────────────────────────────────

@jit
def rms_norm_grad(x, rms, gamma, v):
    """
    RMSNorm Jacobian-vector product:  dL/dx = J_RMSNorm(x)^T  @  v

    Full derivation
    ───────────────
    Forward:  y_i = (x_i / rms) * gamma_i
              rms = sqrt( (1/n) * sum_j(x_j^2)  +  eps )

    We want dL/dx_i given incoming gradient v = dL/dy.

    By the chain rule:
        dL/dx_i = sum_j [ v_j * (dy_j / dx_i) ]

    Computing dy_j/dx_i has two cases:

        Case j == i:
            dy_i/dx_i = (gamma_i / rms) * (1  -  x_i^2 / (n * rms^2))

        Case j != i:
            dy_j/dx_i = (gamma_j / rms) * (-x_j * x_i / (n * rms^2))

    Substituting and collecting:
        dL/dx_i = (gamma_i / rms) * v_i
                - (x_i / (n * rms^3)) * sum_j( gamma_j * v_j * x_j )

    Let x_norm = x / rms.  Then x_i / rms^3 = x_norm_i / rms^2  and
    sum_j(gamma_j * v_j * x_j) = rms * sum_j(gamma_j * v_j * x_norm_j).

    Factoring out (gamma / rms):
        dL/dx = (gamma / rms) * ( v  -  x_norm * mean(x_norm * (gamma * v)) )

    This is the expression implemented below.

    Args:
        x     : forward input saved during advance,  shape (batch, n_embed)
        rms   : rms saved during advance,            shape (batch, 1)
        gamma : scale vector from RMSNorm,            shape (n_embed,)
        v     : incoming error / gradient dL/dy,     shape (batch, n_embed)

    Returns:
        dx    : gradient w.r.t. x,                   shape (batch, n_embed)
    """
    gamma  = gamma.reshape((1,) * (v.ndim - 1) + (-1,)).astype(x.dtype)
    x_norm = x / rms                                           # (batch, n_embed)
    gv     = gamma.reshape((1,) * (v.ndim - 1) + (-1,)) * v                                      # (batch, n_embed)
    inner  = jnp.mean(x_norm * gv, axis=-1, keepdims=True)   # (batch, 1)
    dx     = (gamma / rms) * (v - x_norm * inner)
    return dx


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm component  (forward)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(JaxComponent):
    """
    RMS normalisation — forward pass only.

    Stores .inputs and .rms so the paired RMSNormGrad can use them
    in the backward pass without recomputing.

    Compartments
    ─────────────
    .inputs   : pre-normalisation tensor        (batch, n_embed)
    .outputs  : post-normalisation tensor       (batch, n_embed)
    .rms      : per-row rms value               (batch, 1)
                saved every advance tick for the backward component
    """

    def __init__(self, name, n_embed, batch_size, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed    = n_embed
        self.batch_size = batch_size
        self.gamma      = jnp.ones((n_embed,))

        self.inputs  = Compartment(jnp.zeros((batch_size, n_embed)))
        self.outputs = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms     = Compartment(jnp.ones((batch_size, 1)))

    @compilable
    def advance_state(self):
        x        = self.inputs.get()
        out, rms = rms_normalize(x, self.gamma)
        self.outputs.set(out)
        self.rms.set(rms)          # persisted for RMSNormGrad

    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros((self.batch_size, self.n_embed)))
        self.outputs.set(jnp.zeros((self.batch_size, self.n_embed)))
        self.rms.set(jnp.ones((self.batch_size, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# RMSNormGrad component  (backward)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNormGrad(JaxComponent):
    """
    Backward counterpart of RMSNorm.

    Computes the RMSNorm Jacobian-vector product and then multiplies by an
    optional attention-specific gradient (dq, dk, or dv from attn_block).
    This is the "one additional compartment" required by the mentor task.

    Signal flow
    ───────────
    For the attention (ln1) backward path — one instance per Q/K/V:

        ln1.inputs          >>  ln1_grad_q.mu         (forward x)
        ln1.rms             >>  ln1_grad_q.rms         (forward rms)
        e_attn.dmu          >>  ln1_grad_q.dmu         (incoming error)
        attn_block.dq       >>  ln1_grad_q.dmu_attn    (attention gradient)
        ln1_grad_q.dmu_     >>  z_qkv.jq               (corrected signal out)
        ln1_grad_q.dmu_     >>  W_q.post                (Hebbian post signal)

    For the MLP (ln2) backward path — one instance, no attention gradient:

        ln2.inputs          >>  ln2_grad.mu
        ln2.rms             >>  ln2_grad.rms
        e_mlp1.dmu          >>  ln2_grad.dmu
        (dmu_attn left as ones — no Q/K/V split in MLP)
        ln2_grad.dmu_       >>  z_mlp.j
        ln2_grad.dmu_       >>  W_mlp1.post

    Compartments
    ─────────────
    .mu        : forward input x (from ln.inputs)         (batch, n_embed)
    .rms       : forward rms    (from ln.rms)              (batch, 1)
    .dmu       : incoming error signal (from e_cell.dmu)   (batch, n_embed)
    .dmu_attn  : attention-specific gradient               (batch, n_embed)
                 for ln1: receives dq / dk / dv from attn_block
                 for ln2: leave unwired — defaults to ones (no-op multiply)
    .dmu_      : corrected output = rms_norm_grad(...) * dmu_attn
                 wired to z.j  and  W.post
    """

    def __init__(self, name, n_embed, batch_size, gamma=None, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed    = n_embed
        self.batch_size = batch_size
        self.gamma      = gamma if gamma is not None else jnp.ones((n_embed,))

        self.mu        = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms       = Compartment(jnp.ones((batch_size, 1)))
        self.dmu       = Compartment(jnp.zeros((batch_size, n_embed)))

        self.dmu_attn  = Compartment(jnp.ones((batch_size, n_embed)))
    
        self.dmu_      = Compartment(jnp.zeros((batch_size, n_embed)))

    @compilable
    def advance_state(self):
        x         = self.mu.get()
        rms       = self.rms.get()
        v         = self.dmu.get()
        attn_grad = self.dmu_attn.get()     # dq / dk / dv  (or ones for MLP)

        # Step 1: apply RMSNorm Jacobian to the incoming error
        dx = rms_norm_grad(x, rms, self.gamma, v)

        # Step 2: multiply by the attention-specific gradient
        # For Q path:  dx * dq   → corrects for what Q contributed
        # For K path:  dx * dk
        # For V path:  dx * dv
        # For MLP path: dx * ones = dx  (no change)
        self.dmu_.set(dx * attn_grad)

    @compilable
    def reset(self):
        zeros = jnp.zeros((self.batch_size, self.n_embed))
        ones  = jnp.ones((self.batch_size, self.n_embed))
        self.mu.set(zeros)
        self.rms.set(jnp.ones((self.batch_size, 1)))
        self.dmu.set(zeros)
        self.dmu_attn.set(ones)    # reset to ones so MLP path is always a no-op
        self.dmu_.set(zeros)