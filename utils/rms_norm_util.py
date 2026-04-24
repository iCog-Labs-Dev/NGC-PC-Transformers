import jax.numpy as jnp
from jax import jit

from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


@jit
def rms_normalize(x, gamma, eps=1e-6):
    """
    RMS Normalization forward pass.

    Formula:
        rms   = sqrt( mean(x^2) + eps )
        y     = (x / rms) * gamma
    """
    x_float  = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    rms      = jnp.sqrt(variance + eps).astype(x.dtype)
    out      = x * (1.0 / rms) * gamma.astype(x.dtype)
    return out, rms


@jit
def rms_norm_grad(x, rms, gamma, v):
    """
    RMSNorm Jacobian-vector product:  dx = J_RMSNorm(x)^T @ v

    Derivation:
        y_i = (x_i / rms) * gamma_i
        rms = sqrt( mean(x^2) + eps )

        dL/dx = (gamma / rms) * ( v - x_norm * mean(x_norm * gamma * v) )
        where x_norm = x / rms

    This is the derivation applied once to v — single multiplication
    """
    gamma_r = gamma.reshape((1,) * (v.ndim - 1) + (-1,)).astype(x.dtype)
    x_norm  = x / rms
    scale   = gamma_r / rms
    inner   = jnp.mean(x_norm * (gamma_r * v), axis=-1, keepdims=True)
    dx      = scale * (v - x_norm * inner)
    return dx


class RMSNorm(JaxComponent):
    """
    RMS normalisation — forward pass only.

    Compartments:
        .inputs  : pre-norm tensor   (batch, n_embed)
        .outputs : post-norm tensor  (batch, n_embed)
        .rms     : saved rms value   (batch, 1)  — used by RMSNormGrad
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
        self.rms.set(rms)

    @compilable
    def reset(self):
        zeros = jnp.zeros((self.batch_size, self.n_embed))
        self.inputs.set(zeros)
        self.outputs.set(zeros)
        self.rms.set(jnp.ones((self.batch_size, 1)))


class RMSNormGrad(JaxComponent):
    """
    Backward counterpart of RMSNorm.

    Applies rms_norm_grad (the derivation) once to the incoming error signal.
    The result is stored in both .dmu_ and .dmu_mlp1 

    Compartments:
        .mu        : forward input x  (from ln.inputs)          (batch, n_embed)
        .rms       : saved rms        (from ln.rms)             (batch, 1)
        .dmu       : incoming error   (from E.outputs or e.dmu) (batch, n_embed)
        .dmu_attn  : attention gradient (dq/dk/dv from attn_block or ones)
                     kept for wiring compatibility — value not used in grad
        .dmu_      : corrected output = rms_norm_grad(mu, rms, gamma, dmu)
        .dmu_mlp1  : same as dmu_ 
    """

    def __init__(self, name, n_embed, batch_size, gamma=None, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed    = n_embed
        self.batch_size = batch_size
        self.gamma      = gamma if gamma is not None else jnp.ones((n_embed,))

        self.mu        = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms       = Compartment(jnp.ones((batch_size, 1)))
        self.dmu       = Compartment(jnp.zeros((batch_size,4* n_embed)))
        self.dmu_attn  = Compartment(jnp.ones((batch_size, n_embed)))
        self.dmu_      = Compartment(jnp.zeros((batch_size, n_embed)))
        self.dmu_mlp1  = Compartment(jnp.zeros((batch_size, n_embed)))

    @compilable
    def advance_state(self):
        x   = self.mu.get()
        rms = self.rms.get()
        v   = self.dmu.get()

        # Apply the RMSNorm Jacobian once — derivation applied to v only
        dx  = rms_norm_grad(x, rms, self.gamma, v)

        self.dmu_.set(dx)
        self.dmu_mlp1.set(dx)

    @compilable
    def reset(self):
        zeros = jnp.zeros((self.batch_size, self.n_embed))
        ones  = jnp.ones((self.batch_size, self.n_embed))
        self.mu.set(zeros)
        self.rms.set(jnp.ones((self.batch_size, 1)))
        self.dmu.set(zeros)
        self.dmu_attn.set(ones)
        self.dmu_.set(zeros)
        self.dmu_mlp1.set(zeros)