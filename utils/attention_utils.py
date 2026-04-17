from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from jax import numpy as jnp, random, jit
from ngclearn import compilable
import jax
from functools import partial
import jax.numpy as jnp
from utils.model_util import d_softmax_vjp

@partial(jit, static_argnums=[4, 5, 6, 7, 8])
def _compute_attention(Q, K, V, mask, n_heads, d_head, dropout_rate, seq_len, batch_size, key):
    """
    Compute multi-head attention.

    Shapes:
        Q, K, V : (batch_size, seq_len, n_embed)
        mask    : (seq_len, seq_len)  causal lower-triangular
        returns : attention (batch, seq_len, n_embed)
                  s_c      (batch, n_heads, seq_len, seq_len)  raw scores
                  q, k, v  (batch, n_heads, seq_len, d_head)   head-split inputs
    """
    B = batch_size
    S = seq_len

    q = Q.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = K.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    v = V.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])

    s_c = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)

    _mask = mask[None, None, :, :]
    s_c   = jnp.where(_mask, s_c, -1e9)

    score = jax.nn.softmax(s_c, axis=-1).astype(q.dtype)

    if dropout_rate > 0.0:
        dkey  = random.fold_in(key, 0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)

    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))

    return attention, s_c, q, k, v


@partial(jit, static_argnums=[6, 7, 8, 9, 10])
def compute_grads(Q, K, V, mask, s_c, dmu, n_heads, d_head, dropout_rate, seq_len, batch_size, key):
    """
    Compute separate gradients for Q, K, V using the softmax Jacobian.

    This is the NGC equivalent of the backward pass through scaled dot-product
    attention.  Instead of automatic differentiation the gradients are derived
    analytically using the softmax JVP already available in model_util.

    Derivation outline
    ──────────────────
    Let   A  = softmax( QK^T / sqrt(d) )          attention weights
          O  = A @ V                               attention output   (B,H,S,D)
    Given dO = dmu reshaped to (B,H,S,D) (error from e_qkv.dmu):

        dV  = A^T @ dO                             (B,H,S,D)
        dA  = dO  @ V^T                            (B,H,S,S)
        ds  = J_softmax(s_c)^T @ dA  /  sqrt(d)   softmax JVP, scaled
        dQ  = ds  @ K                              (B,H,S,D)
        dK  = ds^T @ Q                             (B,H,S,D)

    Returns dq, dk, dv each reshaped to (B*S, H*D) = (batch*seq, n_embed).
    These are wired into ln1_grad_q/k/v.dmu_attn in model.py.

    Args:
        Q, K, V    : head-split tensors from _compute_attention  (B,H,S,D)
        mask       : causal mask                                  (S,S)
        s_c        : raw attention scores before softmax          (B,H,S,S)
        dmu        : error signal from e_qkv.dmu                 (B*S, H*D)
        n_heads, d_head, dropout_rate, seq_len, batch_size, key

    Returns:
        dq, dk, dv : gradients for Q, K, V projections           (B*S, n_embed)
    """
    B = batch_size
    S = seq_len
    H = n_heads
    D = d_head

    P, jvp_fn = d_softmax_vjp(s_c, tau=0.0)          # P: (B,H,S,S)

    dmu_reshaped = dmu.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # (B,H,S,D)

    dV = jnp.einsum("bhkq,bhkd->bhqd", P, dmu_reshaped)           # (B,H,S,D)
    da = jnp.einsum("bhkd,bhqd->bhqk", dmu_reshaped, V)            # (B,H,S,S)
    ds = jvp_fn(da) / jnp.sqrt(D)

    _mask = mask[None, None, :, :]
    ds    = jnp.where(_mask, ds, 0.)

    dQ = jnp.einsum("bhqk,bhkd->bhqd", ds, K)                     # (B,H,S,D)
    dK = jnp.einsum("bhkq,bhqd->bhkd", ds, Q)                     # (B,H,S,D)

    dq = dQ.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    dk = dK.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    dv = dV.transpose(0, 2, 1, 3).reshape(B * S, H * D)

    return dq, dk, dv


class AttentionBlock(JaxComponent):
    """
    Multi-head self-attention block for the NGC transformer.

    Handles both the forward pass (computing attended output) and the backward
    pass (computing per-projection gradients dq, dk, dv for the predictive
    coding update).

    Compartments
    ─────────────
    inputs_q, inputs_k, inputs_v  : Q/K/V inputs    (batch, seq, n_embed)
    outputs                        : attention output (batch, seq, n_embed)
    dmu                            : incoming error from e_qkv.dmu
                                     wired:  e_qkv.dmu >> attn_block.dmu
    dq, dk, dv                     : per-projection gradients
                                     wired:  attn_block.dq >> ln1_grad_q.dmu_attn
                                             attn_block.dk >> ln1_grad_k.dmu_attn
                                             attn_block.dv >> ln1_grad_v.dmu_attn
    dtarget_q/k/v                  : negated gradients  = -dq / -dk / -dv
                                     preserved from original wiring —
                                     wired:  attn_block.dtarget_q >> W_q.post
                                     (used in old arch; ln1_grad replaces this
                                      in new arch but kept for compatibility)
    """

    def __init__(self, name, n_heads, n_embed, seq_len, dropout_rate, batch_size, **kwargs):
        super().__init__(name, **kwargs)

        self.n_heads      = n_heads
        self.n_embed      = n_embed
        self.dropout_rate = dropout_rate
        self.batch_size   = batch_size
        self.seq_len      = seq_len
        self.causal_mask  = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

        if self.n_embed % self.n_heads != 0:
            raise ValueError(f"n_embed={n_embed} must be divisible by n_heads={n_heads}")
        self.d_head = n_embed // n_heads

        zeros_2d = jnp.zeros((batch_size * seq_len, n_embed))
        zeros_3d = jnp.zeros((batch_size, seq_len, n_embed))

        self.inputs_q   = Compartment(zeros_3d)
        self.inputs_k   = Compartment(zeros_3d)
        self.inputs_v   = Compartment(zeros_3d)
        self.outputs    = Compartment(zeros_3d)
        self.dmu        = Compartment(zeros_2d)
        self.dq         = Compartment(zeros_2d)
        self.dk         = Compartment(zeros_2d)
        self.dv         = Compartment(zeros_2d)
        # Preserved from original wiring (dtarget = -dq/dk/dv)
        self.dtarget_q  = Compartment(zeros_2d)
        self.dtarget_k  = Compartment(zeros_2d)
        self.dtarget_v  = Compartment(zeros_2d)
        self.key        = Compartment(random.PRNGKey(0))

    @compilable
    def advance_state(self):
        inputs_q = self.inputs_q.get()
        inputs_k = self.inputs_k.get()
        inputs_v = self.inputs_v.get()
        dmu      = self.dmu.get()
        key      = self.key.get()

        # ── Forward pass ─────────────────────────────────────────────────────
        attention, s_c, q, k, v = _compute_attention(
            inputs_q, inputs_k, inputs_v,
            self.causal_mask,
            self.n_heads, self.d_head, self.dropout_rate,
            self.seq_len, self.batch_size, key
        )
        self.outputs.set(attention)

        # ── Backward pass — per-projection gradients ─────────────────────────
        # dq, dk, dv are wired into ln1_grad_q/k/v.dmu_attn in model.py
        # so the RMSNormGrad component multiplies the LN Jacobian by each one.
        dq, dk, dv = compute_grads(
            q, k, v,
            self.causal_mask, s_c, dmu,
            self.n_heads, self.d_head, self.dropout_rate,
            self.seq_len, self.batch_size, key
        )
        self.dq.set(dq)
        self.dk.set(dk)
        self.dv.set(dv)

        # Preserved: dtarget = -dq/dk/dv  (original wiring compatibility)
        self.dtarget_q.set(-dq)
        self.dtarget_k.set(-dk)
        self.dtarget_v.set(-dv)

    @compilable
    def reset(self):
        zeros_2d = jnp.zeros((self.batch_size * self.seq_len, self.n_embed))
        zeros_3d = jnp.zeros((self.batch_size, self.seq_len, self.n_embed))
        self.inputs_q.set(zeros_3d)
        self.inputs_k.set(zeros_3d)
        self.inputs_v.set(zeros_3d)
        self.outputs.set(zeros_3d)
        self.dmu.set(zeros_2d)
        self.dq.set(zeros_2d)
        self.dk.set(zeros_2d)
        self.dv.set(zeros_2d)
        # ── fixed: dtarget compartments were missing from original reset ─────
        self.dtarget_q.set(zeros_2d)
        self.dtarget_k.set(zeros_2d)
        self.dtarget_v.set(zeros_2d)

    @classmethod
    def help(cls):
        properties = {
            "component_type": "AttentionBlock - multi-head self-attention with causal mask and per-projection gradients"
        }
        compartment_props = {
            "inputs":    {"inputs_q/k/v": "(batch, seq_len, n_embed)"},
            "outputs":   {"outputs": "(batch, seq_len, n_embed)"},
            "gradients": {
                "dmu":       "incoming error from e_qkv.dmu  (batch*seq, n_embed)",
                "dq/dk/dv":  "per-projection gradients       (batch*seq, n_embed)",
                "dtarget_q/k/v": "negated gradients, preserved from original arch"
            }
        }
        hyperparams = {
            "n_heads": "number of attention heads",
            "n_embed": "embedding dimension",
            "seq_len": "sequence length",
            "dropout_rate": "attention dropout rate",
            "batch_size": "batch size"
        }
        return {cls.__name__: properties,
                "compartments": compartment_props,
                "hyperparameters": hyperparams}