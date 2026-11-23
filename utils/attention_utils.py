from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment 
from jax import numpy as jnp, random, jit
from ngclearn import compilable
import jax
from functools import partial
import jax.numpy as jnp
from utils.model_util import d_softmax_vjp
from utils.rope_utils import apply_rotary_emb, apply_rotary_emb_inv, precompute_freqs_cis_real

DISABLE_ATTN_DROPOUT = False


@partial(jit, static_argnums=[6, 7, 8, 9, 10, 11])
def _compute_attention(Q, K, V, cos, sin, mask, n_heads, d_head, dropout_rate, seq_len, batch_size, use_rope, key):
    """
    Compute multi-head attention 
    """
    B = batch_size
    S = seq_len
    D = Q.shape[-1]
    
    # Q_3d = Q.reshape(B, S, D)
    # K_3d = K.reshape(B, S, D)
    # V_3d = V.reshape(B, S, D)
    
    # Reshape for multi-head attention
    q = Q.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = K.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3]) 
    v = V.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    
    if use_rope:
        q_used, k_used = apply_rotary_emb(q, k, cos, sin)
    else:
        q_used, k_used = q,k
    
    
    # Scaled dot-product attention
    s_c = jnp.einsum("BHTE,BHSE->BHTS", q_used, k_used) / jnp.sqrt(d_head)
    
  
    _mask = mask[None, None, :, :]  
    s_c = jnp.where(_mask, s_c, -1e9)
        
    score = jax.nn.softmax(s_c, axis=-1)
    score = score.astype(q.dtype)
    
    if dropout_rate > 0.0 and not DISABLE_ATTN_DROPOUT:
        dkey = random.fold_in(key, 0)
        # dkey = random.PRNGKey(0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)
        
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))
    
    return attention, s_c, q_used, k_used, v


@partial(jit, static_argnums=[8, 9, 10, 11, 12,13])
def compute_grads(Q_rot, K_rot, V, cos, sin, mask, s_c, dmu, n_heads, d_head, dropout_rate, seq_len, batch_size,use_rope, key):
    """Compute gradients for Q, K, V using d_softmax and attention scores
    """
    B = batch_size
    S = seq_len
    H = n_heads
    D = d_head
    
    P, jvp_fn = d_softmax_vjp(s_c, tau=0.0)  # P: (B, H, S, S)
    
    # Reshape error
    dmu_reshaped = dmu.reshape(B, S, H, D).transpose(0, 2, 1, 3)  ## (B, H, S, D)
    dV = jnp.einsum("bhkq,bhkd->bhqd", P, dmu_reshaped)           ## (B, H, S, D)
    da = jnp.einsum("bhte,bhse->bhts", dmu_reshaped, V)           ## (B, H, S, D)
    ds = jvp_fn(da)  
    ds = ds / jnp.sqrt(D)
    
    _mask = mask[None, None, :, :]  # (1, 1, S, S)
    ds = jnp.where(_mask, ds, 0.) 
    
    # Backpropagate through scaled dot-product attention
    dQ_rot = jnp.einsum("bhqk,bhkd->bhqd", ds, K_rot)  # (B, H, S, D)
    dK_rot = jnp.einsum("bhkq,bhqd->bhkd", ds, Q_rot)  # (B, H, S, D)
    
    # Inverse RoPE to get raw Q and K gradients if use_rope = true
    if use_rope:
        dQ_raw, dK_raw = apply_rotary_emb_inv(dQ_rot, dK_rot, cos, sin)
    else:
        dQ_raw, dK_raw = dQ_rot, dK_rot
    
    # 6. Reshape to flattened format
    dq = dQ_raw.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    dk = dK_raw.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    dv = dV.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    
    return dq, dk, dv
    
class AttentionBlock(JaxComponent):
    """
    Multi-head attention block for NGC attention.
    
    Takes Q, K, V inputs and computes scaled dot-product attention 
    with optional masking and dropout.
    
    | --- Compartments: ---
    | inputs_q - query inputs
    | inputs_k - key inputs  
    | inputs_v - value inputs
    | outputs - attention outputs
    | key - JAX PRNG key
    | dq - gradient w.r.t. query inputs
    | dk - gradient w.r.t. key inputs
    | dv - gradient w.r.t. value inputs
    | dmu - = gradient w.r.t. mu inputs

    Args:
        name: Component name
        n_heads: Number of attention heads
        n_embed: Embedding dimension
        seq_len: Sequence length
        dropout_rate: Attention dropout rate
        batch_size: Batch size
        position_encoding: "rope" or "positional"
        rope_theta: RoPE frequency base (used only when position_encoding="rope")
        
    """
    
    def __init__(self, name, n_heads, n_embed, seq_len, dropout_rate, batch_size, position_encoding="rope", rope_theta=10000.0, **kwargs):
        super().__init__(name, **kwargs)

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        self.position_encoding = position_encoding
        self.use_rope = position_encoding == "rope"
        self.rope_theta = rope_theta

        if self.n_embed % self.n_heads != 0:
            raise ValueError(f"n_embed={n_embed} must be divisible by n_heads={n_heads}")
        self.d_head = n_embed // n_heads

        if self.use_rope and self.d_head % 2 != 0:
            raise ValueError(f"With RoPE, d_head={self.d_head} must be even for complex number representation.")
        if self.use_rope:
            # Precompute RoPE
            cos_rope, sin_rope = precompute_freqs_cis_real(self.d_head, self.seq_len, theta= self.rope_theta)
        else:
            cos_rope = jnp.ones((self.seq_len, self.d_head))
            sin_rope = jnp.zeros((self.seq_len, self.d_head))
        
        self.cos = Compartment(cos_rope)
        self.sin = Compartment(sin_rope)

        # Input compartments
        self.inputs_q = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_k = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_v = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.dq = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dk = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dv = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dmu = Compartment(jnp.zeros((batch_size * seq_len, n_embed)))
        self.key = Compartment(random.PRNGKey(0))
        # Output compartment
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))    
    
    @compilable
    def advance_state(self):
        """
        Compute multi-head attention
        and compute gradients for inputs and Jacobian using d_softmax
        """
        inputs_q=self.inputs_q.get()
        inputs_k=self.inputs_k.get()
        inputs_v=self.inputs_v.get()
        mask=self.causal_mask
        # S=self.S.get()
        dmu=self.dmu.get()
        n_heads=self.n_heads
        d_head=self.d_head
        dropout_rate=self.dropout_rate
        key=self.key.get()
        cos=self.cos.get()
        sin=self.sin.get()
        attention, s_c, q, k, v = _compute_attention(
            inputs_q, inputs_k, inputs_v, cos, sin, mask,        
            self.n_heads,        
            self.d_head,        
            self.dropout_rate, 
            self.seq_len,
            self.batch_size,  
            self.use_rope,
            key                  
        )
        # self.S.set(S)
        dq, dk, dv = compute_grads(q, k, v, cos, sin, mask, s_c, dmu, n_heads, d_head, dropout_rate, self.seq_len, self.batch_size,self.use_rope, key)
        self.dq.set(dq)
        self.dk.set(dk)
        self.dv.set(dv)
        self.outputs.set(attention)
  
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """
        batch_size=self.batch_size
        seq_len=self.seq_len
        n_embed=self.n_embed
        zeros_2d = jnp.zeros((batch_size*seq_len, n_embed))
        zeros_3d = jnp.zeros((batch_size, seq_len, n_embed))
        # return zeros_3d, zeros_3d, zeros_3d, zeros_3d
        self.inputs_q.set(zeros_3d)
        self.inputs_k.set(zeros_3d)
        self.inputs_v.set(zeros_3d)
        self.outputs.set(zeros_3d)
        self.dq.set(zeros_2d)
        self.dk.set(zeros_2d)
        self.dv.set(zeros_2d)
        self.dmu.set(zeros_2d)

    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "component_type": "AttentionBlock - multi-head self-attention with built-in causal mask and optional rotary position encoding"
        }
        compartment_props = {
            "inputs": 
                {"inputs_q": "Query inputs (batch_size, seq_len, n_embed)",
                "inputs_k": "Key inputs (batch_size, seq_len, n_embed)", 
                "inputs_v": "Value inputs (batch_size, seq_len, n_embed)"},
            "states": {
                "cos": (
                    "Precomputed RoPE cosine values "
                    "(seq_len, d_head); identity values are used when "
                    "position_encoding='positional'"
                ),
                "sin": (
                    "Precomputed RoPE sine values "
                    "(seq_len, d_head); zero values are used when "
                    "position_encoding='positional'"
                ),
                "key": (
                    "JAX pseudo-random key used for attention dropout"
                ),
            },
            "gradients":
                {"dq": "Gradient w.r.t Q (batch_size*seq_len, n_embed)",
                "dk": "Gradient w.r.t K (batch_size*seq_len, n_embed)",
                "dv": "Gradient w.r.t V (batch_size*seq_len, n_embed)",
                "dmu": "Gradient w.r.t mu (batch_size*seq_len, n_embed)"},
            "outputs":
                {"outputs": "Attention outputs (batch_size, seq_len, n_embed)"},
        }
        hyperparams = {
            "n_heads": "Number of attention heads",
            "n_embed": "Embedding dimension", 
            "seq_len": "Sequence length",
            "dropout_rate": "Attention dropout rate",
            "batch_size": "Batch size dimension",
            "position_encoding": (
                "Position-encoding mode: 'rope' or 'positional'"
            ),
            "rope_theta": (
                "RoPE frequency base; used only when "
                "position_encoding='rope'"
            ),
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": (
                    "rope: outputs = CausalMultiHeadAttention("
                    "RoPE(Q), RoPE(K), V); "
                    "positional: outputs = CausalMultiHeadAttention(Q, K, V)"
                ),
                "hyperparameters": hyperparams}
        return info
