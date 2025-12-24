from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment 
from jax import numpy as jnp, random, jit
from ngclearn import compilable
import jax
from functools import partial
from config import Config as config
import jax.random as random
@partial(jit, static_argnums=[4, 5, 6])
def _compute_attention(Q, K, V, mask, n_heads, d_head, dropout_rate, key):
    """
    Compute multi-head attention 
    """
    if Q.ndim == 2:
        # 2D input: (batch_size * seq_len, n_embed) -> reshape to 3D
        M, D = Q.shape
        S = config.seq_len        
        B = M // S  # batch_size
        Q_3d = Q.reshape(B, S, D)
        K_3d = K.reshape(B, S, D)
        V_3d = V.reshape(B, S, D)
    else:
        # 3D input: (batch_size, seq_len, n_embed)
        B, S, D = Q.shape
        Q_3d, K_3d, V_3d = Q, K, V
    # Reshape for multi-head attention
    q = Q.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = K.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3]) 
    v = V.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    
    # Scaled dot-product attention
    score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)
    
    if mask is not None:
        Tq, Tk = q.shape[2], k.shape[2]
        _mask = mask.reshape((B, 1, Tq, Tk))
        score = jnp.where(_mask, score, -1e-9)
        
    score = jax.nn.softmax(score, axis=-1)
    score = score.astype(q.dtype)
    
    if dropout_rate > 0.0:
        # dkey = random.fold_in(key, 0)
        dkey = random.PRNGKey(0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)
        
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))
    
    return attention

class AttentionBlock(JaxComponent):
    """
    Multi-head attention block for NGC attention.
    
    Takes Q, K, V inputs and computes scaled dot-product attention 
    with optional masking and dropout.
    
    | --- Compartments: ---
    | inputs_q - query inputs
    | inputs_k - key inputs  
    | inputs_v - value inputs
    | mask - attention mask
    | outputs - attention outputs
    | key - JAX PRNG key

    Args:
        name: Component name
        n_heads: Number of attention heads
        n_embed: Embedding dimension
        seq_len: Sequence length
        dropout_rate: Attention dropout rate
        batch_size: Batch size
    """
    
    def __init__(self, name, n_heads, n_embed, seq_len, dropout_rate=0.0, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_head = n_embed // n_heads

        # Input compartments
        self.inputs_q = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_k = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_v = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.mask = Compartment(jnp.zeros((batch_size, seq_len, seq_len), dtype=bool))
        
        self.key = Compartment(random.PRNGKey(0))
        # Output compartment
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
    @compilable
    def advance_state(self):
        """
        Compute multi-head attention
        """
        inputs_q=self.inputs_q.get()
        inputs_k=self.inputs_k.get()
        inputs_v=self.inputs_v.get()
        mask=self.mask.get()
        n_heads=self.n_heads.get()
        d_head=self.d_head.get()
        dropout_rate=self.dropout_rate.get()
        key=self.key.get()
        attention = _compute_attention(
            inputs_q, inputs_k, inputs_v, mask, n_heads, d_head, dropout_rate, key
        )
        
        self.outputs.set(attention)
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """
        batch_size=self.batch_size.get()
        seq_len=self.seq_len.get()
        n_embed=self.n_embed.get()
        zeros_3d = jnp.zeros((batch_size, seq_len, n_embed))
        mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=bool)
        # return zeros_3d, zeros_3d, zeros_3d, mask, zeros_3d
        self.inputs_q.set(zeros_3d)
        self.inputs_k.set(zeros_3d)
        self.inputs_v.set(zeros_3d)
        self.mask.set(mask)
        self.outputs.set(zeros_3d)

    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "component_type": "AttentionBlock - multi-head self-attention mechanism"
        }
        compartment_props = {
            "inputs": 
                {"inputs_q": "Query inputs (batch_size, seq_len, n_embed)",
                 "inputs_k": "Key inputs (batch_size, seq_len, n_embed)", 
                 "inputs_v": "Value inputs (batch_size, seq_len, n_embed)",
                 "mask": "Attention mask (batch_size, seq_len, seq_len)"},
            "outputs":
                {"outputs": "Attention outputs (batch_size, seq_len, n_embed)"},
        }
        hyperparams = {
            "n_heads": "Number of attention heads",
            "n_embed": "Embedding dimension", 
            "seq_len": "Sequence length",
            "dropout_rate": "Attention dropout rate",
            "batch_size": "Batch size dimension"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = MultiHeadAttention(Q, K, V, mask)",
                "hyperparameters": hyperparams}
        return info
    













    