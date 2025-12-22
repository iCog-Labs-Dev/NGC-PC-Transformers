from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from jax import numpy as jnp, random, jit
import jax
from config import Config as config
from utils.attention_utils import AttentionBlock



class Attention:
    """
    NGC Attention layer implementing multi-head self-attention
    with Hebbian learning and error propagation.
    
    Architecture:
    - Single z_qkv source projected to Q, K, V
    - Multi-head attention computation  
    - Output projection with predictive coding error signals
    - Hebbian learning for all synaptic weights
    
    Args:
        dkey: JAX PRNG key
        n_embed: Embedding dimension
        seq_len: Sequence length  
        batch_size: Batch size
        n_heads: Number of attention heads
        dropout_rate: Attention dropout rate
        eta: Learning rate for Hebbian synapses
    """
        
    def __init__(self, dkey, n_embed, seq_len, batch_size, n_heads, dropout_rate, eta, optim_type, wub, wlb, prefix, tau_m, **kwargs):
    
        dkey, *subkeys = random.split(dkey, 10)

        self.z_qkv = RateCell(f"{prefix}z_qkv", n_units=n_embed, tau_m=tau_m, 
                            act_fx="identity", batch_size=batch_size * seq_len )
      
        self.W_q = HebbianSynapse(f"{prefix}W_q", shape=(n_embed, n_embed), batch_size=batch_size * seq_len, eta=eta,
                                weight_init=dist.uniform(amin=wlb, amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., 
                                optim_type=optim_type, sign_value= 1.0, key=subkeys[0])
        
        self.W_k = HebbianSynapse(f"{prefix}W_k", shape=(n_embed, n_embed), batch_size=batch_size * seq_len, eta=eta,
                                weight_init=dist.uniform(amin=wlb, amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., 
                                optim_type=optim_type, sign_value= 1.0, key=subkeys[1])
        
        self.W_v = HebbianSynapse(f"{prefix}W_v", shape=(n_embed, n_embed), batch_size=batch_size * seq_len, eta=eta,
                                weight_init=dist.uniform(amin=wlb, amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., 
                                optim_type=optim_type, sign_value= 1.0, key=subkeys[2])
       
        self.attn_block = AttentionBlock(f"{prefix}attn_block", n_heads=n_heads, 
                                       n_embed=n_embed, seq_len=seq_len,
                                       dropout_rate=dropout_rate, 
                                       batch_size=batch_size)
        
        self.W_attn_out = HebbianSynapse(f"{prefix}W_attn_out", shape=(n_embed, n_embed), batch_size=batch_size * seq_len, eta=eta,
                            weight_init=dist.uniform(amin=wlb, amax=wub),
                            bias_init=dist.constant(value=0.), w_bound=0., 
                            optim_type=optim_type, sign_value= 1.0, key=subkeys[3])
        
        self.e_attn = ErrorCell(f"{prefix}e_attn", n_units=n_embed, 
                                  batch_size=batch_size * seq_len) # shape=(seq_len, n_embed, 1),
        
        self.E_attn = StaticSynapse(f"{prefix}E_attn", shape=(n_embed, n_embed),
                        weight_init=dist.uniform(low=wlb, high=wub),  key=subkeys[4])
