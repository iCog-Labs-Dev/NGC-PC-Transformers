from jax import numpy as jnp, random
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config


class Output:
     """
    NGC Output Layer for final projection to vocabulary space.
    
    Projects hidden representations to vocabulary distribution with
    Hebbian learning and predictive coding error propagation.
    
    Args:
        dkey: JAX PRNG key
        target: Target tokens for error computation  
        td_error: Top-down error signal from previous layer
        n_embed: Embedding dimension
        seq_len: Sequence length
        batch_size: Batch size
        vocab_size: Vocabulary size
        eta: Learning rate for Hebbian synapses
    """
     def __init__(self, dkey, n_embed, seq_len, batch_size, vocab_size, eta, optim_type, wub, wlb, tau_m,  **kwargs):
     
        dkey, *subkeys = random.split(dkey, 10)
      
        self.z_out = RateCell("z_out", n_units=n_embed, tau_m=tau_m, act_fx="identity", batch_size=batch_size * seq_len)
        
        self.W_out = HebbianSynapse(
                    "W_out", shape=(n_embed, vocab_size), batch_size= batch_size * seq_len, eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value= 1.0, key=subkeys[4])
        self.e_out = ErrorCell("e_out", n_units=vocab_size, 
                                  batch_size=batch_size * seq_len) # shape=(seq_len, vocab_size, 1),
        self.E_out = StaticSynapse(
                    "E_out", shape=(vocab_size, n_embed), weight_init=dist.uniform(low=wlb, high=wub), key=subkeys[4])
