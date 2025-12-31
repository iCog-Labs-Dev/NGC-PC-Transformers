
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config
from utils.embed_utils import EmbeddingSynapse
from jax import random

class EMBEDDING:
    """
   embedding layer using the EmbeddingSynapse
    """
    def __init__(self, dkey, vocab_size, seq_len, embed_dim, batch_size, pos_learnable, eta, optim_type, **kwargs):
        
        dkey, *subkeys = random.split(dkey, 4)
    
        # RateCell expects a 3D shape tuple for image components (seq_len, embed_dim, channels)so here we use the third dim as a placeholder
        self.z_embed = RateCell("z_embed", n_units=seq_len, tau_m=0., 
                                  act_fx="identity", batch_size=batch_size)            
            # EmbeddingSynapse (handles both word + position internally)
        self.W_embed = EmbeddingSynapse(
                "W_embed", 
                vocab_size=vocab_size,
                seq_len=seq_len,
                embed_dim=embed_dim, 
                batch_size=batch_size,
                pos_learnable=pos_learnable,
                eta=eta,
                optim_type=optim_type,
                key=subkeys[0])
            
        self.e_embed = ErrorCell("e_embed", n_units=embed_dim, 
                                  batch_size=batch_size * seq_len) # shape=(seq_len, embed_dim, 1),
    
            

