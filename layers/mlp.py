import jax
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config

class MLP:
    """
    NGC MLP layer with two Hebbian synapses and error cell.

    Minimal working implementation: creates z_mlp RateCell, two Hebbian synapses,
    an ErrorCell and a StaticSynapse for feedback. The constructor accepts
    `target` and `td_error` to match how the model wires layers.
    """

    def __init__(self, dkey,n_embed, seq_len, batch_size, eta, optim_type, wub , wlb, prefix, tau_m, **kwargs):
        dkey, *subkeys = random.split(dkey, 10)
       

        self.z_mlp = RateCell(f"{prefix}z_mlp", n_units=n_embed, tau_m=tau_m, act_fx="identity", batch_size=batch_size * seq_len)
        self.z_mlp2 = RateCell(f"{prefix}z_mlp2", n_units= 4* n_embed, tau_m= tau_m, act_fx="gelu", batch_size=batch_size * seq_len)
        
        self.W_mlp1 = HebbianSynapse(f"{prefix}W_mlp1", shape=(n_embed, 4*n_embed), batch_size = batch_size * seq_len, eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=1.0, key=subkeys[4])
        self.W_mlp2 = HebbianSynapse(
                    f"{prefix}W_mlp2", shape=(4*n_embed, n_embed), batch_size= batch_size * seq_len, eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=1.0, key=subkeys[5])
        self.e_mlp = ErrorCell(f"{prefix}e_mlp", n_units=n_embed, 
                                  batch_size=batch_size * seq_len) # shape=(seq_len, n_embed, 1),   
        self.e_mlp1 = ErrorCell(f"{prefix}e_mlp1", n_units= 4* n_embed, 
                                  batch_size=batch_size * seq_len)
        
        
        self.E_mlp1 = StaticSynapse(f"{prefix}E_mlp1", shape=(4 * n_embed,n_embed), weight_init=dist.uniform(low=wlb, high=wub), key=subkeys[4])
        self.E_mlp = StaticSynapse(f"{prefix}E_mlp", shape=(n_embed, 4 * n_embed), weight_init=dist.uniform(low=wlb, high=wub), key=subkeys[4])
    def get_components(self):
        """Return all components for easy access"""
        return {
            'z_mlp': self.z_mlp,
            'W_mlp1': self.W_mlp1,
            'W_mlp2': self.W_mlp2,
            'e_mlp': self.e_mlp,
            'E_mlp': self.E_mlp,
            'E_mlp1': self.E_mlp1,
            'e_mlp1': self.e_mlp1,
            'z_mlp2': self.z_mlp2
        }

                