from config import Config as config
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
from utils.attention_utils import _compute_attention, AttentionBlock
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from layers.mlp import MLP
from jax import jit, random
import jax.numpy as jnp
from utils.model_util import ReshapeComponent


class ProjBlock:
    def __init__(self, dkey, block_id, n_embed, seq_len, vocab_size,
                 batch_size, n_heads, dropout_rate, eta, optim_type, wub, wlb, **kwargs):
        
        dkey, *subkeys = random.split(dkey, 20)
        prefix = f"block_proj{block_id}_"
     
        self.q_qkv_Ratecell = RateCell(f"{prefix}q_qkv_Ratecell", n_units=n_embed, tau_m=0., act_fx="identity",
                          batch_size=batch_size * seq_len)
        self.q_mlp_Ratecell = RateCell(f"{prefix}q_mlp_Ratecell", n_units= n_embed, tau_m=0., act_fx="identity",
                           batch_size= batch_size * seq_len)
        self.q_mlp2_Ratecell = RateCell(f"{prefix}q_mlp2_Ratecell", n_units=4 * n_embed, tau_m=0., act_fx="relu",
                           batch_size= batch_size * seq_len)
        self.Q_q = StaticSynapse(f"{prefix}Q_q", shape=(n_embed, n_embed),
                         bias_init=dist.constant(value=0.), key=subkeys[6])
                
        self.Q_k = StaticSynapse(f"{prefix}Q_k", shape=(n_embed, n_embed),
                          bias_init=dist.constant(value=0.), key=subkeys[7])
                
        self.Q_v = StaticSynapse(f"{prefix}Q_v", shape=(n_embed, n_embed),
                          bias_init=dist.constant(value=0.), key=subkeys[8])
        
        self.q_attn_block = AttentionBlock(f"{prefix}q_attn_block",

                                   n_heads=n_heads, n_embed=n_embed, seq_len=seq_len,
                                   dropout_rate=dropout_rate, batch_size=batch_size)
                
        
        self.Q_attn_out = StaticSynapse(f"{prefix}Q_attn_out", shape=(n_embed, n_embed),
                                 bias_init=dist.constant(value=0.),
 key=subkeys[0])
                
        self.Q_mlp1 = StaticSynapse(f"{prefix}Q_mlp1", shape=(n_embed, 4 * n_embed),
                             bias_init=dist.constant(value=0.), key=subkeys[0])
                
        self.Q_mlp2 = StaticSynapse(f"{prefix}Q_mlp2", shape=(4* n_embed, n_embed),
                             bias_init=dist.constant(value=0.), key=subkeys[0])
                        
        
        self.reshape_3d_to_2d_proj1= ReshapeComponent(f"{prefix}reshape_3d_to_2d_proj1",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))