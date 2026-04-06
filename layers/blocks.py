from config import Config as config
from layers.attention import Attention
from layers.mlp import MLP
from jax import random
import jax.numpy as jnp
from utils.model_util import ReshapeComponent
from utils.rms_norm_util import RMSNorm, RMSNormGrad

class Block:
    def __init__(self, dkey, block_id, n_embed, seq_len, vocab_size,
                 batch_size, n_heads, dropout_rate, eta, optim_type, wub, wlb, tau_m, **kwargs):
        
        dkey, attn_key, mlp_key = random.split(dkey, 3)
        prefix = f"block{block_id}_"

        # Layer norms (forward) 

        self.ln1 = RMSNorm(f"{prefix}ln1", n_embed=n_embed, batch_size= batch_size * seq_len)
        self.ln2 = RMSNorm(f"{prefix}ln2", n_embed=n_embed, batch_size= batch_size * seq_len)

        #Backward norm-grad components 
        self.ln1_grad = RMSNormGrad(f"{prefix}ln1_grad", n_embed=n_embed,
                                    batch_size=batch_size * seq_len,
                                    gamma=self.ln1.gamma)
        # self.ln2_grad = RMSNormGrad(f"{prefix}ln2_grad", n_embed=n_embed,
        #                             batch_size=batch_size * seq_len,
        #                             gamma=self.ln2.gamma)
        #  Attention and MLP sub-layers

        self.attention = Attention(dkey=attn_key, n_embed=n_embed, seq_len=seq_len,
                                 batch_size=batch_size, n_heads=n_heads,
                                 dropout_rate=dropout_rate, eta=eta, optim_type= optim_type, wub=wub, wlb=wlb, prefix=prefix, tau_m=tau_m)
        
        self.mlp = MLP(dkey=mlp_key, n_embed=n_embed, seq_len=seq_len,
                      batch_size=batch_size, eta=eta, optim_type=optim_type, wub=wub, wlb=wlb, prefix=prefix, tau_m=tau_m)

         # reshape helpers
        self.reshape_2d_to_3d_q = ReshapeComponent(f"{prefix}reshape_2d_to_3d_q",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_2d_to_3d_k = ReshapeComponent(f"{prefix}reshape_2d_to_3d_k",    
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))    
        self.reshape_2d_to_3d_v = ReshapeComponent(f"{prefix}reshape_2d_to_3d_v",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_3d_to_2d_attnout= ReshapeComponent(f"{prefix}reshape_3d_to_2d_attnout",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))
        self.reshape_3d_to_2d = ReshapeComponent(f"{prefix}reshape_3d_to_2d",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))        