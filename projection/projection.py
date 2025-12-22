from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, StaticSynapse
from utils.model_util import ReshapeComponent
from utils.embed_utils import EmbeddingSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from projection.proj_block import ProjBlock
from jax import random
class Projection():
    def __init__(self, dkey, n_embed, seq_len, batch_size, vocab_size, eta, optim_type, pos_learnable, wub, wlb, n_blocks, n_heads, dropout_rate,  **kwargs):
        dkey, *subkeys = random.split(dkey, 20)
        
        self.q_embed_Ratecell = RateCell("q_embed_Ratecell", n_units=seq_len, tau_m=0., act_fx="identity",
                            batch_size=batch_size)
        
        self.q_out_Ratecell = RateCell("q_out_Ratecell", n_units=n_embed, tau_m=0., act_fx="identity",
                          batch_size= batch_size * seq_len)
        self.q_target_Ratecell = RateCell("q_target", n_units=vocab_size, tau_m=0., act_fx="softmax",
                               batch_size=batch_size * seq_len)
               
        self.Q_embed = EmbeddingSynapse("Q_embed", vocab_size=vocab_size, seq_len=seq_len,
                                embed_dim=n_embed, batch_size= batch_size,
                                pos_learnable=pos_learnable, eta=eta,
                                 optim_type=optim_type, key=subkeys[5])
                
        self.blocks=[]
        for k in range(n_blocks):
            block = ProjBlock(dkey=subkeys[k+1],
                          block_id=k,
                          n_embed=n_embed,
                          seq_len=seq_len,
                          vocab_size=vocab_size,
                          batch_size=batch_size,
                          n_heads=n_heads,
                          dropout_rate=dropout_rate,
                          eta=eta,
                          optim_type=optim_type,
                          wub=wub,
                          wlb=wlb)
            self.blocks.append(block)       
        
        self.Q_out = StaticSynapse("Q_out", shape=(n_embed, vocab_size),  bias_init=dist.constant(value=0.), key=subkeys[12])
                
        self.eq_target = ErrorCell("eq_target", n_units=vocab_size, batch_size=batch_size * seq_len)
                
        self.reshape_3d_to_2d_proj= ReshapeComponent("reshape_3d_to_2d_proj",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))        
       