#import
import jax
from ngclearn import Context, MethodProcess
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config
from layers.embedding import EMBEDDING
from layers.attention import Attention
from layers.blocks import Block
from utils.attention_utils import AttentionBlock
from utils.embed_utils import EmbeddingSynapse
from layers.mlp import MLP
from layers.output import Output
from utils.model_util import ReshapeComponent
from projection.projection import Projection
import numpy as np



class NGCTransformer:
    """
    Predictive Coding Transformer following PCN architecture from:
    Whittington & Bogacz (2017) - "An approximation of the error backpropagation 
    algorithm in a predictive coding network with local hebbian synaptic plasticity"

    Architecture:
    z_embed -(W_embed)-> e_embed, z_qkv -(W_q,W_k,W_v - > W_attn_out)-> e_attn, z_mlp -(W_mlp1,W_mlp2)-> e_mlp, z_out -(W_out)-> e_out
    e_attn -(E_attn)-> z_qkv <- e_embed, e_mlp -(E_mlp)-> z_mlp <- e_attn, e_out -(E_out)-> z_out <- e_mlp

    Args:
        dkey: JAX seeding key
        vocab_size: vocabulary size
        seq_len: sequence length
        n_embed: embedding dimension
        n_heads: number of attention heads
        batch_size: batch size
        n_layers: number of transformer blocks
        dt: integration time constant
        tau_m: membrane time constant
        eta: learning rate for Hebbian synapses
        exp_dir: experimental directory
        model_name: unique model name
    """

   
    def __init__(self, dkey, batch_size, seq_len, n_embed, vocab_size, n_layers, n_heads, T, dt, tau_m, act_fx, eta, dropout_rate, exp_dir, model_name, loadDir=None, pos_learnable=False, optim_type="adam", wub=1.0, wlb=0.0, **kwargs):

        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        self.n_layers = n_layers
        self.T = T
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 50)
       
        with Context("Circuit") as self.circuit:
                
            self.embedding = EMBEDDING(dkey=subkeys[0], vocab_size=vocab_size, seq_len=seq_len, embed_dim=n_embed, batch_size=batch_size, pos_learnable=pos_learnable, eta=eta, optim_type=optim_type)
                
            self.blocks = []
            for i in range(n_layers):
                key, subkey = random.split(subkeys[1 + i])
                block=Block(dkey=subkey, block_id= i, n_embed=n_embed, seq_len=seq_len,
                                batch_size=batch_size, vocab_size=vocab_size, n_heads=n_heads, dropout_rate=dropout_rate, eta=eta, optim_type=optim_type, wub=wub, wlb=wlb, tau_m=tau_m)
                self.blocks.append(block)   
                    
            self.output = Output(dkey=subkeys[3], n_embed=n_embed, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, eta=eta, optim_type=optim_type, wlb=wlb, wub=wub, tau_m=tau_m)
                
            self.z_target=RateCell("z_target", n_units= vocab_size, tau_m=0., act_fx="identity", batch_size=batch_size * seq_len) 
            self.z_actfx= RateCell("z_actfx", n_units= vocab_size, tau_m=0., act_fx="softmax", batch_size=batch_size * seq_len)
            self.projection = Projection(dkey=subkeys[29], n_embed=n_embed, seq_len=seq_len, batch_size=batch_size,
                                             vocab_size=vocab_size, eta=eta, optim_type=optim_type, pos_learnable=pos_learnable, wub=wub, wlb=wlb, n_blocks=n_layers, n_heads=n_heads, dropout_rate=dropout_rate)
            self.reshape_4d_to_2d = ReshapeComponent("reshape_4d_to_2d",
                                            input_shape=(batch_size, seq_len, n_embed, 1),
                                            output_shape=(batch_size * seq_len, n_embed))
                
            self.reshape_3d_to_2d_embed = ReshapeComponent("reshape_3d_to_2d_embed",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))
            self.reshape_2d_to_3d_embed= ReshapeComponent("reshape_2d_to_3d_embed",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
                
                
        if loadDir is not None:
   
            self.load_from_disk(loadDir,n_layers=n_layers)
          
        else:
            with Context("Circuit") as self.circuit:
            
                
                self.embedding.W_embed.inputs >> self.embedding.z_embed.zF  
                self.reshape_3d_to_2d_embed.inputs >> self.embedding.W_embed.outputs   
                self.embedding.e_embed.mu >> self.reshape_3d_to_2d_embed.outputs
                self.embedding.e_embed.target >> self.blocks[0].attention.z_qkv.z
                
                # self.reshape_4d_to_2d.inputs >> self.attention.z_qkv.zF
                for blocks in range(n_layers):
                    block= self.blocks[blocks]
                    block.attention.z_qkv.zF  >>  block.attention.W_q.inputs
                    block.attention.z_qkv.zF >>   block.attention.W_k.inputs 
                    block.attention.z_qkv.zF >>   block.attention.W_v.inputs
                    
                    block.attention.W_q.outputs >> block.reshape_2d_to_3d_q.inputs 
                    block.attention.W_k.outputs >> block.reshape_2d_to_3d_k.inputs 
                    block.attention.W_v.outputs >> block.reshape_2d_to_3d_v.inputs 
                    
                    block.reshape_2d_to_3d_q.outputs >> block.attention.attn_block.inputs_q
                    block.reshape_2d_to_3d_k.outputs >> block.attention.attn_block.inputs_k
                    block.reshape_2d_to_3d_v.outputs >> block.attention.attn_block.inputs_v
                    block.attention.attn_block.outputs >> block.reshape_3d_to_2d.inputs

                    block.reshape_3d_to_2d.outputs >> block.attention.W_attn_out.inputs
                    block.attention.W_attn_out.outputs >> block.attention.e_attn.mu
                    block.mlp.z_mlp.z >> block.attention.e_attn.target


                    block.mlp.z_mlp.zF >> block.mlp.W_mlp1.inputs
                    block.mlp.W_mlp1.outputs >> block.mlp.e_mlp1.mu
                    block.mlp.z_mlp2.z >> block.mlp.e_mlp1.target


                    block.mlp.z_mlp2.zF >> block.mlp.W_mlp2.inputs
                    block.mlp.W_mlp2.outputs >> block.mlp.e_mlp.mu

     
                    
                    if blocks == n_layers - 1:
                        self.output.z_out.z >> block.mlp.e_mlp.target
                    else:
                        self.blocks[blocks + 1].attention.z_qkv.z >> block.mlp.e_mlp.target


                    block.attention.e_attn.dmu >> block.attention.E_attn.inputs

                    block.mlp.e_mlp1.dmu >> block.mlp.E_mlp1.inputs
                    block.mlp.e_mlp.dmu  >> block.mlp.E_mlp.inputs

                    block.attention.E_attn.outputs >> block.attention.z_qkv.j


                    if blocks == 0:
                        self.embedding.e_embed.dtarget >> block.attention.z_qkv.j_td
                    else:
                        block.mlp.e_mlp.dtarget >> block.attention.z_qkv.j_td


                    block.mlp.E_mlp.outputs  >> block.mlp.z_mlp2.j
                    block.mlp.E_mlp1.outputs >> block.mlp.z_mlp.j

                    block.attention.e_attn.dtarget >> block.mlp.z_mlp.j_td
                    block.mlp.e_mlp1.dtarget       >> block.mlp.z_mlp2.j_td


                    block.attention.z_qkv.zF >> block.attention.W_q.pre
                    block.attention.e_attn.dmu >> block.attention.W_q.post

                    block.attention.z_qkv.zF >> block.attention.W_k.pre
                    block.attention.e_attn.dmu >> block.attention.W_k.post

                    block.attention.z_qkv.zF >> block.attention.W_v.pre
                    block.attention.e_attn.dmu >> block.attention.W_v.post


                    block.attention.attn_block.outputs >> block.reshape_3d_to_2d_attnout.inputs
                    block.reshape_3d_to_2d_attnout.outputs >> block.attention.W_attn_out.pre
                    block.attention.e_attn.dmu >> block.attention.W_attn_out.post


                    block.mlp.z_mlp.zF  >> block.mlp.W_mlp1.pre
                    block.mlp.e_mlp1.dmu >> block.mlp.W_mlp1.post

                    block.mlp.z_mlp2.zF >> block.mlp.W_mlp2.pre
                    block.mlp.e_mlp.dmu  >> block.mlp.W_mlp2.post

                        
                self.output.z_out.zF >> self.output.W_out.inputs
                self.output.W_out.outputs >> self.z_actfx.j

                self.z_actfx.zF >> self.output.e_out.mu
                self.z_target.z >> self.output.e_out.target

                self.output.e_out.dmu >> self.output.E_out.inputs


                self.output.E_out.outputs >> self.output.z_out.j
                self.blocks[n_layers - 1].mlp.e_mlp.dtarget >> self.output.z_out.j_td


                self.embedding.e_embed.dmu >> self.reshape_2d_to_3d_embed.inputs
                self.reshape_2d_to_3d_embed.outputs >> self.embedding.W_embed.post


                self.output.z_out.zF >> self.output.W_out.pre
                self.output.e_out.dmu >> self.output.W_out.post

                        
                        
                ## PROJECTION PHASE ##
                
                
                
                self.projection.q_embed_Ratecell.zF >> self.projection.Q_embed.inputs
                self.projection.Q_embed.outputs >> self.projection.reshape_3d_to_2d_proj.inputs

                for b in range(n_layers):
                    block_proj = self.projection.blocks[b]

                    if b == 0:
                        self.projection.reshape_3d_to_2d_proj.outputs >> block_proj.q_qkv_Ratecell.j
                    else:
                        self.projection.blocks[b - 1].Q_mlp2.outputs >> block_proj.q_qkv_Ratecell.j

                    block_proj.q_qkv_Ratecell.zF >> block_proj.Q_q.inputs
                    block_proj.q_qkv_Ratecell.zF >> block_proj.Q_k.inputs
                    block_proj.q_qkv_Ratecell.zF >> block_proj.Q_v.inputs

                    block_proj.Q_q.outputs >> block_proj.q_attn_block.inputs_q
                    block_proj.Q_k.outputs >> block_proj.q_attn_block.inputs_k
                    block_proj.Q_v.outputs >> block_proj.q_attn_block.inputs_v

                    block_proj.q_attn_block.outputs >> block_proj.reshape_3d_to_2d_proj1.inputs
                    block_proj.reshape_3d_to_2d_proj1.outputs >> block_proj.Q_attn_out.inputs
                    block_proj.Q_attn_out.outputs >> block_proj.q_mlp_Ratecell.j

                    block_proj.q_mlp_Ratecell.zF >> block_proj.Q_mlp1.inputs
                    block_proj.Q_mlp1.outputs >> block_proj.q_mlp2_Ratecell.j
                    block_proj.q_mlp2_Ratecell.zF >> block_proj.Q_mlp2.inputs

                self.projection.blocks[n_layers - 1].Q_mlp2.outputs >> self.projection.q_out_Ratecell.j
                self.projection.q_out_Ratecell.zF >> self.projection.Q_out.inputs
                self.projection.Q_out.outputs >> self.projection.q_target_Ratecell.j

                self.projection.q_target_Ratecell.z >> self.projection.eq_target.mu

                
                # Create the processes by iterating through all blocks
                advance_process = MethodProcess(name="advance_process")
                reset_process = MethodProcess(name="reset_process") 
                embedding_evolve_process = MethodProcess(name="embedding_evolve_process")
                                           
                evolve_process = MethodProcess(name="evolve_process")
                project_process = MethodProcess(name="project_process")
                embedding_evolve_process  >> self.embedding.W_embed.evolve



                advance_process >> self.embedding.z_embed.advance_state
                advance_process >> self.embedding.W_embed.advance_state
                advance_process >> self.reshape_3d_to_2d_embed.advance_state
                advance_process >> self.reshape_2d_to_3d_embed.advance_state
                advance_process >> self.embedding.e_embed.advance_state

                for i in range(n_layers):
                    block = self.blocks[i]
                    
                    advance_process >> block.attention.E_attn.advance_state
                    advance_process >> block.mlp.E_mlp.advance_state
                    advance_process >> block.attention.z_qkv.advance_state
                    advance_process >> block.mlp.z_mlp.advance_state
                    advance_process >> block.mlp.z_mlp2.advance_state
                    advance_process >> block.attention.W_q.advance_state
                    advance_process >> block.attention.W_k.advance_state
                    advance_process >> block.attention.W_v.advance_state
                    advance_process >> block.reshape_2d_to_3d_q.advance_state
                    advance_process >> block.reshape_2d_to_3d_k.advance_state
                    advance_process >> block.reshape_2d_to_3d_v.advance_state
                    advance_process >> block.attention.attn_block.advance_state
                    advance_process >> block.reshape_3d_to_2d.advance_state
                    advance_process >> block.reshape_3d_to_2d_attnout.advance_state
                    advance_process >> block.attention.W_attn_out.advance_state
                    advance_process >> block.mlp.W_mlp1.advance_state
                    advance_process >> block.mlp.W_mlp2.advance_state
                    advance_process >> block.attention.e_attn.advance_state
                    advance_process >> block.mlp.e_mlp.advance_state
                    
                    reset_process >> block.attention.z_qkv.reset
                    reset_process >> block.mlp.z_mlp.reset
                    reset_process >> block.mlp.z_mlp2.reset
                    reset_process >> block.attention.e_attn.reset
                    reset_process >> block.mlp.e_mlp.reset
                    reset_process >> block.mlp.e_mlp1.reset
                    reset_process >> block.reshape_3d_to_2d.reset
                    reset_process >> block.reshape_2d_to_3d_q.reset
                    reset_process >> block.reshape_2d_to_3d_k.reset
                    reset_process >> block.reshape_2d_to_3d_v.reset
                    reset_process >> block.reshape_3d_to_2d_attnout.reset
                    
                    evolve_process >> block.attention.W_q.evolve
                    evolve_process >> block.attention.W_k.evolve
                    evolve_process >> block.attention.W_v.evolve
                    evolve_process >> block.attention.W_attn_out.evolve
                    evolve_process >> block.mlp.W_mlp1.evolve
                    evolve_process >> block.mlp.W_mlp2.evolve

                # Add non-block components to advance_process, reset_process, evolve_process
                advance_process >> self.output.E_out.advance_state
                advance_process >> self.output.z_out.advance_state
                advance_process >> self.output.W_out.advance_state
                advance_process >> self.z_actfx.advance_state
                advance_process >> self.output.e_out.advance_state

                reset_process >> self.projection.q_embed_Ratecell.reset
                reset_process >> self.projection.q_out_Ratecell.reset
                reset_process >> self.projection.q_target_Ratecell.reset
                reset_process >> self.projection.eq_target.reset
                reset_process >> self.embedding.z_embed.reset
                reset_process >> self.output.z_out.reset
                reset_process >> self.z_target.reset
                reset_process >> self.z_actfx.reset
                reset_process >> self.embedding.e_embed.reset
                reset_process >> self.output.e_out.reset
                reset_process >> self.reshape_3d_to_2d_embed.reset
                reset_process >> self.reshape_2d_to_3d_embed.reset

                evolve_process >> self.output.W_out.evolve
                project_process >> self.projection.q_embed_Ratecell.advance_state
                project_process >> self.projection.Q_embed.advance_state
                project_process >> self.projection.reshape_3d_to_2d_proj.advance_state
                for b in range(n_layers):
                    block_proj= self.projection.blocks[b]
                    project_process >> block_proj.q_qkv_Ratecell.advance_state
                    project_process >> block_proj.Q_q.advance_state
                    project_process >> block_proj.Q_k.advance_state
                    project_process >> block_proj.Q_v.advance_state
                    project_process >> block_proj.q_attn_block.advance_state
                    project_process >> block_proj.reshape_3d_to_2d_proj1.advance_state
                    project_process >> block_proj.Q_attn_out.advance_state
                    project_process >> block_proj.q_mlp_Ratecell.advance_state
                    project_process >> block_proj.q_mlp2_Ratecell.advance_state
                    project_process >> block_proj.Q_mlp1.advance_state
                    project_process >> block_proj.Q_mlp2.advance_state
                    reset_process >> block_proj.q_qkv_Ratecell.reset
                    reset_process >> block_proj.q_attn_block.reset
                    reset_process >> block_proj.q_mlp_Ratecell.reset
                    reset_process >> block_proj.q_mlp2_Ratecell.reset 
                project_process >> self.projection.q_out_Ratecell.advance_state
                project_process >> self.projection.Q_out.advance_state
                project_process >> self.projection.q_target_Ratecell.advance_state
                project_process >> self.projection.eq_target.advance_state
                
               
                self.reset = reset_process
                self.advance = advance_process
                self.evolve = evolve_process
                self.project = project_process
                self.embedding_evolve=embedding_evolve_process

                


    
    def clamp_input(self,x):
        self.embedding.z_embed.j.set(x)
        self.projection.q_embed_Ratecell.j.set(x) 
        
    
    def clamp_target(self,y):
        self.z_target.j.set(y)

    
    def clamp_infer_target(self,y):
        self.projection.eq_target.target.set(y)
        
    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter get()s to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.embedding.W_embed.save(model_dir)
            self.blocks = []
            for j in range(self.n_layers):
                block = self.circuit.get_components(f"block{j}_W_q")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_k")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_v")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_attn_out")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_mlp1")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_mlp2")
                block.save(model_dir)    
            self.output.W_out.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            

    def load_from_disk(self, model_directory, n_layers=1):
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
       
        processes = self.circuit.get_objects_by_type("process")
        self.advance = processes.get("advance_process")
        self.reset   = processes.get("reset_process")
        self.evolve  = processes.get("evolve_process")
        self.project = processes.get("project_process")

        self.embedding_evolve = processes.get("embedding_evolve_process", self.evolve) 

        self.embedding.W_embed.word_weights.set(self.circuit.get_components("W_embed").word_weights.get())
        self.embedding.W_embed.pos_weights.set(self.circuit.get_components("W_embed").pos_weights.get())
        self.output.W_out.weights.set(self.circuit.get_components("W_out").weights.get())
        self.output.W_out.biases.set(self.circuit.get_components("W_out").biases.get())
      
        self.projection.q_out_Ratecell.z.set( self.circuit.get_components("q_out_Ratecell").z.get())
        self.projection.eq_target.dmu.set( self.circuit.get_components("eq_target").dmu.get())
        self.projection.eq_target.dtarget.set( self.circuit.get_components("eq_target").dtarget.get())
   
        self.projection.q_target_Ratecell.z.set(self.circuit.get_components("q_target").z.get())
        self.output.W_out.outputs.set( self.circuit.get_components("W_out").outputs.get())
        self.embedding.e_embed.L.set( self.circuit.get_components("e_embed").L.get())
        self.output.e_out.L.set( self.circuit.get_components("e_out").L.get())
        self.projection.reshape_3d_to_2d_proj.inputs.set(self.circuit.get_components("reshape_3d_to_2d_proj").inputs.get())
        self.projection.reshape_3d_to_2d_proj.outputs.set(self.circuit.get_components("reshape_3d_to_2d_proj").outputs.get())
      

        # --- B. Map Block Components (Loop) ---
        for i in range(n_layers):
         
            b_prefix = f"block{i}"
            p_prefix = f"block_proj{i}"
            block_proj= self.projection.blocks[i]
            block= self.blocks[i] 
            
            # --- Map Attention Sub-block ---
          
            block.attention.W_q.weights.set( self.circuit.get_components(f"{b_prefix}_W_q").weights.get())
            block.attention.W_k.weights.set( self.circuit.get_components(f"{b_prefix}_W_k").weights.get())
            block.attention.W_v.weights.set( self.circuit.get_components(f"{b_prefix}_W_v").weights.get())
            block.attention.W_q.biases.set( self.circuit.get_components(f"{b_prefix}_W_q").biases.get())
            block.attention.W_k.biases.set( self.circuit.get_components(f"{b_prefix}_W_k").biases.get())
            block.attention.W_v.biases.set( self.circuit.get_components(f"{b_prefix}_W_v").biases.get())
            block.attention.attn_block.inputs_q.set(self.circuit.get_components(f"{b_prefix}_attn_block").inputs_q.get())
            block.attention.attn_block.inputs_k.set(self.circuit.get_components(f"{b_prefix}_attn_block").inputs_k.get())
            block.attention.attn_block.inputs_v.set(self.circuit.get_components(f"{b_prefix}_attn_block").inputs_v.get())
            block.attention.W_attn_out.weights.set(self.circuit.get_components(f"{b_prefix}_W_attn_out").weights.get())
            block.attention.W_attn_out.biases.set(self.circuit.get_components(f"{b_prefix}_W_attn_out").biases.get())

            block.attention.e_attn.L.set(self.circuit.get_components(f"{b_prefix}_e_attn").L.get())
            block.mlp.e_mlp.L.set(self.circuit.get_components(f"{b_prefix}_e_mlp").L.get())
            block.mlp.e_mlp1.L.set(self.circuit.get_components(f"{b_prefix}_e_mlp1").L.get())
            # --- Map MLP Sub-block ---
            block.mlp.z_mlp.z.set(   self.circuit.get_components(f"{b_prefix}_z_mlp"))
            block.mlp.z_mlp2.z.set(  self.circuit.get_components(f"{b_prefix}_z_mlp2"))
            block.mlp.W_mlp1.weights.set(self.circuit.get_components(f"{b_prefix}_W_mlp1").weights.get())
            block.mlp.W_mlp2.weights.set(self.circuit.get_components(f"{b_prefix}_W_mlp2").weights.get())
            block.mlp.W_mlp1.biases.set(self.circuit.get_components(f"{b_prefix}_W_mlp1").biases.get())
            block.mlp.W_mlp2.biases.set(self.circuit.get_components(f"{b_prefix}_W_mlp2").biases.get())

            # --- Map Projection Block ---
            block_proj.q_qkv_Ratecell.z.set(  self.circuit.get_components(f"{p_prefix}_q_qkv_Ratecell").z.get())
            block_proj.q_mlp_Ratecell.z.set(  self.circuit.get_components(f"{p_prefix}_q_mlp_Ratecell").z.get())
            block_proj.q_mlp2_Ratecell.z.set(  self.circuit.get_components(f"{p_prefix}_q_mlp2_Ratecell").z.get())
            block_proj.reshape_3d_to_2d_proj1.inputs.set(self.circuit.get_components(f"{p_prefix}_reshape_3d_to_2d_proj1").inputs.get())
            
            block_proj.reshape_3d_to_2d_proj1.outputs.set(self.circuit.get_components(f"{p_prefix}_reshape_3d_to_2d_proj1").outputs.get())
            block_proj.q_attn_block = self.circuit.get_components(f"{p_prefix}_q_attn_block")
          
            
            
  



    def process(self, obs, lab, adapt_synapses=True):
        
   
     
        self.reset.run()
        self.projection.Q_embed.word_weights.set(self.embedding.W_embed.word_weights.get())
        if self.embedding.W_embed.pos_learnable:
           self.projection.Q_embed.pos_weights.set(self.embedding.W_embed.pos_weights.get())
        for i in range(self.n_layers):
            block_proj= self.projection.blocks[i]
            block= self.blocks[i] 
            block_proj.Q_q.weights.set(block.attention.W_q.weights.get())
            block_proj.Q_q.biases.set(block.attention.W_q.biases.get())
            block_proj.Q_k.weights.set(block.attention.W_k.weights.get())
            block_proj.Q_k.biases.set(block.attention.W_k.biases.get())
            block_proj.Q_v.weights.set(block.attention.W_v.weights.get())
            block_proj.Q_v.biases.set(block.attention.W_v.biases.get())
            block_proj.Q_attn_out.weights.set(block.attention.W_attn_out.weights.get())
            block_proj.q_attn_block.inputs_q.set(block.attention.attn_block.inputs_q.get())
            block_proj.q_attn_block.inputs_k.set(block.attention.attn_block.inputs_k.get())
            block_proj.q_attn_block.inputs_v.set(block.attention.attn_block.inputs_v.get())
            block_proj.Q_attn_out.biases.set(block.attention.W_attn_out.biases.get())
            block_proj.Q_mlp1.weights.set(block.mlp.W_mlp1.weights.get())
            block_proj.Q_mlp1.biases.set(block.mlp.W_mlp1.biases.get())
            block_proj.Q_mlp2.weights.set(block.mlp.W_mlp2.weights.get())
            block_proj.Q_mlp2.biases.set(block.mlp.W_mlp2.biases.get())

            block.attention.z_qkv.z.set(block_proj.q_qkv_Ratecell.z.get())
            block.mlp.z_mlp.z.set(block_proj.q_mlp_Ratecell.z.get())
            block.mlp.z_mlp2.z.set(block_proj.q_mlp2_Ratecell.z.get())
            block.attention.E_attn.weights.set(jnp.transpose(block.attention.W_attn_out.weights.get()))
            block.mlp.E_mlp.weights.set(jnp.transpose(block.mlp.W_mlp2.weights.get()))  
            block.mlp.E_mlp1.weights.set(jnp.transpose(block.mlp.W_mlp1.weights.get()))
  
        self.projection.Q_out.weights.set(self.output.W_out.weights.get())
        self.projection.Q_out.biases.set(self.output.W_out.biases.get())
        self.projection.q_target_Ratecell.j_td.set(jnp.zeros((config.batch_size * config.seq_len, config.vocab_size)))
        
       

        self.output.E_out.weights.set(jnp.transpose(self.output.W_out.weights.get()))
       
        self.clamp_input(obs)
        self.clamp_infer_target(lab)




        
        self.project.run(t=0., dt=1.)



        
        self.output.z_out.z.set(self.projection.q_out_Ratecell.z.get())
        self.output.e_out.dmu.set(self.projection.eq_target.dmu.get())
        self.output.e_out.dtarget.set(self.projection.eq_target.dtarget.get())
        
        
        ## get projected prediction (from the P-step)
        y_mu_inf = self.projection.q_target_Ratecell.z.get()
    
        EFE = 0. 
        y_mu = 0.
        if adapt_synapses:
            for ts in range(0, self.T):
        
                self.clamp_input(obs)
                self.clamp_target(lab)
             
                self.advance.run(t=ts,dt=1.)
           
        y_mu = self.output.W_out.outputs.get() 

        L1 = self.embedding.e_embed.L.get()
        L4 = self.output.e_out.L.get()
        
        block_errors = 0.
        for i in range(self.n_layers):
                block = self.blocks[i]
                block_errors += block.attention.e_attn.L.get() + block.mlp.e_mlp.L.get() + block.mlp.e_mlp1.L.get()

        EFE = L4 + block_errors + L1

        if adapt_synapses == True:
                self.embedding_evolve.run()
                self.evolve.run(t=self.T,dt=1.)
                
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE 

    def get_latents(self):
        return self.q_out_Ratecell.z.get()
  