from jax import random, numpy as jnp, jit
import jax
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable
from ngclearn.utils import tensorstats
import os
from pathlib import Path

@partial(jit, static_argnums=[0, 1])
def _create_sinusoidal_embeddings(seq_len, embed_dim):
    """Create fixed sinusoidal position embeddings"""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * 
                      (-jnp.log(10000.0) / embed_dim))
    
    embeddings = jnp.zeros((seq_len, embed_dim))
    embeddings = embeddings.at[:, 0::2].set(jnp.sin(position * div_term))
    embeddings = embeddings.at[:, 1::2].set(jnp.cos(position * div_term))
    return embeddings

@partial(jit, static_argnums=[4, 5, 6, 7])
def _compute_embedding_updates(inputs, post, word_weights, pos_weights, 
                              vocab_size, seq_len, embed_dim, batch_size, pos_learnable):
    """
    Compute updates for word and position embeddings
    """
    
    # Flatten for processing
    flat_tokens = inputs.reshape(-1)
    flat_errors = post.reshape(batch_size * seq_len, embed_dim)
     
    # Word embeddings update - accumulate gradients for each token
    d_word_weights = jnp.zeros((vocab_size, embed_dim))
    
    
    d_word_weights = d_word_weights.at[flat_tokens].add(flat_errors)
    
    # Position embeddings update (if learnable)
    d_pos_weights = jnp.zeros((seq_len, embed_dim))
    
    batch_positions = jnp.tile(jnp.arange(seq_len), batch_size).astype(jnp.int32)
    d_pos_weights = jax.lax.cond(
    pos_learnable, lambda: d_pos_weights.at[jnp.tile(jnp.arange(seq_len), batch_size)].add(flat_errors), lambda: d_pos_weights
    )
            
    return d_word_weights, d_pos_weights

class EmbeddingSynapse(JaxComponent):
    """
    A synaptic cable that handles both word and position embeddings.
    Combines word embeddings with learnable or fixed positional embeddings.

    | --- Synapse Compartments: ---
    | inputs - input token indices (takes in external signals)
    | outputs - output embedding signals (combined word + position embeddings)
    | word_weights - word embedding matrix
    | pos_weights - position embedding matrix  
    | post - post-synaptic signals for learning (takes in external signals)
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | dWordWeights - current delta matrix for word embedding changes
    | dPosWeights - current delta matrix for position embedding changes
    | word_opt_params - optimizer statistics for word embeddings
    | pos_opt_params - optimizer statistics for position embeddings

    Args:
        name: the string name of this component

        vocab_size: size of vocabulary for word embeddings

        seq_len: sequence length for position embeddings

        embed_dim: dimensionality of embeddings

        batch_size: batch size dimension

        pos_learnable: whether position embeddings are learnable or fixed

        eta: global learning rate 

        optim_type: optimization scheme (Default: "sgd")

        weight_scale: scaling factor for weight initialization (Default: 0.02)
    """

    def __init__(
            self, name, vocab_size, seq_len, embed_dim, batch_size,
            pos_learnable, eta, optim_type, weight_scale=0.02,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.pos_learnable = pos_learnable
        self.eta = eta
        self.weight_scale = weight_scale
        self.optim_type = optim_type

        key =random.PRNGKey(1234)
        word_key, pos_key = random.split(key, 2)
        
        word_weights = random.normal(word_key, (vocab_size, embed_dim)) * weight_scale
        
        if pos_learnable:
            pos_weights = random.normal(pos_key, (seq_len, embed_dim)) * weight_scale
        else:
            pos_weights = _create_sinusoidal_embeddings(seq_len, embed_dim)

        ## Compartments
        self.inputs = Compartment(jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, embed_dim)))
        self.word_weights = Compartment(word_weights)
        self.pos_weights = Compartment(pos_weights)
        self.post = Compartment(jnp.zeros((batch_size, seq_len, embed_dim)))
        
        self.dWordWeights = Compartment(jnp.zeros((vocab_size, embed_dim)))
        self.dPosWeights = Compartment(jnp.zeros((seq_len, embed_dim)))
        
        # Optimization
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)
        self.word_opt_params = Compartment(
            get_opt_init_fn(optim_type)([self.word_weights.get()])
        )
        if pos_learnable:
            self.pos_opt_params = Compartment(
                get_opt_init_fn(optim_type)([self.pos_weights.get()])
            )
        else:
            self.pos_opt_params = Compartment(None)
    @compilable
    def advance_state(self):
        """
        Forward pass: output = word_embedding[inputs] + position_embedding[positions]
        """
        inputs=self.inputs.get()
        word_weights=self.word_weights.get()
        pos_weights=self.pos_weights.get()
        seq_len=self.seq_len.get()
        embed_dim=self.embed_dim.get()
        batch_size = inputs.shape[0]
        
        flat_tokens = inputs.reshape(-1).astype(jnp.int32)
        word_embeds_flat = word_weights[flat_tokens]
        word_embeds = word_embeds_flat.reshape(batch_size, seq_len, embed_dim)
        
        positions = jnp.arange(seq_len)
        pos_embeds = pos_weights[positions]
        pos_embeds_batch = jnp.broadcast_to(pos_embeds, (batch_size, seq_len, embed_dim))
        
        combined_embeddings = word_embeds + pos_embeds_batch
        # return combined_embeddings
        self.outputs.set(combined_embeddings)

  
    @compilable
    def evolve(self):
        """
        Learning step: Hebbian updates for both word and position embeddings
        """
        opt = self.opt.get()
        # pos_learnable = self.pos_learnable.get()
        vocab_size = self.vocab_size.get()
        seq_len = self.seq_len.get()
        embed_dim = self.embed_dim.get()
        batch_size = self.batch_size.get()
        inputs = self.inputs.get()
        post = self.post.get()
        word_weights = self.word_weights.get()
        pos_weights = self.pos_weights.get()
        word_opt_params = self.word_opt_params.get()
        pos_opt_params = self.pos_opt_params.get()

        # Compute embedding updates
        inputs= inputs.astype(jnp.int32)
        d_word_weights, d_pos_weights = _compute_embedding_updates(
            inputs, post, word_weights, pos_weights, vocab_size, seq_len, 
            embed_dim, batch_size, self.pos_learnable
        )
        
        word_opt_params, [new_word_weights] = opt(
            word_opt_params, [word_weights], [d_word_weights]
        )
        
        new_pos_weights = pos_weights
        new_pos_opt_params = pos_opt_params
        
        if self.pos_learnable:
            pos_opt_params, [new_pos_weights] = opt(
                pos_opt_params, [pos_weights], [d_pos_weights]
            )
            new_pos_opt_params = pos_opt_params
        
        # return new_word_weights, new_pos_weights, d_word_weights, d_pos_weights, word_opt_params, new_pos_opt_params
        self.word_weights.set(new_word_weights)
        self.pos_weights.set(new_pos_weights)
        self.dWordWeights.set(d_word_weights)
        self.dPosWeights.set(d_pos_weights)
        self.word_opt_params.set(word_opt_params)
        self.pos_opt_params.set(new_pos_opt_params)
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """
        batch_size = self.batch_size.get()
        seq_len = self.seq_len.get()
        embed_dim = self.embed_dim.get()
        vocab_size = self.vocab_size.get()

        inputs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        outputs = jnp.zeros((batch_size, seq_len, embed_dim))
        post = jnp.zeros((batch_size, seq_len, embed_dim))
        dWordWeights = jnp.zeros((vocab_size, embed_dim))
        dPosWeights = jnp.zeros((seq_len, embed_dim))
        # return inputs, outputs, post, dWordWeights, dPosWeights
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.post.set(post)
        self.dWordWeights.set(dWordWeights)
        self.dPosWeights.set(dPosWeights)


    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "synapse_type": "EmbeddingSynapse - combines word and position embeddings "
                           "into a single embedding representation"
        }
        compartment_props = {
            "inputs": 
                {"inputs": "Input token indices (batch_size, seq_len)",
                 "post": "Post-synaptic error signals for learning"},
            "states":
                {"word_weights": "Word embedding matrix (vocab_size, embed_dim)",
                 "pos_weights": "Position embedding matrix (seq_len, embed_dim)",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWordWeights": "Word embedding adjustment matrix",
                 "dPosWeights": "Position embedding adjustment matrix"},
            "outputs":
                {"outputs": "Combined embeddings (batch_size, seq_len, embed_dim)"},
        }
        hyperparams = {
            "vocab_size": "Size of vocabulary",
            "seq_len": "Maximum sequence length", 
            "embed_dim": "Dimensionality of embeddings",
            "batch_size": "Batch size dimension",
            "pos_learnable": "Whether position embeddings are learnable",
            "eta": "Global learning rate",
            "optim_type": "Optimization scheme",
            "weight_scale": "Weight initialization scale"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = word_embedding[inputs] + position_embedding[positions]",
                "hyperparameters": hyperparams}
        return info
    def __repr__(self):
        # FIX: Replaced the non-existent Compartment.is_compartment with isinstance(..., Compartment)
        comps = [varname for varname in dir(self) if isinstance(getattr(self, varname), Compartment)]
        
        if not comps:
            # Handle the case where no compartments are found to avoid max() on an empty sequence
            return f"[{self.__class__.__name__}] PATH: {self.name}\n  No Compartments Found"

        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        
        # Iterate over the valid compartment names
        for c in comps:
            # Get the actual Compartment object
            compartment_obj = getattr(self, c) 
            
            # Get tensor statistics (assuming tensorstats is correctly imported)
            stats = tensorstats(compartment_obj.get())
            
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
                
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
            
        return lines


    def save(self, directory, **kwargs):
        """Save word and (optional) position embedding parameters to disk."""
        
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(directory, f"{self.name}.npz")
        
        if self.pos_learnable:
            jnp.savez(
                file_name,
                word_weights=self.word_weights.get(),
                pos_weights=self.pos_weights.get()
            )
        else:
            jnp.savez(
                file_name,
                word_weights=self.word_weights.get()
                # pos_weights are fixed (sinusoidal), so not saved
            )
      

    def load(self, directory, **kwargs):
        """Load word and (optional) position embedding parameters from disk."""
        import os
        file_name = os.path.join(directory, f"{self.name}.npz")
        data = jnp.load(file_name)
        
        self.word_weights.set(data['word_weights'])
        
        if self.pos_learnable and 'pos_weights' in data:
            self.pos_weights.set(data['pos_weights'])
        # If pos_learnable=False, pos_weights are recomputed via sinusoidal — no need to load