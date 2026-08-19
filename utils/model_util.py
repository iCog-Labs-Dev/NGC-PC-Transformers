import jax
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable
from jax import numpy as jnp, random, jit
from functools import partial
from jax import vmap


def d_softmax_vjp(x, tau=0.0):
    """
    Memory-efficient softmax derivative using JVP (Jacobian-Vector Product).
    
    Returns probabilities and a function that computes J @ v without forming 
    the full Jacobian matrix.
    
    Args:
        x: Input tensor of any shape (attention scores, output logits, etc.)
           Examples:
           - Attention: (batch_size, n_heads, seq_len, seq_len)
           - Output: (batch_size, vocab_size) or (batch_size*seq_len, vocab_size)
        tau: Temperature parameter for softmax
    
    Returns:
        P: Softmax probabilities (same shape as x)
        jvp_fn: Function that computes J @ v for any v of same shape as P
               jvp_fn(v) = p * (v - (p @ v))
    """
    if tau > 0.0:
        x = x / tau
    
    # Compute probabilities once
    P = jax.nn.softmax(x, axis=-1)
    
    def jvp_fn(v):
        """
        Compute J @ v efficiently using the identity:
        J @ v = p * (v - (p @ v))
        
        Args:
            v: Vector to multiply Jacobian with (same shape as P)
               In practice, this is dL/dP from upstream
        
        Returns:
            J @ v with same shape as v (which is dL/dx)
        """
        # p @ v along last dimension (sum over that dimension)
        p_dot_v = jnp.sum(P * v, axis=-1, keepdims=True)
        return P * (v - p_dot_v)
    
    return P, jvp_fn

class ReshapeComponent(JaxComponent):
    """Component that reshapes tensors for ngc-learn wiring"""
    
    def __init__(self, name, input_shape, output_shape, **kwargs):
        super().__init__(name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.inputs = Compartment(jnp.zeros(input_shape))
        self.outputs = Compartment(jnp.zeros(output_shape))
    
    @compilable
    def advance_state(self):
        output=self.inputs.reshape(self.output_shape)
        self.outputs.set(output)
    
    
    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros(self.input_shape))
        self.outputs.set(jnp.zeros(self.output_shape))

class Outgrad(JaxComponent):
    """Compute the Jacobian matrix multiplication for the logits gradients
    This computes: dL/dmu = J_softmax(mu) @ dL/dP
    where mu are the logits (pre-softmax)
    """
    
    def __init__(self, name, batch_size, seq_len, vocab_size, **kwargs):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.mu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu_ = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
   
    @compilable   
    def advance_state(self):
        """Compute the output gradients: dL/dmu = J_softmax(mu) @ dL/dP"""
        
        mu = self.mu.get()        
        dmu = self.dmu.get()      
        
        P, jvp_fn = d_softmax_vjp(mu, tau=0.0)
        
        dmu_out = jvp_fn(dmu)
        
        self.dmu_.set(dmu_out)
        
    @compilable
    def reset(self):
        """Reset compartments to zeros"""
        zeros = jnp.zeros((self.batch_size * self.seq_len, self.vocab_size))
        self.mu.set(zeros)
        self.dmu.set(zeros)
        self.dmu_.set(zeros)