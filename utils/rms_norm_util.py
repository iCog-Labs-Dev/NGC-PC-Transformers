from statistics import variance

import jax.numpy as jnp
from jax import jit

from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


@jit
def rms_normalize(x, gamma, eps=1e-6):
    """
    RMS Normalization - Normalizes the input using root mean square.
    """
    # Cast inputs to float32 for numerical stability during variance calc
    x_float = x.astype(jnp.float32)

    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    # compute reciprocal sqrt explicitly, b/c jax.numpy not provide rsqrt in this version
    #scale = 1.0 / jnp.sqrt(variance + eps)
    
    #scale = scale.astype(x.dtype)
    #gamma_casted = gamma.astype(x.dtype)

    #return x * scale * gamma_casted
    rms = jnp.sqrt(variance + eps).astype(x.dtype)   # (batch, 1)
    out = x * (1.0 / rms) * gamma.astype(x.dtype)
    return out, rms

@jit
def rms_norm_grad(x, rms, gamma, v):
    """
    RMSNorm Jacobian-vector product  dL/dx = J_RMSNorm(x)^T @ v
 
    Derivation:
        y_i = (x_i / rms) * gamma_i
        rms = sqrt( mean(x^2) + eps )
 
        dL/dx_i = (gamma_i / rms) * v_i
                - (gamma_i * x_i / rms^3) * (1/n) * sum_j(gamma_j * x_j * v_j)
 
    Simplified (factor out gamma/rms):
        dL/dx = (gamma / rms) * ( v - x_norm * mean(x_norm * (gamma * v)) )
        where x_norm = x / rms
 
    Args:
        x     : forward input,        shape (batch, n_embed)
        rms   : saved rms from fwd,   shape (batch, 1)
        gamma : scale vector,          shape (n_embed,)
        v     : incoming gradient,    shape (batch, n_embed)
 
    Returns:
        dL/dx  same shape as x
    """
    gamma = gamma.astype(x.dtype)
    x_norm = x / rms                                          # (batch, n_embed)
    #gv     = gamma * v
    gv     = gamma.reshape((1,) * (v.ndim - 1) + (-1,)) * v                                        # (batch, n_embed)
    inner  = jnp.mean(x_norm * gv, axis=-1, keepdims=True)  # (batch, 1)
    #dx     = (gamma / rms) * (v - x_norm * inner)
    dx     = (gamma.reshape((1,) * (v.ndim - 1) + (-1,)) / rms) * (v - x_norm * inner)
    return dx


class RMSNorm(JaxComponent):
    """A small ngclearn-compatible RMS normalization component.
    
    Parameters
    - n_embed: number of features along last axis
    - batch_size: The effective batch size (B * S)
    """

    def __init__(self, name, n_embed, batch_size, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed = n_embed
        self.batch_size = batch_size
        self.gamma = jnp.ones((n_embed,))

        self.inputs = Compartment(jnp.zeros((batch_size, n_embed)))
        self.outputs = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms     = Compartment(jnp.ones((batch_size, 1)))

    @compilable
    def advance_state(self):
        x = self.inputs.get()
        # apply RMS normalization across last axis
        out, rms = rms_normalize(x, self.gamma)
        self.outputs.set(out)
        self.rms.set(rms)

    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros((self.batch_size, self.n_embed)))
        self.outputs.set(jnp.zeros((self.batch_size, self.n_embed)))
        self.rms.set(jnp.ones((self.batch_size, 1)))


class RMSNormGrad(JaxComponent):
    """
    Backward counterpart of RMSNorm.
 
    Computes the corrected gradient that accounts for the RMSNorm layer:
        dmu_ = J_RMSNorm(x)^T @ dmu
 
    Wiring pattern (mirrors Outgrad):
        ln.inputs  >> ln_grad.mu      # forward x saved during advance
        ln.rms     >> ln_grad.rms     # rms saved during advance
        e_cell.dmu >> ln_grad.dmu     # incoming error signal
        ln_grad.dmu_ >> z.j           # corrected gradient into hidden state
    """
 
    def __init__(self, name, n_embed, batch_size, gamma=None, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed    = n_embed
        self.batch_size = batch_size
        self.gamma      = gamma if gamma is not None else jnp.ones((n_embed,))
 
        self.mu   = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms  = Compartment(jnp.ones((batch_size, 1)))
        self.dmu  = Compartment(jnp.zeros((batch_size, n_embed)))
        self.dmu_ = Compartment(jnp.zeros((batch_size, n_embed)))
 
    @compilable
    def advance_state(self):
        x   = self.mu.get()
        rms = self.rms.get()
        v   = self.dmu.get()
        dx  = rms_norm_grad(x, rms, self.gamma, v)
        self.dmu_.set(dx)
 
    @compilable
    def reset(self):
        self.mu.set(jnp.zeros((self.batch_size, self.n_embed)))
        self.rms.set(jnp.ones((self.batch_size, 1)))
        self.dmu.set(jnp.zeros((self.batch_size, self.n_embed)))
        self.dmu_.set(jnp.zeros((self.batch_size, self.n_embed)))