import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=[0, 1])
def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute RoPE cos/sin of shape [end, dim] for easy broadcasting.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)  # [end, dim//2]
    
    # Interleave to full dimension
    cos = jnp.zeros((end, dim))
    sin = jnp.zeros((end, dim))
    # Fill even and odd indices
    cos = cos.at[:, 0::2].set(jnp.cos(freqs))
    cos = cos.at[:, 1::2].set(jnp.cos(freqs))
    sin = sin.at[:, 0::2].set(jnp.sin(freqs))
    sin = sin.at[:, 1::2].set(jnp.sin(freqs))
    
    return cos, sin

@jit
def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    Used for the RoPE 'real' implementation trick.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    res = jnp.zeros_like(x)
    res = res.at[..., 0::2].set(-x2)
    res = res.at[..., 1::2].set(x1)
    return res

@jit
def apply_rotary_emb(xq, xk, cos, sin):
    """
    Apply rotary embeddings using the Sine-Cosine rewrite.
    xq, xk expected shape: (B, H, S, D) or (B, S, H, D) depending on integration.
    cos, sin expected shape: (S, D)
    """
    # Reshape cos/sin for broadcasting
    # Assuming xq, xk is (B, H, S, D) or (B, S, H, D), we need cos/sin to be alignable.
    # In attention_utils, q, k are flattened out or transposed to (B, H, S, D)
    cos = cos[None, None, :, :]  # shape: [1, 1, S, D]
    sin = sin[None, None, :, :]  # shape: [1, 1, S, D]
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out

@jit
def apply_rotary_emb_inv(xq_rot, xk_rot, cos, sin):
    """
    Inverse Rotary Embedding for Backpropagation / Error extraction
    """
    cos = cos[None, None, :, :]  # shape: [1, 1, S, D]
    sin = sin[None, None, :, :]  # shape: [1, 1, S, D]
    
    # The transpose of the rotation matrix is its inverse, 
    # which corresponds to replacing theta with -theta.
    # cos(-theta) = cos(theta), sin(-theta) = -sin(theta)
    xq_out = (xq_rot * cos) - (rotate_half(xq_rot) * sin)
    xk_out = (xk_rot * cos) - (rotate_half(xk_rot) * sin)
    return xq_out, xk_out
