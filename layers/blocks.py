from config import Config as config
from layers.attention import Attention
from layers.mlp import MLP
from jax import random
import jax.numpy as jnp
from utils.model_util import ReshapeComponent
from utils.rms_norm_util import RMSNorm, RMSNormGrad


class Block:
    """
    One transformer block with RMSNorm in both forward and backward paths.

    Forward:
        z_qkv.zF  →  ln1  →  W_q / W_k / W_v  →  attn_block  →  ...
        z_mlp.zF  →  ln2  →  W_mlp1            →  z_mlp2      →  ...

    Backward 
        For Q path:
            ln1_grad_q.mu       ← z_qkv.z          (forward input to ln1)
            ln1_grad_q.rms      ← ln1.rms           (saved rms)
            ln1_grad_q.dmu      ← e_attn.dmu        (incoming error)
            ln1_grad_q.dmu_attn ← attn_block.dq     (Q-specific gradient)
            ln1_grad_q.dmu_     →  E_q  →  z_qkv.jq (state update)
            ln1_grad_q.dmu_mlp1 →  W_q.post          (Hebbian post)

        Same pattern for K and V paths using ln1_grad_k, ln1_grad_v.

        For MLP path:
            ln2_grad.mu         ← z_mlp.z
            ln2_grad.rms        ← ln2.rms
            ln2_grad.dmu        ← e_mlp1.dmu
            (dmu_attn not wired — stays ones)
            ln2_grad.dmu_       →  E_mlp1  →  z_mlp.j
            ln2_grad.dmu_       →  W_mlp1.post
    """

    def __init__(self, dkey, block_id, n_embed, seq_len, vocab_size,
                 batch_size, n_heads, dropout_rate, eta, optim_type,
                 wub, wlb, tau_m, **kwargs):

        dkey, attn_key, mlp_key = random.split(dkey, 3)
        prefix = f"block{block_id}_"
        bs     = batch_size * seq_len

        # ── Layer norms (forward) ─────────────────────────────────────────────
        self.ln1 = RMSNorm(f"{prefix}ln1", n_embed=n_embed, batch_size=bs)
        self.ln2 = RMSNorm(f"{prefix}ln2", n_embed=n_embed, batch_size=bs)

        # ── Backward norm-grad — three for attention (Q/K/V), one for MLP ─────
        self.ln1_grad_q = RMSNormGrad(f"{prefix}ln1_grad_q", n_embed=n_embed,
                                      batch_size=bs, gamma=self.ln1.gamma)
        self.ln1_grad_k = RMSNormGrad(f"{prefix}ln1_grad_k", n_embed=n_embed,
                                      batch_size=bs, gamma=self.ln1.gamma)
        self.ln1_grad_v = RMSNormGrad(f"{prefix}ln1_grad_v", n_embed=n_embed,
                                      batch_size=bs, gamma=self.ln1.gamma)
        # ln2_grad: dmu_attn never wired → stays ones → MLP path unaffected
        self.ln2_grad = RMSNormGrad(f"{prefix}ln2_grad", n_embed=n_embed,
                                    batch_size=bs, gamma=self.ln2.gamma)

        # ── Attention and MLP sub-layers ──────────────────────────────────────
        self.attention = Attention(
            dkey=attn_key, n_embed=n_embed, seq_len=seq_len,
            batch_size=batch_size, n_heads=n_heads,
            dropout_rate=dropout_rate, eta=eta, optim_type=optim_type,
            wub=wub, wlb=wlb, prefix=prefix, tau_m=tau_m)

        self.mlp = MLP(
            dkey=mlp_key, n_embed=n_embed, seq_len=seq_len,
            batch_size=batch_size, eta=eta, optim_type=optim_type,
            wub=wub, wlb=wlb, prefix=prefix, tau_m=tau_m)

        # ── Reshape helpers ───────────────────────────────────────────────────
        self.reshape_2d_to_3d_q = ReshapeComponent(
            f"{prefix}reshape_2d_to_3d_q",
            input_shape=(bs, n_embed), output_shape=(batch_size, seq_len, n_embed))
        self.reshape_2d_to_3d_k = ReshapeComponent(
            f"{prefix}reshape_2d_to_3d_k",
            input_shape=(bs, n_embed), output_shape=(batch_size, seq_len, n_embed))
        self.reshape_2d_to_3d_v = ReshapeComponent(
            f"{prefix}reshape_2d_to_3d_v",
            input_shape=(bs, n_embed), output_shape=(batch_size, seq_len, n_embed))
        self.reshape_3d_to_2d_attnout = ReshapeComponent(
            f"{prefix}reshape_3d_to_2d_attnout",
            input_shape=(batch_size, seq_len, n_embed), output_shape=(bs, n_embed))
        self.reshape_3d_to_2d = ReshapeComponent(
            f"{prefix}reshape_3d_to_2d",
            input_shape=(batch_size, seq_len, n_embed), output_shape=(bs, n_embed))