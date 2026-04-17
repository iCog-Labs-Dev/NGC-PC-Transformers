from config import Config as config
from layers.attention import Attention
from layers.mlp import MLP
from jax import random
import jax.numpy as jnp
from utils.model_util import ReshapeComponent
from utils.rms_norm_util import RMSNorm, RMSNormGrad


class Block:
    """
    One transformer block: ln1 → Attention → ln2 → MLP

    Layer norm placement
    ─────────────────────
    Forward:
        z_qkv.zF  →  ln1  →  W_q / W_k / W_v  →  attn_block  →  ...
        z_mlp.zF  →  ln2  →  W_mlp1            →  z_mlp2      →  ...

    Backward (new — mentor task):
        For each of Q, K, V:
            e_attn.dmu  →  ln1_grad_q/k/v.dmu       (incoming error)
            ln1.inputs  →  ln1_grad_q/k/v.mu         (saved forward x)
            ln1.rms     →  ln1_grad_q/k/v.rms        (saved forward rms)
            attn_block.dq/dk/dv  →  ln1_grad_q/k/v.dmu_attn  ← NEW compartment
            ln1_grad_q/k/v.dmu_  →  z_qkv.jq/jk/jv  (corrected state update)
            ln1_grad_q/k/v.dmu_  →  W_q/k/v.post     (corrected Hebbian post)

        For MLP:
            e_mlp1.dmu  →  ln2_grad.dmu
            ln2.inputs  →  ln2_grad.mu
            ln2.rms     →  ln2_grad.rms
            (ln2_grad.dmu_attn left unwired — stays ones, so multiply is no-op)
            ln2_grad.dmu_  →  z_mlp.j
            ln2_grad.dmu_  →  W_mlp1.post
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

        # ── Backward norm-grad components ─────────────────────────────────────
        # Three separate instances for Q / K / V paths.
        # Each receives its own attention gradient (dq / dk / dv) via
        # the new .dmu_attn compartment in RMSNormGrad.
        self.ln1_grad_q = RMSNormGrad(
            f"{prefix}ln1_grad_q", n_embed=n_embed,
            batch_size=bs, gamma=self.ln1.gamma)
        self.ln1_grad_k = RMSNormGrad(
            f"{prefix}ln1_grad_k", n_embed=n_embed,
            batch_size=bs, gamma=self.ln1.gamma)
        self.ln1_grad_v = RMSNormGrad(
            f"{prefix}ln1_grad_v", n_embed=n_embed,
            batch_size=bs, gamma=self.ln1.gamma)

        # One instance for the MLP path.
        # dmu_attn is never wired → stays ones → multiply is identity.
        self.ln2_grad = RMSNormGrad(
            f"{prefix}ln2_grad", n_embed=n_embed,
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
            input_shape=(bs, n_embed),
            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_2d_to_3d_k = ReshapeComponent(
            f"{prefix}reshape_2d_to_3d_k",
            input_shape=(bs, n_embed),
            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_2d_to_3d_v = ReshapeComponent(
            f"{prefix}reshape_2d_to_3d_v",
            input_shape=(bs, n_embed),
            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_3d_to_2d_attnout = ReshapeComponent(
            f"{prefix}reshape_3d_to_2d_attnout",
            input_shape=(batch_size, seq_len, n_embed),
            output_shape=(bs, n_embed))
        self.reshape_3d_to_2d = ReshapeComponent(
            f"{prefix}reshape_3d_to_2d",
            input_shape=(batch_size, seq_len, n_embed),
            output_shape=(bs, n_embed))