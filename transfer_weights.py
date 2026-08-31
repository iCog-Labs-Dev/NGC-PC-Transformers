"""
transfer_weights.py
===================
Transfers trained weights from the backprop Transformer checkpoint (.npz)
into the NGC-PC Transformer model, then saves it to disk.

After running this script, you can launch `train.py` or `eval.py` with
`loadDir="exp"` and the NGC-PC model will start from pre-trained weights
rather than random initialisation.

Usage
-----
    # 1. Train the backprop transformer first
    python backprop_transformer.py

    # 2. Transfer weights to NGC-PC model
    python transfer_weights.py

    # 3. Fine-tune with PC / evaluate
    python train.py       # starts from pre-trained weights
    python eval.py
    python generation.py

Weight Mapping
--------------
    Backprop param key          →  NGC-PC component attribute
    ─────────────────────────────────────────────────────────
    embed/word                  →  W_embed.word_weights
    embed/pos                   →  W_embed.pos_weights
    blocks/i/W_q                →  blocks[i].attention.W_q.weights
    blocks/i/b_q                →  blocks[i].attention.W_q.biases
    blocks/i/W_k                →  blocks[i].attention.W_k.weights
    blocks/i/b_k                →  blocks[i].attention.W_k.biases
    blocks/i/W_v                →  blocks[i].attention.W_v.weights
    blocks/i/b_v                →  blocks[i].attention.W_v.biases
    blocks/i/W_o                →  blocks[i].attention.W_attn_out.weights
    blocks/i/b_o                →  blocks[i].attention.W_attn_out.biases
    blocks/i/W_mlp1             →  blocks[i].mlp.W_mlp1.weights
    blocks/i/b_mlp1             →  blocks[i].mlp.W_mlp1.biases
    blocks/i/W_mlp2             →  blocks[i].mlp.W_mlp2.weights
    blocks/i/b_mlp2             →  blocks[i].mlp.W_mlp2.biases
    out/W                       →  W_out.weights
    out/b                       →  W_out.biases
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from config import Config as config
from model import NGCTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BACKPROP_CKPT = Path("exp_backprop") / "best_params.npz"    # trained by backprop_transformer.py
NGC_SAVE_DIR  = "exp"                                        # standard NGC-PC save directory
NGC_MODEL_NAME = "ngc_transformer"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: verify shape compatibility before transferring
# ─────────────────────────────────────────────────────────────────────────────
def _check_shape(name, src, dst):
    if np.array(src).shape != np.array(dst.get()).shape:
        raise ValueError(
            f"Shape mismatch for '{name}': "
            f"backprop={np.array(src).shape}  NGC-PC={np.array(dst.get()).shape}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load backprop checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_backprop_checkpoint(path):
    """
    Load the flat .npz checkpoint saved by backprop_transformer.py.
    Returns a dict keyed by the flat parameter names (e.g. 'blocks/0/W_q').
    Weights are clipped to NGC-PC's [wlb, wub] bounds to prevent Hebbian
    dynamics from exploding (NaN) during the first settling iterations.
    """
    if not Path(str(path)).exists():
        raise FileNotFoundError(
            f"Backprop checkpoint not found at: {path}\n"
            "Run `python backprop_transformer.py` first."
        )
    data = np.load(str(path))

    # Clip all weight tensors to NGC-PC's expected range
    wlb = config.wlb  # -0.073186
    wub = config.wub  #  0.035284
    params = {}
    for k in data.files:
        v = jnp.array(data[k])
        # Only clip weight matrices (not biases — biases stay at 0 initially)
        if not k.endswith(('/b_q', '/b_k', '/b_v', '/b_o',
                            '/b_mlp1', '/b_mlp2', 'out/b')):
            v = jnp.clip(v, wlb, wub)
        params[k] = v

    print(f"[transfer] Loaded backprop checkpoint: {path}")
    print(f"           {len(params)} weight tensors  (clipped to [{wlb:.4f}, {wub:.4f}])")
    return params



# ─────────────────────────────────────────────────────────────────────────────
# Build NGC-PC model (fresh, no loadDir so weights are fresh placeholders)
# ─────────────────────────────────────────────────────────────────────────────
def build_ngc_model():
    dkey = jax.random.PRNGKey(0)
    model = NGCTransformer(
        dkey,
        batch_size    = config.batch_size,
        seq_len       = config.seq_len,
        n_embed       = config.n_embed,
        vocab_size    = config.vocab_size,
        n_layers      = config.n_layers,
        n_heads       = config.n_heads,
        T             = config.n_iter,
        dt            = 1.,
        tau_m         = config.tau_m,
        act_fx        = config.act_fx,
        eta           = config.eta,
        dropout_rate  = config.dropout_rate,
        exp_dir       = NGC_SAVE_DIR,
        loadDir       = None,          # fresh init — we will overwrite weights below
        pos_learnable = config.pos_learnable,
        optim_type    = config.optim_type,
        wub           = config.wub,
        wlb           = config.wlb,
        model_name    = NGC_MODEL_NAME,
        generate      = False,
    )
    print(f"[transfer] NGC-PC model built  ({model.count_parameters()/1e6:.3f} M params)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Transfer weights
# ─────────────────────────────────────────────────────────────────────────────
def transfer(model, bp):
    """
    Copy every backprop weight into the corresponding NGC-PC compartment.

    bp : flat dict from load_backprop_checkpoint()
    """
    n_transferred = 0

    # ── Embedding ─────────────────────────────────────────────────────────────
    _check_shape("W_embed.word_weights", bp['embed/word'], model.embedding.W_embed.word_weights)
    model.embedding.W_embed.word_weights.set(bp['embed/word'])
    print("  [✓] embed/word  →  W_embed.word_weights")
    n_transferred += 1

    _check_shape("W_embed.pos_weights", bp['embed/pos'], model.embedding.W_embed.pos_weights)
    model.embedding.W_embed.pos_weights.set(bp['embed/pos'])
    print("  [✓] embed/pos   →  W_embed.pos_weights")
    n_transferred += 1

    # Also mirror into the projection circuit (needed for single-sweep inference)
    model.projection.Q_embed.word_weights.set(bp['embed/word'])
    model.projection.Q_embed.pos_weights.set(bp['embed/pos'])
    print("  [✓] embed       →  projection Q_embed (mirror)")

    # ── Transformer Blocks ────────────────────────────────────────────────────
    for i, block in enumerate(model.blocks):
        blk = f"blocks/{i}"

        # Attention — Q
        _check_shape(f"block{i} W_q", bp[f'{blk}/W_q'], block.attention.W_q.weights)
        block.attention.W_q.weights.set(bp[f'{blk}/W_q'])
        block.attention.W_q.biases.set(bp[f'{blk}/b_q'])
        print(f"  [✓] {blk}/W_q,b_q   →  block{i}.attention.W_q")
        n_transferred += 2

        # Attention — K
        _check_shape(f"block{i} W_k", bp[f'{blk}/W_k'], block.attention.W_k.weights)
        block.attention.W_k.weights.set(bp[f'{blk}/W_k'])
        block.attention.W_k.biases.set(bp[f'{blk}/b_k'])
        print(f"  [✓] {blk}/W_k,b_k   →  block{i}.attention.W_k")
        n_transferred += 2

        # Attention — V
        _check_shape(f"block{i} W_v", bp[f'{blk}/W_v'], block.attention.W_v.weights)
        block.attention.W_v.weights.set(bp[f'{blk}/W_v'])
        block.attention.W_v.biases.set(bp[f'{blk}/b_v'])
        print(f"  [✓] {blk}/W_v,b_v   →  block{i}.attention.W_v")
        n_transferred += 2

        # Attention output projection
        _check_shape(f"block{i} W_attn_out", bp[f'{blk}/W_o'], block.attention.W_attn_out.weights)
        block.attention.W_attn_out.weights.set(bp[f'{blk}/W_o'])
        block.attention.W_attn_out.biases.set(bp[f'{blk}/b_o'])
        print(f"  [✓] {blk}/W_o,b_o   →  block{i}.attention.W_attn_out")
        n_transferred += 2

        # MLP — W_mlp1
        _check_shape(f"block{i} W_mlp1", bp[f'{blk}/W_mlp1'], block.mlp.W_mlp1.weights)
        block.mlp.W_mlp1.weights.set(bp[f'{blk}/W_mlp1'])
        block.mlp.W_mlp1.biases.set(bp[f'{blk}/b_mlp1'])
        print(f"  [✓] {blk}/W_mlp1,b  →  block{i}.mlp.W_mlp1")
        n_transferred += 2

        # MLP — W_mlp2
        _check_shape(f"block{i} W_mlp2", bp[f'{blk}/W_mlp2'], block.mlp.W_mlp2.weights)
        block.mlp.W_mlp2.weights.set(bp[f'{blk}/W_mlp2'])
        block.mlp.W_mlp2.biases.set(bp[f'{blk}/b_mlp2'])
        print(f"  [✓] {blk}/W_mlp2,b  →  block{i}.mlp.W_mlp2")
        n_transferred += 2

        # Mirror into projection circuit blocks as well
        pb = model.projection.blocks[i]
        pb.Q_q.weights.set(bp[f'{blk}/W_q'])
        pb.Q_q.biases.set(bp[f'{blk}/b_q'])
        pb.Q_k.weights.set(bp[f'{blk}/W_k'])
        pb.Q_k.biases.set(bp[f'{blk}/b_k'])
        pb.Q_v.weights.set(bp[f'{blk}/W_v'])
        pb.Q_v.biases.set(bp[f'{blk}/b_v'])
        pb.Q_attn_out.weights.set(bp[f'{blk}/W_o'])
        pb.Q_attn_out.biases.set(bp[f'{blk}/b_o'])
        pb.Q_mlp1.weights.set(bp[f'{blk}/W_mlp1'])
        pb.Q_mlp1.biases.set(bp[f'{blk}/b_mlp1'])
        pb.Q_mlp2.weights.set(bp[f'{blk}/W_mlp2'])
        pb.Q_mlp2.biases.set(bp[f'{blk}/b_mlp2'])
        print(f"  [✓] block{i} projection circuit mirrored")

    # ── Output head ───────────────────────────────────────────────────────────
    _check_shape("W_out.weights", bp['out/W'], model.output.W_out.weights)
    model.output.W_out.weights.set(bp['out/W'])
    model.output.W_out.biases.set(bp['out/b'])
    model.projection.Q_out.weights.set(bp['out/W'])
    model.projection.Q_out.biases.set(bp['out/b'])
    print("  [✓] out/W,b  →  W_out  (+ projection Q_out)")
    n_transferred += 2

    print(f"\n[transfer] Done. {n_transferred} weight tensors transferred.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Verification: compare a single forward pass
# ─────────────────────────────────────────────────────────────────────────────
def verify(model, bp):
    """
    Run one projection pass with a dummy input and report that the model
    is producing non-trivial (non-zero) output, confirming the transfer worked.
    """
    import jax.numpy as jnp
    dummy_input   = jnp.zeros((config.batch_size, config.seq_len), dtype=jnp.int32)
    dummy_target  = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))

    y_mu_inf, _, _ = model.process(dummy_input, dummy_target, adapt_synapses=False)

    max_logit = float(jnp.max(jnp.abs(y_mu_inf)))
    print(f"\n[verify] Max |logit| after transfer = {max_logit:.6f}  "
          f"({'OK — non-zero output' if max_logit > 1e-6 else 'WARNING — output is zero!'})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Backprop → NGC-PC Weight Transfer")
    print("=" * 60)

    # 1. Load backprop checkpoint
    bp = load_backprop_checkpoint(BACKPROP_CKPT)

    # 2. Build fresh NGC-PC model
    model = build_ngc_model()

    # 3. Transfer all weights
    print("\n[transfer] Transferring weights...\n")
    model = transfer(model, bp)

    # 4. Quick sanity check
    verify(model, bp)

    # 5. Save to exp/ so train.py / eval.py / generation.py can load it
    model.save_to_disk(params_only=False)
    print(f"\n[transfer] NGC-PC model saved to → {NGC_SAVE_DIR}/{NGC_MODEL_NAME}/")
    print("\nYou can now run:")
    print("  python train.py       # fine-tune with PC from pre-trained weights")
    print("  python eval.py        # evaluate pre-trained weights")
    print("  python generation.py  # generate text with pre-trained weights")


if __name__ == "__main__":
    main()
