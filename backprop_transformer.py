"""
backprop_transformer.py
=======================
Standard backpropagation Transformer using pure JAX + Optax.

Architecture is deliberately kept IDENTICAL to the NGC-PC Transformer so
that every trained weight can be transferred 1-to-1 to the NGC-PC model
via `transfer_weights.py`.

Matches config.py:
    vocab_size  = 11711
    n_embed     = 128
    n_heads     = 8
    n_layers    = 4
    seq_len     = 32
    batch_size  = 8
    mlp_hidden  = 4 * n_embed = 512   (matches z_mlp2 shape in MLP layer)
    dropout     = 0.1
    act_fx(mlp) = GELU               (matches z_mlp2 act_fx='gelu')

No LayerNorm is used because the NGC-PC model has RMSNorm commented out.

Weight layout (params pytree):
    params['embed']['word']        (vocab_size, n_embed)
    params['embed']['pos']         (seq_len, n_embed)
    params['blocks'][i]['W_q']     (n_embed, n_embed)
    params['blocks'][i]['b_q']     (n_embed,)
    params['blocks'][i]['W_k']     (n_embed, n_embed)
    params['blocks'][i]['b_k']     (n_embed,)
    params['blocks'][i]['W_v']     (n_embed, n_embed)
    params['blocks'][i]['b_v']     (n_embed,)
    params['blocks'][i]['W_o']     (n_embed, n_embed)
    params['blocks'][i]['b_o']     (n_embed,)
    params['blocks'][i]['W_mlp1']  (n_embed, 4*n_embed)
    params['blocks'][i]['b_mlp1']  (4*n_embed,)
    params['blocks'][i]['W_mlp2']  (4*n_embed, n_embed)
    params['blocks'][i]['b_mlp2']  (n_embed,)
    params['out']['W']             (n_embed, vocab_size)
    params['out']['b']             (vocab_size,)
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
import optax
import numpy as np
from functools import partial
from config import Config as config
from data_preprocess.data_loader import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (mirror config.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE    = config.vocab_size      # 11711
N_EMBED       = config.n_embed         # 128
N_HEADS       = config.n_heads         # 8
N_LAYERS      = config.n_layers        # 4
SEQ_LEN       = config.seq_len         # 32
BATCH_SIZE    = config.batch_size      # 8
D_HEAD        = N_EMBED // N_HEADS     # 16
MLP_DIM       = 4 * N_EMBED            # 512
DROPOUT_RATE  = config.dropout_rate    # 0.1
EPOCHS        = config.epoch           # 5
LR            = 3e-4                   # AdamW learning rate

SAVE_DIR      = Path("exp_backprop")
SAVE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialisation
# ─────────────────────────────────────────────────────────────────────────────
def fan_in_gaussian(key, shape):
    """Fan-in Gaussian init matching NGC-PC's dist.fan_in_gaussian()."""
    fan_in = shape[0]
    std = 1.0 / jnp.sqrt(float(fan_in))
    return jax.random.normal(key, shape) * std


def init_params(key):
    """
    Initialise all parameters.
    Returns a nested Python dict of JAX arrays.
    """
    n_keys_needed = 2 + N_LAYERS * 6 + 2
    keys = jax.random.split(key, n_keys_needed)
    ki = 0

    params = {}

    # ── Embedding ────────────────────────────────────────────────────────────
    params['embed'] = {
        'word': fan_in_gaussian(keys[ki],   (VOCAB_SIZE, N_EMBED)),
        'pos' : fan_in_gaussian(keys[ki+1], (SEQ_LEN,    N_EMBED)),
    }
    ki += 2

    # ── Transformer Blocks ───────────────────────────────────────────────────
    params['blocks'] = []
    for _ in range(N_LAYERS):
        block = {
            # Attention projections
            'W_q':    fan_in_gaussian(keys[ki],   (N_EMBED, N_EMBED)),
            'b_q':    jnp.zeros((N_EMBED,)),
            'W_k':    fan_in_gaussian(keys[ki+1], (N_EMBED, N_EMBED)),
            'b_k':    jnp.zeros((N_EMBED,)),
            'W_v':    fan_in_gaussian(keys[ki+2], (N_EMBED, N_EMBED)),
            'b_v':    jnp.zeros((N_EMBED,)),
            # Attention output projection (W_attn_out in NGC-PC)
            'W_o':    fan_in_gaussian(keys[ki+3], (N_EMBED, N_EMBED)),
            'b_o':    jnp.zeros((N_EMBED,)),
            # MLP — W_mlp1 (embed → 4*embed, GELU) then W_mlp2 (4*embed → embed)
            'W_mlp1': fan_in_gaussian(keys[ki+4], (N_EMBED, MLP_DIM)),
            'b_mlp1': jnp.zeros((MLP_DIM,)),
            'W_mlp2': fan_in_gaussian(keys[ki+5], (MLP_DIM, N_EMBED)),
            'b_mlp2': jnp.zeros((N_EMBED,)),
        }
        params['blocks'].append(block)
        ki += 6

    # ── Output Head ──────────────────────────────────────────────────────────
    params['out'] = {
        'W': fan_in_gaussian(keys[ki],   (N_EMBED, VOCAB_SIZE)),
        'b': jnp.zeros((VOCAB_SIZE,)),
    }

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Forward Pass
# ─────────────────────────────────────────────────────────────────────────────
_causal_mask = jnp.tril(jnp.ones((SEQ_LEN, SEQ_LEN), dtype=bool))


def attention_block(block_params, x, training, key):
    """
    Multi-head causal self-attention.
    x : (B, S, D)  →  out : (B, S, D)
    """
    B, S, D = x.shape

    # Linear projections
    Q = x @ block_params['W_q'] + block_params['b_q']   # (B, S, D)
    K = x @ block_params['W_k'] + block_params['b_k']
    V = x @ block_params['W_v'] + block_params['b_v']

    # Split into heads: (B, H, S, d_head)
    Q = Q.reshape(B, S, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale  = jnp.sqrt(float(D_HEAD))
    scores = jnp.einsum('bhte,bhse->bhts', Q, K) / scale   # (B, H, S, S)

    # Causal mask
    scores = jnp.where(_causal_mask[None, None, :S, :S], scores, -1e9)
    attn   = jax.nn.softmax(scores, axis=-1)                # (B, H, S, S)

    # Attention dropout (training only)
    if training and DROPOUT_RATE > 0.0:
        key, subkey = jax.random.split(key)
        keep = jax.random.bernoulli(subkey, 1.0 - DROPOUT_RATE, attn.shape)
        attn = attn * keep / (1.0 - DROPOUT_RATE)

    # Weighted sum and reshape: (B, H, S, d) → (B, S, D)
    out = jnp.einsum('bhts,bhse->bhte', attn, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)

    # Output projection (W_attn_out equivalent)
    out = out @ block_params['W_o'] + block_params['b_o']
    return out


def mlp_block(block_params, x):
    """
    Two-layer MLP: Linear(D→4D, GELU) → Linear(4D→D).
    Matches NGC-PC: W_mlp1 (GELU activation on z_mlp2) → W_mlp2.
    x : (B, S, D)  →  out : (B, S, D)
    """
    h = x @ block_params['W_mlp1'] + block_params['b_mlp1']  # (B, S, 4D)
    h = jax.nn.gelu(h)
    return h @ block_params['W_mlp2'] + block_params['b_mlp2']  # (B, S, D)


def forward(params, tokens, training, key):
    """
    Full transformer forward pass.
    tokens  : (B, S)  integer token ids
    returns : logits  (B, S, vocab_size)
    """
    B, S = tokens.shape

    # Token + positional embeddings
    x = params['embed']['word'][tokens]                   # (B, S, D)
    x = x + params['embed']['pos'][None, :S, :]          # (B, S, D)

    # Transformer blocks — no residual by default (matching use_residual=False)
    # Note: standard transformers do use residual; we add residual here to match
    # the mathematical function computed by the NGC-PC model at convergence.
    for block_params in params['blocks']:
        key, k_attn = jax.random.split(key)
        x = x + attention_block(block_params, x, training, k_attn)
        x = x + mlp_block(block_params, x)

    # Output logits
    logits = x @ params['out']['W'] + params['out']['b']  # (B, S, V)
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(logits, targets, mask=None):
    """
    logits  : (B, S, vocab_size)
    targets : (B, S)  integer token ids
    mask    : (B, S)  float32, 1=valid token, 0=padding
    """
    B, S, V = logits.shape
    log_probs = jax.nn.log_softmax(logits, axis=-1)          # (B, S, V)
    target_log_probs = log_probs[jnp.arange(B)[:, None],
                                 jnp.arange(S)[None, :],
                                 targets]                    # (B, S)
    nll = -target_log_probs
    if mask is not None:
        nll = nll * mask
        return nll.sum() / (mask.sum() + 1e-8)
    return nll.mean()


# ─────────────────────────────────────────────────────────────────────────────
# JIT-compiled training / eval steps
# ─────────────────────────────────────────────────────────────────────────────
def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, tokens, targets, mask, key):
        def loss_fn(p):
            logits = forward(p, tokens, training=True, key=key)
            return cross_entropy_loss(logits, targets, mask)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return train_step


@jax.jit
def eval_step(params, tokens, targets, mask):
    logits = forward(params, tokens, training=False, key=jax.random.PRNGKey(0))
    loss   = cross_entropy_loss(logits, targets, mask)
    return loss, jnp.exp(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load utilities
# ─────────────────────────────────────────────────────────────────────────────
def _flatten_params(params, prefix='', out=None):
    """Recursively flatten nested dict/list → flat string-keyed dict."""
    if out is None:
        out = {}
    if isinstance(params, dict):
        for k, v in params.items():
            _flatten_params(v, f"{prefix}/{k}" if prefix else str(k), out)
    elif isinstance(params, list):
        for i, v in enumerate(params):
            _flatten_params(v, f"{prefix}/{i}" if prefix else str(i), out)
    else:
        out[prefix] = np.array(params)
    return out


def _unflatten_params(flat):
    """Reconstruct nested dict/list from flat string-keyed dict."""
    root = {}
    for dotkey, val in flat.items():
        parts = dotkey.split('/')
        d = root
        for part in parts[:-1]:
            if part.isdigit():
                part = int(part)
            if isinstance(d, dict):
                if part not in d:
                    d[part] = {}
                d = d[part]
        last = parts[-1]
        if last.isdigit():
            last = int(last)
        d[last] = jnp.array(val)

    def _convert(obj):
        if isinstance(obj, dict):
            if obj and all(isinstance(k, int) for k in obj):
                return [_convert(obj[i]) for i in range(max(obj) + 1)]
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(root)


def save_params(params, path):
    """Save params pytree to a .npz file. Appends .npz automatically."""
    flat = _flatten_params(params)
    np.savez(path, **flat)
    print(f"[backprop] Saved params → {path}.npz")


def load_params(path):
    """Load params from a .npz file (path without extension)."""
    data = np.load(str(path) + '.npz')
    flat = {k: data[k] for k in data.files}
    return _unflatten_params(flat)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print(" Backprop Transformer Pre-Training")
    print(f"  vocab={VOCAB_SIZE}  embed={N_EMBED}  heads={N_HEADS}"
          f"  layers={N_LAYERS}  seq={SEQ_LEN}  batch={BATCH_SIZE}")
    print("=" * 60)

    # Data
    dl = DataLoader(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    train_loader, valid_loader, _ = dl.load_and_prepare_data()

    # Model
    key = jax.random.PRNGKey(config.SEED)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)

    n_params = sum(v.size for v in _flatten_params(params).values())
    print(f"  {n_params / 1e6:.3f} M parameters")

    # Optimiser
    optimizer  = optax.adamw(learning_rate=LR, weight_decay=1e-2)
    opt_state  = optimizer.init(params)
    train_step = make_train_step(optimizer)

    best_val_ppl = float('inf')

    for epoch in range(EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────
        train_loss_acc, n_batches = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            tokens  = jnp.array(batch[0][1], dtype=jnp.int32)
            targets = jnp.array(batch[1][1], dtype=jnp.int32)
            mask    = jnp.array(batch[2][1], dtype=jnp.float32)

            key, step_key = jax.random.split(key)
            params, opt_state, loss = train_step(
                params, opt_state, tokens, targets, mask, step_key
            )
            train_loss_acc += float(loss)
            n_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx:4d} | "
                      f"loss={float(loss):.4f}  ppl={np.exp(float(loss)):.2f}")

        avg_train_loss = train_loss_acc / max(n_batches, 1)
        train_ppl      = np.exp(avg_train_loss)

        # ── Validation ────────────────────────────────────────────────────
        val_loss_acc, val_n = 0.0, 0
        for batch in valid_loader:
            tokens  = jnp.array(batch[0][1], dtype=jnp.int32)
            targets = jnp.array(batch[1][1], dtype=jnp.int32)
            mask    = jnp.array(batch[2][1], dtype=jnp.float32)
            loss, _ = eval_step(params, tokens, targets, mask)
            val_loss_acc += float(loss)
            val_n += 1

        avg_val_loss = val_loss_acc / max(val_n, 1)
        val_ppl      = np.exp(avg_val_loss)

        print(f"\nEpoch {epoch} | Train loss={avg_train_loss:.4f} ppl={train_ppl:.2f}"
              f"  |  Val loss={avg_val_loss:.4f} ppl={val_ppl:.2f}\n")

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_params(params, str(SAVE_DIR / "best_params"))
            print(f"  ✓ Best val ppl={val_ppl:.2f} → checkpoint saved.\n")

    save_params(params, str(SAVE_DIR / "final_params"))
    print("Training complete.")
    return params


if __name__ == "__main__":
    train()
