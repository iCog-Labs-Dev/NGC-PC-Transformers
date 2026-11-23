"""
Export Generation Weights and Metadata for Standalone Decoder.
Extracts the projection-equivalent parameter tensors and architecture metadata
from an NGCTransformer model instance (trained checkpoint or synthetic initialization),
validates tensor shapes and dtypes, and exports generation_weights.npz and generation_meta.json.
"""

import os
import sys
import json
import hashlib
import argparse
import inspect
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import Config as config
from model import NGCTransformer


def compute_config_hash(cfg_dict: dict) -> str:
    """Compute a deterministic SHA-256 hash of configuration settings."""
    serialized = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def get_current_config_hash() -> str:
    """Compute a deterministic hash of all non-callable config parameters."""
    keys = sorted([k for k in dir(config) if not k.startswith("__") and not callable(getattr(config, k))])
    d = {k: getattr(config, k) for k in keys}
    return compute_config_hash(d)


def get_checkpoint_files_meta(checkpoint_dir: str) -> dict:
    """Collect file sizes and modification times for files in the checkpoint directory."""
    cp = Path(checkpoint_dir)
    files = {}
    if cp.exists() and cp.is_dir():
        for p in sorted(cp.glob("**/*")):
            if p.is_file():
                rel = str(p.relative_to(cp)).replace("\\", "/")
                st = p.stat()
                files[rel] = {"size": st.st_size, "mtime": st.st_mtime}
    return files


def extract_weights_and_meta_from_model(model, checkpoint_dir: str = "exp"):
    """
    Extracts the projection-equivalent parameter tensors and architecture metadata
    directly from an NGCTransformer model instance (source of truth).
    """
    # 1. Read architecture dimensions directly from the model instance
    n_layers = int(getattr(model, "n_layers", config.n_layers))
    batch_size = int(getattr(model, "batch_size", config.batch_size))
    seq_len = int(getattr(model, "seq_len", config.seq_len))
    vocab_size = int(getattr(model, "vocab_size", config.vocab_size))
    n_embed = int(getattr(model, "n_embed", config.n_embed))

    # Read n_heads and d_head from first block or model
    if hasattr(model, "blocks") and len(model.blocks) > 0 and hasattr(model.blocks[0], "attention"):
        attn = model.blocks[0].attention
        if hasattr(attn, "attn_block") and hasattr(attn.attn_block, "n_heads"):
            n_heads = int(attn.attn_block.n_heads)
            d_head = int(attn.attn_block.d_head)
        else:
            n_heads = int(getattr(model, "n_heads", config.n_heads))
            d_head = n_embed // n_heads
    else:
        n_heads = int(getattr(model, "n_heads", config.n_heads))
        d_head = n_embed // n_heads

    # 2. Determine position encoding configuration from model instance (source of truth)
    if hasattr(model, "position_encoding"):
        pos_encoding = str(model.position_encoding).lower()
    elif hasattr(model, "use_positional"):
        pos_encoding = "positional" if model.use_positional else "rope"
    else:
        pos_encoding = "positional"

    # Read pos_learnable from model or embedding
    if hasattr(model, "pos_learnable"):
        pos_learnable = bool(model.pos_learnable)
    elif hasattr(model, "embedding") and hasattr(model.embedding, "W_embed") and hasattr(model.embedding.W_embed, "pos_learnable"):
        pos_learnable = bool(model.embedding.W_embed.pos_learnable)
    else:
        pos_learnable = bool(getattr(config, "pos_learnable", True))

    # Read rope_theta
    if hasattr(model, "rope_theta"):
        rope_theta = float(model.rope_theta)
    else:
        rope_theta = float(getattr(config, "rope_theta", 10000.0))

    # Validate position encoding combinations explicitly
    if pos_encoding not in ("positional", "rope"):
        raise ValueError(
            f"Unknown position_encoding '{pos_encoding}' on model instance. "
            "Must be one of {'positional', 'rope'}."
        )

    # Relaxed rope+pos_learnable check: only error if pos_weights would actually be used
    if pos_encoding == "rope" and pos_learnable:
        if getattr(model, "use_positional", False):
            raise ValueError(
                "Invalid configuration: position_encoding='rope' cannot be used when positional embeddings (pos_weights) are actively enabled."
            )

    # 3. Extract parameter tensors
    weights_dict = {}

    # Word embeddings
    word_w = np.array(model.embedding.W_embed.word_weights.get())
    weights_dict["word_weights"] = word_w

    # Positional embeddings
    if pos_encoding == "positional":
        pos_w = np.array(model.embedding.W_embed.pos_weights.get())
        weights_dict["pos_weights"] = pos_w
    else:
        # For RoPE, position embeddings are computed dynamically via rotary angles
        pass

    # Per-layer attention and MLP weights + biases
    for i in range(n_layers):
        block = model.blocks[i]
        weights_dict[f"block{i}_W_q"] = np.array(block.attention.W_q.weights.get())
        weights_dict[f"block{i}_b_q"] = np.array(block.attention.W_q.biases.get()).squeeze()
        weights_dict[f"block{i}_W_k"] = np.array(block.attention.W_k.weights.get())
        weights_dict[f"block{i}_b_k"] = np.array(block.attention.W_k.biases.get()).squeeze()
        weights_dict[f"block{i}_W_v"] = np.array(block.attention.W_v.weights.get())
        weights_dict[f"block{i}_b_v"] = np.array(block.attention.W_v.biases.get()).squeeze()

        weights_dict[f"block{i}_W_attn_out"] = np.array(block.attention.W_attn_out.weights.get())
        weights_dict[f"block{i}_b_attn_out"] = np.array(block.attention.W_attn_out.biases.get()).squeeze()

        weights_dict[f"block{i}_W_mlp1"] = np.array(block.mlp.W_mlp1.weights.get())
        weights_dict[f"block{i}_b_mlp1"] = np.array(block.mlp.W_mlp1.biases.get()).squeeze()
        weights_dict[f"block{i}_W_mlp2"] = np.array(block.mlp.W_mlp2.weights.get())
        weights_dict[f"block{i}_b_mlp2"] = np.array(block.mlp.W_mlp2.biases.get()).squeeze()

    # Final Output Head
    weights_dict["W_out"] = np.array(model.output.W_out.weights.get())
    weights_dict["b_out"] = np.array(model.output.W_out.biases.get()).squeeze()

    # Act_fx
    act_fx = str(getattr(model, "act_fx", getattr(config, "act_fx", "identity")))

    # 4. Construct metadata
    cfg_summary = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_embed": n_embed,
        "d_head": d_head,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "position_encoding": pos_encoding,
        "pos_learnable": pos_learnable,
        "rope_theta": rope_theta if pos_encoding == "rope" else None,
        "act_fx": act_fx,
        "act_fx_mlp2": "gelu",
        "act_fx_out": "softmax",
        "tau_m_projection": 0.0,
    }
    cfg_hash = get_current_config_hash()
    cp_files = get_checkpoint_files_meta(checkpoint_dir) if str(checkpoint_dir) != "in_memory" else {}

    metadata = {
        **cfg_summary,
        "config_hash": cfg_hash,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": cp_files,
        "cells": {
            "q_embed_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_qkv_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_mlp_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_mlp2_Ratecell": {"tau_m": 0.0, "act_fx": "gelu", "prior": ["gaussian", 0.0]},
            "q_out_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_target_Ratecell": {"tau_m": 0.0, "act_fx": "softmax", "prior": ["gaussian", 0.0]},
        },
        "export_source": "NGCTransformer.process() projection path",
        "exported_from_dir": str(checkpoint_dir),
    }

    return weights_dict, metadata


def export_from_model(model, output_npz: str, output_meta: str, checkpoint_dir: str = "in_memory"):
    """
    Extracts weights and metadata from a provided model instance and writes to disk.
    """
    weights_dict, metadata = extract_weights_and_meta_from_model(model, checkpoint_dir=checkpoint_dir)

    # Validate weights
    for name, arr in weights_dict.items():
        if np.isnan(arr).any():
            raise RuntimeError(f"Weight tensor '{name}' contains NaN values! Cannot export invalid weights.")

    npz_path = Path(output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **weights_dict)

    meta_path = Path(output_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return weights_dict, metadata


def export_weights(
    checkpoint_dir: str = "exp",
    model_name: str = "ngc_transformer",
    output_npz: str = "generation_weights.npz",
    output_meta: str = "generation_meta.json",
    use_synthetic: bool = False,
    device: str = None,
    **kwargs
):
    """
    Unified export function supporting both CLI and programmatic calls.
    Signature handles (checkpoint_dir, model_name, output_npz, output_meta, use_synthetic).
    Also supports legacy keyword aliases: exp_dir, out_weights, out_meta.
    """
    checkpoint_dir = kwargs.get("exp_dir", checkpoint_dir)
    output_npz = kwargs.get("out_weights", output_npz)
    output_meta = kwargs.get("out_meta", output_meta)

    if device is not None:
        jax.config.update("jax_platform_name", device)

    print("=" * 75)
    print("NGC TRANSFORMER: WEIGHT EXPORT FOR FAST STANDALONE DECODER")
    print("=" * 75)

    if not use_synthetic:
        cp_path = Path(checkpoint_dir)
        has_checkpoint = False
        if cp_path.exists() and cp_path.is_dir():
            has_checkpoint = any(cp_path.glob("**/*.npz")) or any(cp_path.glob("**/circuit.json"))
        if not has_checkpoint:
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_dir}' does not exist or contains no valid checkpoint files.\n"
                f"To export weights from a trained model, provide a valid directory containing trained weights.\n"
                f"To export a randomly initialized model for testing, explicitly pass use_synthetic=True (or --synthetic on CLI)."
            )
    else:
        print("\n" + "!" * 75)
        print("[WARNING] use_synthetic=True: EXPORTING UNTRAINED RANDOM WEIGHTS!")
        print("These weights are synthetic and should only be used for testing, not real generation.")
        print("!" * 75 + "\n")

    dkey = jax.random.PRNGKey(0)
    load_dir = None if use_synthetic else checkpoint_dir

    print(f"[1/4] Initializing NGCTransformer (loadDir='{load_dir}', synthetic={use_synthetic})...")

    from model import build_model_from_config
    model = build_model_from_config(
        dkey=dkey,
        exp_dir=checkpoint_dir,
        loadDir=load_dir,
        model_name=model_name,
        generate=True,
    )

    print("[2/4] Extracting projection-equivalent parameter tensors from model instance...")
    weights_dict, metadata = export_from_model(
        model=model,
        output_npz=output_npz,
        output_meta=output_meta,
        checkpoint_dir=str(checkpoint_dir)
    )

    print("\n" + "-" * 75)
    print(f"{'TENSOR NAME':<30} | {'SHAPE':<20} | {'DTYPE':<10} | {'HAS NAN':<8}")
    print("-" * 75)
    for name, arr in weights_dict.items():
        has_nan = bool(np.isnan(arr).any())
        print(f"{name:<30} | {str(arr.shape):<20} | {str(arr.dtype):<10} | {str(has_nan):<8}")
    print("-" * 75)

    npz_path = Path(output_npz)
    meta_path = Path(output_meta)
    print(f"\n[3/4] Successfully saved generation weights to: '{npz_path.resolve()}'")
    print(f"      Size: {npz_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"[4/4] Successfully saved generation metadata to: '{meta_path.resolve()}'")
    print("\nWeight export completed successfully.\n")

    return weights_dict, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Export generation weights and metadata from checkpoint or synthetic model.")
    parser.add_argument(
        "--checkpoint-dir", "--exp-dir",
        dest="checkpoint_dir",
        type=str,
        default="exp",
        help="Checkpoint directory path (default: 'exp')."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ngc_transformer",
        help="Model name (default: 'ngc_transformer')."
    )
    parser.add_argument(
        "--output-npz", "--out-weights", "--weights",
        dest="output_npz",
        type=str,
        default="generation_weights.npz",
        help="Output .npz file path."
    )
    parser.add_argument(
        "--output-meta", "--out-meta", "--meta",
        dest="output_meta",
        type=str,
        default="generation_meta.json",
        help="Output .json metadata path."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic weights if no trained checkpoint exists."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'gpu', or auto-detect if None)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_weights(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        output_npz=args.output_npz,
        output_meta=args.output_meta,
        use_synthetic=args.synthetic,
        device=args.device
    )
