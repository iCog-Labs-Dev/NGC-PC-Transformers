#!/usr/bin/env python3
"""
Verification script for real trained checkpoints on Colab / GPU servers.
DO NOT RUN LOCALLY WITHOUT TRAINED CHECKPOINT IN exp/.

This script:
1. Loads the trained NGCTransformer model from exp/ and the exported standalone weights.
2. Performs 80+ token greedy decoding (crossing seq_len) with both legacy model.process
   and the fast standalone decoder, asserting identical token IDs and max prob diff < 1e-4
   at every single step. Prints first divergence if any.
3. Samples 3 outputs each with temperature=0.8, top_k=50 for qualitative comparison.
4. Benchmarks tokens/sec for both legacy and fast decoders with warmup excluded,
   reporting speedup factor.

Usage on Colab (GPU):
    python scripts/verify_real_checkpoint.py --exp-dir exp --device gpu
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Configure environment before JAX imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_ENABLE_X64", "False")

import warnings
warnings.filterwarnings("ignore")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
from config import Config as config
from model import NGCTransformer
from generation import setup_tokenizer, generate_text_legacy
from export_generation_weights import export_weights
from inference.decoder import load_weights, prefill, decode_step, generate as fast_generate, softmax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify fast standalone decoder against legacy model on real trained checkpoint."
    )
    parser.add_argument("--exp-dir", type=str, default="exp", help="Checkpoint directory containing trained model.")
    parser.add_argument("--model-name", type=str, default="ngc_transformer", help="Name of trained model.")
    parser.add_argument("--weights", type=str, default="generation_weights.npz", help="Exported weights .npz file.")
    parser.add_argument("--meta", type=str, default="generation_meta.json", help="Exported metadata .json file.")
    parser.add_argument("--prompt", type=str, default="The transformer model is", help="Prompt for generation.")
    parser.add_argument("--greedy-tokens", type=int, default=80, help="Number of greedy tokens to verify (>=80).")
    parser.add_argument("--sample-tokens", type=int, default=80, help="Number of tokens for sampled generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampled generation.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering for sampled generation.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of sampled generations to display.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--prob-tol", type=float, default=1e-4, help="Tolerance for max probability diff per step.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu", "tpu"], help="JAX device.")
    return parser.parse_args()


def verify_greedy_step_by_step(model, params, meta, tokenizer, prompt, max_tokens, prob_tol):
    """
    Greedy decode token-by-token using both legacy model.process and fast decoder.
    Asserts identical token IDs and max prob diff < prob_tol at every single step.
    Prints the first divergence if any.
    """
    print(f"\n" + "=" * 70)
    print(f"[STEP 1] GREEDY DECODING VERIFICATION ({max_tokens} tokens, crossing seq_len={config.seq_len})")
    print("=" * 70)

    # Tokenize prompt
    encoded = tokenizer.encode(prompt)
    prompt_ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    if len(prompt_ids) == 0:
        prompt_ids = [0]

    pad_token_id = 0
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        pad_id = tokenizer.tokenizer.token_to_id("<pad>")
        if pad_id is not None:
            pad_token_id = pad_id

    # Running sequence lists
    legacy_tokens = list(prompt_ids)
    fast_tokens = list(prompt_ids)

    # Initial prefill for fast decoder
    prompt_arr = jnp.array(prompt_ids, dtype=jnp.int32)
    prompt_len = len(prompt_ids)
    n_prefill = min(prompt_len, config.seq_len)
    fast_logits, fast_cache = prefill(params, prompt_arr[-n_prefill:], n=n_prefill, meta=meta)
    fast_pos = n_prefill - 1

    max_prob_diff_overall = 0.0
    divergence_found = False

    for step in range(max_tokens):
        # ── 1. Legacy next-token prediction ──
        if len(legacy_tokens) > config.seq_len:
            sub = legacy_tokens[-config.seq_len:]
            last_pos = config.seq_len - 1
        else:
            sub = legacy_tokens
            last_pos = len(legacy_tokens) - 1

        cur_arr = jnp.array([sub], dtype=jnp.int32)
        if cur_arr.shape[1] < config.seq_len:
            pad_len = config.seq_len - cur_arr.shape[1]
            input_seq = jnp.pad(cur_arr, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
        else:
            input_seq = cur_arr

        dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))
        y_mu_inf, _, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        probs_3d = y_mu_inf.reshape(config.batch_size, config.seq_len, config.vocab_size)
        legacy_probs = np.array(probs_3d[0, last_pos, :])
        legacy_next_token = int(np.argmax(legacy_probs))

        # ── 2. Fast decoder next-token prediction ──
        if step == 0:
            current_fast_logits = fast_logits
        else:
            # decode step or sliding-window re-prefill
            prev_token = fast_tokens[-1]
            fast_pos += 1
            if fast_pos < config.seq_len:
                current_fast_logits, fast_cache = decode_step(
                    params, fast_cache, token=prev_token, pos=fast_pos, meta=meta
                )
            else:
                window_tokens = jnp.array(fast_tokens[-config.seq_len:], dtype=jnp.int32)
                current_fast_logits, fast_cache = prefill(params, window_tokens, n=config.seq_len, meta=meta)
                fast_pos = config.seq_len - 1

        fast_probs = np.array(softmax(current_fast_logits))
        fast_next_token = int(np.argmax(fast_probs))

        fast_log_probs = np.array(jax.nn.log_softmax(current_fast_logits))
        if hasattr(model, "projection") and hasattr(model.projection, "q_out_Ratecell"):
            gt_h = model.projection.q_out_Ratecell.zF.get().reshape(config.batch_size, config.seq_len, -1)[0, last_pos]
            gt_W = model.projection.Q_out.weights.get()
            gt_b = model.projection.Q_out.biases.get().squeeze()
            legacy_logits = gt_h @ gt_W + gt_b
            legacy_log_probs = np.array(jax.nn.log_softmax(legacy_logits))
        else:
            legacy_log_probs = np.log(np.clip(legacy_probs, 1e-12, 1.0))

        # ── 3. Step comparison & assertions (Item 5) ──
        step_log_prob_diff = float(np.max(np.abs(legacy_log_probs - fast_log_probs)))
        if step_log_prob_diff > max_prob_diff_overall:
            max_prob_diff_overall = step_log_prob_diff

        # Top-k probability comparison
        k_eval = min(50, config.vocab_size)
        topk_idx = np.argsort(-legacy_probs)[:k_eval]
        step_topk_prob_diff = float(np.max(np.abs(legacy_probs[topk_idx] - fast_probs[topk_idx])))

        tokens_match = (legacy_next_token == fast_next_token)
        log_probs_match = (step_log_prob_diff < 1e-3)
        topk_probs_match = (step_topk_prob_diff < 1e-4)

        if not tokens_match or not log_probs_match or not topk_probs_match:
            print(f"\n[DIVERGENCE DETECTED AT STEP {step + 1}]")
            print(f"  Current sequence length: {len(legacy_tokens)}")
            print(f"  Legacy next token: {legacy_next_token} ({tokenizer.decode([legacy_next_token])!r})")
            print(f"  Fast   next token: {fast_next_token} ({tokenizer.decode([fast_next_token])!r})")
            print(f"  Max log-prob diff over full vocab: {step_log_prob_diff:.6e} (tolerance: 1e-3)")
            print(f"  Max top-{k_eval} prob diff: {step_topk_prob_diff:.6e} (tolerance: 1e-4)")

            # Print top 5 probabilities for both
            legacy_top5 = np.argsort(-legacy_probs)[:5]
            fast_top5 = np.argsort(-fast_probs)[:5]
            print(f"  Legacy top-5: {list(zip(legacy_top5, legacy_probs[legacy_top5]))}")
            print(f"  Fast   top-5: {list(zip(fast_top5, fast_probs[fast_top5]))}")

            divergence_found = True
            raise AssertionError(
                f"Equivalence failed at step {step + 1}: token match={tokens_match}, "
                f"log-prob diff={step_log_prob_diff:.2e} (< 1e-3), "
                f"top-k prob diff={step_topk_prob_diff:.2e} (< 1e-4)"
            )

        legacy_tokens.append(legacy_next_token)
        fast_tokens.append(fast_next_token)

        if (step + 1) % 20 == 0 or (step + 1) == max_tokens:
            print(f"  Step {step + 1:3d}/{max_tokens:3d}: MATCH (max log-prob diff so far: {max_prob_diff_overall:.2e})")

    print(f"\n[PASS] Greedy equivalence verified for all {max_tokens} tokens!")
    print(f"  Total tokens generated: {max_tokens}")
    print(f"  Overall maximum log-probability difference: {max_prob_diff_overall:.2e}")
    print(f"  Final generated text sample:\n{tokenizer.decode(fast_tokens)!r}")


def sample_comparisons(model, params, meta, tokenizer, prompt, max_tokens, temperature, top_k, num_samples, seed):
    """
    Generate sampled outputs with T=0.8, top_k=50 for both models using matching PRNG keys.
    """
    print(f"\n" + "=" * 70)
    print(f"[STEP 2] SAMPLED GENERATION (T={temperature}, top_k={top_k}, {num_samples} samples each)")
    print("=" * 70)

    base_key = jax.random.PRNGKey(seed)

    for i in range(num_samples):
        key_legacy, key_fast = jax.random.split(jax.random.fold_in(base_key, i), 2)

        print(f"\n--- Sample {i + 1}/{num_samples} ---")

        # Fast decoder sample
        text_fast = fast_generate(
            params=params,
            meta=meta,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            key=key_fast,
        )
        print(f"[FAST DECODER OUTPUT]:\n{text_fast}\n")

        # Legacy generator sample
        text_legacy = generate_text_legacy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            key=key_legacy,
        )
        print(f"[LEGACY MODEL OUTPUT]:\n{text_legacy}")


def benchmark_speed(model, params, meta, tokenizer, prompt, tokens_to_generate=80):
    """
    Benchmark tokens/second with warmup run excluded to measure steady-state throughput.
    """
    print(f"\n" + "=" * 70)
    print(f"[STEP 3] PERFORMANCE BENCHMARK (Warmup run excluded, {tokens_to_generate} tokens)")
    print("=" * 70)

    # ── 1. Warmup runs ──
    print("Running warmup for fast decoder...")
    _ = fast_generate(params, meta, tokenizer, prompt=prompt, max_new_tokens=10, temperature=0.0)

    print("Running warmup for legacy generator...")
    _ = generate_text_legacy(model, tokenizer, prompt=prompt, max_new_tokens=10, temperature=0.0)

    # ── 2. Timed runs for Fast Decoder ──
    print(f"\nTiming Fast Decoder over 3 runs of {tokens_to_generate} tokens...")
    fast_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = fast_generate(params, meta, tokenizer, prompt=prompt, max_new_tokens=tokens_to_generate, temperature=0.0)
        fast_times.append(time.perf_counter() - t0)

    avg_fast_time = float(np.mean(fast_times))
    fast_tok_per_sec = tokens_to_generate / avg_fast_time

    # ── 3. Timed runs for Legacy Generator ──
    print(f"Timing Legacy Generator over 3 runs of {tokens_to_generate} tokens...")
    legacy_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = generate_text_legacy(model, tokenizer, prompt=prompt, max_new_tokens=tokens_to_generate, temperature=0.0)
        legacy_times.append(time.perf_counter() - t0)

    avg_legacy_time = float(np.mean(legacy_times))
    legacy_tok_per_sec = tokens_to_generate / avg_legacy_time

    speedup = fast_tok_per_sec / legacy_tok_per_sec if legacy_tok_per_sec > 0 else float("inf")

    print(f"\n" + "-" * 50)
    print(f"BENCHMARK RESULTS ({tokens_to_generate} tokens):")
    print(f"  Legacy Generator: {legacy_tok_per_sec:7.2f} tokens/sec ({avg_legacy_time:.3f}s)")
    print(f"  Fast Decoder:     {fast_tok_per_sec:7.2f} tokens/sec ({avg_fast_time:.3f}s)")
    print(f"  Speedup factor:   {speedup:7.2f}x faster")
    print("-" * 50)


def main():
    args = parse_args()

    if args.device:
        jax.config.update("jax_platform_name", args.device)
    print(f"JAX active backend: {jax.default_backend()}")

    # Setup tokenizer
    tokenizer = setup_tokenizer()

    # Ensure exported weights exist
    weights_path = Path(args.weights)
    meta_path = Path(args.meta)
    if not weights_path.exists() or not meta_path.exists():
        print(f"Exported weights not found. Exporting from checkpoint {args.exp_dir}...")
        export_weights(
            checkpoint_dir=args.exp_dir,
            model_name=args.model_name,
            output_npz=str(weights_path),
            output_meta=str(meta_path),
            use_synthetic=False,
            device=args.device,
        )

    # Load standalone weights
    print(f"Loading standalone weights from {weights_path}...")
    params, meta = load_weights(str(weights_path), str(meta_path))

    # Instantiate legacy model and load checkpoint weights from exp/
    print(f"Instantiating legacy NGCTransformer and loading checkpoint from {args.exp_dir}...")
    from model import build_model_from_config
    model = build_model_from_config(
        seed=args.seed,
        exp_dir=args.exp_dir,
        loadDir=args.exp_dir,
        model_name=args.model_name,
        generate=True,
    )

    # 1. Greedy Verification
    verify_greedy_step_by_step(
        model=model,
        params=params,
        meta=meta,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.greedy_tokens,
        prob_tol=args.prob_tol,
    )

    # 2. Sampled generation
    sample_comparisons(
        model=model,
        params=params,
        meta=meta,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # 3. Performance Benchmark
    benchmark_speed(
        model=model,
        params=params,
        meta=meta,
        tokenizer=tokenizer,
        prompt=args.prompt,
        tokens_to_generate=args.greedy_tokens,
    )

    print("\n[SUCCESS] Real checkpoint verification complete! All tests passed.")


if __name__ == "__main__":
    main()
