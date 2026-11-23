import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# MUST BE SET BEFORE ANY JAX/TENSORFLOW IMPORTS
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false --xla_cpu_use_thunk_runtime=false')
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('JAX_ENABLE_X64', 'False')

# Redirect stderr to suppress XLA warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings('ignore')

# Now import JAX and other libraries
import jax
if 'JAX_PLATFORM_NAME' in os.environ:
    jax.config.update('jax_platform_name', os.environ['JAX_PLATFORM_NAME'])
jax.config.update('jax_log_compiles', False)

# Restore stderr after JAX initialization
sys.stderr = stderr

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax.numpy as jnp
from config import Config as config
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer, CharacterTokenizer
from inference.decoder import load_weights, generate as fast_generate, sample


def setup_tokenizer():
    """Load and initialize the tokenizer configured in config.py."""
    tokenizer = get_tokenizer(config)
    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
        vocab_file = getattr(config, "tokenizer_vocab_file", None)
        if vocab_file is None:
            from data_preprocess.datasets_registry import prepare_dataset
            _, output_dir = prepare_dataset(config.dataset)
            default_path = output_dir / "tokenizer" / "bpe_tokenizer.json"
            if default_path.exists():
                vocab_file = str(default_path)
                print(f"Auto-loading BPE tokenizer from default path: {vocab_file}")

        if vocab_file and Path(vocab_file).exists():
            tokenizer.load_tokenizer(vocab_file)
            print(f"Loaded BPE tokenizer (vocab size: {tokenizer.get_vocab_size()})")
        else:
            raise RuntimeError(
                "BPE tokenizer not trained or loaded!\n"
                "Run `python data_preprocess/tokenizer.py` first to train/tokenize."
            )
    return tokenizer


def generate_text_legacy(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None
):
    """
    Legacy generation loop using NGCTransformer.process().
    Maintains exact original sliding-window semantics with bugfix for temperature scaling:
    projection output is softmax probabilities from q_target_Ratecell, so temperature
    is applied to log(probs + 1e-9) / temperature.
    """
    if pad_token_id is None:
        if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
            pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        elif isinstance(tokenizer, CharacterTokenizer):
            pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, "_enc") and hasattr(tokenizer._enc, "eot_token"):
            pad_token_id = tokenizer._enc.eot_token
        else:
            pad_token_id = 0

    start_token_id = None
    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
        start_token_id = tokenizer.tokenizer.token_to_id("<bos>")
    elif isinstance(tokenizer, CharacterTokenizer):
        start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = pad_token_id

    # Initialize sequence with prompt or start token ID
    if prompt:
        encoded = tokenizer.encode(prompt)
        input_ids = encoded.ids if hasattr(encoded, "ids") else list(encoded)
        if len(input_ids) == 0:
            input_ids = [start_token_id]
    else:
        input_ids = [start_token_id]

    tokens_list = list(input_ids)
    current_key = key

    for _ in range(max_new_tokens):
        # Truncate context to fit model's seq_len
        if len(tokens_list) > config.seq_len:
            sub = tokens_list[-config.seq_len:]
        else:
            sub = tokens_list

        cur_arr = jnp.array([sub], dtype=jnp.int32)
        if cur_arr.shape[1] < config.seq_len:
            pad_len = config.seq_len - cur_arr.shape[1]
            input_seq = jnp.pad(cur_arr, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
        else:
            input_seq = cur_arr

        # Forward pass (no target clamping during inference)
        dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))
        y_mu_inf, _, _ = model.process(input_seq, dummy_target, adapt_synapses=False)

        # Extract pre-softmax logits directly from the projection layer
        if hasattr(model, "get_logits"):
            logits_2d = model.get_logits()
        elif hasattr(model, "projection") and hasattr(model.projection, "Q_out"):
            logits_2d = model.projection.Q_out.outputs.get()
        else:
            logits_2d = jnp.log(y_mu_inf + 1e-9)

        logits_3d = logits_2d.reshape(config.batch_size, config.seq_len, config.vocab_size)

        # Get logits for the last *real* token (excluding padding)
        if len(tokens_list) > config.seq_len:
            last_pos = config.seq_len - 1
        else:
            last_pos = len(tokens_list) - 1
        last_logits = logits_3d[0, last_pos, :]

        if current_key is not None:
            current_key, subkey = jax.random.split(current_key)
        else:
            subkey = None

        next_token = sample(last_logits, key=subkey, temperature=temperature, top_k=top_k)
        tokens_list.append(next_token)

    # Decode generated IDs back to text
    return tokenizer.decode(tokens_list)


# Backward-compatibility alias
generate_text = generate_text_legacy


def benchmark_generation(gen_fn, num_runs=3):
    """
    Run generation with an initial warmup pass (capturing JIT compilation overhead),
    followed by `num_runs` timed passes for pure inference throughput.
    """
    print("  Warmup run (compiling JIT graphs)...", end="", flush=True)
    t_warm_0 = time.perf_counter()
    warm_text, warm_tokens = gen_fn()
    t_warm_1 = time.perf_counter()
    jit_time = t_warm_1 - t_warm_0
    print(f" done in {jit_time:.3f}s")

    durations = []
    tokens_per_sec = []
    text_out = warm_text

    for r in range(num_runs):
        t0 = time.perf_counter()
        text_out, n_tok = gen_fn()
        t1 = time.perf_counter()
        dt = t1 - t0
        tps = n_tok / max(dt, 1e-6)
        durations.append(dt)
        tokens_per_sec.append(tps)
        print(f"  Run {r + 1}: {n_tok} tokens in {dt:.3f}s ({tps:.2f} tok/s)")

    median_tps = float(np.median(tokens_per_sec))
    min_tps = float(np.min(tokens_per_sec))
    max_tps = float(np.max(tokens_per_sec))
    print(f"\nBenchmark Summary ({num_runs} runs, excluding warmup):")
    print(f"  JIT compile / Warmup time: {jit_time:.3f}s")
    print(f"  Tokens/sec - Median: {median_tps:.2f} | Min: {min_tps:.2f} | Max: {max_tps:.2f}")
    return text_out, median_tps, min_tps, max_tps


def main():
    parser = argparse.ArgumentParser(description="NGC Transformer Inference Generator")
    parser.add_argument("--legacy", action="store_true", help="Use legacy NGCTransformer.process() generation loop")
    parser.add_argument("--benchmark", action="store_true", help="Run 3+ benchmark passes and report tokens/sec")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (0 to disable)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--weights", type=str, default="generation_weights.npz", help="Weights file for standalone decoder")
    parser.add_argument("--meta", type=str, default="generation_meta.json", help="Meta JSON file for standalone decoder")
    parser.add_argument("--exp-dir", type=str, default="exp", help="Checkpoint directory path")
    parser.add_argument("--model-name", type=str, default="ngc_transformer", help="Model name")
    parser.add_argument("--synthetic", action="store_true", help="Explicitly allow exporting synthetic random weights if weights/meta are missing (test only)")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--seed", type=int, default=0, help="PRNG key seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'gpu', or auto-detect)")
    args = parser.parse_args()

    if args.device is not None:
        jax.config.update("jax_platform_name", args.device)

    tokenizer = setup_tokenizer()
    rng = jax.random.PRNGKey(args.seed)

    if args.legacy:
        print("\n[Mode] Running LEGACY generation loop (NGCTransformer.process)...")
        # Lazy import of model.py only when --legacy is requested
        from model import build_model_from_config

        model = build_model_from_config(
            seed=args.seed,
            exp_dir=args.exp_dir,
            loadDir=args.exp_dir,
            model_name=args.model_name,
            generate=True
        )

        def run_legacy():
            key = None
            if args.temperature > 0.0:
                _, k = jax.random.split(rng)
                key = k
            text = generate_text_legacy(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                key=key,
            )
            return text, args.max_new_tokens

        if args.benchmark:
            print(f"\nBenchmarking legacy generator over {args.runs} runs ({args.max_new_tokens} tokens each)...")
            output, _, _, _ = benchmark_generation(run_legacy, num_runs=args.runs)
        else:
            output, _ = run_legacy()

        print("\nFINAL GENERATED (LEGACY):\n")
        print(output)

    else:
        print("\n[Mode] Running FAST STANDALONE decoder...")
        weights_path = Path(args.weights)
        meta_path = Path(args.meta)
        if not weights_path.exists() or not meta_path.exists():
            from export_generation_weights import export_weights
            if args.synthetic:
                print("\n" + "!" * 75)
                print("[WARNING] --synthetic flag was explicitly passed: EXPORTING UNTRAINED RANDOM WEIGHTS!")
                print("Generated text will be completely untrained gibberish.")
                print("!" * 75 + "\n")
                export_weights(
                    checkpoint_dir=args.exp_dir,
                    model_name=args.model_name,
                    output_npz=str(weights_path),
                    output_meta=str(meta_path),
                    use_synthetic=True,
                    device=args.device
                )
            else:
                cp_dir = Path(args.exp_dir)
                has_checkpoint = False
                if cp_dir.exists() and cp_dir.is_dir():
                    has_checkpoint = any(cp_dir.glob("**/*.npz")) or any(cp_dir.glob("**/circuit.json"))
                if not has_checkpoint:
                    raise FileNotFoundError(
                        f"Cannot auto-export generation weights: checkpoint directory '{args.exp_dir}' "
                        f"does not exist or contains no saved checkpoint files.\n"
                        f"To generate text with a trained model, please train a checkpoint first (via train.py) "
                        f"or specify an existing checkpoint directory with --exp-dir.\n"
                        f"(If you only want to test the pipeline with random weights, explicitly pass --synthetic.)"
                    )
                print(f"Standalone weights not found. Exporting from checkpoint at '{args.exp_dir}'...")
                export_weights(
                    checkpoint_dir=args.exp_dir,
                    model_name=args.model_name,
                    output_npz=str(weights_path),
                    output_meta=str(meta_path),
                    use_synthetic=False,
                    device=args.device
                )

        params, meta = load_weights(str(weights_path), str(meta_path))

        def run_fast():
            key = None
            if args.temperature > 0.0:
                _, k = jax.random.split(rng)
                key = k
            text = fast_generate(
                params=params,
                meta=meta,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                key=key,
            )
            return text, args.max_new_tokens

        if args.benchmark:
            print(f"\nBenchmarking fast standalone decoder over {args.runs} runs ({args.max_new_tokens} tokens each)...")
            output, _, _, _ = benchmark_generation(run_fast, num_runs=args.runs)
        else:
            output, _ = run_fast()

        print("\nFINAL GENERATED (FAST DECODER):\n")
        print(output)


if __name__ == "__main__":
    main()
