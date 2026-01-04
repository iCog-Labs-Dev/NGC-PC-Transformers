import alogos as al
import csv
import jax
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc
import os
import time
import traceback
from itertools import count
from pathlib import Path
from config import Config as config

# --- 1. Environment & Setup ---
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Import Error: {e}. Check project path.")
    sys.exit(1)

LOG_FILE = "search_progress.log"
RESULT_FILE = "tuning_results.txt"
RESULT_CSV = "trial_metrics.csv"
BEST_FILE = "best_trial.txt"
TRIAL_COUNTER = count(1)
BEST_RECORD = {"val_ce": float("inf"), "avg_ppl": None, "avg_efe": None, "avg_ce": None, "phenotype": None, "trial_id": None}


def _fmt_metric(value):
    if value is None or (isinstance(value, float) and np.isinf(value)):
        return "N/A"
    return f"{value:.6f}" if isinstance(value, float) else str(value)

def log_message(message, end="\n"):
    print(message, end=end, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + end)


def clean_memory():
    """Aggressively free JAX/device and host allocations."""
    try:
        clear_caches()
        jax.block_until_ready(jnp.array(0))
    except Exception:
        pass
    gc.collect()


def set_model_eta(model, eta_value):
    """Best-effort setter to propagate eta to major submodules."""
    def _try_set(obj):
        if obj is not None and hasattr(obj, "eta"):
            try:
                obj.eta = eta_value
            except Exception:
                pass

    _try_set(model)
    for attr in ["embedding", "output", "projection"]:
        _try_set(getattr(model, attr, None))
    blocks = getattr(model, "blocks", []) or []
    for blk in blocks:
        _try_set(blk)
        _try_set(getattr(blk, "attention", None))
        _try_set(getattr(blk, "mlp", None))

with open(LOG_FILE, "w") as f:
    f.write("=== New Search Session ===\n")

# Create CSV header if missing
if not Path(RESULT_CSV).exists() or Path(RESULT_CSV).stat().st_size == 0:
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "status", "phenotype", "avg_efe", "avg_ppl", "avg_ce", "val_ce", "eval_time_sec"])

# --- 2. Grammar Guided Search Space (CLEANED) ---
# CRITICAL FIX: No comments (#) allowed inside this string!
bnf_text = """
<hparams>     ::= <arch_pair> "," <block_conf> "," <depth_conf> "," <steps> "," <learn_rate> "," <drop> "," <warmup> "," <bounds> "," <tau_m> "," <act_fx>
<arch_pair>   ::= "n_embed=64,n_heads=4" | "n_embed=128,n_heads=4" | "n_embed=128,n_heads=8" | "n_embed=256,n_heads=8"
<block_conf>  ::= "block_size=64" | "block_size=128" | "block_size=256"
<depth_conf>  ::= "n_layers=2" | "n_layers=4" | "n_layers=6"
<steps>       ::= "T=4" | "T=5" | "T=6" | "T=7" | "T=8"
<learn_rate>  ::= "eta=0.0004" | "eta=0.0016" | "eta=0.0064" | "eta=0.001" 
<drop>        ::= "dropout=0.1" | "dropout=0.2" | "dropout=0.3"
<warmup>      ::= "warmup_epochs=1" | "warmup_epochs=2" | "warmup_epochs=3" | "warmup_epochs=5"
<bounds>      ::= "wlb=-0.05,wub=0.05" | "wlb=-0.02,wub=0.02" | "wlb=-0.01,wub=0.01"
<tau_m>       ::= "tau_m=25" | "tau_m=10" | "tau_m=20"
<act_fx>      ::= "act_fx=tanh" | "act_fx=softplus" | "act_fx=identity"
"""

def get_dynamic_batch_size(n_embed, block_size):
    """
    Calculates batch size to maintain constant memory usage.
    """
    complexity_score = n_embed * block_size
    # More conservative sizing to avoid OOM seen in prior runs
    if complexity_score >= 32000:
        return 4
    elif complexity_score >= 20000:
        return 8
    elif complexity_score >= 12000:
        return 12
    elif complexity_score >= 6000:
        return 16
    else:
        return 24

# --- 3. The Objective Function (The "Trial") ---
def objective_function(phenotype_string):
    start_time = time.time()
    trial_id = next(TRIAL_COUNTER)
    
    # 1. CLEANING THE STRING
    # Removes quotes and spaces to handle "n_embed=64" format
    clean_string = phenotype_string.replace('"', '').replace(' ', '').replace('\n', '')
    
    try:
        # 2. ROBUST PARSING
        # Splits by comma and creates key-value pairs
        params = {}
        for item in clean_string.split(','):
            if '=' in item:
                k, v = item.split('=')
                params[k] = v
            else:
                # If we encounter garbage (like comments), we ignore or fail safely
                continue
                
        # Extract Key Params
        n_embed = int(params['n_embed'])
        n_heads = int(params['n_heads'])
        seq_len = int(params['block_size'])
        n_layers = int(params['n_layers'])
        T = int(params['T'])
        eta = float(params['eta'])
        dropout = float(params['dropout'])
        wlb = float(params['wlb'])
        wub = float(params['wub'])
        warmup_epochs = int(params['warmup_epochs'])
        tau_m = float(params['tau_m'])
        act_fx = str(params['act_fx'])

    except Exception as e:
        # If parsing fails, print exactly why so we don't get silent "0.0 runtime"
        log_message(f"[!] Parsing Failed: {clean_string}")
        log_message(f"    Error: {e}")
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "parse_fail", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9 # Penalty

    if n_embed % n_heads != 0:
        log_message(f"[!] Invalid Config: n_embed ({n_embed}) not divisible by n_heads ({n_heads}). Skipping trial.")
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "invalid_config", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9
    if tau_m <= 0:
        log_message(f"[!] Invalid tau_m ({tau_m}); must be > 0. Skipping trial.")
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "invalid_tau_m", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9
    allowed_act_fx = {"tanh", "softplus", "identity"}
    if act_fx not in allowed_act_fx:
        log_message(f"[!] Invalid act_fx ({act_fx}); must be one of {sorted(allowed_act_fx)}. Skipping trial.")
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "invalid_act_fx", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9
    if wlb >= wub:
        log_message(f"[!] Invalid Bounds: wlb ({wlb}) must be < wub ({wub}). Skipping trial.")
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "invalid_bounds", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9
    # Clip bounds to a tighter envelope to prevent extreme init leading to NaNs
    safe_wlb = max(wlb, -0.05)
    safe_wub = min(wub, 0.05)
    if (safe_wlb != wlb) or (safe_wub != wub):
        log_message(f"[i] Bounds clipped to safe range: wlb={safe_wlb}, wub={safe_wub}")
    wlb, wub = safe_wlb, safe_wub

    # Optional: adjust hyperparams for extreme configs (disabled per user request)

    log_message(f"\n" + "="*60)
    log_message(f"[Trial Start] Config: {clean_string}")
    
    curr_batch_size = get_dynamic_batch_size(n_embed, seq_len)
    log_message(f" >> Dynamic Sizing: Seq_Len={seq_len} | Embed={n_embed} -> Batch_Size={curr_batch_size}")
    log_message(f" >> Warmup Epochs: {warmup_epochs}")
    log_message(f" >> tau_m={tau_m} | act_fx={act_fx}")

    # Sync global config so downstream components relying on it stay consistent with tuned params
    config.seq_len = seq_len
    config.batch_size = curr_batch_size
    config.n_embed = n_embed
    config.n_heads = n_heads
    config.tau_m = tau_m
    config.act_fx = act_fx

    model = None
    try:
        # Reload Data with new Block Size
        loader = DataLoader(seq_len=seq_len, batch_size=curr_batch_size)
        train_loader, valid_loader, _ = loader.load_and_prepare_data()
        clean_memory()
        
        # Initialize Model
        dkey = random.PRNGKey(int(time.time()))
        model = NGCTransformer(
            dkey, 
            batch_size=curr_batch_size, 
            seq_len=seq_len, 
            n_embed=n_embed,
            vocab_size=config.vocab_size, 
            n_layers=n_layers, 
            n_heads=n_heads,
            T=T, 
            dt=1.0, 
            tau_m=tau_m, 
            act_fx=act_fx, 
            eta=eta,
            dropout_rate=dropout, 
            pos_learnable=config.pos_learnable,
            optim_type=config.optim_type,
            wub=wub, 
            wlb=wlb, 
            exp_dir="exp",
            model_name="ngc_evo_temp"
        )

        # Training Loop
        log_message(f"\n {'Batch':<6} | {'Energy (EFE)':<14} | {'PPL':<12} | {'CL (Loss)':<12}")
        log_message(f" {'-'*52}")

        total_efe = 0.0
        total_ppl = 0.0
        total_ce = 0.0
        batch_count = 0
        max_batches_per_trial = 10
        warmup_steps = max(1, warmup_epochs)
        warmup_factors = np.linspace(0.1, 1.0, warmup_steps)
        
        train_iter = iter(train_loader)
        
        for i in range(max_batches_per_trial):
            try:
                batch = next(train_iter)
            except StopIteration:
                break

            # Linear warmup on eta: 10% -> 100%
            if i < warmup_steps:
                eta_scale = float(warmup_factors[i])
            else:
                eta_scale = 1.0
            effective_eta = eta * eta_scale
            set_model_eta(model, effective_eta)

            inputs = batch[0][1] 
            targets = batch[1][1]
            targets_onehot = jnp.eye(config.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, config.vocab_size)
            
            # Forward + Update
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            y_pred = yMu_inf.reshape(-1, config.vocab_size)
            
            # NaN Check
            if jnp.isnan(_EFE) or jnp.isnan(y_pred).any():
                log_message(f" [!] Trial Killed: NaN detected at batch {i}")
                return 1e9 
                
            batch_ce = float(measure_CatNLL(y_pred, targets_flat).mean())
            batch_ppl = float(np.exp(batch_ce))
            batch_efe = float(_EFE)

            
            log_message(f" {i:<6} | {batch_efe:<14.4f} | {batch_ppl:<12.2f} | {batch_ce:<12.4f}")
            
            total_efe += batch_efe
            total_ppl += batch_ppl
            total_ce += batch_ce
            batch_count += 1
            clean_memory()

        # Calculate Averages
        avg_efe = total_efe / batch_count if batch_count > 0 else 0
        avg_ppl = total_ppl / batch_count if batch_count > 0 else 0
        avg_ce = total_ce / batch_count if batch_count > 0 else 0
        eval_time = time.time() - start_time

        log_message(f"\n[Summary] Avg Energy: {avg_efe:.4f} | Avg PPL: {avg_ppl:.2f} | Avg CE: {avg_ce:.4f} | Time: {eval_time:.2f}s")

        # Validation (The Fitness Function)
        val_ce, _ = eval_model(model, valid_loader, config.vocab_size)
        log_message(f"[Result] Validation Score (CE): {val_ce:.4f}")

        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_id,
                "ok",
                clean_string,
                f"{avg_efe:.6f}",
                f"{avg_ppl:.6f}",
                f"{avg_ce:.6f}",
                f"{val_ce:.6f}",
                f"{eval_time:.2f}"
            ])

        if val_ce < BEST_RECORD["val_ce"]:
            BEST_RECORD.update({
                "val_ce": val_ce,
                "avg_ppl": avg_ppl,
                "avg_efe": avg_efe,
                "avg_ce": avg_ce,
                "phenotype": clean_string,
                "trial_id": trial_id
            })
            with open(BEST_FILE, "w") as f:
                f.write(
                    "Best Trial So Far\n"
                    f"Trial ID: {trial_id}\n"
                    f"Phenotype: {clean_string}\n"
                    f"Validation CE: {val_ce:.6f}\n"
                    f"Avg CE: {avg_ce:.6f}\n"
                    f"Avg PPL: {avg_ppl:.6f}\n"
                    f"Avg EFE: {avg_efe:.6f}\n"
                    f"Elapsed: {eval_time:.2f}s\n"
                )
        
        clean_memory()
        return float(val_ce)

    except KeyboardInterrupt:
        clean_memory()
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "keyboard_interrupt", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9
    except Exception as e:
        log_message(f"\n[!] ERROR in Trial: {e}")
        error_trace = traceback.format_exc()
        # log_message(error_trace) # Uncomment if you need deep debugging
        clean_memory()
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "exception", clean_string, "", "", "", "", f"{time.time() - start_time:.2f}"])
        return 1e9 
    finally:
        if model: del model
        clean_memory()
        try:
            del loader, train_loader, valid_loader, train_iter
        except Exception:
            pass
        clean_memory()


def generate_visualization(csv_path=RESULT_CSV, out_path="trial_metrics.png"):
    if not Path(csv_path).exists():
        log_message(f"[viz] No CSV found at {csv_path}; skipping plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_message("[viz] matplotlib not installed; skipping plot.")
        return

    trials = []
    val_ce = []
    avg_ppl = []
    avg_efe = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                trials.append(int(row["trial_id"]))
                val_ce.append(float(row["val_ce"]))
                avg_ppl.append(float(row["avg_ppl"]))
                avg_efe.append(float(row["avg_efe"]))
            except Exception:
                continue

    if not trials:
        log_message("[viz] No successful trials to plot.")
        return

    best_idx = int(np.argmin(np.array(val_ce)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(trials, val_ce, marker="o", label="Validation CE")
    axes[0].scatter(trials[best_idx], val_ce[best_idx], color="red", label="Best")
    axes[0].set_ylabel("Validation CE")
    axes[0].legend()

    axes[1].plot(trials, avg_ppl, marker="o", color="purple", label="Avg PPL")
    axes[1].set_ylabel("Avg PPL")
    axes[1].legend()

    axes[2].plot(trials, avg_efe, marker="o", color="green", label="Avg EFE")
    axes[2].set_ylabel("Avg EFE")
    axes[2].set_xlabel("Trial ID")
    axes[2].legend()

    fig.suptitle("Trial Metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    log_message(f"[viz] Saved visualization to {out_path}")

# --- 4. The Genetic Algorithm Execution ---
def main():
    # Load Grammar
    grammar = al.Grammar(bnf_text=bnf_text)
    
    # Define Evolutionary Algorithm
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        population_size=6,   
        max_generations=4,    
        verbose=True
    )
    
    log_message("\n" + "#"*50)
    log_message("STARTING GRAMMAR-GUIDED EVOLUTION")
    log_message("#"*50)
    
    try:
        best_ind = ea.run()
        
        # --- SAVE RESULTS ---
        log_message("\n" + "#"*50)
        log_message("OPTIMIZATION COMPLETED")
        log_message("#"*50)
        
        report = f"""
Optimization Success.

Best Fitness (Validation CE): {best_ind.fitness:.5f}
Best Parameters Found:
{best_ind.phenotype}

    Best Running Metrics Tracked:
    Trial ID: {_fmt_metric(BEST_RECORD['trial_id'])}
    Phenotype: {BEST_RECORD['phenotype'] or 'N/A'}
    Validation CE: {_fmt_metric(BEST_RECORD['val_ce'])}
    Avg CE: {_fmt_metric(BEST_RECORD['avg_ce'])}
    Avg PPL: {_fmt_metric(BEST_RECORD['avg_ppl'])}
    Avg EFE: {_fmt_metric(BEST_RECORD['avg_efe'])}
"""
        print(report)
        with open(RESULT_FILE, "w") as f:
            f.write(report)

        generate_visualization()
            
    except KeyboardInterrupt:
        print("\nOptimization stopped manually.")

if __name__ == "__main__":
    main()