import alogos as al
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc
import os
from config import Config as config

# Prevent JAX memory pre-allocation issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Import Error: {e}. Check project path.")
    sys.exit(1)

# ==========================================
# 1. PARAMETERS FROM CONFIG
# ==========================================
FIXED_BS = config.batch_size
FIXED_BLOCK = config.seq_len
FIXED_VOCAB = config.vocab_size

bnf_text = """
<hparams>      ::= <embed> "," <heads> "," <layers> "," <eta> "," <act> "," <w_init>

<embed>        ::= "n_embed=64" | "n_embed=128" | "n_embed=256"
<heads>        ::= "n_heads=4" | "n_heads=6" | "n_heads=8"
<layers>       ::= "n_layers=2" | "n_layers=4" | "n_layers=6"

<eta>          ::= "eta=0.01" | "eta=0.005" | "eta=0.001"

<act>          ::= "act_fx=identity" | "act_fx=lrelu" | "act_fx=tanh"
<w_init>       ::= "w_val=0.01" | "w_val=0.05" | "w_val=0.1"
"""

# Initialize Data
data_loader = DataLoader(seq_len=FIXED_BLOCK, batch_size=FIXED_BS)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

# ==========================================
# 2. OBJECTIVE FUNCTION
# ==========================================
def objective_function(phenotype_string):
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    print(f"\n[Testing Config]: {clean_string}")
    
    model = None
    
    try:
        # Parse params
        params = {p.split('=')[0]: p.split('=')[1] for p in clean_string.split(',')}
        
        dkey = random.PRNGKey(42)
        model = NGCTransformer(
            dkey, 
            batch_size=FIXED_BS, 
            seq_len=FIXED_BLOCK, 
            n_embed=int(params['n_embed']),
            vocab_size=FIXED_VOCAB, 
            n_layers=int(params['n_layers']), 
            n_heads=int(params['n_heads']),
            T=int(params['T']), 
            dt=1.0, 
            tau_m=config.tau_m, 
            act_fx=params['act_fx'], 
            eta=float(params['eta']),
            dropout_rate=float(params['dropout_rate']), 
            pos_learnable=config.pos_learnable,
            optim_type=config.optim_type,
            wub=float(params['w_val']), 
            wlb=-float(params['w_val']), 
        )

        # Fast training steps
        train_iter = iter(train_loader)
        for _ in range(10):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            inputs = batch[0][1][:FIXED_BS, :FIXED_BLOCK]
            targets = batch[1][1][:FIXED_BS, :FIXED_BLOCK]
            targets_flat = jnp.eye(FIXED_VOCAB)[targets].reshape(-1, FIXED_VOCAB)
            
            model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

        # --- Evaluation (CE and PPL) ---
        val_ce, _ = eval_model(model, valid_loader, FIXED_VOCAB)
        
        if np.isnan(val_ce) or np.isinf(val_ce):
            return 2000.0
        
        # Calculate Perplexity
        val_ppl = np.exp(val_ce)
            
        print(f"   >>> Result CE: {val_ce:.4f} | PPL: {val_ppl:.4f}")
        return float(val_ce)

    except Exception as e:
        print(f"   [!] Individual Failed: {e}")
        return 5000.0

    finally:
        # Explicit Memory Cleanup for JAX
        if model is not None:
            del model
        clear_caches()
        gc.collect()

# ==========================================
# 3. MAIN SEARCH LOOP
# ==========================================
def main():
    grammar = al.Grammar(bnf_text=bnf_text)
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        population_size=10, 
        max_generations=5
    )

    print("\n" + "="*40)
    print("STARTING SEARCH (CE + PPL TRACKING)")
    print(f"Fixed Params: BS={FIXED_BS}, Seq={FIXED_BLOCK}, Vocab={FIXED_VOCAB}")
    print("="*40)
    
    best_ind = ea.run()

    print("\n" + "="*40)
    print("BEST CONFIG FOUND")
    print("="*40)
    print(f"Params: {best_ind.phenotype}")
    print(f"Best CE: {best_ind.fitness:.4f}")
    print(f"Best PPL: {np.exp(best_ind.fitness):.4f}")

if __name__ == "__main__":
    main()