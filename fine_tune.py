import alogos as al
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc  # Garbage Collector for memory cleanup

# Set JAX to not pre-allocate 100% of VRAM immediately
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- Project Imports ---
try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Import Error: {e}. Ensure script is in the project root.")
    sys.exit(1)

# ==========================================
# 1. FIXED PARAMETERS (The "Safe" Zone)
# ==========================================
FIXED_BS = 8        # Lowered to prevent OOM
FIXED_BLOCK = 16    
FIXED_VOCAB = 5000  
FIXED_EMBED = 64    

# ==========================================
# 2. CORRECTED GRAMMAR
# ==========================================
bnf_text = """
<hparams>      ::= <heads> "," <layers> "," <dropout> "," <eta> "," <t_step> "," <act> "," <w_init>

<heads>        ::= "n_heads=4" | "n_heads=8"
<layers>       ::= "n_layers=2" | "n_layers=4" | "n_layers=6"
<dropout>      ::= "dropout_rate=0.0" | "dropout_rate=0.1"
<eta>          ::= "eta=0.01" | "eta=0.005" | "eta=0.001"
<t_step>       ::= "T=10" | "T=20"
<act>          ::= "act_fx=identity" | "act_fx=lrelu" | "act_fx=tanh"
<w_init>       ::= "w_val=0.01" | "w_val=0.05" | "w_val=0.1"
"""

# ==========================================
# 3. GLOBAL DATA LOADING
# ==========================================
print("--- Loading Dataset ---")
data_loader = DataLoader(seq_len=FIXED_BLOCK, batch_size=FIXED_BS)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

# ==========================================
# 4. OBJECTIVE FUNCTION (With Forced Cleanup)
# ==========================================
def objective_function(phenotype_string):
    # Clean the input string from alogos
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    print(f"\n[Testing Config]: {clean_string}")
    
    model = None  # Initialize for finally block cleanup
    
    try:
        # Parse params into dictionary
        params = {p.split('=')[0]: p.split('=')[1] for p in clean_string.split(',')}
        
        # Initialize NGCTransformer
        dkey = random.PRNGKey(42)
        model = NGCTransformer(
            dkey, 
            batch_size=FIXED_BS, 
            seq_len=FIXED_BLOCK, 
            n_embed=FIXED_EMBED,
            vocab_size=FIXED_VOCAB, 
            n_layers=int(params['n_layers']), 
            n_heads=int(params['n_heads']),
            T=int(params['T']), 
            dt=1.0, 
            tau_m=10.0, 
            act_fx=params['act_fx'], 
            eta=float(params['eta']),
            dropout_rate=float(params['dropout_rate']), 
            exp_dir="exp_evo",
            wub=float(params['w_val']), 
            wlb=-float(params['w_val']), 
            model_name="temp_model"
        )

        # Small training burst to evaluate performance
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

        # Evaluation
        val_ce, _ = eval_model(model, valid_loader, FIXED_VOCAB)
        
        if np.isnan(val_ce) or np.isinf(val_ce):
            return 2000.0
            
        print(f"   >>> Result CE: {val_ce:.4f}")
        return float(val_ce)

    except Exception as e:
        print(f"   [!] Individual Failed: {e}")
        return 5000.0

    finally:
        # --- CRITICAL MEMORY CLEANUP ---
        if model is not None:
            del model
        
        # Clear JAX compilation cache (XLA)
        clear_caches()
        
        # Force Python's Garbage Collector
        gc.collect()
        # -------------------------------

# ==========================================
# 5. EVOLUTIONARY RUN
# ==========================================
def main():
    grammar = al.Grammar(bnf_text=bnf_text)
    
    # Keeping population small to save memory and time
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        population_size=8, 
        max_generations=3
    )

    print("\n" + "="*40)
    print("STARTING MEMORY-SAFE SEARCH")
    print("="*40)
    
    best_ind = ea.run()

    print("\n" + "="*40)
    print("BEST CONFIGURATION")
    print("="*40)
    print(f"Params: {best_ind.phenotype}")
    print(f"Fitness: {best_ind.fitness:.4f}")

if __name__ == "__main__":
    main()