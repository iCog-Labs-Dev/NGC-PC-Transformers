import alogos as al
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import re

# --- Project Imports ---
try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Import Error: {e}. Ensure this script is in the root of your project.")
    sys.exit(1)

# ==========================================
# 1. GRAMMAR DEFINITION
# ==========================================
# We define the search space. alogos will pick one option from each rule.
bnf_text = """
<hparams>      ::= <bs> "," <embed> "," <layers> "," <dropout> "," <block> "," <vocab> "," <eta> "," <t_step> "," <act> "," <w_init>

<bs>           ::= "batch_size=" <bs_v>
<bs_v>         ::= "16" | "32" | "64"

<embed>        ::= "n_embed=128,n_heads=4" | "n_embed=128,n_heads=8" | "n_embed=256,n_heads=4" | "n_embed=256,n_heads=8" | "n_embed=512,n_heads=8"

<layers>       ::= "n_layers=" <l_v>
<l_v>          ::= "2" | "3" | "4" | "6"

<dropout>      ::= "dropout_rate=" <d_v>
<d_v>          ::= "0.0" | "0.1" | "0.3"

<block>        ::= "block_size=" <bl_v>
<bl_v>         ::= "32" | "64" | "128"

<vocab>        ::= "vocab_size=" <v_v>
<v_v>          ::= "1000" | "5000" | "10000"

<eta>          ::= "eta=" <e_v>
<e_v>          ::= "0.01" | "0.005" | "0.001"

<t_step>       ::= "T=" <t_v>
<t_v>          ::= "10" | "20"

<act>          ::= "act_fx=" <a_v>
<a_v>          ::= "identity" | "lrelu" | "tanh"

<w_init>       ::= "w_val=" <w_v>
<w_v>          ::= "0.01" | "0.05" | "0.1" | "0.2"
"""

# ==========================================
# 2. GLOBAL DATA LOADING
# ==========================================
print("--- Loading Dataset ---")
# Use large enough defaults for the loader; the objective function will slice them
data_loader = DataLoader(seq_len=128, batch_size=64)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

# ==========================================
# 3. OBJECTIVE FUNCTION (Fitness Evaluation)
# ==========================================
def objective_function(phenotype_string):
    # Fix the alogos quote bug: remove all double quotes and extra spaces
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    print(f"\n[Testing Config]: {clean_string}")
    
    try:
        # A. Parse the clean string into a dictionary
        params = {}
        parts = clean_string.split(',')
        for part in parts:
            if '=' in part:
                k, v = part.split('=')
                # Type conversion
                if k in ['act_fx']:
                    params[k] = v
                elif k in ['dropout_rate', 'eta', 'w_val']:
                    params[k] = float(v)
                else:
                    params[k] = int(v)

        # B. Setup Model Symmetries
        # As per your requirement: wub = val, wlb = -val
        w_init = params.get('w_val', 0.1)
        wub = w_init
        wlb = -w_init

        # C. Initialize NGCTransformer
        dkey = random.PRNGKey(42)
        model = NGCTransformer(
            dkey,
            batch_size=params['batch_size'],
            seq_len=params['block_size'],
            n_embed=params['n_embed'],
            vocab_size=params['vocab_size'],
            n_layers=params['n_layers'],
            n_heads=params['n_heads'],
            T=params['T'],
            dt=1.,
            tau_m=10.0,
            act_fx=params['act_fx'],
            eta=params['eta'],
            dropout_rate=params['dropout_rate'],
            exp_dir="exp_tuning",
            wub=wub,
            wlb=wlb,
            model_name="tuning_model"
        )

        # D. Fast Training (Evaluation Loop)
        # We run a small number of steps to see if the model learns
        NUM_STEPS = 40 
        train_iter = iter(train_loader)
        
        for _ in range(NUM_STEPS):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Slice data to match the current individual's hyperparameters
            inputs = batch[0][1][:params['batch_size'], :params['block_size']]
            targets = batch[1][1][:params['batch_size'], :params['block_size']]
            
            # One-hot encoding
            targets_flat = jnp.eye(params['vocab_size'])[targets].reshape(-1, params['vocab_size'])
            
            # Model process (Update synapses)
            model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

        # E. Calculate Fitness (Validation CE)
        # We want to minimize Cross Entropy
        val_ce, _ = eval_model(model, valid_loader, params['vocab_size'])
        
        if np.isnan(val_ce) or np.isinf(val_ce):
            return 1000.0 # Return penalty for unstable models
            
        print(f"   >>> Result CE: {val_ce:.4f}")
        return float(val_ce)

    except Exception as e:
        print(f"   [!] Individual Failed: {e}")
        return 1000.0

# ==========================================
# 4. EVOLUTIONARY RUN
# ==========================================
def main():
    # 1. Create Grammar object
    grammar = al.Grammar(bnf_text=bnf_text)

    # 2. Configure Genetic Algorithm
    # Adjust population_size and max_generations based on your time/GPU
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        population_size=12, 
        max_generations=5
    )

    # 3. Start Search
    print("\n" + "="*40)
    print("STARTING GRAMMAR-GUIDED SEARCH")
    print("="*40)
    best_ind = ea.run()

    # 4. Final Results
    print("\n" + "="*40)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*40)
    
    clean_best = best_ind.phenotype.replace('"', '').replace(' ', '')
    print(f"Config: {clean_best}")
    print(f"Validation CE: {best_ind.fitness:.4f}")
    
    # Final conversion logic for your config file
    print("\nSuggested config.py values:")
    for part in clean_best.split(','):
        if 'w_val' in part:
            v = float(part.split('=')[1])
            print(f"wub = {v}")
            print(f"wlb = {-v}")
        else:
            print(part)

if __name__ == "__main__":
    main()