import alogos as al
import jax.numpy as jnp
from jax import random
import numpy as np
import sys

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
# 1. FIXED PARAMETERS (The "Safe" Zone)
# ==========================================
FIXED_BS = 64
FIXED_BLOCK = 12   # Locked to 12 to satisfy the (12,12) reshape logic
FIXED_VOCAB = 5000
FIXED_EMBED = 128

# ==========================================
# 2. GRAMMAR DEFINITION (Searching for Logic/Dynamics)
# ==========================================
bnf_text = """
<hparams>      ::= <heads> "," <layers> "," <dropout> "," <eta> "," <t_step> "," <act> "," <w_init>

<heads>        ::= "n_heads=4" | "n_heads=8"
<layers>       ::= "n_layers=2" | "n_layers=4" | "n_layers=6"
<dropout>      ::= "dropout_rate=0.0" | "dropout_rate=0.1"
<eta>          ::= "eta=0.01" | "0.005" | "0.001"
<t_step>       ::= "T=10" | "T=20"
<act>          ::= "act_fx=identity" | "act_fx=lrelu" | "act_fx=tanh"
<w_init>       ::= "w_val=0.01" | "w_val=0.05" | "w_val=0.1"
"""

# ==========================================
# 3. GLOBAL DATA LOADING
# ==========================================
print("--- Loading Dataset ---")
data_loader = DataLoader(seq_len=128, batch_size=FIXED_BS)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

# ==========================================
# 4. OBJECTIVE FUNCTION
# ==========================================
def objective_function(phenotype_string):
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    print(f"\n[Testing Config]: {clean_string}")
    
    try:
        # Parse searchable params
        params = {}
        for part in clean_string.split(','):
            if '=' in part:
                k, v = part.split('=')
                if k == 'act_fx': params[k] = v
                elif k in ['dropout_rate', 'eta', 'w_val']: params[k] = float(v)
                else: params[k] = int(v)

        dkey = random.PRNGKey(42)
        
        # Initialize with FIXED + SEARCHED params
        model = NGCTransformer(
            dkey,
            batch_size=FIXED_BS,
            seq_len=FIXED_BLOCK,
            n_embed=FIXED_EMBED,
            vocab_size=FIXED_VOCAB,
            n_layers=params['n_layers'],
            n_heads=params['n_heads'],
            T=params['T'],
            dt=1.,
            tau_m=10.0,
            act_fx=params['act_fx'],
            eta=params['eta'],
            dropout_rate=params['dropout_rate'],
            exp_dir="exp_tuning",
            wub=params['w_val'],
            wlb=-params['w_val'],
            model_name="tuning_model"
        )

        # Fast training steps
        train_iter = iter(train_loader)
        for _ in range(20):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Slicing inputs to FIXED_BLOCK
            inputs = batch[0][1][:FIXED_BS, :FIXED_BLOCK]
            targets = batch[1][1][:FIXED_BS, :FIXED_BLOCK]
            targets_flat = jnp.eye(FIXED_VOCAB)[targets].reshape(-1, FIXED_VOCAB)
            
            model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

        # Validation CE
        val_ce, _ = eval_model(model, valid_loader, FIXED_VOCAB)
        
        if np.isnan(val_ce) or np.isinf(val_ce):
            return 2000.0
            
        print(f"   >>> Result CE: {val_ce:.4f}")
        return float(val_ce)

    except Exception as e:
        print(f"   [!] Individual Failed: {e}")
        return 5000.0

# ==========================================
# 5. EVOLUTIONARY RUN
# ==========================================
def main():
    grammar = al.Grammar(bnf_text=bnf_text)
    ea = al.EvolutionaryAlgorithm(
        grammar, objective_function, 'min', 
        population_size=10, 
        max_generations=5
    )

    print("\n" + "="*40)
    print("STARTING SEARCH (BS=64, Seq=12, Embed=128)")
    print("="*40)
    best_ind = ea.run()

    print("\n" + "="*40)
    print("BEST TUNABLE PARAMS FOUND")
    print("="*40)
    print(f"Phenotype: {best_ind.phenotype}")
    print(f"Fitness: {best_ind.fitness:.4f}")

if __name__ == "__main__":
    main()