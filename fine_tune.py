import alogos as al
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os

# --- Imports from your project structure ---
# Ensure these match your actual file names
try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please make sure fine_tune.py is in the same folder as model.py and data_preprocess/")
    sys.exit(1)

# ==========================================
# 1. GRAMMAR DEFINITION (Hyperparameter Search Space)
# ==========================================
# This grammar enforces:
# 1. n_embed is divisible by n_heads (pre-defined valid pairs)
# 2. wlb is always negative of wub (symmetric initialization)
# ==========================================

bnf_text = """
<hparams>      ::= <batch_size> "," <embed_config> "," <n_layers> "," <dropout_rate> "," <block_size> "," <vocab_size> "," <eta> "," <T> "," <act_fx> "," <weight_init>

<batch_size>   ::= "batch_size=" <bs_val>
<bs_val>       ::= "16" | "32" | "64"

<embed_config> ::= "n_embed=64, n_heads=2" | "n_embed=64, n_heads=4" | "n_embed=128, n_heads=4" | "n_embed=128, n_heads=8" | "n_embed=256, n_heads=4" | "n_embed=256, n_heads=8" | "n_embed=512, n_heads=8"

<n_layers>     ::= "n_layers=" <nl_val>
<nl_val>       ::= "2" | "3" | "4" | "6"

<dropout_rate> ::= "dropout_rate=" <dr_val>
<dr_val>       ::= "0.0" | "0.1" | "0.2" | "0.3" | "0.5"

<block_size>   ::= "block_size=" <sl_val>
<sl_val>       ::= "32" | "64" | "128"

<vocab_size>   ::= "vocab_size=" <vs_val>
<vs_val>       ::= "1000" | "5000" | "10000"

<eta>          ::= "eta=" <eta_val>
<eta_val>      ::= "0.05" | "0.02" | "0.01" | "0.005" | "0.001"

<T>            ::= "T=" <t_val>
<t_val>        ::= "5" | "10" | "15" | "20"

<act_fx>       ::= "act_fx='" <act_val> "'"
<act_val>      ::= "identity" | "lrelu" | "tanh"

<weight_init>  ::= "w_init=" <w_val>
<w_val>        ::= "0.01" | "0.02" | "0.05" | "0.1" | "0.2"
"""

# ==========================================
# 2. DATA PREPARATION (Load Once)
# ==========================================
print("--- Loading Data (Global) ---")
# We load data once globally so we don't reload for every genetic individual
# We use a standard max sequence length here; the model will handle slicing
GLOBAL_SEQ_LEN = 128 
GLOBAL_BATCH_SIZE = 64 # This is just for the loader; model batch size varies

data_loader = DataLoader(seq_len=GLOBAL_SEQ_LEN, batch_size=GLOBAL_BATCH_SIZE)
train_loader, valid_loader, test_loader = data_loader.load_and_prepare_data()

# ==========================================
# 3. OBJECTIVE FUNCTION
# ==========================================
def objective_function(phenotype_string):
    """
    1. Parses the grammar string (phenotype).
    2. Initializes the NGCTransformer.
    3. Runs a short training loop.
    4. Returns Validation Loss (to be minimized).
    """
    print(f"\n[Testing] Config: {phenotype_string}")
    
    # --- A. Parse Parameters ---
    try:
        params = {}
        # Example string: "batch_size=32, n_embed=64, n_heads=4, ..."
        parts = phenotype_string.split(',')
        
        for part in parts:
            key, value = part.strip().split('=')
            
            if key == 'act_fx':
                params[key] = value.replace("'", "")
            elif key == 'w_init':
                # Special logic: wub = val, wlb = -val
                val = float(value)
                params['wub'] = val
                params['wlb'] = -val
            elif key in ['dropout_rate', 'eta']:
                params[key] = float(value)
            else:
                params[key] = int(value)

        # --- B. Initialize Model ---
        dkey = random.PRNGKey(1234)
        
        # We enforce a shorter sequence length for training if block_size changed
        seq_len = params['block_size']
        vocab_size = params['vocab_size']
        
        model = NGCTransformer(
            dkey,
            batch_size=params['batch_size'],
            seq_len=seq_len,
            n_embed=params['n_embed'],
            vocab_size=vocab_size,
            n_layers=params['n_layers'],
            n_heads=params['n_heads'],
            T=params['T'],
            dt=1.,
            tau_m=10., 
            act_fx=params['act_fx'],
            eta=params['eta'],
            dropout_rate=params['dropout_rate'],
            exp_dir="exp_evo",
            loadDir=None,
            pos_learnable=True,
            optim_type="adam", 
            wub=params['wub'],
            wlb=params['wlb'],
            model_name="ngc_evo_temp"
        )

        # --- C. Short Training Loop (Fitness Check) ---
        # We run fewer iterations than 'train.py' just to see if parameters are promising
        SEARCH_ITERATIONS = 50 # Enough to see if loss goes down
        
        # Create a simple iterator for the loader
        train_iter = iter(train_loader)
        
        total_nll = 0.0
        
        for i in range(SEARCH_ITERATIONS):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Extract data
            inputs = batch[0][1] # Shape: (Loader_Batch, Seq_Len)
            targets = batch[1][1]

            # 1. Adjust Batch Size (if model BS < Loader BS)
            curr_bs = params['batch_size']
            inputs = inputs[:curr_bs]
            targets = targets[:curr_bs]
            
            # 2. Adjust Sequence Length (if model Block_Size < Loader Seq_Len)
            # We slice the sequence to match the model's block_size
            inputs = inputs[:, :seq_len]
            targets = targets[:, :seq_len]

            # 3. One Hot Encoding
            targets_onehot = jnp.eye(vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, vocab_size)

            # 4. Train Step (adapt_synapses=True)
            yMu_inf, _, _ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

        # --- D. Validation Step ---
        # Evaluate on the validation set
        # Note: We must ensure eval_model handles the slicing/batch size logic too, 
        # but usually eval_model iterates the loader. 
        # If eval_model relies on global config, it might break. 
        # Assuming eval_model takes the model object and uses its internal settings:
        
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        
        # Check for NaN (model exploded)
        if np.isnan(dev_ce) or np.isinf(dev_ce):
            return 1000.0 # High penalty
            
        return float(dev_ce)

    except Exception as e:
        # If parameters are invalid or OOM occurs
        print(f"  [!] Failed config: {e}")
        return 1000.0

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # 1. Create Grammar
    grammar = al.Grammar(bnf_text=bnf_text)

    # 2. Setup Evolutionary Algorithm
    # 'min' because we want to minimize Cross Entropy
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        max_generations=5,    # Increase this for better results (e.g., 20)
        population_size=10,   # Increase this for better diversity (e.g., 20-50)
        offspring_size=10,
        verbose=True
    )

    print("Starting Evolutionary Hyperparameter Search...")
    best_individual = ea.run()

    # 3. Output Results
    print("\n" + "="*40)
    print("SEARCH COMPLETE")
    print("="*40)
    print(f"Best Fitness (Validation CE): {best_individual.fitness:.4f}")
    print("Best Hyperparameters String:")
    print(best_individual.phenotype)
    print("="*40)
    
    # Clean parsing for the user to copy
    print("\nCopy these into config.py:")
    parts = best_individual.phenotype.split(',')
    for part in parts:
        if 'w_init' in part:
            val = float(part.split('=')[1])
            print(f"wub = {val}")
            print(f"wlb = {-val}")
        else:
            print(part.strip())

if __name__ == "__main__":
    main()