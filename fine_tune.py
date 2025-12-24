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
    print(f"Import Error: {e}")
    sys.exit(1)

# ==========================================
# 1. FIXED PARAMETERS
# ==========================================
FIXED_BS = 64
FIXED_BLOCK = 128
FIXED_VOCAB = 11710 
FIXED_EMBED = 128

# ==========================================
# 2. FIXED GRAMMAR (Corrected eta prefix)
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
data_loader = DataLoader(seq_len=FIXED_BLOCK, batch_size=FIXED_BS)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

# ==========================================
# 4. OBJECTIVE FUNCTION
# ==========================================
def objective_function(phenotype_string):
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    print(f"\n[Testing Config]: {clean_string}")
    
    try:
        # Parse params
        params = {p.split('=')[0]: p.split('=')[1] for p in clean_string.split(',')}
        
        # Convert types
        p_layers = int(params['n_layers'])
        p_heads = int(params['n_heads'])
        p_dropout = float(params['dropout_rate'])
        p_eta = float(params['eta'])
        p_T = int(params['T'])
        p_act = params['act_fx']
        p_w = float(params['w_val'])

        dkey = random.PRNGKey(42)
        model = NGCTransformer(
            dkey, batch_size=FIXED_BS, seq_len=FIXED_BLOCK, n_embed=FIXED_EMBED,
            vocab_size=FIXED_VOCAB, n_layers=p_layers, n_heads=p_heads,
            T=p_T, dt=1., tau_m=10.0, act_fx=p_act, eta=p_eta,
            dropout_rate=p_dropout, exp_dir="exp",
            wub=p_w, wlb=-p_w, model_name="ngc_transformer"
        )

        train_iter = iter(train_loader)
        for _ in range(10): # Reduced steps for speed
            batch = next(train_iter)
            inputs = batch[0][1][:FIXED_BS, :FIXED_BLOCK]
            targets = batch[1][1][:FIXED_BS, :FIXED_BLOCK]
            
            # The model is crashing on label dimensions. 
            # We ensure targets_flat matches the expected output shape
            targets_flat = jnp.eye(FIXED_VOCAB)[targets].reshape(-1, FIXED_VOCAB)
            
            model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

        val_ce, _ = eval_model(model, valid_loader, FIXED_VOCAB)
        return float(val_ce) if not (np.isnan(val_ce) or np.isinf(val_ce)) else 2000.0

    except Exception as e:
        print(f"   [!] Individual Failed: {e}")
        return 5000.0

# ==========================================
# 5. MAIN
# ==========================================
def main():
    grammar = al.Grammar(bnf_text=bnf_text)
    ea = al.EvolutionaryAlgorithm(grammar, objective_function, 'min', population_size=8, max_generations=3)
    best_ind = ea.run()
    print(f"\nBEST CONFIG: {best_ind.phenotype}")

if __name__ == "__main__":
    main()