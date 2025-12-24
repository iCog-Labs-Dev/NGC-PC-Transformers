import alogos as al
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Import Error: {e}. Check project path.")
    sys.exit(1)


FIXED_BS = 8
FIXED_BLOCK = 16
FIXED_VOCAB = 5000

bnf_text = """
<hparams>      ::= <embed> "," <heads> "," <layers> "," <dropout> "," <eta> "," <t_step> "," <act> "," <w_init>

<embed>        ::= "n_embed=64" | "n_embed=128" | "n_embed=256"
<heads>        ::= "n_heads=4" | "n_heads=6" | "n_heads=8"
<layers>       ::= "n_layers=2" | "n_layers=4" | "n_layers=6"
<dropout>      ::= "dropout_rate=0.0" | "dropout_rate=0.1" | "dropout_rate=0.5"
<eta>          ::= "eta=0.01" | "eta=0.005" | "eta=0.001"
<t_step>       ::= "T=10" | "T=20"
<act>          ::= "act_fx=identity" | "act_fx=lrelu" | "act_fx=tanh"
<w_init>       ::= "w_val=0.01" | "w_val=0.05" | "w_val=0.1"
"""


data_loader = DataLoader(seq_len=FIXED_BLOCK, batch_size=FIXED_BS)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

def objective_function(phenotype_string):
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    print(f"\n[Testing Config]: {clean_string}")
    
    model = None
    
    try:
        # Parse all params from the string
        params = {p.split('=')[0]: p.split('=')[1] for p in clean_string.split(',')}
        
        dkey = random.PRNGKey(42)
        model = NGCTransformer(
            dkey, 
            batch_size=FIXED_BS, 
            seq_len=FIXED_BLOCK, 
            n_embed=int(params['n_embed']), # Now searchable
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
            model_name="tuning_model"
        )

        # Training process
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

        # Evaluate fitness
        val_ce, _ = eval_model(model, valid_loader, FIXED_VOCAB)
        
        if np.isnan(val_ce) or np.isinf(val_ce):
            return 2000.0
            
        print(f"   >>> Result CE: {val_ce:.4f}")
        return float(val_ce)

    except Exception as e:
        print(f"   [!] Individual Failed: {e}")
        return 5000.0

    finally:
        # Memory cleanup
        if model is not None:
            del model
        clear_caches()
        gc.collect()

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
    print("STARTING UPDATED SEARCH")
    print(f"Fixed: BS={FIXED_BS}, Seq={FIXED_BLOCK}, Vocab={FIXED_VOCAB}")
    print("="*40)
    
    best_ind = ea.run()

    print("\n" + "="*40)
    print("BEST CONFIG FOUND")
    print("="*40)
    print(f"Params: {best_ind.phenotype}")
    print(f"Loss: {best_ind.fitness:.4f}")

if __name__ == "__main__":
    main()