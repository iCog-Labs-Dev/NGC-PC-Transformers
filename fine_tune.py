import os
# CRITICAL FOR JAX TUNING: Prevent JAX from eating all GPU RAM at start
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import optuna
import time
import logging
import gc
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Import your model and data modules
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from eval import eval_model
from ngclearn.utils.metric_utils import measure_CatNLL

# Logging Setup
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_memory():
    """JAX-specific memory cleanup"""
    gc.collect()
    # JAX handles memory via XLA backend, usually clearing Python references 
    # and running gc.collect() is enough if preallocation is disabled.
    try:
        jax.clear_backends()
    except:
        pass

def get_config_from_trial(trial):
    """Define the Search Space for the NGC Transformer"""
    
    # 1. Structural Parameters
    n_embed_candidates = list(range(64, 513, 64)) # [64, 128, ..., 512]
    n_embed = n_embed_candidates[trial.suggest_int('embed_idx', 0, len(n_embed_candidates) - 1)]

    # Ensure n_heads divides n_embed evenly
    valid_heads = [h for h in [2, 4, 8] if n_embed % h == 0]
    if not valid_heads:
        valid_heads = [1]
    n_heads = valid_heads[trial.suggest_int('head_idx', 0, len(valid_heads) - 1)]

    config = {
        'seq_len': 50,  # Keeping fixed or make tunable
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'n_embed': n_embed,
        'n_heads': n_heads,
        'n_layers': trial.suggest_int('n_layers', 2, 6),
        
        # 2. NGC Specific Dynamics
        'n_iter': trial.suggest_int('T', 5, 20), # Steps of inference
        'eta': trial.suggest_float('eta', 1e-4, 5e-2, log=True), # Learning rate
        'tau_m': trial.suggest_int('tau_m', 10, 100), # Membrane time constant
        
        # 3. Regularization & Bounds
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'wub': trial.suggest_float('wub', 0.1, 1.0), # Weight Upper Bound
        'wlb': trial.suggest_float('wlb', -1.0, -0.1), # Weight Lower Bound
        
        # 4. Fixed/Categorical
        'optim_type': "adam",
        'act_fx': "gelu",
        'pos_learnable': True
    }
    
    logger.info(f"Trial Config: Embed={n_embed}, Heads={n_heads}, Layers={config['n_layers']}, Eta={config['eta']:.4f}")
    return config

def train_one_epoch(model, train_loader, vocab_size, epoch_idx):
    """Training loop helper for JAX"""
    total_nll, total_tokens = 0., 0
    batch_count = 0
    
    # Note: In JAX/NGC, usually 'eta' is fixed in the model state 
    # unless you explicitly expose it in model.process arguments.
    # Assuming standard calling convention from your snippet:
    
    for batch in train_loader:
        inputs = batch[0][1]
        targets = batch[1][1]
        
        targets_onehot = jnp.eye(vocab_size)[targets]
        targets_flat = targets_onehot.reshape(-1, vocab_size)
        
        # NGC Update
        yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
        
        # Calculate Stats for logging
        y_pred = yMu_inf.reshape(-1, vocab_size)
        
        # Optimization: Calculate NLL only periodically to save compute
        if batch_count % 10 == 0:
            batch_nll = measure_CatNLL(y_pred, targets_flat)
            # Check for NaNs (common in JAX if LR is too high)
            if jnp.isnan(batch_nll):
                return float('inf') 
                
        batch_count += 1

    return 0.0 # Return value not critical here, we rely on eval

def objective(trial):
    """The Main Bayesian Objective"""
    start_time = time.time()
    model = None
    
    try:
        cleanup_memory()
        cfg = get_config_from_trial(trial)
        
        # Load Data (Re-loading every trial is safer for memory, though slower)
        # Ideally, load this ONCE globally if your dataset is small (fits in RAM).
        data_loader = DataLoader(seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
        
        # Get Vocab Size from loader or config
        vocab_size = 5000 # Replace with: train_loader.dataset.vocab_size if available
        
        # Init JAX Model
        dkey = random.PRNGKey(42 + trial.number) # VITAL: Change seed per trial
        
        model = NGCTransformer(
            dkey, 
            batch_size=cfg['batch_size'], 
            seq_len=cfg['seq_len'], 
            n_embed=cfg['n_embed'], 
            vocab_size=vocab_size, 
            n_layers=cfg['n_layers'], 
            n_heads=cfg['n_heads'],
            T=cfg['n_iter'], 
            dt=1., 
            tau_m=cfg['tau_m'], 
            act_fx=cfg['act_fx'], 
            eta=cfg['eta'], 
            dropout_rate=cfg['dropout_rate'], 
            exp_dir=f"exp_trial_{trial.number}",
            loadDir=None, 
            pos_learnable=cfg['pos_learnable'], 
            optim_type=cfg['optim_type'], 
            wub=cfg['wub'], 
            wlb=cfg['wlb'], 
            model_name="ngc_transformer"
        )

        # Short Training Loop for Tuning (e.g., 2 epochs instead of full training)
        tuning_epochs = 2 
        
        for epoch in range(tuning_epochs):
            loss = train_one_epoch(model, train_loader, vocab_size, epoch)
            
            # Pruning check: If loss exploded (NaN), stop immediately
            if loss == float('inf'):
                logger.warning("Trial failed with NaN loss")
                return float('inf')

            # Evaluate
            dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
            
            # Report to Optuna for early stopping (Pruning)
            trial.report(dev_ce, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        logger.info(f"Trial {trial.number} finished. PPL: {dev_ppl:.4f}")
        return dev_ce # We want to minimize Cross Entropy

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf')
    finally:
        del model
        cleanup_memory()

def run_tuning(n_trials=10, study_name="jax_ngc_tuning"):
    
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    )
    
    logger.info(f"Starting JAX tuning for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
        
    return study

if __name__ == "__main__":
    run_tuning(n_trials=10)