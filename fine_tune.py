import os
import sys
import gc
import logging
import jax
import jax.numpy as jnp
from jax import random
import optuna
import jax.extend
# --- GPU MEMORY CONFIGURATION ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Import custom project modules
from config import Config
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from eval import eval_model
from ngclearn.utils.metric_utils import measure_CatNLL

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Force JAX and Python to release GPU memory."""
    gc.collect()
    jax.clear_caches()

def objective(trial):
    cleanup_memory()
    
    # --- 1. OPTUNA HYPERPARAMETER SEARCH SPACE ---
    n_embed = trial.suggest_categorical("n_embed", [32, 64, 128])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    eta = trial.suggest_float("eta", 1e-6, 1e-3, log=True)
    tau_m = trial.suggest_float("tau_m", 5.0, 20.0)
    wub = trial.suggest_float("wub", 0.05, 0.3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
    
    # Map trial suggestions to a config dict
    cfg = {
        'n_embed': n_embed,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'n_iter': Config.n_iter,      # Settling steps (T)
        'num_epochs': Config.num_iter, # Number of training iterations
        'eta': eta,
        'tau_m': tau_m,
        'wub': wub,
        'dropout_rate': dropout_rate,
        'act_fx': Config.act_fx,
        'seq_len': Config.seq_len,
        'batch_size': Config.batch_size,
        'vocab_size': Config.vocab_size,
        'pos_learnable': Config.pos_learnable,
        'optim_type': Config.optim_type,
        'exp_dir': f"tuning_run_trial_{trial.number}"
    }
    
    try:
        # --- 2. DATA LOADING ---
        # Uses your custom DataLoader which respects train_sample_size
        data_loader = DataLoader(seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
        
        dkey = random.PRNGKey(Config.SEED + trial.number)
        
        # --- 3. MODEL INITIALIZATION ---
        model = NGCTransformer(
            dkey,
            batch_size=cfg['batch_size'],
            seq_len=cfg['seq_len'],
            n_embed=cfg['n_embed'],
            vocab_size=cfg['vocab_size'],
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            T=cfg['n_iter'],
            dt=1.0,
            tau_m=cfg['tau_m'],
            act_fx=cfg['act_fx'],
            eta=cfg['eta'],
            dropout_rate=cfg['dropout_rate'],
            exp_dir=cfg['exp_dir'],
            loadDir=None,
            pos_learnable=cfg['pos_learnable'],
            optim_type=cfg['optim_type'],
            wub=cfg['wub'],
            wlb=-cfg['wub'],
            model_name=f"trial_{trial.number}"
        )
        
        logger.info(f"--- STARTING TRIAL {trial.number} | Params: {trial.params} ---")

        # --- 4. TRAINING LOOP (Aligned with train.py) ---
        final_val_ce = float('inf')

        for epoch in range(cfg['num_epochs']):
            train_EFE = 0.
            total_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch[0][1]
                targets = batch[1][1]
                
                # Convert targets to one-hot and flatten
                targets_onehot = jnp.eye(cfg['vocab_size'])[targets]
                targets_flat = targets_onehot.reshape(-1, cfg['vocab_size'])
                
                # Model Process (adapt_synapses=True for training)
                yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                
                # Instability check
                if jnp.isnan(_EFE) or jnp.isinf(_EFE):
                    logger.warning(f"Trial {trial.number} encountered NaN/Inf EFE. Pruning.")
                    return float('inf')

                train_EFE += _EFE
                total_batches += 1

                # Periodic batch logging
                if batch_idx % 10 == 0:
                    y_pred = yMu_inf.reshape(-1, cfg['vocab_size'])
                    y_true = targets_flat
                    
                    batch_nll = measure_CatNLL(y_pred, y_true)
                    batch_ce = batch_nll.mean()
                    batch_ppl = jnp.exp(batch_ce)
                    
                    logger.info(f"T{trial.number} | Ep {epoch} | Bat {batch_idx}: "
                                f"EFE={_EFE:.4f}, CE={batch_ce:.4f}, PPL={batch_ppl:.4f}")

            # --- 5. VALIDATION ---
            avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0
            val_ce, val_ppl = eval_model(model, valid_loader, cfg['vocab_size'])
            
            logger.info(f"--- Trial {trial.number} Epoch {epoch} Summary ---")
            logger.info(f"Val CE: {val_ce:.4f} | Val PPL: {val_ppl:.4f} | Avg Train EFE: {avg_train_EFE:.4f}")
            
            final_val_ce = val_ce

            # Report to Optuna for pruning decisions
            trial.report(val_ce, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.TrialPruned()

        return final_val_ce

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return float('inf')
    finally:
        cleanup_memory()

def run_tuning():
    # Modern way to check the backend platform (GPU/CPU/TPU)
    try:
        # This checks the primary device platform
        device_platform = jax.devices()[0].platform.upper()
        logger.info(f"JAX Backend: {device_platform}")
    except Exception:
        logger.warning("Could not explicitly determine JAX backend; proceeding with default.")

    # Initialize Optuna Study
    study = optuna.create_study(
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=10)
    
    if study.best_trial:
        print("\n" + "="*50)
        print("BEST HYPERPARAMETERS FOUND")
        print(f"Trial Number: {study.best_trial.number}")
        print(f"Validation CE: {study.best_trial.value:.5f}")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print("="*50)