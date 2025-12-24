import os
import sys
import gc
import logging
import jax
import jax.numpy as jnp
from jax import random
import optuna

# Memory management for T4 GPU
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Import your custom project modules
from config import Config
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from eval import eval_model
from ngclearn.utils.metric_utils import measure_CatNLL  # Added for metric calculation

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def cleanup_memory():
    gc.collect()
    jax.clear_caches()

def objective(trial):
    cleanup_memory()
    
    # --- OPTUNA SEARCH SPACE ---
    # You can expand these ranges if needed
    n_embed = trial.suggest_categorical("n_embed", [32, 64, 128])
    n_heads = trial.suggest_int("n_heads", 1, 4)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    eta = trial.suggest_float("eta", 1e-5, 1e-3, log=True)
    tau_m = trial.suggest_int("tau_m", 5, 20)
    wub = trial.suggest_float("wub", 0.01, 0.1)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
    
    # Configuration dictionary for this trial
    cfg = {
        'n_embed': n_embed,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'n_iter': Config.n_iter,      # Inference steps (T)
        'num_epochs': Config.num_iter, # Training epochs
        'eta': eta,
        'tau_m': tau_m,
        'wub': wub,
        'dropout_rate': dropout_rate,
        'act_fx': "gelu",
        'seq_len': Config.seq_len,
        'batch_size': Config.batch_size,
        'vocab_size': Config.vocab_size,
        'pos_learnable': Config.pos_learnable,
        'optim_type': Config.optim_type,
        'exp_dir': f"test_run_trial_{trial.number}"
    }
    
    try:
        # Load Data
        # Note: Your DataLoader class handles limiting samples via train_sample_size internally
        data_loader = DataLoader(seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
        
        dkey = random.PRNGKey(Config.SEED + trial.number)
        
        # Initialize Model
        model = NGCTransformer(
            dkey,
            batch_size=cfg['batch_size'],
            seq_len=cfg['seq_len'],
            n_embed=cfg['n_embed'],
            vocab_size=cfg['vocab_size'],
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            T=cfg['n_iter'],
            dt=1.,
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
            model_name="test_model"
        )
        
        logger.info(f"--- STARTING TRIAL {trial.number} ---")
        
        # --- TRAINING LOOP (Matches logic in train.py) ---
        final_val_ce = float('inf')

        for epoch in range(cfg['num_epochs']):
            train_EFE = 0.
            total_batches = 0
            
            # Loop over the full train_loader (no hard break at 50)
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch[0][1], batch[1][1]
                
                # Prepare targets
                targets_onehot = jnp.eye(cfg['vocab_size'])[targets]
                targets_flat = targets_onehot.reshape(-1, cfg['vocab_size'])
                
                # Process Model (Train Step)
                yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                
                # Check for Numerical Instability
                if jnp.isnan(yMu_inf).any() or jnp.isnan(_EFE):
                    logger.warning(f"Trial {trial.number} pruned due to NaN values.")
                    return float('inf')

                train_EFE += _EFE
                total_batches += 1
                
                # Log metrics every 10 batches (Mirroring train.py)
                if batch_idx % 10 == 0:
                    y_pred = yMu_inf.reshape(-1, cfg['vocab_size'])
                    y_true = jnp.eye(cfg['vocab_size'])[targets.flatten()]
                    
                    batch_nll = measure_CatNLL(y_pred, y_true)
                    batch_ce_loss = batch_nll.mean()
                    batch_ppl = jnp.exp(batch_ce_loss)
                    
                    logger.info(f"Trial {trial.number} | Epoch {epoch} | Batch {batch_idx}: "
                                f"EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")

            # --- End of Epoch Summary ---
            avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0
            
            # Validation Step
            val_ce, val_ppl = eval_model(model, valid_loader, cfg['vocab_size'])
            logger.info(f"Trial {trial.number} | Epoch {epoch} Summary: "
                        f"Val CE = {val_ce:.4f}, Val PPL = {val_ppl:.4f}, Avg Train EFE = {avg_train_EFE:.4f}")
            
            final_val_ce = val_ce

            # Optional: Report intermediate value to Optuna for pruning
            trial.report(val_ce, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned by Optuna.")
                raise optuna.TrialPruned()

        return final_val_ce

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} crashed: {e}")
        return float('inf')
    finally:
        cleanup_memory()

def run_tuning():
    # 'minimize' because we want lower Cross Entropy (CE)
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Run optimization
    study.optimize(objective, n_trials=10) # Set n_trials to however many you want
    
    if study.best_trial:
        print("\n" + "="*40)
        print("HYPERPARAMETER TUNING COMPLETE")
        print(f"Best Trial Number: {study.best_trial.number}")
        print(f"Best Loss (Val CE): {study.best_trial.value:.5f}")
        print("Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print("="*40)

if __name__ == "__main__":
    run_tuning()