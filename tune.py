import os
import sys
import gc
import logging


# Memory management for T4 GPU
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import jax
import jax.numpy as jnp
from jax import random
import optuna


# Import your custom project modules
from config import Config
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from eval import eval_model


# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def cleanup_memory():
    gc.collect()
    jax.clear_caches()


def objective(trial):
    cleanup_memory()
   
    # --- OPTUNA SEARCH SPACE ---
    n_embed = trial.suggest_categorical("n_embed", [32, 64, 128])
    n_heads = trial.suggest_int("n_heads", 1, 4)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    eta = trial.suggest_float("eta", 1e-4, 1e-3, log=True)
    tau_m = trial.suggest_int("tau_m", 5, 20)
    wub = trial.suggest_float("wub", 0.01, 0.05)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.2)
   
    cfg = {
        'n_embed': n_embed,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'n_iter': 2,  # Keep minimal for fast test
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
       
        logger.info(f"--- STARTING FAST TEST TRIAL {trial.number} ---")


        # --- Minimal training: only 2 batches ---
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:
                break
            inputs, targets = batch[0][1], batch[1][1]
            targets_flat = jnp.eye(cfg['vocab_size'])[targets].reshape(-1, cfg['vocab_size'])
            yMu_inf, _, _ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            if jnp.isnan(yMu_inf).any():
                return float('inf')
            logger.info(f"Batch {batch_idx} completed.")


        # --- Validation ---
        val_ce, _ = eval_model(model, valid_loader, cfg['vocab_size'])
        logger.info(f"Trial {trial.number} Val CE: {val_ce}")


        return val_ce


    except Exception as e:
        logger.error(f"Trial crashed: {e}")
        return float('inf')
    finally:
        cleanup_memory()


def run_tuning():
    study = optuna.create_study(direction='minimize')
    # Only 5 trials for quick parameter suggestion
    study.optimize(objective, n_trials=5)
   
    if study.best_trial:
        print("\n" + "="*30)
        print("FAST HYPERPARAMETER SUGGESTION SUCCESSFUL")
        print(f"Best Parameters: {study.best_trial.params}")
        print(f"Resulting Loss: {study.best_trial.value:.5f}")
        print("="*30)


if __name__ == "__main__":
    run_tuning() #