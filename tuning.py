import os
import sys
import warnings
import logging
import optuna

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import time
import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from eval import eval_model
from config import Config as base_config
from ngclearn.utils.metric_utils import measure_CatNLL
import gc

EFE_STABILITY_THRESHOLD = 300


def define_search_space(trial):
    # Heads and embedding: ensure n_embed divisible by n_heads
    n_heads = trial.suggest_int("n_heads", 2, 8)
    embed_mult = trial.suggest_int("embed_mult", 8, 16, step=4)
    n_embed =  n_heads * embed_mult
    n_embed = trial.suggest_int("n_embed", n_embed, n_embed)
    batch_size = trial.suggest_int("batch_size", 2, 12)
    seq_len = trial.suggest_int("seq_len", 8, 32)

    return {
        "n_layers": trial.suggest_int("n_layers", 1, 8),
        "pos_learnable": trial.suggest_categorical("pos_learnable", [True, False]),
        "eta": trial.suggest_float("eta", 1e-6, 1e-4, log=True),
        "tau_m": trial.suggest_int("tau_m", 10, 20),
        "n_iter": trial.suggest_int("n_iter", 1, 30),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.),
        "wub": trial.suggest_float("wub", 0.01, 0.1),
        "wlb": trial.suggest_float("wlb", -0.1, -0.01),
        "optim_type": trial.suggest_categorical("optim_type", ["adam", "sgd"]),
        "act_fx": trial.suggest_categorical("act_fx", ["identity", "relu"]),
        "n_heads": n_heads,
        "n_embed": n_embed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_mult": embed_mult
    }

def define_search_space_phase2(trial, best_params):
    """Phase 2: Only tune continuous parameters, keep others fixed from Phase 1"""
    
    # Extract Phase 1 best values
    eta_best = best_params.get("eta", 1e-5)
    dropout_rate_best = best_params.get("dropout_rate", 0.0)
    wub_best = best_params.get("wub", 0.05)
    wlb_best = best_params.get("wlb", -0.05)
    
    # Only tune these continuous parameters with narrow search
    return {
        "eta": trial.suggest_float("eta",
                                   eta_best * 0.2,      
                                   eta_best * 5.0,      
                                   log=True),
        "dropout_rate": trial.suggest_float("dropout_rate",
                                           max(0.0, dropout_rate_best - 0.05),
                                           min(0.3, dropout_rate_best + 0.05)),
        "wub": trial.suggest_float("wub",
                                  max(0.01, wub_best - 0.02),
                                  min(0.1, wub_best + 0.02)),
        
        "wlb": trial.suggest_float("wlb",
                                  max(-0.1, wlb_best - 0.02),
                                  min(-0.01, wlb_best + 0.02)),
    }
    
    # ALL OTHER PARAMETERS ARE FIXED FROM PHASE 1 BEST
    

def create_model_with_all_params(trial_number, params, cfg):
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
    dkey = random.PRNGKey(trial_number * 1000 + 42)

    model_args = {
        "dkey": dkey,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "n_embed": cfg.n_embed,
        "vocab_size": cfg.vocab_size,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "T": cfg.n_iter,
        "dt": 1.0,
        "tau_m": cfg.tau_m,
        "act_fx": cfg.act_fx,
        "eta": cfg.eta,
        "dropout_rate": cfg.dropout_rate,
        "exp_dir": None,
        "loadDir": None,
        "pos_learnable": cfg.pos_learnable,
        "optim_type": cfg.optim_type,
        "wub": cfg.wub,
        "wlb": cfg.wlb,
        "model_name": f"trial_{trial_number}"
    }

    model = NGCTransformer(**model_args)
    return model, train_loader, valid_loader
def run_single_trial_efe(trial):
    try:
        params = define_search_space(trial)
        print(f"[EFE Phase] Trial {trial.number} | params: {params}")

        cfg = type('Config', (), {})()
        for key, value in base_config.__dict__.items():
            if not key.startswith('_'):
                setattr(cfg, key, value)
        for key, value in params.items():
            setattr(cfg, key, value)
        if not hasattr(cfg, 'vocab_size'):
            cfg.vocab_size = base_config.vocab_size

        try:
            model, train_loader, valid_loader = create_model_with_all_params(trial.number, params, cfg)
        except Exception as e:
            reason = f"Failed to create model: {e}"
            trial.set_user_attr("prune_reason", reason)
            print(reason)
            raise optuna.TrialPruned()

        total_EFE = 0.0
        batches_processed = 0
        start_time = time.time()
        max_batches = 20
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            inputs = batch[0][1]
            targets = batch[1][1]
            targets_flat = jax.nn.one_hot(targets.flatten(), cfg.vocab_size)


            try:
                _, _, EFE, *_ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                EFE = abs(float(EFE))
            except Exception as e:
                reason = f"model.process failed: {e}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()

            if jnp.isnan(EFE) or jnp.isinf(EFE) or EFE > EFE_STABILITY_THRESHOLD:
                reason = f"Unstable EFE: {EFE}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()

            total_EFE += EFE
            batches_processed += 1
            current_efe = total_EFE / batches_processed

            trial.report(current_efe, batch_idx)
            if trial.should_prune():
                reason = f"TPE pruned at batch {batch_idx} | current EFE={current_efe:.4f}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()

            if batch_idx % 2 == 0:
                elapsed = time.time() - start_time
                print(f"Batch {batch_idx} | EFE={EFE:.4f} | Avg EFE={current_efe:.4f} | Time={elapsed:.1f}s")

        try:
            final_ce, final_ppl = eval_model(model, valid_loader, cfg.vocab_size)
        except:
            final_ce = 1000.0
            final_ppl = float('inf')

        final_efe = total_EFE / batches_processed if batches_processed > 0 else 1000.0
        total_time = time.time() - start_time

        trial.set_user_attr("ce", float(final_ce))
        trial.set_user_attr("ppl", float(final_ppl))
        trial.set_user_attr("time", total_time)

        for key, value in params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(f"Trial {trial.number} Complete | EFE={final_efe:.4f} | CE={final_ce:.4f} | Time={total_time:.1f}s")
        return float(final_efe)
    finally:
        
        # Delete Python objects
        for obj_name in ['model', 'train_loader', 'valid_loader']:
            if obj_name in locals() and locals()[obj_name] is not None:
                del locals()[obj_name]
        
        # Force garbage collection
        for _ in range(2):
            gc.collect()
        
        # Clear JAX caches
        try:
            jax.clear_caches()
        except:
            pass

def run_phase2_trial(trial, best_params):
    """Phase 2: Only tune continuous parameters, keep others fixed from Phase 1"""
    continuous_params = define_search_space_phase2(trial, best_params)
    params = {**best_params, **continuous_params}
    tuning_params = {k: v for k, v in params.items() if k in ['eta', 'dropout_rate', 'wub', 'wlb']}
    print(f"[CE Phase - Continuous Only] Trial {trial.number} | params: {tuning_params}")
    print(f"[CE Phase - Fixed] Architecture: n_layers={params['n_layers']}, n_heads={params['n_heads']}, "
          f"tau_m={params['tau_m']}, n_iter={params['n_iter']}")

    cfg = type('Config', (), {})()
    for key, value in base_config.__dict__.items():
        if not key.startswith('_'):
            setattr(cfg, key, value)
    for key, value in params.items():
        setattr(cfg, key, value)
    if not hasattr(cfg, 'vocab_size'):
        cfg.vocab_size = base_config.vocab_size

    try:
        model, train_loader, valid_loader = create_model_with_all_params(trial.number, params, cfg)
    except Exception as e:
        reason = f"Failed to create model: {e}"
        trial.set_user_attr("prune_reason", reason)
        print(reason)
        raise optuna.TrialPruned()

    total_train_ce = 0.0  
    batches_processed = 0
    start_time = time.time()
    max_batches = 20
    best_train_ce = float('inf')
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        inputs = batch[0][1]
        targets = batch[1][1]
        targets_flat = jax.nn.one_hot(targets.flatten(), cfg.vocab_size)

        try:
            yMu_inf, y_mu, EFE, *_ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            EFE = abs(float(EFE))
            
            y_pred = y_mu.reshape(-1, cfg.vocab_size)
            batch_nll = measure_CatNLL(y_pred, targets_flat) * targets_flat.shape[0]
            batch_train_ce = batch_nll / targets_flat.shape[0]
            
            if jnp.isnan(EFE) or jnp.isinf(EFE) or EFE > EFE_STABILITY_THRESHOLD:
                reason = f"Unstable EFE during CE: {EFE}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()
        except Exception as e:
            reason = f"model.process failed during CE: {e}"
            trial.set_user_attr("prune_reason", reason)
            print(reason)
            raise optuna.TrialPruned()

        total_train_ce += float(batch_train_ce)
        batches_processed += 1
        avg_train_ce = total_train_ce / batches_processed

        trial.report(avg_train_ce, batch_idx)
        if trial.should_prune():
            reason = f"TPE pruned at batch {batch_idx} | Avg Train CE={avg_train_ce:.4f}"
            trial.set_user_attr("prune_reason", reason)
            print(reason)
            raise optuna.TrialPruned()
        if float(batch_train_ce) < best_train_ce:
            best_train_ce = float(batch_train_ce)
        if batch_idx % 2 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx} | CE={float(batch_train_ce):.4f} | Avg Train CE={avg_train_ce:.4f} | Time={elapsed:.1f}s")

    try:
        final_ce, final_ppl = eval_model(model, valid_loader, cfg.vocab_size)
        final_ce = float(final_ce)
    except:
        final_ce = avg_train_ce if batches_processed > 0 else 100.0
        final_ppl = float('inf')

    total_time = time.time() - start_time
    trial.set_user_attr("ppl", float(final_ppl))
    trial.set_user_attr("time", total_time)

    for key, value in params.items():
        trial.set_user_attr(f"param_{key}", value)

    print(f"Trial {trial.number} Complete | Final Val CE={final_ce:.4f} | Time={total_time:.1f}s")
    return float(final_ce)  

def case1_efe_to_ce_complete():
    Path("tuning").mkdir(exist_ok=True)

    print("PHASE 1: TPE optimizing EFE (all parameters)")
    study_efe = optuna.create_study(
        study_name="case1_complete_phase1_efe",
        storage="sqlite:///tuning/case1_complete_phase1_efe.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2),
        pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=15, reduction_factor=2)
    )

    study_efe.optimize(run_single_trial_efe, n_trials=10, n_jobs= 1, show_progress_bar=False)

    if study_efe.best_trial:
        best_efe = study_efe.best_value
        best_efe_ce = study_efe.best_trial.user_attrs.get("ce", "N/A")
        best_params = study_efe.best_trial.params
        
        print(f"\n{'='*60}")
        print("PHASE 1 COMPLETE")
        print(f"{'='*60}")
        print(f"Best EFE: {best_efe:.4f}")
        print(f"Corresponding CE: {best_efe_ce}")
        print(f"\nBest Architecture Parameters (FIXED for Phase 2):")
        for key in ['n_layers', 'n_heads', 'n_embed', 'tau_m', 'n_iter', 
                   'batch_size', 'seq_len', 'pos_learnable', 'optim_type', 'act_fx']:
            print(f"  {key}: {best_params.get(key)}")
        print(f"\nContinuous Parameters to be fine-tuned in Phase 2:")
        for key in ['eta', 'dropout_rate', 'wub', 'wlb']:
            print(f"  {key}: {best_params.get(key)}")
    else:
        return None

    print(f"\n{'='*60}")
    print("PHASE 2: TPE optimizing CE (continuous parameters only)")
    print(f"{'='*60}")
    print("Strategy: Keep architecture/discrete/categorical parameters FIXED")
    print("Only fine-tune: eta, dropout_rate, wub, wlb")
    
    study_ce = optuna.create_study(
    study_name="case1_complete_phase2_ce",
    storage="sqlite:///tuning/case1_complete_phase2_ce.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=5,
        multivariate=True,    
        group=True,          
        prior_weight=1.0,     
        consider_endpoints=True  
    ),
    pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=15)
    )
    
    print(f"Resuming with {len(study_ce.trials)} previous trials")

    def phase2_trial_wrapper(trial):
        return run_phase2_trial(trial, best_params)

    study_ce.optimize(phase2_trial_wrapper, n_trials=25, n_jobs= 1, show_progress_bar=False)

    if study_ce.best_trial:
        best_ce = study_ce.best_value
        final_params = study_ce.best_trial.params
        
        print(f"\n{'='*60}")
        print("PHASE 2 COMPLETE")
        print(f"{'='*60}")
        print(f"Best CE achieved: {best_ce:.4f}")
        
        print(f"\nParameter changes from Phase 1 → Phase 2:")
        for key in ['eta', 'dropout_rate', 'wub', 'wlb']:
            phase1_val = best_params.get(key)
            phase2_val = final_params.get(key)
            change_pct = ((phase2_val - phase1_val) / phase1_val * 100) if phase1_val != 0 else 0
            print(f"  {key}: {phase1_val:.6f} → {phase2_val:.6f} ({change_pct:+.1f}%)")
        
        with open("tuning/best_hyperparameters.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("BEST HYPERPARAMETERS\n")
            f.write("="*60 + "\n\n")
            
            f.write("PHASE 1 - BEST FOR EFE:\n")
            f.write("-"*40 + "\n")
            f.write(f"Best EFE: {best_efe:.6f}\n")
            f.write(f"Corresponding CE: {best_efe_ce:.6f}\n")
            f.write("-"*40 + "\n")
            
            for key, value in best_params.items():
                f.write(f"{key} = {value}\n")
            
            f.write("\n" + "="*60 + "\n\n")
            
            f.write("PHASE 2 - BEST FOR CE:\n")
            f.write("-"*40 + "\n")
            f.write(f"Best CE: {best_ce:.6f}\n")
            f.write("-"*40 + "\n")
            
            for key, value in final_params.items():
                f.write(f"{key} = {value}\n")
        
        print(f"\n✓ Best hyperparameters saved to: tuning/best_hyperparameters.txt")
        
        return {
            "phase1_best_efe": best_efe,
            "phase1_best_ce": best_efe_ce,
            "phase2_best_ce": best_ce,
            "phase1_parameters": best_params,
            "phase2_parameters": final_params,
            "improvement_pct": ((best_efe_ce - best_ce) / best_efe_ce * 100) if best_efe_ce > 0 else 0
        }
    return None

def main():
    print("PC TRANSFORMER - TWO-PHASE HYPERPARAMETER TUNING")
    print("="*60)
    print("PHASE 1: Find stable architecture (minimize EFE)")
    print("PHASE 2: Fine-tune continuous parameters (minimize CE)")
    print("="*60)

    try:
        results = case1_efe_to_ce_complete()
        if results:
            print(f"\n{'='*60}")
            print("TUNING COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Final Results:")
            print(f"- Phase 1 Best EFE: {results['phase1_best_efe']:.4f}")
            print(f"- Phase 1 Corresponding CE: {results['phase1_best_ce']:.4f}")
            print(f"- Phase 2 Best CE: {results['phase2_best_ce']:.4f}")
            if 'improvement_pct' in results:
                print(f"- Improvement: {results['improvement_pct']:+.1f}%")
            print(f"\n Parameters saved to: tuning/best_hyperparameters.txt")
        else:
            print("Tuning failed or was interrupted.")
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")
    except Exception as e:
        print(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()