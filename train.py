import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
    
import jax
from jax import numpy as jnp, random
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model
import time


jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
def main():
    seq_len, batch_size, n_embed, vocab_size, n_layers, n_heads, n_iter, optim_type = config.seq_len, config.batch_size, config.n_embed, config.vocab_size, config.n_layers, config.n_heads, config.n_iter, config.optim_type
    pos_learnable= config.pos_learnable
    epoch= config.epoch
    wub= config.wub 
    wlb= config.wlb
    eta = config.eta
    T = n_iter
    tau_m= config.tau_m
    act_fx= config.act_fx
    dropout_rate= config.dropout_rate
    dkey = random.PRNGKey(1234)
    
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
    
    # loadDir="exp" loads pre-trained weights transferred by transfer_weights.py.
    # Set loadDir=None to train from random initialisation instead.
    pretrained_dir = "exp"
    import os
    loadDir = pretrained_dir if os.path.isdir(pretrained_dir) else None
    if loadDir:
        print(f"[train] Loading pre-trained weights from '{loadDir}/'")
    else:
        print("[train] No pre-trained weights found — starting from random init.")

    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=n_embed, vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads,
                          T=T, dt=1., tau_m=tau_m , act_fx=act_fx, eta=eta, dropout_rate= dropout_rate, exp_dir="exp",
                  loadDir=loadDir, pos_learnable= pos_learnable, optim_type=optim_type, wub = wub, wlb= wlb, model_name="ngc_transformer", generate =False )

    print(f" {model.count_parameters()/1e6:.2f} M parameters")

    def train_model(data_loader):
        train_EFE = 0.
        total_nll, total_tokens = 0., 0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch[0][1]
            targets = batch[1][1]

            targets_flat = jax.nn.one_hot(targets, vocab_size).reshape(-1, vocab_size)

            _, y_mu, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            train_EFE += _EFE

            y_pred = y_mu.reshape(-1, vocab_size)
            batch_ce_loss = measure_CatNLL(y_pred, targets_flat).mean()
            total_nll += batch_ce_loss * targets_flat.shape[0]
            total_tokens += targets_flat.shape[0]

            if batch_idx % 10 == 0:
                batch_ppl = jnp.exp(batch_ce_loss)
                print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")

        num_batches = batch_idx + 1
        avg_train_EFE = train_EFE / num_batches
        ce_loss = total_nll / total_tokens
        ppl = jnp.exp(ce_loss)
        return avg_train_EFE, ce_loss, ppl

    start_time = time.time()

    for i in range(epoch):
        print(f"\nEpoch {i}:")

        avg_train_EFE, train_ce, train_ppl = train_model(train_loader)

        dev_ce, dev_ppl,_ = eval_model(model, valid_loader, vocab_size)
        print(f"Epoch {i} Summary: Train CE = {train_ce:.4f}, Train PPL = {train_ppl:.4f}, Val CE = {dev_ce:.4f}, Val PPL = {dev_ppl:.4f}, Avg EFE = {avg_train_EFE:.4f}")
        if i == (epoch-1):
          model.save_to_disk(params_only=False) # save final state of model to disk
    total_time = time.time() - start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} min)")
   
if __name__ == "__main__":
    main()