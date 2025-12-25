from jax import numpy as jnp, random
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

def main():
    seq_len, batch_size, n_embed, vocab_size, n_layers, n_heads, n_iter, optim_type = config.seq_len, config.batch_size, config.n_embed, config.vocab_size, config.n_layers, config.n_heads, config.n_iter, config.optim_type
    pos_learnable= config.pos_learnable
    num_iter= config.num_iter
    wub= config.wub 
    wlb= config.wlb
    eta = config.eta
    T = config.n_iter
    tau_m= config.tau_m
    act_fx= config.act_fx
    dropout_rate= config.dropout_rate
    dkey = random.PRNGKey(1234)
    
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, test_loader = data_loader.load_and_prepare_data()
    
    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=n_embed, vocab_size=vocab_size, n_layers=n_layers, n_heads=config.n_heads,
                          T=T, dt=1., tau_m=tau_m , act_fx=act_fx, eta=eta, dropout_rate= dropout_rate, exp_dir="exp",
                  loadDir= None, pos_learnable= pos_learnable, optim_type=optim_type, wub = wub, wlb= wlb, model_name="ngc_transformer" )

    def train_model(data_loader):
        total_nll, total_tokens = 0., 0
        
        for batch in data_loader:
            inputs = batch[0][1]
            targets = batch[1][1]
            
            targets_onehot = jnp.eye(vocab_size)[targets]  # (B, S, V)
            targets_flat = targets_onehot.reshape(-1, vocab_size)  # (B*S, V)

            yMu_inf, y_mu, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=False)
            
            y_pred = yMu_inf.reshape(-1, vocab_size)
            y_true = targets_flat
            
            total_nll += measure_CatNLL(y_pred, y_true) * y_true.shape[0]
            total_tokens += y_true.shape[0]
        
        ce_loss = total_nll / total_tokens
        return ce_loss, jnp.exp(ce_loss)

    for i in range(num_iter):
        train_EFE = 0.
        total_batches = 0
        
        print(f"\n iter {i}:")
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0][1]
            targets = batch[1][1]
            
            #Convert targets to one-hot and flatten
            targets_onehot = jnp.eye(vocab_size)[targets]  # (B, S, V)
            targets_flat = targets_onehot.reshape(-1, vocab_size)  # (B*S, V)

            
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            train_EFE += _EFE
            total_batches += 1

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = jnp.eye(vocab_size)[targets.flatten()]
                
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()  
                batch_ppl = jnp.exp(batch_ce_loss)
                
                print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")
        
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0
        
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"Iter {i} Summary: CE = {dev_ce:.4f}, PPL = {dev_ppl:.4f}, Avg EFE = {avg_train_EFE:.4f}")
        if  i == (num_iter-1):
          model.save_to_disk(params_only=False) # save final state of model to disk

   
if __name__ == "__main__":
    main()