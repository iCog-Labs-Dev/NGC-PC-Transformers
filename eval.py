
import os
import jax.numpy as jnp
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
import jax.random as random

def eval_model(model: NGCTransformer, data_loader, vocab_size: int):
    """
    Runs inference-only forward pass on a data loader and returns
    cross-entropy and perplexity.
    """
    total_nll = 0.0
    total_tokens = 0

    for batch in data_loader:
        inputs = batch[0][1]         
        targets = batch[1][1]        

       
        targets_onehot = jnp.eye(vocab_size)[targets]             
        targets_flat = targets_onehot.reshape(-1, vocab_size)      

        yMu_inf, _, _ = model.process(obs=inputs,
                                      lab=targets_flat,
                                      adapt_synapses=False)

        y_pred = yMu_inf.reshape(-1, vocab_size)   

        total_nll += measure_CatNLL(y_pred, targets_flat) * targets_flat.shape[0]
        total_tokens += targets_flat.shape[0]

    ce = total_nll / total_tokens
    ppl = jnp.exp(ce)
    return ce, ppl



def load_weights_into_model(model, model_dir):
    custom_dir = os.path.join(model_dir, "custom")
    print(f"Loading weights from: {custom_dir}")

    embed_data = jnp.load(os.path.join(custom_dir, "W_embed.npz"))
    model.embedding.W_embed.word_weights.set(embed_data["word_weights"])
    if model.embedding.W_embed.pos_learnable:
        model.embedding.W_embed.pos_weights.set(embed_data["pos_weights"])

    for i in range(model.n_layers):
        for name in ["W_q", "W_k", "W_v", "W_attn_out", "W_mlp1", "W_mlp2"]:
            path = os.path.join(custom_dir, f"block{i}_{name}.npz")
            data = jnp.load(path)
            if name.startswith("W_mlp"):
                comp = getattr(model.blocks[i].mlp, name)
            else:
                comp = getattr(model.blocks[i].attention, name)
            comp.weights.set(data["weights"])
            if "biases" in data:
                comp.biases.set(data["biases"])

    out_data = jnp.load(os.path.join(custom_dir, "W_out.npz"))
    model.output.W_out.weights.set(out_data["weights"])
    if "biases" in out_data:
        model.output.W_out.biases.set(out_data["biases"])
    print("Weights loaded successfully.")



if __name__ == "__main__":
    
    dkey = random.PRNGKey(0)
    model = NGCTransformer(
        dkey=dkey,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_embed=config.n_embed,
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        T=config.n_iter,
        dt=1., tau_m=config.tau_m,
        act_fx=config.act_fx,
        eta=config.eta,
        dropout_rate=config.dropout_rate,
        exp_dir="exp",
        model_name="ngc_transformer",
        loadDir="exp",  
        pos_learnable=config.pos_learnable,
        optim_type=config.optim_type,
        wub=config.wub,
        wlb=config.wlb,
    )
    data_loader = DataLoader(seq_len=config.seq_len, batch_size=config.batch_size)
    _, _, test_loader = data_loader.load_and_prepare_data()

    test_ce, test_ppl = eval_model(model, test_loader, config.vocab_size)
    print("\nFinal Test Evaluation:")
    print(f"\nCE: {test_ce:.4f} | PPL: {test_ppl:.4f}")