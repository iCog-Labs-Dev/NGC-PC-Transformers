class Config:
    SEED = 42

    seq_len =  12
    n_embed = 12
    
    batch_size = 8
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 4
    dropout_rate = 0.1
    eta = 0.0001
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    num_iter = 3
    n_iter= 0
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.05
    wlb = -0.05
    tau_m = 10.
    act_fx = "tanh"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None