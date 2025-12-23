class Config:
    SEED = 42
    n_embed = 12
    seq_len =  12
    n_embed = 12
    seq_len =  12
    batch_size = 5
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 4
    dropout_rate = 0.0
    eta = 0.0000089
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    num_iter = 2
    n_iter= 10
    wub = 0.2
    wlb = -0.2
    tau_m = 10.
    act_fx = "identity"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None