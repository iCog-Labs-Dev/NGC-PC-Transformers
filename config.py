class Config:
    SEED = 42
    seq_len =64
    n_embed = 96
    batch_size = 12
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 2
    dropout_rate = 0.0
    eta = 4.919042890915579e-06
    eta_o= 2.9e-03
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "sgd"
    epoch = 20
    n_iter= 26
    tau_o = 2
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.035284728580901155
    wlb =  -0.07318664527441558
    wu = 0.035284728580901155
    wl = -0.035284728580901155
    tau_m = 2.7
    act_fx = "identity"
    act_fx_o = "identity"

    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "tiktoken"
    tokenizer_encoding = "o200k_base"

    tokenizer_vocab_file = None
