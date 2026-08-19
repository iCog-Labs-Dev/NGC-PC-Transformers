class Config:
    SEED = 42
    seq_len =32
    n_embed = 128
    batch_size = 8
    vocab_size = 11711# data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 4
    dropout_rate = 0.1
    eta = 4.919042890915579e-06
    eta_o= 4.919042890915579e-03
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "sgd"
    epoch = 5
    n_iter= 26
    tau_o = 20
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.035284728580901155
    wlb =  -0.07318664527441558
    wu = 0.035284728580901155
    wl = -0.035284728580901155
    tau_m = 20
    act_fx = "identity"
    act_fx_o = "identity"
    
    # RateCell soft-thresholding regularization:
    # shrinks small state activations toward zero to test stability effects
    threshold_type = "soft_threshold"    
    threshold_lambda = 0.0125

    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None

    # set True to Use jax.lax.scan fused advance loop (faster, minor floating-point differences from the normal python loop)
    fused_advance = True
