class Config:
    # 🔥 MINIMAL CONFIG FOR VERY FAST TESTING 🔥

    SEED = 42

    # Model size (VERY SMALL)
    n_embed = 4
    n_heads = 1
    n_layers = 1

    # Sequence / batch (VERY SMALL)
    seq_len = 4
    batch_size = 2

    # Vocabulary (REDUCED — must be >= max token id used)
    vocab_size = 128

    # Training
    dropout_rate = 0.0
    eta = 1e-4
    optim_type = "adam"

    # Iterations (VERY FEW)
    num_iter = 1
    n_iter = 1

    # Weight bounds
    wub = 0.05
    wlb = -0.05

    # Dynamics
    tau_m = 5.0
    act_fx = "identity"

    # Experiment
    exp_dir = "exp_test"

    # Position encoding
    pos_learnable = False

    # Tokenizer (FASTEST PATH)
    tokenizer = "BPE"
    tokenizer_name = "gpt2"
    tokenizer_vocab_file = None
