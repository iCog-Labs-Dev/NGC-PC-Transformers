# class Config:
#     SEED = 42
#     n_embed = 12
#     seq_len =  12
#     n_embed = 12
#     seq_len =  12
#     batch_size = 5
#     vocab_size = 11710# data vocab size + special tokens = 11706 + 4
#     n_heads = 2
#     n_layers = 4
#     dropout_rate = 0.0
#     eta = 0.00001
#     exp_dir = "exp" 
#     pos_learnable = True
#     optim_type = "adam"
#     num_iter = 2
#     n_iter= 20
#     wub = 0.02
#     wlb = -0.02
#     tau_m = 5.
#     act_fx = "identity"
#     # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
#     tokenizer = "BPE"
#     # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
#     tokenizer_name = "gpt2"
# # 
#     # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
#     # Optional: set to None to use a simple fallback whitespace tokenizer.
#     tokenizer_vocab_file = None
class Config:
    # --- Global ---
    SEED = 42
    exp_dir = "exp"
    
    # --- Data & Tokenizer ---
    seq_len = 12        # Current default
    batch_size = 5      # Current default
    vocab_size = 11710  # 11706 + 4 special tokens
    
    # Tokenizer settings
    tokenizer = "BPE" 
    tokenizer_name = "gpt2" 
    tokenizer_vocab_file = None 

    # --- Model Architecture (Defaults) ---
    # These will be OVERWRITTEN by Optuna during tuning
    n_embed = 12
    n_heads = 2
    n_layers = 4
    dropout_rate = 0.0
    act_fx = "identity" 
    pos_learnable = True

    # --- NGC / Predictive Coding Dynamics ---
    # n_iter: How many inference steps (T) per token
    n_iter = 20    
    # eta: The learning rate / step size
    eta = 0.00001
    # tau_m: Membrane time constant
    tau_m = 5.
    # wub/wlb: Weight bounds
    wub = 0.02
    wlb = -0.02

    # --- Optimization ---
    optim_type = "adam"
    # num_iter: Number of Training Epochs (Different from n_iter!)
    num_iter = 2