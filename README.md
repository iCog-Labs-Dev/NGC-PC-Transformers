# NGC-PC-Transformers
Implementation of a Predictive Coding Transformer built on the Neural Generative Coding framework.

## Installation

Clone the Repository

```bash
git clone https://github.com/iCog-Labs-Dev/NGC-PC-Transformers.git
```
Navigate to the Project Directory

```bash
cd NGC-PC-Transformers
```
Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``` 
Install required packages:

```bash
pip install -r requirements.txt
```
Run the tokenization
```bash
python -m data_preprocess.tokenizer
```
Run the training
```bash
python train.py
```
Run the evaluation
```bash
python eval.py
```
Run the generation
```bash
python generation.py
```

## Fast Standalone Inference Decoder

Generation historically called `model.process()`, which executed the full training-time predictive coding circuit: `reset.run()`, 52 parameter copies from the training model into the projection model, error cell updates, transposed feedback synapse updates, and $T=26$ iterative relaxation steps over a full padded `batch_size x seq_len` window for every single generated token.

The standalone inference decoder in `inference/decoder.py` isolates the exact feedforward predictive coding projection path (`y_mu_inf = projection.q_target_Ratecell.zF`) in pure JAX with zero runtime imports of `model.py`.

### Computational Complexity & KV Cache Semantics
- **Within context window (`total_tokens <= seq_len`)**: Per-token step decoding executes $O(1)$ matrix multiplications with cached projection states, with $O(\text{seq\_len})$ attention and KV cache slice updates per step (`decode_step`).
- **Past context window (`total_tokens > seq_len`)**: To strictly preserve 100% mathematical parity with the original model's sliding-window dynamics, the decoder performs a full prefill over the trailing `seq_len` window for each new token ($O(\text{seq\_len}^2)$ computation per token).

### 1. One-Time Weight Export
Extract parameters and architecture metadata from a checkpoint into standalone files:
```bash
python export_generation_weights.py \
    --checkpoint-dir exp \
    --model-name ngc_transformer \
    --output-npz generation_weights.npz \
    --output-meta generation_meta.json
```
*(Aliases `--exp-dir`, `--out-weights`, `--out-meta` are supported. For verification without a trained checkpoint, pass `--synthetic`.)*

### 2. Fast Text Generation
```bash
# Fast generation using the standalone decoder (default)
python generation.py --max-new-tokens 200 --temperature 0.8 --top-k 50

# Deterministic greedy generation
python generation.py --max-new-tokens 100 --temperature 0.0

# Optional device specification (cpu / gpu)
python generation.py --device gpu --max-new-tokens 100

# Legacy fallback (calls NGCTransformer.process)
python generation.py --legacy --max-new-tokens 50
```

### 3. Benchmarking
Benchmark token throughput over multiple runs (warmup run excluded, JIT compilation reported separately):
```bash
# Benchmark the fast standalone decoder
python generation.py --benchmark --max-new-tokens 100 --runs 3

# Benchmark the legacy generator
python generation.py --legacy --benchmark --max-new-tokens 20 --runs 3
```

**Trained Checkpoint Performance:**
*To be measured on Colab after training (see verification instructions below).*

### 4. Running the Equivalence Test Suite
Run the comprehensive numerical verification suite covering positional and RoPE position encodings, per-node intermediate activations against model cell `zF` values (`q_embed`, `q_qkv`, `q_attn`, `q_mlp`, `q_mlp2`, `q_out`, `q_target`), softmax probabilities, incremental KV cache parity, sliding-window boundary crossings, export round-trip, and sampling tests:
```bash
python -m unittest tests/test_decoder_equivalence.py -v
```
All tests run on small randomly initialized models without training or checkpoint loading, finishing in ~2 minutes on CPU.

---

## Run this on Colab after training

Once model training finishes on Colab, run these exact commands to export weights and verify full numerical equivalence, sampled generation, and tokens/sec throughput:

```bash
# 1. Export standalone weights from trained checkpoint
python export_generation_weights.py \
    --exp-dir exp \
    --model-name ngc_transformer \
    --out-weights generation_weights.npz \
    --out-meta generation_meta.json \
    --device gpu

# 2. Verify numerical equivalence (80+ greedy tokens crossing seq_len), sample outputs, and benchmark throughput
python scripts/verify_real_checkpoint.py \
    --exp-dir exp \
    --model-name ngc_transformer \
    --weights generation_weights.npz \
    --meta generation_meta.json \
    --device gpu \
    --greedy-tokens 80 \
    --temperature 0.8 \
    --top-k 50
```


