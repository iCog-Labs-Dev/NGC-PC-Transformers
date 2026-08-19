from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import sys
import os
""" to run: python -m data_preprocess.tokenizer """

DIR = Path(__file__).parent

try:
    sys.path.append(str(DIR.parent))
    from config import Config as config
    VOCAB_SIZE = config.vocab_size
except ImportError:
    VOCAB_SIZE = 12000
    print("Using default vocab_size: 12000")

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class BPETokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def load_data(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DIR / "data"
        else:
            data_dir = DIR / data_dir

        data_dir = Path(data_dir)

        with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(data_dir / "valid.txt", "r", encoding="utf-8") as f:
            valid_text = f.read()
        with open(data_dir / "test.txt", "r", encoding="utf-8") as f:
            test_text = f.read()

        all_text = train_text + valid_text + test_text
        return train_text, valid_text, test_text, all_text

    def train_tokenizer(self, all_text: str):
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2
        )

        self.tokenizer.train_from_iterator([all_text], trainer=trainer)

    def load_tokenizer(self, path: str):
        """
        Load a saved tokenizers Tokenizer JSON file (e.g. outputs/tokenizer/bpe_tokenizer.json).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> jnp.ndarray:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        encoded = self.tokenizer.encode(text)
        return jnp.array(encoded.ids, dtype=jnp.int32)

    def decode(self, tokens: jnp.ndarray) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def tokenize_splits(self, train_text: str, valid_text: str, test_text: str):
        train_tokens = self.encode(train_text)
        valid_tokens = self.encode(valid_text)
        test_tokens = self.encode(test_text)
        return train_tokens, valid_tokens, test_tokens

    def get_vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        return self.tokenizer.get_vocab_size()

    def save_tokenizer(self, save_path: str = None):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        if save_path is None:
            save_path = DIR / "outputs" / "tokenizer"
        else:
            save_path = DIR / save_path

        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(f"{save_path}/bpe_tokenizer.json")

    def save_data(self, train_tokens: jnp.ndarray, valid_tokens: jnp.ndarray, test_tokens: jnp.ndarray):
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", np.array(train_tokens))
        np.save(f"{save_dir}/valid_tokens.npy", np.array(valid_tokens))
        np.save(f"{save_dir}/test_tokens.npy", np.array(test_tokens))


class TiktokenAdapter:
    """
    Adapter exposing the same encode/decode API as BPETokenizer but backed by tiktoken.
    encode returns a jnp.ndarray of dtype int32 to match BPETokenizer.encode.
    """
    def __init__(self, encoding):
        self._enc = encoding

    def encode(self, text: str) -> jnp.ndarray:
        # tiktoken encoders expose .encode(...)
        ids = self._enc.encode(text)
        return jnp.array(ids, dtype=jnp.int32)

    def decode(self, tokens: jnp.ndarray) -> str:
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return self._enc.decode(tokens)

    def get_vocab_size(self) -> int:
        return getattr(self._enc, "n_vocab", None) or getattr(self._enc, "vocab_size", 0)


def get_tokenizer(cfg: config = None):
    """
    Factory: returns a tokenizer instance according to cfg.tokenizer.
    cfg.tokenizer: "BPE" (default) or "tiktoken" (case-insensitive).
    If "BPE" and cfg.tokenizer_vocab_file is set, BPETokenizer.load_tokenizer(...) will be used.
    """
    if cfg is None:
        cfg = config()

    backend = getattr(cfg, "tokenizer", "BPE")
    if isinstance(backend, str) and backend.lower() == "tiktoken":
        enc_name = getattr(cfg, "tokenizer_name", "gpt2")
        print(f"Using tokenizer backend: tiktoken (encoding='{enc_name}')")
        try:
            import tiktoken
        except Exception as e:
            raise RuntimeError("tiktoken requested but not installed. Install with: pip install tiktoken") from e
        enc = tiktoken.get_encoding(enc_name)
        return TiktokenAdapter(enc)

    # Default: BPETokenizer
    print("Using tokenizer backend: custom BPE")
    bpe = BPETokenizer(vocab_size=getattr(cfg, "vocab_size", VOCAB_SIZE))
    vocab_file = getattr(cfg, "tokenizer_vocab_file", None)
    if vocab_file:
        # try loading a saved tokenizer json
        try:
            bpe.load_tokenizer(vocab_file)
        except Exception:
            # ignore and let user call train_tokenizer manually
            pass
    return bpe


def _tokenize_with_progress(adapter, text: str, desc: str = "Tokenize words"):
    """
    Tokenize 'text' word-by-word using adapter.encode(...) and show a progress bar.
    Returns a numpy array of token ids.
    """
    # split on whitespace to mimic BPE training progress (counts words)
    words = text.split()
    ids_out = []
    if tqdm is not None:
        for w in tqdm(words, desc=desc, unit="word"):
            toks = adapter.encode(w)
            # adapter.encode may return jnp.ndarray or list
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            ids_out.extend(list(toks))
    else:
        total = len(words)
        for i, w in enumerate(words, 1):
            toks = adapter.encode(w)
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            ids_out.extend(list(toks))
            if i % max(1, total // 20) == 0 or i == total:
                print(f"{desc}: {i} / {total}")
    return np.array(ids_out, dtype=np.int32)


def main():
    # determine backend from config and print selection
    cfg = config
    tokenizer = get_tokenizer(cfg)

    # load raw text using BPETokenizer helper (no training implied)
    loader = BPETokenizer()
    train_text, valid_text, test_text, all_text = loader.load_data()

    if isinstance(tokenizer, BPETokenizer):
        # train custom BPE, save tokenizer and tokenized data
        print("Training custom BPE tokenizer and tokenizing data...")
        tokenizer.train_tokenizer(all_text)
        train_tokens, valid_tokens, test_tokens = tokenizer.tokenize_splits(train_text, valid_text, test_text)
        tokenizer.save_tokenizer()
        tokenizer.save_data(train_tokens, valid_tokens, test_tokens)
    else:
        # tokenizer is a TiktokenAdapter
        enc_name = getattr(cfg, "tokenizer_name", "gpt2")
        print(f"Tokenizing data with tiktoken encoding='{enc_name}'")
        # tokenise with progress bars similar to BPE flow
        train_ids = _tokenize_with_progress(tokenizer, train_text, desc="Tokenize train")
        valid_ids = _tokenize_with_progress(tokenizer, valid_text, desc="Tokenize valid")
        test_ids = _tokenize_with_progress(tokenizer, test_text, desc="Tokenize test")
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", train_ids)
        np.save(f"{save_dir}/valid_tokens.npy", valid_ids)
        np.save(f"{save_dir}/test_tokens.npy", test_ids)


if __name__ == "__main__":
    main()