from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import sys
import tiktoken

""" to run: python -m data_preprocess.tokenizer """

DIR = Path(__file__).parent

try:
    sys.path.append(str(DIR.parent))
    from config import Config as config
    VOCAB_SIZE = config.vocab_size
except ImportError:
    VOCAB_SIZE = 12000
    print("Using default vocab_size: 12000")


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
        Load a saved tokenizers Tokenizer JSON file
        (e.g. outputs/tokenizer/bpe_tokenizer.json).
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

    def decode(self, tokens) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        if hasattr(tokens, "tolist"):
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

    def save_data(self, train_tokens, valid_tokens, test_tokens):
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", np.array(train_tokens))
        np.save(f"{save_dir}/valid_tokens.npy", np.array(valid_tokens))
        np.save(f"{save_dir}/test_tokens.npy", np.array(test_tokens))



_BEST_ENCODING = "o200k_base"


class TiktokenTokenizer:
    """
    Tokenizer backed by OpenAI's tiktoken library.
    Default encoding is 'o200k_base' (GPT-4o), the most recent and largest
    vocabulary available in tiktoken.

    Drop-in replacement for BPETokenizer: same encode / decode /
    get_vocab_size / tokenize_splits / save_data API.
    """

    def __init__(self, encoding: str = _BEST_ENCODING):
        self.encoding = encoding
        self._enc = tiktoken.get_encoding(encoding)
        print(
            f"[TiktokenTokenizer] encoding='{encoding}'  "
            f"vocab_size={self._enc.n_vocab}"
        )

    # ------------------------------------------------------------------
    def encode(self, text: str) -> jnp.ndarray:
        ids = self._enc.encode(text)
        return jnp.array(ids, dtype=jnp.int32)

    def decode(self, tokens) -> str:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self._enc.decode(tokens)

    def get_vocab_size(self) -> int:
        return self._enc.n_vocab

    def tokenize_splits(self, train_text: str, valid_text: str, test_text: str):
        train_tokens = self.encode(train_text)
        valid_tokens = self.encode(valid_text)
        test_tokens = self.encode(test_text)
        return train_tokens, valid_tokens, test_tokens

    def save_data(self, train_tokens, valid_tokens, test_tokens):
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", np.array(train_tokens))
        np.save(f"{save_dir}/valid_tokens.npy", np.array(valid_tokens))
        np.save(f"{save_dir}/test_tokens.npy", np.array(test_tokens))