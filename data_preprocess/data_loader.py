import jax.numpy as jnp
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys

DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))
from config import Config as config


class DataLoader:
    def __init__(
        self,
        data_dir=DIR / "outputs" / "tokenized_data",
        seq_len=4,          # VERY SMALL sequence length
        batch_size=2        # VERY SMALL batch size
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = 0

    def load_and_prepare_data(self):
        """Load tokenized data and prepare for FAST testing"""

        # Load full arrays
        train_tokens = jnp.load(self.data_dir / "train_tokens.npy")
        valid_tokens = jnp.load(self.data_dir / "valid_tokens.npy")
        test_tokens = jnp.load(self.data_dir / "test_tokens.npy")

        # 🔥 USE VERY SMALL DATA (FAST)
        train_tokens = train_tokens[:20]
        valid_tokens = valid_tokens[:20]
        test_tokens = test_tokens[:20]

        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):
        """Create tiny sequences for fast execution"""

        window_size = self.seq_len + 1
        num_sequences = max(1, len(tokens) - window_size + 1)

        sequences = []
        for i in range(num_sequences):
            window = tokens[i:i + window_size]
            if len(window) < window_size:
                window = jnp.concatenate(
                    [window, jnp.full((window_size - len(window),), self.pad_token)]
                )
            sequences.append(window)

        sequences = jnp.stack(sequences)

        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        return NGCDataLoader(
            design_matrices=[("inputs", inputs), ("targets", targets)],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )
