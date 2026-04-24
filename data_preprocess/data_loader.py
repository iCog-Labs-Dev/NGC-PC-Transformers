import jax.numpy as jnp
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys
import numpy as np

DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))

class DataLoader:
    def __init__(self, seq_len, batch_size, stride=None, data_dir= DIR / "outputs" / "tokenized_data"):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.stride = stride if stride is not None else seq_len  # Default to non-overlapping
        self.pad_token = -1

    def load_and_prepare_data(self):
        """Load tokenized data and prepare for training"""
        train_tokens = jnp.load(self.data_dir / "train_tokens.npy")
        valid_tokens = jnp.load(self.data_dir / "valid_tokens.npy")
        test_tokens = jnp.load(self.data_dir / "test_tokens.npy")

        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):
        """Create sequences with configurable stride"""
        window_size = self.seq_len + 1  # 209 tokens per sequence
        
        if self.stride == self.seq_len:
            # Non-overlapping (PyTorch style) - more efficient
            total_len = (len(tokens) // window_size) * window_size
            tokens = tokens[:total_len]
            num_sequences = total_len // window_size
            sequences = tokens.reshape(num_sequences, window_size)
        else:
            # Sliding window with custom stride
            num_sequences = (len(tokens) - window_size) // self.stride + 1
            if num_sequences <= 0:
                # Handle insufficient data
                sequences = jnp.full((1, window_size), self.pad_token)
                sequences[0, :len(tokens)] = tokens
            else:
                # Extract sequences with given stride
                sequences = []
                for i in range(0, len(tokens) - window_size + 1, self.stride):
                    sequences.append(tokens[i:i + window_size])
                sequences = jnp.stack(sequences)
        
        # Split into inputs and targets
        inputs = sequences[:, :-1]    # Shape: [num_sequences, seq_len]
        targets = sequences[:, 1:]    # Shape: [num_sequences, seq_len]
        
        # Create mask (ignore padding tokens if any)
        mask = (targets != self.pad_token).astype(jnp.float32)
        
        return NGCDataLoader(
            design_matrices=[
                ("inputs", inputs), 
                ("targets", targets),
                ("mask", mask)
            ],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )
