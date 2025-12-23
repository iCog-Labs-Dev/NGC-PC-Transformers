import jax.numpy as jnp
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys

DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))
from config import Config as config

class DataLoader:
    def __init__(self, data_dir= DIR / "outputs" / "tokenized_data", seq_len=config.seq_len, batch_size=config.batch_size):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = 0

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
        """Create sequences and return NGC DataLoader"""
        window_size = self.seq_len + 1 
        num_sequences = (len(tokens) - window_size + 1) // 1  
        
        if num_sequences <= 0:
            padded_tokens = jnp.concatenate([
                tokens, 
                jnp.full((window_size - len(tokens),), self.pad_token)
            ])
            sequences = padded_tokens.reshape(1, -1)  
        else:
            sequences = []
            for i in range(num_sequences):
                window = tokens[i:i + window_size]
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
