from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data")
    max_length: int = 50          # filter long sentences
    vocab_size: int = 8000        # token vocabulary
    val_size: int = 2000
    test_size: int = 2000

    # Model
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_epochs: int = 30
    gradient_clip: float = 1.0
    teacher_forcing_start: float = 1.0   # Start at 100%
    teacher_forcing_end: float = 0.3     # En at 30%
    label_smoothing: float = 0.1

    # Logging
    log_every: int = 50           
    eval_every_epoch: int = 1     
    checkpoint_dir: Path = Path("checkpoints")
    tensorboard_dir: Path = Path("runs")

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"

    @property
    def encoder_dim(self) -> int:
        return 2 * self.hidden_dim  # BiLSTM concatenation forward + backward
