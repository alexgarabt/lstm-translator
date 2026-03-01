from dataclasses import dataclass

@dataclass
class ModelConfig:
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    attention_type: str = "dot"  # "dot" o "additive"

@dataclass  
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_epochs: int = 30
    gradient_clip: float = 1.0
    teacher_forcing_ratio: float = 1.0  # start 1.0, down side
    label_smoothing: float = 0.1
    
@dataclass
class DataConfig:
    dataset: str = "tatoeba"  # o "europarl"
    max_length: int = 50      # filtrar frases más largas
    min_freq: int = 2         # palabras con freq < 2 → <UNK>
    tokenizer: str = "word"   # "word" o "bpe"
    bpe_vocab_size: int = 8000
