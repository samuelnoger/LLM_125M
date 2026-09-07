from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 1
    max_steps: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    device: str = "auto"
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 20
    grad_accum_steps: int = 4
    checkpoint_dir: str = "LLM/checkpoints/pretrain_125M"
    pretrained_path: str = ""
    warmup_steps: int = 1000
    
