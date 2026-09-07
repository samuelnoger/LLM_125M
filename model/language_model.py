import torch
from torch import nn

from ..config import ModelConfig
from .block import RMSNorm, TransformerBlock


class CausalLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE frequency caches up to block_size once
        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        t = torch.arange(config.block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Pre-unsqueeze to [1, 1, block_size, head_dim] right here
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(1), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(1), persistent=False)

        self.apply(self._init_weights)
        
        # Tie weights AFTER initialization to ensure exact pointer matching
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")

        # Slice the pre-shaped cache directly
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss