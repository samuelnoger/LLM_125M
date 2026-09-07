from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator
import sys

# Ensure the parent directory containing the 'LLM' package is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

from LLM.config import ModelConfig
from LLM.model import CausalLanguageModel
from LLM.utils import load_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained LLM checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)  # Kept safer for 256 block size
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)  # <--- Added repetition penalty arg
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_checkpoint_bundle(checkpoint_dir: str | Path, device: str = "auto") -> tuple[CausalLanguageModel, object, torch.device]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    tokenizer = load_bpe_tokenizer(checkpoint_path / "tokenizer.json")
    model_config = ModelConfig(**metadata["model_config"])

    resolved_device = resolve_device(device)
    model = CausalLanguageModel(model_config).to(resolved_device)
    checkpoint = torch.load(checkpoint_path / "last.pt", map_location=resolved_device)
    model.load_state_dict(checkpoint["model"])
    model.valid_vocab_size = int(metadata.get("data_vocab_size", tokenizer.vocab_size))
    return model, tokenizer, resolved_device


def apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for i in range(logits.size(0)):
        unique_tokens = tokens[i].unique()
        for token_id in unique_tokens:
            score = logits[i, token_id]
            if score < 0:
                logits[i, token_id] = score * penalty
            else:
                logits[i, token_id] = score / penalty
    return logits


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, valid_vocab_size: int | None = None) -> torch.Tensor:
    if valid_vocab_size is not None and valid_vocab_size < logits.size(-1):
        logits = logits.clone()
        logits[..., valid_vocab_size:] = float("-inf")
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def iter_generate_tokens(
    model: CausalLanguageModel,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.15,  # <--- Add default value here
) -> Iterator[torch.Tensor]:
    model.eval()
    tokens = prompt_tokens.clone()
    yield tokens
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]
        
        next_token_logits = apply_repetition_penalty(next_token_logits, tokens, repetition_penalty)
        
        valid_vocab_size = getattr(model, "valid_vocab_size", next_token_logits.size(-1))
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            valid_vocab_size=valid_vocab_size,
        )
        tokens = torch.cat([tokens, next_token], dim=1)
        yield tokens


@torch.no_grad()
def generate_text(
    model: CausalLanguageModel,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.15,  # <--- Add default value here too
) -> torch.Tensor:
    tokens = prompt_tokens.clone()
    for tokens in iter_generate_tokens(
        model,
        prompt_tokens=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    ):
        pass
    return tokens


def main() -> None:
    args = parse_args()
    model, tokenizer, _ = load_checkpoint_bundle(args.checkpoint_dir, device=args.device)

    prompt_tokens = tokenizer.encode(args.prompt).unsqueeze(0).to(next(model.parameters()).device)
    output_tokens = generate_text(
        model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    print(tokenizer.decode(output_tokens[0].cpu()))


if __name__ == "__main__":
    main()