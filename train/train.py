from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from LLM.config import ModelConfig, TrainConfig
    from LLM.model import CausalLanguageModel
    from LLM.utils import (
        BPETokenizerWrapper,
        TextDataset,
        prepare_character_corpus,
        prepare_wikitext2_corpus,
        prepare_c4_corpus,
        prepare_fineweb_edu_corpus,
        prepare_dolly_corpus,
        prepare_dailydialog_corpus,
        save_bpe_tokenizer,
    )
    from LLM.utils.arguments import parse_train_args
else:
    from ..config import ModelConfig, TrainConfig
    from ..model import CausalLanguageModel
    from ..utils import (
        BPETokenizerWrapper,
        TextDataset,
        prepare_character_corpus,
        prepare_wikitext2_corpus,
        prepare_c4_corpus,
        prepare_dolly_corpus,
        prepare_fineweb_edu_corpus,
        prepare_dailydialog_corpus,
        save_bpe_tokenizer,
    )
    from ..utils.arguments import parse_train_args


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, loss_history: list, epoch_idx: int, loader_idx: int, tokens_seen: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, 
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "loss_history": loss_history,
        "epoch_idx": epoch_idx,
        "loader_idx": loader_idx,
        "tokens_seen": tokens_seen, 
    }, path)


def load_corpus(args: argparse.Namespace):
    train_token_limit = args.train_token_limit if args.train_token_limit > 0 else None
    val_token_limit = args.val_token_limit if args.val_token_limit > 0 else None

    if args.dataset == "dolly":
        return prepare_dolly_corpus()
    
    if args.dataset == "dailydialog":
        return prepare_dailydialog_corpus()

    if args.dataset == "wikitext2":
        return prepare_wikitext2_corpus(
            train_token_limit=train_token_limit,
            val_token_limit=val_token_limit,
        )

    if args.dataset == "fineweb-edu":
        return prepare_fineweb_edu_corpus(
            train_token_limit=train_token_limit,
            val_token_limit=val_token_limit,
            train_token_offset=args.train_token_offset,
            subset="sample-10BT",
        )
    
    if args.dataset == "c4":
        return prepare_c4_corpus(
            train_token_limit=train_token_limit,
            val_token_limit=val_token_limit,
            train_token_offset=args.train_token_offset
        )

    if args.text_path is None:
        raise ValueError("--text-path is required when --dataset local")
    return prepare_character_corpus(
        args.text_path,
        val_fraction=args.val_fraction,
        train_token_limit=train_token_limit,
        val_token_limit=val_token_limit,
    )


def pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def get_lr(
    step: int, 
    warmup_steps: int, 
    max_steps: int, 
    max_lr: float, 
    min_lr: float = 1e-5
) -> float:
    """Universal schedule: Automatically computes the correct learning rate 
       for any step, whether starting at 0 or resuming mid-training.
    """
    if step >= max_steps:
        return min_lr

    # Linear warmup phase
    if step < warmup_steps:
        return max_lr * (step / max(1, warmup_steps))

    # Global cosine decay phase
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(
    token_ids: torch.Tensor,
    vocab_size: int,
    tokenizer: BPETokenizerWrapper,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> CausalLanguageModel:
    device = resolve_device(train_config.device)
    print(f"training device: {device}")

    dataset = TextDataset(token_ids=token_ids, block_size=model_config.block_size)

    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError("No training batches available. Reduce block size or batch size.")

    model = CausalLanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "last.pt"
    step = 0
    loss_history = []
    start_epoch = 0
    start_loader_idx = 0
    tokens_seen = 0
    
    if checkpoint_path.exists():
        # Case A: Resuming an already-in-progress SFT run
        print(f"Found existing SFT checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", 0)
        loss_history = checkpoint.get("loss_history", [])
        start_epoch = checkpoint.get("epoch_idx", 0)
        start_loader_idx = checkpoint.get("loader_idx", 0)
        tokens_seen = checkpoint.get("tokens_seen", step * train_config.batch_size * model_config.block_size)
        
    elif hasattr(train_config, "pretrained_path") and train_config.pretrained_path:
        # Case B: Brand new SFT run — load pre-trained weights from step 365k!
        print(f"Loading pre-trained base model weights from {train_config.pretrained_path}...")
        pretrained_ckpt = torch.load(train_config.pretrained_path, map_location=device)
        model.load_state_dict(pretrained_ckpt["model"])
        print("Successfully loaded pre-trained foundation weights. Starting SFT with fresh optimizer.")
    else:
        print("No checkpoint found. Starting training from scratch...")

    metadata = {
        "model_config": model_config.__dict__,
        "train_config": train_config.__dict__,
        "data_vocab_size": vocab_size,
        "model_vocab_size": model_config.vocab_size,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    save_bpe_tokenizer(tokenizer, checkpoint_dir / "tokenizer.json")

    model.train()
    tokens_seen = 0
    max_steps = train_config.max_steps if train_config.max_steps > 0 else train_config.epochs * steps_per_epoch
    total_epochs = math.ceil(max_steps / steps_per_epoch)

    warmup_steps = train_config.warmup_steps
    use_amp = device.type in {"mps", "cpu"}
    amp_dtype = torch.bfloat16

    step_progress = tqdm(total=max_steps, desc="steps", unit="step", position=1, dynamic_ncols=True)
    if step > 0:
        step_progress.update(step)

    grad_accum_steps = train_config.grad_accum_steps

    optimizer.zero_grad(set_to_none=True)

    for epoch_idx in range(total_epochs):
        # Skip old epochs if resuming further along
        if epoch_idx < start_epoch:
            continue

        for loader_idx, (input_ids, targets) in enumerate(loader):
            # Skip old batches within the resume epoch
            if epoch_idx == start_epoch and loader_idx < start_loader_idx:
                continue
            
            # Reset start indices after the first resumed iteration so future loops run normally
            if epoch_idx == start_epoch and loader_idx == start_loader_idx:
                start_loader_idx = 0 

            lr = get_lr(step, warmup_steps, max_steps, train_config.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, targets)

            if loss is None:
                raise RuntimeError("Loss was not computed")

            loss = loss / grad_accum_steps
            loss.backward()

            tokens_seen += input_ids.numel()
            
            is_accum_boundary = ((loader_idx + 1) % grad_accum_steps == 0) or ((loader_idx + 1) == len(loader))
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            current_loss_val = (loss * grad_accum_steps).item()
            
            if step % 50 == 0:
                loss_history.append({"step": step, "loss": round(current_loss_val, 4)})

            if step % train_config.log_every == 0:
                step_progress.set_postfix(
                    loss=f"{current_loss_val:.4f}",
                    lr=f"{lr:.2e}",
                    tokens=f"{tokens_seen / 1_000_000:.2f}M",
                )
            
            if step > 0 and step % train_config.eval_every == 0:
                # Pass current epoch_idx and loader_idx to checkpoint saver
                save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, loss_history, epoch_idx, loader_idx, tokens_seen)
                if device.type == "mps":
                    torch.mps.empty_cache()

            step_progress.update(1)
            step += 1
            if step >= max_steps:
                break

        if step >= max_steps:
            break

    step_progress.close()
    save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, loss_history, epoch_idx, loader_idx, tokens_seen)
    return model


def main() -> None:
    args = parse_train_args()
    corpus = load_corpus(args)
    data_vocab_size = corpus.tokenizer.vocab_size
    model_vocab_size = pad_to_multiple(data_vocab_size, args.vocab_pad_multiple)
    if model_vocab_size != data_vocab_size:
        print(f"vocab padding: data_vocab_size={data_vocab_size} model_vocab_size={model_vocab_size}")

    model_config = ModelConfig(
        vocab_size=model_vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_path=args.pretrained_path,
        warmup_steps=args.warmup_steps, 
        grad_accum_steps=args.grad_accum_steps,
    )

    print(
        f"corpus ready: vocab_size={data_vocab_size} "
        f"train_tokens={len(corpus.train_tokens)} val_tokens={len(corpus.val_tokens)}"
    )
    train(corpus.train_tokens, data_vocab_size, corpus.tokenizer, model_config, train_config)


if __name__ == "__main__":
    main()