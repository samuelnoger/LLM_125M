from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re  

import torch

from .tokenizer import (
    BPETokenizerWrapper,
    train_bpe_tokenizer,
    save_bpe_tokenizer,
    load_bpe_tokenizer,
)


@dataclass
class PreparedCorpus:
    tokenizer: BPETokenizerWrapper
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor


def _get_or_train_bpe_tokenizer(texts: list[str], vocab_size: int = 32768) -> BPETokenizerWrapper:
    """Loads a pre-trained BPE tokenizer if it exists, otherwise trains one on the fly."""
    tokenizer_path = Path("LLM/checkpoints/bpe_tokenizer.json")
    
    if tokenizer_path.exists():
        return load_bpe_tokenizer(tokenizer_path)
    
    # Train new BPE tokenizer from the provided text iterator
    tokenizer = train_bpe_tokenizer(texts, vocab_size=vocab_size)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    save_bpe_tokenizer(tokenizer, tokenizer_path)
    return tokenizer


def prepare_dailydialog_corpus(
    val_fraction: float = 0.1,
    vocab_size: int = 32768,
) -> PreparedCorpus:
    tokenizer_path = Path("LLM/checkpoints/bpe_tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Pre-trained tokenizer not found at {tokenizer_path}! "
            "You must use the exact same tokenizer from your pre-training phase."
        )
    
    # Load your existing pre-trained tokenizer
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Dataset loading requires the 'datasets' package.") from exc

    # New cache file names so you don't overwrite your Dolly data
    cache_file = Path("LLM/checkpoints/dailydialog_train_tokens.pt")
    val_cache_file = Path("LLM/checkpoints/dailydialog_val_tokens.pt")

    if cache_file.exists() and val_cache_file.exists():
        print(f"Loading cached DailyDialog tokens from {cache_file}...")
        return PreparedCorpus(
            tokenizer=tokenizer,
            train_tokens=torch.load(cache_file),
            val_tokens=torch.load(val_cache_file)
        )

    print("Loading daily_dialog dataset from Hugging Face...")
    # trust_remote_code=True is often required for this specific dataset now
    dataset = load_dataset("OpenRL/daily_dialog", split="train")

    token_chunks = []
    total_tokens = 0

    for item in dataset:
        dialog = item.get("dialog", [])
        
        # DailyDialog provides a list of conversation turns.
        # We will loop through them in pairs. Turn 0 is User, Turn 1 is Assistant.
        for i in range(0, len(dialog) - 1, 2):
            user_text = dialog[i].strip()
            assistant_text = dialog[i+1].strip()

            if not user_text or not assistant_text:
                continue

            # Format it exactly how your chat app expects it
            conversation_text = f"User: {user_text}\nAssistant: {assistant_text}<|endoftext|>"

            ids = tokenizer.tokenizer.encode(conversation_text).ids
            if not ids:
                continue

            chunk = torch.tensor(ids, dtype=torch.long)
            token_chunks.append(chunk)
            total_tokens += len(chunk)

    if not token_chunks:
        raise ValueError("No valid training pairs found in DailyDialog dataset.")

    # Concatenate all conversations into one continuous 1D tensor
    all_tokens = torch.cat(token_chunks)
    
    # Split into train and validation sets
    train_tokens, val_tokens = split_tokens(all_tokens, val_fraction=val_fraction)

    # Cache them so you don't have to re-process next time
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, cache_file)
    torch.save(val_tokens, val_cache_file)
    print(f"Saved DailyDialog corpus: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens.")

    return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)


def prepare_dolly_corpus(
    val_fraction: float = 0.1,
    vocab_size: int = 32768,
) -> PreparedCorpus:
    tokenizer_path = Path("LLM/checkpoints/bpe_tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Pre-trained tokenizer not found at {tokenizer_path}! "
            "You must use the exact same tokenizer from your pre-training phase."
        )
    
    # Load your existing pre-trained tokenizer (DO NOT RETRAIN)
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Dataset loading requires the 'datasets' package.") from exc

    cache_file = Path("LLM/checkpoints/dolly_train_tokens.pt")
    val_cache_file = Path("LLM/checkpoints/dolly_val_tokens.pt")

    if cache_file.exists() and val_cache_file.exists():
        print(f"Loading cached Dolly-15k tokens from {cache_file}...")
        return PreparedCorpus(
            tokenizer=tokenizer,
            train_tokens=torch.load(cache_file),
            val_tokens=torch.load(val_cache_file)
        )

    print("Loading databricks/databricks-dolly-15k dataset from Hugging Face...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    token_chunks = []
    total_tokens = 0

    for item in dataset:
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        response = item.get("response", "").strip()

        if not instruction or not response:
            continue

        # Format strictly in order. Include context if Dolly provides it.
        if context:
            conversation_text = f"User: {instruction}\nContext: {context}\nAssistant: {response}<|endoftext|>"
        else:
            conversation_text = f"User: {instruction}\nAssistant: {response}<|endoftext|>"

        ids = tokenizer.tokenizer.encode(conversation_text).ids
        if not ids:
            continue

        chunk = torch.tensor(ids, dtype=torch.long)
        token_chunks.append(chunk)
        total_tokens += len(chunk)

    if not token_chunks:
        raise ValueError("No valid training pairs found in Dolly-15k dataset.")

    # Concatenate all conversations into one continuous 1D tensor
    all_tokens = torch.cat(token_chunks)
    
    # Split into train and validation sets
    train_tokens, val_tokens = split_tokens(all_tokens, val_fraction=val_fraction)

    # Cache them so you don't have to re-process next time
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, cache_file)
    torch.save(val_tokens, val_cache_file)
    print(f"Saved Dolly-15k corpus: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens.")

    return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)

def prepare_fineweb_edu_corpus(
    train_token_limit: int | None = 100_000_000,
    val_token_limit: int | None = 5_000_000,
    vocab_size: int = 32768,
    train_token_offset: int = 0,
    subset: str = "sample-10BT",
) -> PreparedCorpus:
    if train_token_limit is None or train_token_limit <= 0:
        train_token_limit = 100_000_000
    if val_token_limit is None or val_token_limit <= 0:
        val_token_limit = 5_000_000

    tokenizer_path = Path("LLM/checkpoints/bpe_tokenizer.json")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("FineWeb-Edu loading requires the 'datasets' package.") from exc

    # Automatically train and save the Byte-Level tokenizer if it doesn't exist yet
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}. Training a new Byte-Level tokenizer on a FineWeb sample...")
        stream = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)
        sample_texts = []
        for i, item in enumerate(stream):
            if i >= 20000:  # Grab 20k documents to train a robust tokenizer
                break
            if item.get("text"):
                sample_texts.append(clean_text(item["text"]))
                
        tokenizer = train_bpe_tokenizer(sample_texts, vocab_size=vocab_size)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        save_bpe_tokenizer(tokenizer, tokenizer_path)
        print(f"Successfully trained and saved new Byte-Level tokenizer to {tokenizer_path}")
    else:
        tokenizer = load_bpe_tokenizer(tokenizer_path)

    cache_file = Path(f"LLM/checkpoints/fineweb_train_tokens_offset_{train_token_offset}_limit_{train_token_limit}.pt")
    val_cache_file = Path("LLM/checkpoints/fineweb_val_tokens.pt")

    # 1. Exact cache match (Instant load)
    if cache_file.exists() and val_cache_file.exists():
        print(f"Loading exact cached FineWeb-Edu training tokens from {cache_file}...")
        train_tokens = torch.load(cache_file)
        val_tokens = torch.load(val_cache_file)
        return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)

    # 2. Smart Check: Can we reuse a smaller existing FineWeb cache if offset is 0?
    base_tokens = None
    actual_offset = train_token_offset
    
    if train_token_offset == 0:
        checkpoint_dir = Path("LLM/checkpoints")
        if checkpoint_dir.exists():
            for cand in checkpoint_dir.glob("fineweb_train_tokens_offset_0_limit_*.pt"):
                try:
                    cached_limit = int(cand.stem.split("limit_")[1])
                    if cached_limit < train_token_limit:
                        print(f"Found existing smaller FineWeb cache ({cached_limit / 1_000_000:.1f}M tokens). Reusing it...")
                        base_tokens = torch.load(cand)
                        actual_offset = cached_limit
                        break
                except (IndexError, ValueError):
                    continue

    def collect_tokens_from_stream(token_limit: int, token_offset: int = 0) -> torch.Tensor:
        stream = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)
        token_chunks = []
        total_tokens = 0
        skipped_tokens = 0
        
        print(f"Streaming from FineWeb-Edu ({subset}) starting at offset {token_offset / 1_000_000:.1f}M...")
        for item in stream:
            text = clean_text(item["text"])
            if not text:
                continue
                
            ids = tokenizer.tokenizer.encode(text).ids
            if not ids:
                continue
                
            chunk_len = len(ids)
            if skipped_tokens < token_offset:
                if skipped_tokens + chunk_len <= token_offset:
                    skipped_tokens += chunk_len
                    continue
                else:
                    slice_idx = token_offset - skipped_tokens
                    ids = ids[slice_idx:]
                    skipped_tokens = token_offset
            
            chunk = torch.tensor(ids, dtype=torch.long)
            token_chunks.append(chunk)
            total_tokens += len(chunk)
            
            if total_tokens >= token_limit:
                break
                
        if not token_chunks:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(token_chunks)[:token_limit]

    # Handle Validation Tokens: Extract them from the very start of the stream if missing
    if val_cache_file.exists():
        val_tokens = torch.load(val_cache_file)
    else:
        print(f"Extracting first {val_token_limit / 1_000_000:.1f}M tokens of FineWeb-Edu for validation...")
        val_tokens = collect_tokens_from_stream(val_token_limit, token_offset=0)
        torch.save(val_tokens, val_cache_file)

    # Shift training offset past the validation block so train & val don't overlap
    effective_train_offset = actual_offset + (val_token_limit if actual_offset == 0 else 0)

    if base_tokens is not None:
        remainder_limit = train_token_limit - len(base_tokens)
        if remainder_limit > 0:
            new_tokens = collect_tokens_from_stream(remainder_limit, effective_train_offset + len(base_tokens))
            train_tokens = torch.cat([base_tokens, new_tokens])
        else:
            train_tokens = base_tokens[:train_token_limit]
    else:
        train_tokens = collect_tokens_from_stream(train_token_limit, effective_train_offset)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, cache_file)
    print(f"Saved updated FineWeb-Edu token chunk to {cache_file}.")

    return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)

def prepare_c4_corpus(
    train_token_limit: int | None = 100_000_000,
    val_token_limit: int | None = 5_000_000,
    vocab_size: int = 32768,
    train_token_offset: int = 0,
) -> PreparedCorpus:
    if train_token_limit is None or train_token_limit <= 0:
        train_token_limit = 100_000_000
    if val_token_limit is None or val_token_limit <= 0:
        val_token_limit = 5_000_000

    tokenizer_path = Path("LLM/checkpoints/bpe_tokenizer.json")
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    cache_file = Path(f"LLM/checkpoints/c4_train_tokens_offset_{train_token_offset}_limit_{train_token_limit}.pt")
    val_cache_file = Path("LLM/checkpoints/c4_val_tokens.pt")

    # 1. Exact cache match (Instant load)
    if cache_file.exists() and val_cache_file.exists():
        print(f"Loading exact cached training tokens from {cache_file}...")
        train_tokens = torch.load(cache_file)
        val_tokens = torch.load(val_cache_file)
        return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)

    # 2. Smart Check: Can we reuse a smaller existing cache file if offset is 0?
    base_tokens = None
    actual_offset = train_token_offset
    
    if train_token_offset == 0:
        checkpoint_dir = Path("LLM/checkpoints")
        if checkpoint_dir.exists():
            for cand in checkpoint_dir.glob("c4_train_tokens_offset_0_limit_*.pt"):
                try:
                    cached_limit = int(cand.stem.split("limit_")[1])
                    # If we found a smaller cache than what we currently want
                    if cached_limit < train_token_limit:
                        print(f"Found existing smaller cache ({cached_limit / 1_000_000:.1f}M tokens). Reusing it...")
                        base_tokens = torch.load(cand)
                        actual_offset = cached_limit # Stream only the remainder!
                        break
                except (IndexError, ValueError):
                    continue

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("C4 loading requires the 'datasets' package.") from exc

    def collect_tokens_from_stream(split: str, token_limit: int, token_offset: int = 0) -> torch.Tensor:
        if token_limit is None or token_limit <= 0:
            token_limit = 100_000_000 if split == "train" else 5_000_000

        stream = load_dataset("allenai/c4", "en", split=split, streaming=True)
        token_chunks = []
        total_tokens = 0
        skipped_tokens = 0
        
        print(f"Streaming from C4 ({split}) starting at offset {token_offset / 1_000_000:.1f}M...")
        for item in stream:
            text = clean_text(item["text"])
            if not text:
                continue
                
            ids = tokenizer.tokenizer.encode(text).ids
            if not ids:
                continue
                
            chunk_len = len(ids)
            if skipped_tokens < token_offset:
                if skipped_tokens + chunk_len <= token_offset:
                    skipped_tokens += chunk_len
                    continue
                else:
                    slice_idx = token_offset - skipped_tokens
                    ids = ids[slice_idx:]
                    skipped_tokens = token_offset
            
            chunk = torch.tensor(ids, dtype=torch.long)
            token_chunks.append(chunk)
            total_tokens += len(chunk)
            
            if total_tokens >= token_limit:
                break
                
        if not token_chunks:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(token_chunks)[:token_limit]

    # Calculate how many *new* tokens we need to fetch for the remainder
    if base_tokens is not None:
        remainder_limit = train_token_limit - len(base_tokens)
        if remainder_limit > 0:
            new_tokens = collect_tokens_from_stream("train", remainder_limit, actual_offset)
            train_tokens = torch.cat([base_tokens, new_tokens])
        else:
            train_tokens = base_tokens[:train_token_limit]
    else:
        train_tokens = collect_tokens_from_stream("train", train_token_limit, train_token_offset)

    if val_cache_file.exists():
        val_tokens = torch.load(val_cache_file)
    else:
        val_tokens = collect_tokens_from_stream("validation", val_token_limit)

    # Save the new comprehensive cache file for future runs
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, cache_file)
    torch.save(val_tokens, val_cache_file)
    print(f"Saved updated token chunk to {cache_file}.")

    return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)


def load_text_files(paths: str | Path | Iterable[str | Path]) -> str:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    contents: list[str] = []
    for path in paths:
        resolved = Path(path)
        if resolved.is_dir():
            for file_path in sorted(resolved.rglob("*.txt")):
                contents.append(file_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            contents.append(resolved.read_text(encoding="utf-8", errors="ignore"))

    if not contents:
        raise ValueError("No text files were found")

    return "\n\n".join(contents)


def preprocess_wikitext(text: str) -> str:
    text = re.sub(r'^\s*=+[^=\n]+=+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(\s*=\s*){2,}\s*$', '', text, flags=re.MULTILINE)
    text = text.replace(" @,@ ", ", ")
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @.@ ", ". ")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text


def clean_text(text: str) -> str:
    text = preprocess_wikitext(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def split_tokens(token_ids: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    split_index = max(1, int(len(token_ids) * (1.0 - val_fraction)))
    split_index = min(split_index, len(token_ids) - 1)
    train_tokens = token_ids[:split_index]
    val_tokens = token_ids[split_index:]
    return train_tokens, val_tokens


def limit_tokens(token_ids: torch.Tensor, token_limit: int | None) -> torch.Tensor:
    if token_limit is None or token_limit <= 0:
        return token_ids
    return token_ids[: min(len(token_ids), token_limit)]


def prepare_character_corpus(
    paths: str | Path | Iterable[str | Path],
    val_fraction: float = 0.1,
    train_token_limit: int | None = None,
    val_token_limit: int | None = None,
    vocab_size: int = 32768,
) -> PreparedCorpus:
    text = clean_text(load_text_files(paths))
    tokenizer = _get_or_train_bpe_tokenizer([text], vocab_size=vocab_size)
    token_ids = tokenizer.encode(text)
    train_tokens, val_tokens = split_tokens(token_ids, val_fraction=val_fraction)
    train_tokens = limit_tokens(train_tokens, train_token_limit)
    val_tokens = limit_tokens(val_tokens, val_token_limit)
    return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)


def _load_wikitext_split(split: str) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "WikiText-2 loading requires the 'datasets' package. Install it with 'pip install datasets'."
        ) from exc

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [example["text"] for example in dataset if example["text"].strip()]
    if not texts:
        raise ValueError(f"WikiText-2 split '{split}' was empty")
    return clean_text("\n\n".join(texts))


def prepare_wikitext2_corpus(
    train_token_limit: int | None = None,
    val_token_limit: int | None = None,
    vocab_size: int = 32768,
) -> PreparedCorpus:
    train_text = _load_wikitext_split("train")
    val_text = _load_wikitext_split("validation")

    tokenizer = _get_or_train_bpe_tokenizer([train_text, val_text], vocab_size=vocab_size)
    train_tokens = limit_tokens(tokenizer.encode(train_text), train_token_limit)
    val_tokens = limit_tokens(tokenizer.encode(val_text), val_token_limit)
    return PreparedCorpus(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)