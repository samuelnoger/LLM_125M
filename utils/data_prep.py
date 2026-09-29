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


from dataclasses import dataclass

@dataclass
class PreparedSFTCorpus:
    tokenizer: BPETokenizerWrapper
    train_tokens: torch.Tensor
    train_labels: torch.Tensor
    val_tokens: torch.Tensor
    val_labels: torch.Tensor

def prepare_synthesis_corpus(
    val_fraction: float = 0.1,
    vocab_size: int = 32768,
    block_size: int = 512,
) -> PreparedSFTCorpus:
    tokenizer_path = Path("LLM/checkpoints/pretrain_125M_phase_4/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError("Pre-trained tokenizer not found!")
    
    tokenizer = load_bpe_tokenizer(tokenizer_path)
    pad_token_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")
    if pad_token_id is None:
        pad_token_id = 0

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Dataset loading requires the 'datasets' package.") from exc

    import random

    cache_file = Path("LLM/checkpoints/synthesis_sft_train.pt")
    
    if cache_file.exists():
        print(f"Loading cached Synthesis SFT tensors from {cache_file}...")
        cached_data = torch.load(cache_file)
        return PreparedSFTCorpus(tokenizer=tokenizer, **cached_data)

    print("Building Synthesis Blend directly in memory (this may take a minute)...")
    random.seed(42)
    unified_data = []

    # 1. Dolly 15k (All available context-based rows, up to 5000)
    print("Fetching Dolly...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_with_context = dolly.filter(lambda x: x["context"] is not None and len(x["context"].strip()) > 0)
    
    # Dynamically select up to 5000, or the max available (4,467)
    num_dolly = min(5000, len(dolly_with_context))
    for row in dolly_with_context.shuffle(seed=42).select(range(num_dolly)):
        unified_data.append((row["instruction"], row["context"], row["response"]))

    # 2. XSum (5,000 rows)
    print("Fetching XSum...")
    xsum = load_dataset("EdinburghNLP/xsum", split="train")
    
    num_xsum = min(5000, len(xsum))
    for row in xsum.shuffle(seed=42).select(range(num_xsum)):
        unified_data.append(("Synthesize and summarize the following information.", row["document"], row["summary"]))

    # 3. MS MARCO (10,000 rows via streaming)
    print("Fetching MS MARCO (streaming)...")
    ms_marco = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    marco_count = 0
    for row in ms_marco:
        if row["answers"] and len(row["answers"]) > 0:
            selected_passages = [
                p for p, is_sel in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]) 
                if is_sel == 1
            ]
            if selected_passages:
                context = " ".join(selected_passages)
                unified_data.append((row["query"], context, row["answers"][0]))
                marco_count += 1
        if marco_count >= 10000:
            break

    print(f"Shuffling merged dataset ({len(unified_data)} total samples)...")
    random.shuffle(unified_data)

    all_input_ids = []
    all_labels = []

    # --- THE FIX: We need sequence length block_size + 1 to perform the shift ---
    target_len = block_size + 1

    print("Tokenizing and padding sequences...")
    for instruction, context, response in unified_data:
        if not instruction or not response:
            continue

        prompt_text = f"User: {instruction.strip()}\nContext: {context.strip()}\nAssistant:"
        response_text = f" {response.strip()}<|endoftext|>"

        prompt_ids = tokenizer.tokenizer.encode(prompt_text).ids
        response_ids = tokenizer.tokenizer.encode(response_text).ids

        # NEW FIX: Ensure the prompt leaves at least 16 tokens for the response
        if len(prompt_ids) >= block_size - 16:
            continue

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # Truncate to 513
        input_ids = input_ids[:target_len]
        labels = labels[:target_len]

        # Pad to 513 if shorter
        pad_len = target_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_token_id] * pad_len)
            labels.extend([-100] * pad_len) 

        # Shift tokens
        shifted_input_ids = input_ids[:-1]
        shifted_labels = labels[1:]

        all_input_ids.append(shifted_input_ids)
        all_labels.append(shifted_labels)

    inputs_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    val_size = int(len(inputs_tensor) * val_fraction)
    train_tokens = inputs_tensor[val_size:]
    train_labels = labels_tensor[val_size:]
    val_tokens = inputs_tensor[:val_size]
    val_labels = labels_tensor[:val_size]

    save_data = {
        "train_tokens": train_tokens,
        "train_labels": train_labels,
        "val_tokens": val_tokens,
        "val_labels": val_labels
    }
    
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, cache_file)
    print(f"Saved Synthesis SFT corpus: {len(train_tokens)} training sequences.")

    return PreparedSFTCorpus(tokenizer=tokenizer, **save_data)

def prepare_dolly_corpus(
    val_fraction: float = 0.1,
    vocab_size: int = 32768,
    block_size: int = 512,
) -> PreparedSFTCorpus:
    tokenizer_path = Path("LLM/checkpoints/pretrain_125M_phase_4/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError("Pre-trained tokenizer not found!")
    
    tokenizer = load_bpe_tokenizer(tokenizer_path)
    pad_token_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")
    if pad_token_id is None:
        pad_token_id = 0

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Dataset loading requires the 'datasets' package.") from exc

    cache_file = Path("LLM/checkpoints/dolly_sft_train.pt")
    
    if cache_file.exists():
        print(f"Loading cached Dolly SFT tensors from {cache_file}...")
        cached_data = torch.load(cache_file)
        return PreparedSFTCorpus(tokenizer=tokenizer, **cached_data)

    print("Processing Dolly-15k dataset for SFT...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    all_input_ids = []
    all_labels = []

    # --- THE FIX: We need sequence length block_size + 1 to perform the shift ---
    target_len = block_size + 1

    for item in dataset:
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        response = item.get("response", "").strip()

        if not instruction or not response:
            continue

        if context:
            prompt_text = f"User: {instruction}\nContext: {context}\nAssistant:"
        else:
            prompt_text = f"User: {instruction}\nAssistant:"
            
        response_text = f" {response}<|endoftext|>"

        prompt_ids = tokenizer.tokenizer.encode(prompt_text).ids
        response_ids = tokenizer.tokenizer.encode(response_text).ids

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # Truncate to 513
        input_ids = input_ids[:target_len]
        labels = labels[:target_len]

        # Pad to 513 if shorter
        pad_len = target_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_token_id] * pad_len)
            labels.extend([-100] * pad_len) 

        # THE CRITICAL SHIFT: 
        # Inputs get tokens 0 to 511 (length 512)
        # Labels get tokens 1 to 512 (length 512)
        shifted_input_ids = input_ids[:-1]
        shifted_labels = labels[1:]

        all_input_ids.append(shifted_input_ids)
        all_labels.append(shifted_labels)

    inputs_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    val_size = int(len(inputs_tensor) * val_fraction)
    train_tokens = inputs_tensor[val_size:]
    train_labels = labels_tensor[val_size:]
    val_tokens = inputs_tensor[:val_size]
    val_labels = labels_tensor[:val_size]

    save_data = {
        "train_tokens": train_tokens,
        "train_labels": train_labels,
        "val_tokens": val_tokens,
        "val_labels": val_labels
    }
    
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, cache_file)
    print(f"Saved Dolly SFT corpus: {len(train_tokens)} training sequences.")

    return PreparedSFTCorpus(tokenizer=tokenizer, **save_data)

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
