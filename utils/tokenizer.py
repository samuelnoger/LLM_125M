from pathlib import Path
import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class BPETokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

def train_bpe_tokenizer(text_iterator, vocab_size: int = 32768) -> BPETokenizerWrapper:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Industry-standard Byte-Level handling for spaces and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
    )
    
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return BPETokenizerWrapper(tokenizer)

def save_bpe_tokenizer(tokenizer_wrapper: BPETokenizerWrapper, path: str | Path) -> None:
    tokenizer_wrapper.tokenizer.save(str(path))

def load_bpe_tokenizer(path: str | Path) -> BPETokenizerWrapper:
    tokenizer = Tokenizer.from_file(str(path))
    return BPETokenizerWrapper(tokenizer)
