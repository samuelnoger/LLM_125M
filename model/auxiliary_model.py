from __future__ import annotations

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class ExtractorEngine:
    def __init__(self, checkpoint_dir: str | Path, device: str = "auto") -> None:
        self.checkpoint_dir = str(checkpoint_dir)
        
        # Hardware targeting for Apple Silicon inference
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load the trained 66M weights from your hard drive
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.checkpoint_dir)
        self.model.to(self.device)
        
        # Set to evaluation mode (turns off dropout layers for deterministic answers)
        self.model.eval()

    @torch.no_grad()
    def extract_span(self, question: str, context: str) -> tuple[str, float]:
        """Returns (extracted_span, confidence_score)."""
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=384,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        
        start_probs = torch.softmax(outputs.start_logits, dim=-1)
        end_probs = torch.softmax(outputs.end_logits, dim=-1)

        start_idx = torch.argmax(start_probs, dim=-1).item()
        end_idx = torch.argmax(end_probs, dim=-1).item()

        # Index 0 is [CLS] -> DistilBERT believes the answer is NOT in this sentence
        if start_idx == 0 or start_idx > end_idx:
            return "", 0.0

        score = float((start_probs[0, start_idx] * end_probs[0, end_idx]).item())
        answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
        extracted_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        
        return extracted_text, score

def load_extractor(checkpoint_dir: str | Path, device: str = "auto") -> ExtractorEngine:
    """Helper factory that chat_app.py imports to load the engine."""
    return ExtractorEngine(checkpoint_dir, device=device)