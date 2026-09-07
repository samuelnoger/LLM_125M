from __future__ import annotations

import argparse


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the LLM starter model")
    parser.add_argument("--dataset", choices=("wikitext2", "c4", "fineweb-edu", "local", "dolly", "dailydialog"), default="wikitext2")
    parser.add_argument("--text-path", type=str, default=None, help="Path to a .txt file or folder of text files")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split for local text corpora")
    parser.add_argument("--train-token-limit", type=int, default=0, help="Optional cap on training tokens. 0 disables it")
    parser.add_argument("--val-token-limit", type=int, default=0, help="Optional cap on validation tokens. 0 disables it")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--vocab-pad-multiple", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional override. If > 0, stops after this many optimizer steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--train-token-offset", type=int, default=0, help="Optional offset for training tokens. Useful for resuming training from a specific point.")  
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of steps for linear learning rate warmup") 
    parser.add_argument("--pretrained-path", type=str, default="", help="Path to pre-trained checkpoint for SFT")
    parser.add_argument("--grad-accum-steps",type=int,default=4,help="Number of steps to accumulate gradients before updating weights"
)
    return parser


def parse_train_args() -> argparse.Namespace:
    return build_train_parser().parse_args()
