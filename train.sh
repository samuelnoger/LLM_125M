#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../env/bin/python"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- PRE-TRAINING DATASET SETTINGS ---
# Switched to the massive educational dataset to teach the model English
DATASET="fineweb-edu"  
TRAIN_TOKEN_LIMIT=500000000  # Capped at 500M tokens for a reasonable MacBook run
TRAIN_TOKEN_OFFSET=0
TEXT_PATH=""
VAL_FRACTION=0.1
VAL_TOKEN_LIMIT=0

# --- THE 125M SWIGLU ARCHITECTURE ---
BLOCK_SIZE=512         
N_LAYERS=12            
N_HEADS=12             
D_MODEL=768            
D_FF=2048              
VOCAB_PAD_MULTIPLE=64
DROPOUT=0.0            # Set to 0.0 for pre-training. You want it to memorize as much as possible right now.

# --- PRE-TRAINING HYPERPARAMETERS ---
BATCH_SIZE=2          
EPOCHS=1               # Pre-training only ever does 1 pass over massive data
WARMUP_STEPS=2000      # CRITICAL: Increased heavily to prevent gradient explosions on random weights
MAX_STEPS=0
LEARNING_RATE=2e-4    
WEIGHT_DECAY=0.1
GRAD_CLIP_NORM=1.0
DEVICE="mps"           # Apple Silicon GPU
LOG_EVERY=50
EVAL_EVERY=500         # Increased to 500. Evaluating too often wastes time during pre-training
CHECKPOINT_DIR="LLM/checkpoints/pretrain_125M_phase_4"
PRETRAINED_PATH="LLM/checkpoints/pretrain_125M_phase_3/last.pt"     
GRAD_ACCUM_STEPS=128      

if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

cd "$PROJECT_ROOT"

ARGS=(
    --dataset "$DATASET"
    --train-token-limit "$TRAIN_TOKEN_LIMIT"
    --train-token-offset "$TRAIN_TOKEN_OFFSET"
    --val-token-limit "$VAL_TOKEN_LIMIT"
    --block-size "$BLOCK_SIZE"
    --n-layers "$N_LAYERS"
    --n-heads "$N_HEADS"
    --d-model "$D_MODEL"
    --d-ff "$D_FF"
    --vocab-pad-multiple "$VOCAB_PAD_MULTIPLE"
    --dropout "$DROPOUT"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --max-steps "$MAX_STEPS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --grad-clip-norm "$GRAD_CLIP_NORM"
    --device "$DEVICE"
    --log-every "$LOG_EVERY"
    --eval-every "$EVAL_EVERY"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --warmup-steps "$WARMUP_STEPS"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
)

# Only pass pretrained-path if it is not empty
if [[ -n "$PRETRAINED_PATH" ]]; then
    ARGS+=(--pretrained-path "$PRETRAINED_PATH")
fi

if [[ "$DATASET" == "local" ]]; then
    if [[ -z "$TEXT_PATH" ]]; then
        echo "TEXT_PATH is required when DATASET=local"
        exit 1
    fi
    ARGS+=(--text-path "$TEXT_PATH" --val-fraction "$VAL_FRACTION")
fi

exec "$PYTHON" -m LLM.train.train "${ARGS[@]}" "$@"