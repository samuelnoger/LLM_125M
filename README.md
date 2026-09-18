# 125M Parameter LLM Pre-Training Pipeline

This repository contains the architecture and training pipeline for a 125M parameter causal language model, written in PyTorch and optimized for local pre-training on Apple Silicon (MPS). 

The project demonstrates an end-to-end implementation of a modern Transformer architecture, focusing on memory-efficient training strategies to process large-scale datasets on consumer hardware.

## Architecture Highlights
The model implements a GPT-2 style causal language model architecture (125M parameters), enhanced with modern structural improvements:
* **SwiGLU Activation:** Replaced standard GeLU for improved convergence and expressivity.
* **Dimensions:** 12 Layers, 12 Attention Heads, $d_{model} = 768$, $d_{ff} = 2048$.
* **Vocabulary Optimization:** Padded to a multiple of 64 for optimal matrix multiplication efficiency on GPUs.

## Training & Hardware Optimization
The training loop is specifically configured to maximize hardware utilization on Apple Silicon (`device="mps"`). 
* **Dataset:** Pre-trained on the high-quality [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset.
* **Gradient Accumulation:** To bypass VRAM limitations while maintaining training stability, the script uses a micro-batch size of 2 with 128 gradient accumulation steps, yielding an effective batch size of 256.
* **Optimization:** Uses a learning rate of `2e-4` with 2,000 warmup steps to prevent gradient explosions during early stages.

## Repository Structure
* `/LLM/train/` - Contains the main PyTorch training loop, dataset tokenization, and evaluation logic.
* `/LLM/model/` - Contains the neural network architecture (Transformer blocks, Attention mechanisms, SwiGLU).
* `train.sh` - The bash executable for launching the training run with configured hyperparameters.

## How to Run
Ensure your environment is set up with PyTorch configured for MPS or CUDA. 

1. Clone the repository.
2. Install the required packages (e.g., `pip install torch datasets transformers`).
3. Execute the training script:
   ```bash
   ./train.sh
