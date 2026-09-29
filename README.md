# 125M Parameter LLM & Decoupled RAG Architecture

This repository contains the architecture and training pipeline for a 125M parameter causal language model, alongside a lightweight, local Compound AI system. Written in PyTorch and optimized for Apple Silicon (MPS), the project bridges memory-efficient pre-training with a decoupled Retrieval-Augmented Generation (RAG) inference pipeline designed to eliminate multi-turn induction loops.

The project demonstrates an end-to-end implementation from pre-training a SwiGLU-based Transformer to deploying it via a Tkinter-based conversational interface with a DistilBERT semantic gatekeeper. Detailed documentation and architectural notes are consolidated within the LLM Project file.

## Architecture Highlights

**1. Generative Synthesizer (125M Causal LM)**

* **Architecture:** GPT-2 style causal language model featuring 12 Layers, 12 Attention Heads, $d_{model} = 768$, and $d_{ff} = 2048$.
* **Activation:** SwiGLU replaces standard GeLU to improve convergence and representational expressivity.
* **Vocabulary Optimization:** Padded to a multiple of 64 for optimal matrix multiplication efficiency on consumer GPUs.

**2. Compound Orchestration & Decoupled RAG**

* **Extractive Gatekeeper:** Integrates a 66M parameter DistilBERT model for high-fidelity, extractive sentence scoring, guaranteeing the generative model only receives mathematically verified context.
* **Dynamic Query Rewriting:** Uses spaCy to isolate noun chunks and entities from prior conversation turns, injecting missing subjects into the current query (pronoun resolution) without feeding raw chat history into the RAG context.
* **Dual-Mode Routing (chat_app.py):**
* *RAG Mode:* Triggers when system context is present. Feeds the 125M model a strict single-turn prompt containing only DistilBERT-extracted facts, keeping temperature cold ($T=0.2$) for factual synthesis.
* *Free Chat Mode:* Triggers when system context is empty. Bypasses DistilBERT and feeds the generative model exactly one turn of conversation history, allowing natural conversational follow-ups using its internal weights.



## Training, Datasets & Hardware Optimization

The training loop maximizes Apple Silicon (device="mps") utilization through strict memory management. Pre-training and SFT should be executed separately to ensure stability.

* **Pre-Training:** Trained on the Hugging Face FineWeb-Edu dataset using a learning rate of 2e-4 with 2,000 warmup steps. The script leverages a micro-batch size of 2 with 128 gradient accumulation steps to yield an effective batch size of 256.
* **Supervised Fine-Tuning (SFT):** Executed as a separate step after pre-training. The model weights are optimized for strict factual QA and summarization using a composite dataset of MS MARCO and XSum.
* **Conversational Fallback:** A secondary set of weights trained exclusively on the Databricks Dolly dataset is documented for open-ended, creative chat where catastrophic forgetting of general world knowledge limits the strictly fine-tuned RAG weights.
* **Auxiliary Model Training:** The DistilBERT extractor is fine-tuned independently from the generative model using the train_auxiliary_model.py script.

## Repository Structure

* **/LLM/train/** - Main PyTorch training loop, tokenization, gradient accumulation, and evaluation logic.
* **/LLM/model/** - Neural network architecture (Transformer blocks, Attention mechanisms, SwiGLU).
* **/utils/** - Utility scripts, including generate.py for headless inference and text generation.
* **chat_app.py** - The Tkinter-based graphical interface managing the routing logic between Free Chat and RAG mode.
* **compound_pipeline.py** - The orchestrator housing spaCy query rewriting and DistilBERT extraction logic.
* **train.sh** - Bash executable for launching the causal pre-training run.
* **train_auxiliary_model.py** - Script for training the standalone DistilBERT extractor.

## How to Run

Ensure your environment is set up with PyTorch configured for MPS or CUDA.

1. Clone the repository.
2. Install the required dependencies.
3. Execute the pre-training script for the causal model:
```bash
./train.sh
```
4. Run the Supervised Fine-Tuning (SFT) phase separately using the same script with adjusted parameters.
5. Train the DistilBERT extractor model separately by running:
```bash
python train_auxiliary_model.py
```
6. Launch the conversational interface:
```bash
python chat_app.py
```

## Development Methodology

The core Transformer architecture and PyTorch boilerplate for this model were scaffolded with the assistance of AI coding tools. Primary technical contributions focus on configuring the memory-efficient pre-training pipeline, optimizing hardware utilization for Apple Silicon (MPS) via gradient accumulation, hyperparameter tuning, and architecting the decoupled Compound AI pipeline to resolve generative induction loops in small-parameter models.

## Acknowledgements & References

The architecture, datasets, and component models utilized in this project are based on the following works:

* **GPT-2 Architecture:** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI*.
* **SwiGLU Activation:** Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv preprint arXiv:2002.05202*.
* **DistilBERT:** Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *Hugging Face*.
* **spaCy:** Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
* **FineWeb-Edu Dataset:** Hugging Face. (2024). *FineWeb-Edu*.
* **Dolly 15k Dataset:** Conover, M., Hayes, C., Mathur, A., et al. (2023). Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM. *Databricks*.
* **XSum Dataset:** Narayan, S., Cohen, S. B., & Lapata, M. (2018). Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization. *arXiv preprint arXiv:1808.08745*.
* **MS MARCO Dataset:** Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., & Deng, L. (2016). MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.
