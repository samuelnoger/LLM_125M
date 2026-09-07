# LLM

Minimal causal language-model starter package.

## Layout

- `model/`: Transformer building blocks and the causal language model
- `train/`: Training entry point and loop
- `utils/`: Config and data helpers

## How To Expand It

This starter is intentionally small so you can change one layer at a time without rewriting the whole project. A good expansion path is:

1. Replace the character tokenizer with a subword tokenizer such as BPE or SentencePiece.
2. Add a corpus loader that reads large text files, streams them from disk, and builds train/validation splits.
3. Add a proper dataset pipeline that chunks long sequences, shuffles efficiently, and can resume from saved tokenized data.
4. Scale the model by increasing `n_layers`, `n_heads`, `d_model`, and `block_size` in `ModelConfig`.
5. Add text generation helpers for greedy decoding, top-k, top-p, and temperature sampling.
6. Add evaluation code for loss tracking, perplexity, and sample quality on a held-out validation set.
7. Add checkpoint loading, experiment logging, and learning-rate scheduling so training is easier to resume and compare.
8. If the model gets large, add mixed precision, gradient accumulation, and distributed training support.

## Practical Rule

Make each step work before moving to the next one. A simple progression is:

`raw text -> tokenizer -> token dataset -> small model -> training loop -> generation -> evaluation -> scale up`

That keeps the project easy to debug while you search for a better text corpus.

## How To Train It

The trainer can now build the dataset for you. The usual flow is:

1. Gather a large text corpus.
2. Clean it enough to remove obvious noise, duplicates, and broken encodings.
3. Build a tokenizer from that corpus.
4. Convert the text into token IDs.
5. Feed those token IDs into `train()`.

You do not have to do those steps manually if you run the trainer script. It can prepare the corpus on its own in two ways:

- `--dataset wikitext2`: downloads and prepares WikiText-2 from Hugging Face
- `--dataset local --text-path /path/to/texts`: reads your own `.txt` files from disk

For WikiText-2, the helper `prepare_wikitext2_corpus()` in `utils/data_prep.py` loads the train and validation splits, builds the character tokenizer from the training text, and returns token tensors ready for the trainer.

You will need the `datasets` package for that:

```bash
pip install datasets
```

Run training like this from the project root:

```bash
python -m LLM.train.train --dataset wikitext2 --epochs 1
```

If you prefer a shell wrapper, you can also run:

```bash
bash ./train.sh --dataset wikitext2
```

The wrapper also has a config block at the top, so you can set `DATASET` there the same way you do in your other `.sh` scripts.

If the dataset is too large and training takes too long, shrink it with `TRAIN_TOKEN_LIMIT` in [train.sh](train.sh). For example, setting `TRAIN_TOKEN_LIMIT=500000` makes the training corpus much smaller and reduces the time per epoch.

For your own text files:

```bash
python -m LLM.train.train --dataset local --text-path ./my_texts
```

By default, training is epoch-based. You can still force a fixed step budget with `--max-steps N`.

Or with the wrapper:

```bash
bash ./train.sh --dataset local --text-path ./my_texts
```

The simplest training data is a large plain-text file or a folder full of plain-text files. Good sources are books, articles, documentation, forum archives, and code, as long as the license allows you to use them. The model learns next-token prediction, so it does not need labels.

What matters most:

- More text usually helps more than fancy preprocessing.
- Use a validation split so you can measure generalization later.
- If you keep the current character tokenizer, any UTF-8 text works.
- If you move to a subword tokenizer later, the same raw text can still be reused.

## Training Loop

The trainer does this on each batch:

- build fixed-length windows from token IDs
- predict the next token at every position
- compute cross-entropy loss
- backpropagate the gradients
- clip gradients
- update the weights with AdamW
- save `checkpoints/last.pt`

It also shows a `tqdm` progress bar so you can see step count and current loss while training.

Training now shows two progress bars:

- `epochs`: overall epoch progress
- `steps`: optimizer-step progress with live loss and tokens seen

## Performance Notes

On Apple Silicon, these settings are usually helpful:

- use `--device mps`
- prefer larger batch sizes until memory becomes a problem
- avoid very frequent CPU sync (the default `--log-every` is now less frequent)
- pad model vocabulary dimensions for better kernel alignment (`--vocab-pad-multiple 64`)

The trainer can pad model vocabulary size while keeping your real tokenizer vocabulary unchanged. Generation masks padded output IDs so decoded text still uses your real tokenizer characters.

## Testing The Model

Yes, the normal way to test it is to give it a prompt and let it generate more text.

After training, run:

```bash
python -m LLM.generate --prompt "The meaning of life is"
```

That loads `checkpoints/last.pt`, reads the saved tokenizer, and samples new tokens from the prompt.

Useful knobs:

- `--max-new-tokens`: how long the continuation should be
- `--temperature`: higher makes output more random, lower makes it more conservative
- `--top-k`: limits sampling to the k most likely next tokens

For example:

```bash
python -m LLM.generate --prompt "Once upon a time" --max-new-tokens 100 --temperature 0.8 --top-k 50
```

## Live Interface

If you want a small UI with a temperature slider and text that updates while it is being generated, run:

```bash
python -m LLM.interface
```

The window lets you change:

- prompt text
- temperature
- top-k
- number of new tokens
- checkpoint directory

It updates the output box as new tokens are sampled, so you can watch the continuation appear in real time.
