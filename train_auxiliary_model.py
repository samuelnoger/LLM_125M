import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm

def main():
    # 1. Hardware setup for Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load the base Encoder and Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)

    import os
    from datasets import load_dataset
    
    #3. Load official SQuAD v1.1 directly from Hugging Face without dataset script errors
    raw_datasets = load_dataset("rajpurkar/squad")
    
    # We will train on a subset for speed, but you can remove the slicing later
    train_dataset = raw_datasets["train"].select(range(10000))

    # 4. Data Preparation: Map character indices to token indices
    def prepare_train_features(examples):
        # Tokenize with truncation only on the document context, not the question
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # If no answer exists, point to the [CLS] token
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # If the answer is truncated out of the window, point to [CLS]
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        return tokenized_examples

    print("Tokenizing and mapping answers...")
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    # Format for PyTorch DataLoader
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 5. Native PyTorch Training Loop
    optimizer = AdamW(model.parameters(), lr=3e-5)
    epochs = 2
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            
            # Move batch to Apple Silicon GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            # Forward pass: AutoModelForQuestionAnswering computes the dual loss for us automatically
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch} Average Loss: {total_loss / len(train_dataloader)}")

    # 6. Save the Extractor for the UI
    output_dir = "checkpoints/extractor_66M"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()