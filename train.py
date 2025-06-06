import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class ErrorRateCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, num_examples=100, eval_steps=100):
        print("Initializing ErrorRateCallback")
        # Format the dataset first, just like in evaluate.py
        formatted_ds = eval_dataset.map(format_eval_example)
        self.eval_dataset = formatted_ds.select(range(num_examples))
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_length = 512
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_steps = eval_steps


    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"Computing error rate at step {state.global_step}")
            model = kwargs.get('model')
            if model is None:
                print("Model not found in kwargs")
                return
                
            model.eval()
            correct = 0
            total = 0

            for ex in tqdm(self.eval_dataset, desc=f"Computing error rate at step {state.global_step}"):
                # Use the formatted text directly
                input_text = ex["prompt"]
                
                # Tokenize input
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

                # Generate answer
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=1,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        forced_bos_token_id=self.tokenizer.bos_token_id,
                        forced_eos_token_id=self.tokenizer.eos_token_id
                    )
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Extract the answer part from the generated text
                if "### Answer:\n" in generated_text:
                    answer = generated_text.split("### Answer:\n", 1)[1].strip()
                    answer = answer.replace("<eos>", "").strip()
                else:
                    answer = generated_text.strip()

                # Get the ground truth answer
                gt_answer = ex["label"].strip()

                # Truncate generated answer to match ground truth length
                answer = answer[:len(gt_answer)]

                # Compare answers (using startswith)
                if answer.startswith(gt_answer):
                    correct += 1
                total += 1

                # Print example with reasoning for debugging
                if total <= 3:  # Print first 3 examples
                    print("\nExample:")
                    print(f"Input: {ex['input']}")
                    print(f"Generated reasoning and answer: {generated_text}")
                    print(f"Ground truth: {gt_answer}")
                    print(f"Correct: {answer.startswith(gt_answer)}")

            error_rate = 1 - (correct / total)
            print(f"\nStep {state.global_step} - Error rate on {self.num_examples} examples: {error_rate:.4f}")
            model.train()  # Set model back to training mode

    def on_train_end(self, args, state, control, **kwargs):
        print("Training ended")

class KStepRolloutTrainer(Trainer):
    def __init__(self, *args, k_tokens=20, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_tokens = k_tokens
        if not isinstance(self.data_collator, CustomDataCollator):
            raise ValueError("KStepRolloutTrainer requires CustomDataCollator")
        self.tokenizer = self.data_collator.tokenizer
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the input_ids and attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get special token IDs
        begin_reasoning_id = self.tokenizer.encode(SPECIAL_TOKENS["begin_reasoning_token"])[0]
        end_reasoning_id = self.tokenizer.encode(SPECIAL_TOKENS["end_reasoning_token"])[0]
        
        # Initialize sequences with just the prompts
        full_sequences = []
        full_attention_masks = []
        valid_indices = []
        
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # Find the begin_reasoning token position
            begin_reasoning_pos = torch.where(input_ids[i] == begin_reasoning_id)[0]
            end_reasoning_pos = torch.where(input_ids[i] == end_reasoning_id)[0]
            
            if len(begin_reasoning_pos) == 1 and len(end_reasoning_pos) == 1:
                # Get question portion (including <bos>)
                question = input_ids[i, :begin_reasoning_pos[0] + 1]  # Include <begin_reasoning>
                full_sequences.append(question)
                full_attention_masks.append(attention_mask[i, :begin_reasoning_pos[0] + 1])
                valid_indices.append(i)
        
        # If no valid sequences, return zero loss with gradients enabled
        if not valid_indices:
            print("Warning: No valid sequences in batch!")
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # Generate k tokens for reasoning
        with torch.no_grad():
            for i in range(self.k_tokens):
                # Generate next token for each sequence based on its current state
                next_tokens = []
                for j in range(len(full_sequences)):
                    # Generate next token based on question + all previous reasoning tokens
                    outputs = model(
                        input_ids=full_sequences[j].unsqueeze(0),
                        attention_mask=full_attention_masks[j].unsqueeze(0),
                    )
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                    next_tokens.append(next_token[0])
                
                # Append the new token to each sequence
                for j in range(len(full_sequences)):
                    full_sequences[j] = torch.cat([full_sequences[j], next_tokens[j].unsqueeze(-1)], dim=-1)
                    full_attention_masks[j] = torch.cat([full_attention_masks[j], torch.ones_like(next_tokens[j].unsqueeze(-1))], dim=-1)
        
        # Now compute loss only on the target answer portion
        full_sequences_with_target = []
        full_attention_masks_with_target = []
        labels_list = []
        
        for i, idx in enumerate(valid_indices):
            # Get the target answer (after end_reasoning)
            end_reasoning_pos = torch.where(input_ids[idx] == end_reasoning_id)[0][0]
            target_answer = input_ids[idx, end_reasoning_pos:-1]  # Exclude <eos>
            
            # Create full sequence: question + reasoning + answer
            full_sequence = torch.cat([
                full_sequences[i],  # question + reasoning tokens
                torch.tensor([end_reasoning_id], device=target_answer.device),  # <end_reasoning>
                target_answer,      # answer
                torch.tensor([self.tokenizer.eos_token_id], device=target_answer.device),  # <eos>
            ], dim=0)
            
            # Create attention mask
            full_attention = torch.cat([
                full_attention_masks[i],  # question + reasoning tokens
                torch.ones(1, device=target_answer.device),  # <end_reasoning>
                torch.ones_like(target_answer),  # answer
                torch.ones(1, device=target_answer.device),  # <eos>
            ], dim=0)
            
            # Create labels where we only compute loss on the target answer portion
            labels = torch.full_like(full_sequence, -100)
            reasoning_end = full_sequences[i].size(0) + 1  # +1 for <end_reasoning>
            labels[reasoning_end:reasoning_end + target_answer.size(0)] = target_answer
            
            full_sequences_with_target.append(full_sequence)
            full_attention_masks_with_target.append(full_attention)
            labels_list.append(labels)
        
        # Pad all sequences to the same length
        max_len = max(seq.size(0) for seq in full_sequences_with_target)
        padded_sequences = []
        padded_masks = []
        padded_labels = []
        
        for i in range(len(full_sequences_with_target)):
            seq = full_sequences_with_target[i]
            mask = full_attention_masks_with_target[i]
            labels = labels_list[i]
            padding_len = max_len - seq.size(0)
            if padding_len > 0:
                # Add padding at the end
                seq = torch.cat([seq, torch.full((padding_len,), self.tokenizer.pad_token_id, device=seq.device)])
                mask = torch.cat([mask, torch.zeros(padding_len, device=mask.device)])
                labels = torch.cat([labels, torch.full((padding_len,), -100, device=labels.device)])
            padded_sequences.append(seq)
            padded_masks.append(mask)
            padded_labels.append(labels)
        
        # Stack all sequences and masks
        full_sequences_with_target = torch.stack(padded_sequences)
        full_attention_masks_with_target = torch.stack(padded_masks)
        labels = torch.stack(padded_labels)
        
        # Compute loss
        outputs = model(
            input_ids=full_sequences_with_target,
            attention_mask=full_attention_masks_with_target,
            labels=labels,
        )
        
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.tokenizer = tokenizer

    def torch_call(self, examples):
        # Convert examples to tensors
        batch = {
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in examples]),
            "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in examples])
        }
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description='Train Gemma model on shortest path dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint directory to resume training from')
    parser.add_argument('--output_dir', type=str, default='saved_models/gemma-shortest-path',
                      help='Directory to save model checkpoints')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Add timestamp to output directory if training from checkpoint
if args.checkpoint:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{timestamp}"
    print(f"Training from checkpoint. Output will be saved to: {args.output_dir}")

# Model and data configuration
model_id = "google/gemma-2b"
max_length = 512
batch_size = 4
gradient_accumulation_steps = 4

# Load dataset
ds = load_dataset("json", data_files="data/shortest_paths_train.jsonl")["train"]
ds = ds.train_test_split(test_size=0.05, seed=42)

# Keep a copy of the test set before formatting
test_ds = ds["test"]

SYSTEM = "You are a graph‑reasoning assistant."
PROMPT = "{input}\n\nLet's think step by step:\n"
TARGET = "{label}"

# Add special tokens for our task
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "begin_reasoning_token": "<begin_reasoning>",
    "end_reasoning_token": "<end_reasoning>"
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_id if args.checkpoint is None else args.checkpoint,
    local_files_only=True if args.checkpoint else False
)

# Add our special tokens
special_tokens = {
    "additional_special_tokens": [
        SPECIAL_TOKENS["sep_token"],
        SPECIAL_TOKENS["begin_reasoning_token"],
        SPECIAL_TOKENS["end_reasoning_token"]
    ]
}
tokenizer.add_special_tokens(special_tokens)

# Verify special tokens are unique
for token_name, token in SPECIAL_TOKENS.items():
    token_id = tokenizer.encode(token)[0]
    print(f"{token_name} ID: {token_id}")
    print(f"{token_name} decoded: {tokenizer.decode([token_id])}")

# Resize model's token embeddings to match new tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id if args.checkpoint is None else args.checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

def format_example(ex):
    # Format the basic sequence: question + separator + answer
    question = f"{SYSTEM}\n\n{PROMPT.format(**ex)}"
    answer = TARGET.format(**ex)
    
    # Tokenize question and answer separately
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    
    # Get special token IDs
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    begin_reasoning_id = tokenizer.encode(SPECIAL_TOKENS["begin_reasoning_token"])[0]
    end_reasoning_id = tokenizer.encode(SPECIAL_TOKENS["end_reasoning_token"])[0]
    
    # Construct the full sequence with special tokens
    full_sequence = (
        [bos_id] +                # <bos>
        question_tokens +         # question
        [begin_reasoning_id] +    # <begin_reasoning>
        [end_reasoning_id] +     # <end_reasoning>
        answer_tokens +          # answer
        [eos_id]                # <eos>
    )
    
    # Pad to max_length
    if len(full_sequence) < max_length:
        full_sequence = full_sequence + [tokenizer.pad_token_id] * (max_length - len(full_sequence))
    else:
        full_sequence = full_sequence[:max_length]
    
    # Create attention mask (1 for all real tokens, 0 for padding)
    attention_mask = [1] * (len(full_sequence) - full_sequence.count(tokenizer.pad_token_id))
    attention_mask.extend([0] * full_sequence.count(tokenizer.pad_token_id))
    
    return {
        "input_ids": full_sequence,
        "attention_mask": attention_mask
    }

# Format dataset for training
ds = ds.map(format_example, remove_columns=["input", "label"])

# Create a separate formatted dataset for evaluation
def format_eval_example(ex):
    return {
        "input": ex["input"],
        "label": ex["label"],
        "prompt": f"{SYSTEM}\n\n{PROMPT.format(**ex)}"
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    bf16=True,
    report_to="tensorboard",
    save_total_limit=3,
)

# Initialize trainer with k-token rollout
print("Creating trainer with KStepRolloutTrainer")
trainer = KStepRolloutTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=CustomDataCollator(tokenizer=tokenizer, mlm=False),
    callbacks=[ErrorRateCallback(test_ds, tokenizer, num_examples=10, eval_steps=500)],
    k_tokens=1,  # Number of tokens to generate for reasoning
    tokenizer=tokenizer  # Pass tokenizer explicitly
)
print("Trainer created")  # Debug print

# Start training
trainer.train()

# Save the final model
trainer.save_model(os.path.join(args.output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))