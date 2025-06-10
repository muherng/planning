import os
import torch
import warnings
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

# Suppress the specific deprecation warning about Trainer.tokenizer
warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated")

from transformers.modeling_outputs import CausalLMOutputWithPast

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
        # Ensure tokenizer is available
        print('WARNING')
        self.processing_class = tokenizer if tokenizer is not None else self.data_collator.tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        A batched and efficient implementation of the k-step rollout loss.
        """
        # Verify all sequences have the correct length
        for i, seq_ids in enumerate(inputs["input_ids"]):
            if len(seq_ids) != total_sequence_length:
                print(f"\nError: Sequence {i} has length {len(seq_ids)}, expected {total_sequence_length}")
                print(f"Sequence: {seq_ids}")
                raise ValueError(f"All sequences must have length {total_sequence_length}")
        
        # Get special token IDs from the tokenizer
        begin_reasoning_id = self.processing_class.encode(SPECIAL_TOKENS["begin_reasoning_token"], add_special_tokens=False)[0]
        end_reasoning_id = self.processing_class.encode(SPECIAL_TOKENS["end_reasoning_token"], add_special_tokens=False)[0]
        
        print(f"\nDebug - Special token IDs in compute_loss:")
        print(f"begin_reasoning_id: {begin_reasoning_id}")
        print(f"end_reasoning_id: {end_reasoning_id}")
        
        # === Stage 1: Prepare a batch of question prompts for generation ===
        # The input_ids from the dataloader contain the full sequence (q + special_tokens + a).
        # We must first extract just the question part for each item.
        
        question_ids_list = []
        answer_ids_list = []
        
        # This small loop is unavoidable but fast as it's just CPU tensor slicing.
        for i, seq_ids in enumerate(inputs["input_ids"]):
            #print(f"\nDebug - Processing sequence {i}:")
            #print(f"Sequence length: {len(seq_ids)}")
            #print(f"Non-pad tokens: {(seq_ids != self.processing_class.pad_token_id).sum().item()}")
            
            # Find the position of the special tokens
            begin_reasoning_pos = (seq_ids == begin_reasoning_id).nonzero(as_tuple=True)[0]
            end_reasoning_pos = (seq_ids == end_reasoning_id).nonzero(as_tuple=True)[0]
            #print(f"begin_reasoning_pos: {begin_reasoning_pos}, end_reasoning_pos: {end_reasoning_pos}")
            
            if len(begin_reasoning_pos) == 0 or len(end_reasoning_pos) == 0:
                print(f"No begin_reasoning_pos or end_reasoning_pos found in sequence: {seq_ids}")
                raise ValueError('Stop')
            
            if len(begin_reasoning_pos) == 1 and len(end_reasoning_pos) == 1:
                # Extract question: from <bos> up to and including <begin_reasoning>
                question = seq_ids[:begin_reasoning_pos[0] + 1]
                question_ids_list.append(question)
                
                # Extract answer: from after <end_reasoning> up to <eos>
                # We assume <eos> is the last token before padding.
                non_pad_tokens = seq_ids[seq_ids != self.processing_class.pad_token_id]
                eos_pos = (non_pad_tokens == self.processing_class.eos_token_id).nonzero(as_tuple=True)[0]
                answer = non_pad_tokens[end_reasoning_pos[0] + 1 : eos_pos[0]]
                answer_ids_list.append(answer)

        if not question_ids_list:
            print('No valid examples in batch')
            raise ValueError('Stop')
            # If the batch contains no valid examples, return zero loss.
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Pad questions to the same length
        max_question_len = max(len(q) for q in question_ids_list)
        padded_questions = []
        for q in question_ids_list:
            if len(q) < max_question_len:
                padding = torch.full((max_question_len - len(q),), self.processing_class.pad_token_id, device=q.device)
                q = torch.cat([q, padding])
            padded_questions.append(q)
        
        question_batch = {
            "input_ids": torch.stack(padded_questions),
            "attention_mask": (torch.stack(padded_questions) != self.processing_class.pad_token_id).long()
        }

        # === Stage 2: Generate reasoning for the entire batch in one call ===
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids=question_batch["input_ids"],
                attention_mask=question_batch["attention_mask"],
                max_new_tokens=self.k_tokens,
                # Use greedy decoding to match the original logic
                do_sample=False,
                num_beams=1,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )

        # === Stage 3: Combine generated sequences with answers and compute loss ===
        final_input_ids_list = []
        labels_list = []
        
        for i in range(len(generated_sequences)):
            # The generated_sequences tensor now contains `question + reasoning`.
            # We need to find the true end of the sequence before any padding.
            q_and_r_tokens = generated_sequences[i]
            
            # Find where the padding starts
            pad_start_pos = (q_and_r_tokens == self.processing_class.pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_start_pos) > 0:
                q_and_r_tokens = q_and_r_tokens[:pad_start_pos[0]] # Trim padding

            answer_tokens = answer_ids_list[i].to(model.device)

            # Construct the final sequence: [question + reasoning] + [<end_reasoning>] + [answer] + [<eos>]
            final_seq = torch.cat([
                q_and_r_tokens,
                torch.tensor([end_reasoning_id], device=model.device, dtype=torch.long),
                answer_tokens,
                torch.tensor([self.processing_class.eos_token_id], device=model.device, dtype=torch.long),
            ])
            
            # Pad to total_sequence_length
            if len(final_seq) < total_sequence_length:
                padding = torch.full((total_sequence_length - len(final_seq),), self.processing_class.pad_token_id, device=final_seq.device)
                final_seq = torch.cat([final_seq, padding])
            final_input_ids_list.append(final_seq)
            
            # Create the labels tensor, masking everything except the answer.
            labels = torch.full_like(final_seq, -100)
            answer_start_index = len(q_and_r_tokens) + 1  # +1 for <end_reasoning>
            # Shift the answer tokens by one position to the left for next token prediction
            labels[answer_start_index : answer_start_index + len(answer_tokens)] = answer_tokens
            num_supervised = (labels != -100).sum().item()
            #print("supervised tokens in this example:", num_supervised)
            labels_list.append(labels)

        # Stack the sequences into a batch
        final_batch = {
            "input_ids": torch.stack(final_input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": (torch.stack(final_input_ids_list) != self.processing_class.pad_token_id).long()
        }

        # Final forward pass to compute the loss on the correctly masked labels.
        outputs = model(
            input_ids=final_batch["input_ids"],
            attention_mask=final_batch["attention_mask"],
            labels=final_batch["labels"],
        )
        
        return outputs.loss if not return_outputs else (outputs.loss, outputs)

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
max_length = 1024
batch_size = 4
gradient_accumulation_steps = 4

# Load dataset
ds = load_dataset("json", data_files="data/shortest_paths_train.jsonl")["train"]
ds = ds.train_test_split(test_size=0.05, seed=42)

# Keep a copy of the test set before formatting
test_ds = ds["test"]

# Define prompts
SYSTEM = "You are a graph‑reasoning assistant."
PROMPT = "{input}\n\nLet's think step by step:\n"
TARGET = "{label}"

# Analyze token lengths in the dataset
print("\nAnalyzing dataset token lengths...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id if args.checkpoint is None else args.checkpoint,
    local_files_only=True if args.checkpoint else False
)

def analyze_example_lengths(examples):
    input_lengths = []
    label_lengths = []
    for ex in examples:
        input_text = f"{SYSTEM}\n\n{PROMPT.format(**ex)}"
        input_lengths.append(len(tokenizer.encode(input_text, add_special_tokens=False)))
        label_lengths.append(len(tokenizer.encode(TARGET.format(**ex), add_special_tokens=False)))
    return {
        "max_input_length": max(input_lengths),
        "min_input_length": min(input_lengths),
        "avg_input_length": sum(input_lengths) / len(input_lengths),
        "max_label_length": max(label_lengths),
        "min_label_length": min(label_lengths),
        "avg_label_length": sum(label_lengths) / len(label_lengths)
    }

length_stats = analyze_example_lengths(ds["train"])
print("\nDataset length statistics:")
print(f"Input lengths - Max: {length_stats['max_input_length']}, Min: {length_stats['min_input_length']}, Avg: {length_stats['avg_input_length']:.1f}")
print(f"Label lengths - Max: {length_stats['max_label_length']}, Min: {length_stats['min_label_length']}, Avg: {length_stats['avg_label_length']:.1f}")

# Calculate total sequence length needed
# Base sequence: max_input + max_label + special_tokens + k_tokens
k_tokens = 1  # This should match the k_tokens parameter in KStepRolloutTrainer
special_tokens = 4  # bos, begin_reasoning, end_reasoning, eos
total_sequence_length = length_stats['max_input_length'] + length_stats['max_label_length'] + special_tokens + k_tokens

print(f"\nTotal sequence length needed: {total_sequence_length}")

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

# Debug prints for special tokens
print("\nVerifying special tokens:")
for token in special_tokens["additional_special_tokens"]:
    token_id = tokenizer.encode(token, add_special_tokens=False)[0]
    print(f"Token: {token}, ID: {token_id}")
    print(f"Decoded: {tokenizer.decode([token_id])}")

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
    begin_reasoning_id = tokenizer.encode(SPECIAL_TOKENS["begin_reasoning_token"], add_special_tokens=False)[0]
    end_reasoning_id = tokenizer.encode(SPECIAL_TOKENS["end_reasoning_token"], add_special_tokens=False)[0]
    
    # Calculate space needed for reasoning tokens
    reasoning_space = k_tokens  # This will be filled with reasoning tokens during training
    
    # Calculate space needed for special tokens
    special_tokens_space = 4  # bos, begin_reasoning, end_reasoning, eos
    
    # Calculate how much space is available for question and answer
    available_space = total_sequence_length - special_tokens_space - reasoning_space
    
    # Construct the full sequence with special tokens and padding
    full_sequence = (
        [bos_id] +                # <bos>
        question_tokens +         # question
        [begin_reasoning_id] +    # <begin_reasoning>
        [tokenizer.pad_token_id] * reasoning_space +  # Space for reasoning tokens
        [end_reasoning_id] +     # <end_reasoning>
        answer_tokens +          # answer
        [eos_id]                # <eos>
    )
    
    # Pad to total_sequence_length
    if len(full_sequence) < total_sequence_length:
        full_sequence = full_sequence + [tokenizer.pad_token_id] * (total_sequence_length - len(full_sequence))
    
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
    #callbacks=[ErrorRateCallback(test_ds, tokenizer, num_examples=10, eval_steps=500)],
    k_tokens=1,  # Number of tokens to generate for reasoning
    tokenizer=tokenizer  # Pass tokenizer explicitly
)
print("Trainer created")  # Debug print

# Start training
trainer.train()

# Save the final model
trainer.save_model(os.path.join(args.output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))