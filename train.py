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

    #def on_train_begin(self, args, state, control, **kwargs):
    #    print("Training beginning")

    #def on_step_begin(self, args, state, control, **kwargs):
    #    print(f"Step beginning: {state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step end: {state.global_step}")
        print("self.eval_steps: ", self.eval_steps)
        # Only compute error rate every eval_steps
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
                        max_length=self.max_length,
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

            error_rate = 1 - (correct / total)
            print(f"\nStep {state.global_step} - Error rate on {self.num_examples} examples: {error_rate:.4f}")
            model.train()  # Set model back to training mode

    def on_train_end(self, args, state, control, **kwargs):
        print("Training ended")

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
PROMPT = "{input}\n\n### Answer:\n"
TARGET = "{label}"

# First format the dataset for training
def format_example(ex):
    return {
        "text": f"<bos>{SYSTEM}\n\n{PROMPT.format(**ex)}{TARGET.format(**ex)}<eos>"
    }

# Format dataset for training
ds = ds.map(format_example, remove_columns=["input", "label"])

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id if args.checkpoint is None else args.checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Load model from checkpoint or pretrained
if args.checkpoint:
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
else:
    print(f"Loading pretrained model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])

# Create a separate formatted dataset for evaluation
def format_eval_example(ex):
    return {
        "input": ex["input"],
        "label": ex["label"],
        "prompt": f"<bos>{SYSTEM}\n\n{PROMPT.format(**ex)}"
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=3,
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

# Initialize trainer with error rate callback
print("Creating trainer with ErrorRateCallback")  # Debug print
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
    callbacks=[ErrorRateCallback(test_ds, tokenizer, num_examples=10, eval_steps=10)]  # Use the unformatted test set
)
print("Trainer created")  # Debug print

# Start training
trainer.train()

# Save the final model
trainer.save_model(os.path.join(args.output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))