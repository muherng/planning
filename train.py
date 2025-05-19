import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from huggingface_hub import login
import torch

# Login to Hugging Face
login(token="")

# Model and data configuration
model_id = "google/gemma-2b"  # or "google/gemma-7b" if you have enough resources
output_dir = "gemma-shortest-path"
max_length = 512
batch_size = 4  # Adjust based on your GPU memory
gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps

# Load and prepare dataset
ds = load_dataset("json", data_files="shortest_paths.jsonl")["train"]
ds = ds.train_test_split(test_size=0.05, seed=42)

SYSTEM = "You are a graph‑reasoning assistant."
PROMPT = "{input}\n\n### Answer:\n"
TARGET = "{label}"

def format_example(ex):
    return {
        "text": f"<bos>{SYSTEM}\n\n{PROMPT.format(**ex)}{TARGET.format(**ex)}<eos>"
    }

# Format dataset
ds = ds.map(format_example, remove_columns=["input", "label"])

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
    device_map="auto"  # Automatically handle model placement on available devices
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
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
    bf16=True,  # Use bfloat16 mixed precision training
    report_to="tensorboard",
    save_total_limit=3,  # Keep only the last 3 checkpoints
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
)

# Start training
trainer.train()

# Save the final model
trainer.save_model(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))