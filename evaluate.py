import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Gemma model on shortest path dataset')
    parser.add_argument('--checkpoint', type=str, default='saved_models/gemma-shortest-path/checkpoint-8200',
                      help='Path to model checkpoint directory')
    parser.add_argument('--num_examples', type=int, default=100,
                      help='Number of examples to evaluate')
    parser.add_argument('--data_file', type=str, default='data/shortest_paths_test.jsonl',
                      help='Path to the test data file')
    return parser.parse_args()

# Paths and model config
args = parse_args()
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
print(f"Loading model from checkpoint: {args.checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
model.eval()

# Load test set directly
print(f"Loading test data from: {args.data_file}")
test_ds = load_dataset("json", data_files=args.data_file)["train"]
test_ds = test_ds.select(range(args.num_examples))  # Take only specified number of examples

SYSTEM = "You are a graph‑reasoning assistant."
PROMPT = "{input}\n\n### Answer:\n"

def format_example(ex):
    return {
        "input": ex["input"],
        "label": ex["label"],
        "prompt": f"<bos>{SYSTEM}\n\n{PROMPT.format(**ex)}"
    }

test_ds = test_ds.map(format_example)

# Evaluate
correct = 0
total = 0

for ex in tqdm(test_ds, desc="Evaluating"):
    # Prepare input
    input_text = ex["prompt"]
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    # Generate answer with explicit EOS handling
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            forced_bos_token_id=tokenizer.bos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Extract the answer part from the generated text
    if "### Answer:\n" in generated_text:
        answer = generated_text.split("### Answer:\n", 1)[1].strip()
        answer = answer.replace("<eos>", "").strip()
    else:
        answer = generated_text.strip()

    # Extract the ground truth answer
    gt_answer = ex["label"].strip()

    # Truncate generated answer to match ground truth length
    answer = answer[:len(gt_answer)]

    # Compare answers (using startswith)
    if answer.startswith(gt_answer):
        correct += 1
    total += 1

    # Print all examples for the smaller test set
    print("\nExample:")
    print(f"Input: {ex['input']}")
    print(f"Generated: {answer}")
    print(f"Ground truth: {gt_answer}")
    print(f"Correct: {answer.startswith(gt_answer)}")

# Results
accuracy = correct / total
print(f"\nEval accuracy (fraction correct): {accuracy:.4f}")
print(f"Total examples evaluated: {total}") 