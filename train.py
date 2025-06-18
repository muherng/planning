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
import json

# Suppress the specific deprecation warning about Trainer.tokenizer
warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated")

from transformers.modeling_outputs import CausalLMOutputWithPast

def get_formatted_dataset_path(data_file, k_tokens):
    """Generate path for formatted dataset based on input file and k_tokens."""
    data_path = Path(data_file)
    base_name = data_path.stem  # filename without extension
    formatted_name = f"{base_name}_formatted_k{k_tokens}.json"
    return Path("data") / formatted_name

def save_formatted_dataset(dataset, data_file, k_tokens):
    """Save formatted dataset to disk."""
    save_path = get_formatted_dataset_path(data_file, k_tokens)
    save_path.parent.mkdir(exist_ok=True)
    
    print(f"Saving formatted dataset to {save_path}")
    dataset.save_to_disk(str(save_path.with_suffix('')))
    print(f"Formatted dataset saved successfully")

def load_formatted_dataset(data_file, k_tokens):
    """Load formatted dataset from disk if it exists."""
    load_path = get_formatted_dataset_path(data_file, k_tokens)
    load_path_dir = load_path.with_suffix('')
    
    if load_path_dir.exists():
        print(f"Loading pre-formatted dataset from {load_path_dir}")
        from datasets import load_from_disk
        return load_from_disk(str(load_path_dir))
    else:
        print(f"No pre-formatted dataset found at {load_path_dir}")
        return None

class ExactMatchCallback(TrainerCallback):
    """
    Periodically runs full‑pipeline inference on the held‑out set and
    prints exact‑match accuracy (all answer tokens must match).
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        k_tokens: int,
        eval_steps: int = 1000,
        max_answer_tokens: int = 2048,
    ):
        # store raw (un‑formatted) dataset; we re‑format on the fly
        self.eval_ds = eval_dataset
        self.tok = tokenizer
        self.k = k_tokens
        self.eval_steps = eval_steps
        self.max_answer = max_answer_tokens

        # constant IDs
        self.BOS = tokenizer.bos_token_id
        self.PAD = tokenizer.pad_token_id
        self.EOS = tokenizer.eos_token_id
        self.BEGIN = tokenizer.encode(
            SPECIAL_TOKENS["begin_reasoning_token"], add_special_tokens=False
        )[0]
        self.END = tokenizer.encode(
            SPECIAL_TOKENS["end_reasoning_token"], add_special_tokens=False
        )[0]

    # ────────────────────────────────────────────────────────────────
    # helper: build prompt ids  <bos>  question  <begin_reasoning>
    # ────────────────────────────────────────────────────────────────
    def _build_prompt(self, ex):
        q_text = (
            f"{SYSTEM}\n\n{PROMPT.format(**ex)}"
        )  # >>> same recipe as format_example
        q_ids = self.tok.encode(q_text, add_special_tokens=False)
        return torch.tensor([self.BOS] + q_ids + [self.BEGIN], dtype=torch.long)

    # ────────────────────────────────────────────────────────────────
    # main callback hook
    # ────────────────────────────────────────────────────────────────
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return

        model = kwargs["model"]
        device = next(model.parameters()).device
        model.eval()

        exact, total = 0, 0

        for ex in tqdm(self.eval_ds, desc=f"[EM eval @ {state.global_step}]"):
            # ----- 1. build prompt & (optionally) generate K-token reasoning -----
            prompt = self._build_prompt(ex).to(device).unsqueeze(0)

            if self.k > 0:
                # Generate latent reasoning only when K > 0
                with torch.no_grad():
                    gen1 = model.generate(
                        prompt,
                        max_new_tokens=self.k,
                        pad_token_id=self.PAD,
                        eos_token_id=self.EOS,
                        use_cache=True,
                    )

                # Extract the tokens generated AFTER the prompt
                gen_seq = gen1[0]
                prompt_len = prompt.shape[1]
                reasoning_tokens = gen_seq[prompt_len:]

                # Remove EOS and everything after it
                eos_pos = (reasoning_tokens == self.EOS).nonzero(as_tuple=True)
                if eos_pos[0].numel() > 0:
                    reasoning_tokens = reasoning_tokens[: eos_pos[0][0]]

                # Pad to exactly k tokens
                if reasoning_tokens.shape[0] < self.k:
                    pad_len = self.k - reasoning_tokens.shape[0]
                    pad_tensor = torch.full((pad_len,), self.PAD, dtype=reasoning_tokens.dtype, device=device)
                    reasoning_tokens = torch.cat([reasoning_tokens, pad_tensor])

                reasoning = reasoning_tokens.tolist()
            else:
                reasoning = []  # No reasoning tokens when k == 0

            # ----- 2. build second input  prompt + reasoning + <end> -----
            start2 = torch.tensor(
                prompt.tolist()[0] + reasoning + [self.END], device=device
            ).unsqueeze(0)

            # ────────────────────────────────────────────────────────────────
            # Stop generation early when the model has produced as many
            # tokens as the gold answer – beyond this point exact-match is
            # impossible, so we save compute by capping `max_new_tokens`.
            # ────────────────────────────────────────────────────────────────
            gold_answer_len = len(
                self.tok.encode(ex["label"], add_special_tokens=False)
            )

            with torch.no_grad():
                gen2 = model.generate(
                    start2,
                    max_new_tokens=gold_answer_len,  # cap by gold length
                    pad_token_id=self.PAD,
                    eos_token_id=self.EOS,
                    use_cache=True,
                )

            answer_tokens = (
                gen2[0][start2.shape[1] :]  # tokens after <end_reasoning>
            )
            pred_answer = self.tok.decode(answer_tokens, skip_special_tokens=True).strip()
            gold_answer = ex["label"].strip()

            exact += int(pred_answer == gold_answer)
            total += 1

            if total <= 3:  # print a few examples
                print("\n── Example ──")
                print("Q:", ex["input"])
                print("Pred:", pred_answer)
                print("Gold:", gold_answer)
                print("Match:", pred_answer == gold_answer)
                print("Reasoning:", self.tok.decode(reasoning, skip_special_tokens=True))

        acc = exact / total if total else 0.0
        print(
            f"\n[Step {state.global_step}] exact‑match accuracy "
            f"on {total} eval examples: {acc:.4%}"
        )

        model.train()


class KStepRolloutTrainer(Trainer):
    def __init__(
        self,
        *args,
        k_tokens: int = 20,
        tokenizer=None,
        rl_objective: str = "wake_sleep",   # NEW – "reinforce" | "wake_sleep"
        lambda_pg: float = 0.1,               # NEW – weight on policy-gradient term
        **kwargs,
    ):
        """Extend default trainer with optional REINFORCE objective.

        Args:
            k_tokens: number of latent-reasoning tokens to sample / supervise.
            tokenizer: tokenizer used by the data collator (passed explicitly so
                this class remains usable outside the default training script).
            rl_objective: either "wake_sleep" (default supervised path) or
                "reinforce" (policy-gradient on the reasoning tokens).
            lambda_pg: weight on the policy-gradient term when rl_objective ==
                "reinforce".
        """

        super().__init__(*args, **kwargs)

        self.k_tokens = k_tokens
        self.rl_objective = rl_objective
        self.lambda_pg = lambda_pg
        self.running_R = 0.0  # moving-average reward baseline

        if not isinstance(self.data_collator, CustomDataCollator):
            raise ValueError("KStepRolloutTrainer requires CustomDataCollator")

        # Ensure tokenizer is available
        self.processing_class = (
            tokenizer if tokenizer is not None else self.data_collator.processing_class
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Implements a two-stage latent-reasoning pipeline:
        1. Generation pass (no grad): Generate K reasoning tokens
        2. Supervised pass (with grad): Compute loss on answer span
        """
        # Get special token IDs
        BOS_ID = self.processing_class.bos_token_id
        PAD_ID = self.processing_class.pad_token_id
        EOS_ID = self.processing_class.eos_token_id
        BEGIN_ID = self.processing_class.encode(SPECIAL_TOKENS["begin_reasoning_token"], add_special_tokens=False)[0]
        END_ID = self.processing_class.encode(SPECIAL_TOKENS["end_reasoning_token"], add_special_tokens=False)[0]

        # ---------------------------------------------------------------------
        # Guard: REINFORCE objective must have k_tokens > 0
        # ---------------------------------------------------------------------
        if self.rl_objective == "reinforce" and self.k_tokens == 0:
            raise ValueError("REINFORCE needs k_tokens > 0")

        # ────────────────────────────────────────────────────────────────
        # Short-circuit path when k_tokens == 0 (wake-sleep objective)
        # ────────────────────────────────────────────────────────────────
        if self.k_tokens == 0:
            seqs = inputs["input_ids"]                   # (B, L)
            PAD_ID = self.processing_class.pad_token_id
            EOS_ID = self.processing_class.eos_token_id

            # Build supervision labels (only supervise answer span + <eos>)
            new_labels = torch.full_like(seqs, -100)
            for i, row in enumerate(seqs):
                end_pos = (row == END_ID).nonzero(as_tuple=True)[0][0]
                eos_pos = (row == EOS_ID).nonzero(as_tuple=True)[0][0]
                ans_start = end_pos + 1
                new_labels[i, ans_start:eos_pos] = row[ans_start:eos_pos]
                new_labels[i, eos_pos] = EOS_ID

            attn_mask = (seqs != PAD_ID).long()
            outputs = model(input_ids=seqs, attention_mask=attn_mask, labels=new_labels)
            return outputs.loss if not return_outputs else (outputs.loss, outputs)

        # =====================================================================
        # CASE 2 ───────────────────────────────────────────────────────────────
        # REINFORCE objective  (policy-gradient on reasoning tokens)
        # =====================================================================
        if self.rl_objective == "reinforce":
            if self.k_tokens == 0:
                raise ValueError("REINFORCE needs k_tokens > 0")

            # ---------- 1. split prompt vs. reasoning placeholder ----------
            seqs = inputs["input_ids"]  # (B, L_tot)
            begin_pos = (seqs == BEGIN_ID).nonzero(as_tuple=True)[1]  # (B,)
            prompts = [row[:pos + 1] for row, pos in zip(seqs, begin_pos)]
            prompt_batch = self.processing_class.pad(
                {"input_ids": prompts}, return_tensors="pt"
            ).to(model.device)

            # ---------- 2. sample K reasoning tokens WITH gradients ----------
            model.train()  # ensure grads are on
            r_tok, logp_r = self._sample_reasoning(
                model, prompt_batch["input_ids"], self.k_tokens, PAD_ID, EOS_ID
            )

            # ---------- 3. rebuild full sequence q ⊕ r ⊕ END ⊕ answer ----------
            full_ids, labels = [], []
            for orig, r in zip(seqs, r_tok):
                start = (orig == BEGIN_ID).nonzero(as_tuple=True)[0][0] + 1
                rebuilt = orig.clone()
                rebuilt[start : start + self.k_tokens] = r

                lab = torch.full_like(rebuilt, -100)
                end_pos = (rebuilt == END_ID).nonzero(as_tuple=True)[0][0]
                ans_start = end_pos + 1
                eos_pos = (rebuilt == EOS_ID).nonzero(as_tuple=True)[0][0]
                lab[ans_start:eos_pos] = rebuilt[ans_start:eos_pos]
                lab[eos_pos] = EOS_ID

                full_ids.append(rebuilt)
                labels.append(lab)

            full_ids = torch.stack(full_ids)
            labels = torch.stack(labels)
            attn_mask = (full_ids != PAD_ID).long()

            # ---------- 4. forward pass for answer CE ----------
            out = model(input_ids=full_ids, attention_mask=attn_mask, labels=labels)
            answer_loss = out.loss  # scalar

            # ---------- 5. reward, baseline, advantage ----------
            with torch.no_grad():
                reward = (-answer_loss).detach()  # higher is better
            self.running_R = 0.9 * self.running_R + 0.1 * reward.item()
            advantage = reward - self.running_R  # broadcast across batch

            # ---------- 6. policy-gradient loss on reasoning ----------
            policy_loss = -(advantage * logp_r).mean()

            total_loss = answer_loss + self.lambda_pg * policy_loss
            if self.state is not None and self.state.global_step % 10 == 0:
                print(f"Total loss: {total_loss}")
                print(f"Answer loss: {answer_loss}")
                print(f"Policy loss: {policy_loss}")
                print(f"Running reward: {self.running_R}")
                print(f"Advantage: {advantage}")
                print(f"Logp_r: {logp_r}")
                print(f"Reward: {reward}")
                print(f"Full loss: {out.loss}")
            return (total_loss, out) if return_outputs else total_loss

        # ---------------------------------------------------------------------
        # fall-through: wake-sleep objective (original supervised pipeline)
        # ---------------------------------------------------------------------
        if self.rl_objective not in ("wake_sleep", "reinforce"):
            raise ValueError(f"Unknown rl_objective: {self.rl_objective}")

        # ---------- 1. isolate prompts ----------
        seqs = inputs["input_ids"]  # (B, TOTAL_LEN)
        #begin_pos = (seqs == BEGIN_ID).long().argmax(-1)  # first BEGIN per row
        # --- safer begin_position retrieval in compute_loss ---
        begin_idxs = (seqs == BEGIN_ID).nonzero(as_tuple=True)
        if begin_idxs[0].numel() != seqs.size(0):
            raise ValueError("BEGIN_ID missing in some sequences")
        begin_pos = begin_idxs[1]          # shape: (B,)
        
        prompts = []
        for row, pos in zip(seqs, begin_pos):
            prompts.append(row[:pos+1])  # inclusive of BEGIN_ID

        # ---------- 2. left‑pad & generate ----------
        old_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        prompt_batch = self.processing_class.pad(
            {"input_ids": prompts}, return_tensors="pt"
        ).to(model.device)
        model.eval()
        with torch.no_grad():
            gen = model.generate(
                **prompt_batch,
                max_new_tokens=self.k_tokens,
                pad_token_id=PAD_ID,
                eos_token_id=EOS_ID,
                use_cache=True
            )
        model.train()
        self.processing_class.padding_side = old_side

        # --- keep exactly K new tokens (remove early <eos>, pad to k_tokens) ---
        prompt_len = prompt_batch["input_ids"].shape[1]
        gen_reasonings = []
        for row in gen:
            # Tokens the model actually produced (≤ k_tokens by construction)
            reas = row[prompt_len:]

            # Drop everything from the first EOS onward (including EOS)
            eos_pos = (reas == EOS_ID).nonzero(as_tuple=True)
            if eos_pos[0].numel() > 0:
                reas = reas[: eos_pos[0][0]]

            # Pad to exactly k_tokens
            if reas.shape[0] < self.k_tokens:
                pad_len = self.k_tokens - reas.shape[0]
                reas = torch.cat(
                    [
                        reas,
                        torch.full((pad_len,), PAD_ID, dtype=reas.dtype, device=reas.device),
                    ]
                )
            gen_reasonings.append(reas)

        # ---------- 3. re-assemble full sequences ----------
        new_input_ids = []
        new_labels = []
        for orig, reas in zip(seqs, gen_reasonings):
            # Safeguard: ensure reasoning span has exactly k_tokens
            if reas.shape[0] < self.k_tokens:
                pad_len = self.k_tokens - reas.shape[0]
                reas = torch.cat(
                    [
                        reas,
                        torch.full((pad_len,), PAD_ID, dtype=reas.dtype, device=reas.device),
                    ]
                )
            elif reas.shape[0] > self.k_tokens:
                reas = reas[: self.k_tokens]

            # locate placeholder start (just after BEGIN_ID)
            start = (orig == BEGIN_ID).nonzero(as_tuple=True)[0][0] + 1

            # replace the K pads
            rebuilt = orig.clone()
            rebuilt[start : start + self.k_tokens] = reas

            # build labels: supervise only ANSWER (after END_ID) (+ eos)
            labels = torch.full_like(rebuilt, -100)
            end_pos = (rebuilt == END_ID).nonzero(as_tuple=True)[0][0]
            ans_start = end_pos + 1
            eos_pos = (rebuilt == EOS_ID).nonzero(as_tuple=True)[0][0]
            labels[ans_start : eos_pos] = rebuilt[ans_start : eos_pos]
            labels[eos_pos] = EOS_ID  # supervise eos token

            new_input_ids.append(rebuilt)
            new_labels.append(labels)

        input_ids = torch.stack(new_input_ids)
        labels = torch.stack(new_labels)
        attn_mask = (input_ids != PAD_ID).long()

        # ---------- 4. supervised forward ----------
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels
        )

        return outputs.loss if not return_outputs else (outputs.loss, outputs)

    # ────────────────────────────────────────────────────────────────
    # helper: sample K reasoning tokens **with gradients & sanitation**
    # ────────────────────────────────────────────────────────────────
    def _sample_reasoning(self, model, prompt, K, PAD_ID, EOS_ID):
        """
        Autoregressively sample *exactly* `K` tokens while collecting their
        log-probabilities.  Any sampled `<eos>` is replaced by `<pad>` and no
        further log-prob is accumulated for that example.  The returned tensor
        is always length-`K`, padded where necessary.

        Returns
        -------
        r_tok  : (B, K) long
            Clean reasoning span (no EOS; padded with PAD).
        logp_r : (B,)   float
            Sum of log-probs *up to but excluding* the first EOS token for each
            example.  Gradients flow through this sum.
        """

        device = prompt.device
        B = prompt.size(0)

        logp_r = torch.zeros(B, device=device)
        r_tok = []

        past = None
        cur = prompt                      # (B, L₀)
        cur_attn = (prompt != PAD_ID).long()
        alive = torch.ones(B, dtype=torch.bool, device=device)  # mask of sequences that have not hit EOS

        for _ in range(K):
            # Forward pass (with grad kept)
            out = model(
                input_ids=cur,
                attention_mask=cur_attn,
                past_key_values=past,
                use_cache=True,
            )
            logits, past = out.logits[:, -1], out.past_key_values  # (B, V)
            probs = torch.softmax(logits, dim=-1)                 # (B, V)

            # Sample next token independently for each batch element
            tok = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Gather probability of sampled token
            gathered = probs.gather(1, tok).squeeze(1)  # (B,)

            # Add log-prob only for sequences still alive (i.e., before EOS)
            logp_r = logp_r + torch.log(gathered) * alive.float()

            # Detect EOS just generated
            just_eos = (tok.squeeze(1) == EOS_ID)

            # Build a mask for tokens that should become PAD (EOS or already finished)
            pad_mask = (just_eos | (~alive)).unsqueeze(1)  # (B,1)
            tok = tok.masked_fill(pad_mask, PAD_ID)

            r_tok.append(tok)

            # Update alive mask AFTER using it for logp
            alive = alive & (~just_eos)

            # Next-step inputs: sampled token, full attention
            cur, cur_attn = tok, torch.ones_like(tok)

        r_tok = torch.cat(r_tok, dim=1)  # (B, K)
        return r_tok, logp_r

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.processing_class = tokenizer

    def torch_call(self, examples):
        # Dynamically pad to the longest sequence in the current batch
        batch = self.processing_class.pad(
            examples,
            padding="longest",
            return_tensors="pt"
        )
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description='Train Gemma model on shortest path dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint directory to resume training from')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save model checkpoints (default: auto-generated with k_tokens)')
    parser.add_argument('--k_tokens', type=int, default=0,
                      help='Number of tokens to generate for reasoning')
    parser.add_argument('--data_file', type=str, default='data/cfg/arith_depth9_10000_train.jsonl',
                      help='Path to the input data file')
    parser.add_argument('--use_cached_data', action='store_true',
                      help='Load pre-formatted dataset if available')
    parser.add_argument('--save_formatted_data', action='store_true', default=True,
                      help='Save formatted dataset for future use')
    parser.add_argument('--train_mode', type=str, default='wake_sleep',
                      help='Training mode: wake_sleep or reinforce')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Set default output directory with k_tokens if not specified
if args.output_dir is None:
    args.output_dir = f'saved_models/depth9/gemma-cfg-k{args.k_tokens}-{args.train_mode}'

# Add timestamp to output directory if training from checkpoint
if args.checkpoint:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{timestamp}"
    print(f"Training from checkpoint. Output will be saved to: {args.output_dir}")

print(f"Model will be saved to: {args.output_dir}")

# Model and data configuration
model_id = "google/gemma-2b"
max_length = 100000
batch_size = 4
gradient_accumulation_steps = 4

# Load dataset
print(f"Loading dataset from {args.data_file}")

ds = None
test_ds = None

# Try to load pre-formatted dataset if requested
if args.use_cached_data:
    ds = load_formatted_dataset(args.data_file, args.k_tokens)
    if ds is not None:
        # Dataset was loaded from cache
        test_ds = ds["test"]
        print("Using cached formatted dataset")

# If no cached dataset was loaded, process from scratch
if ds is None:
    print("Processing dataset from scratch...")
    ds = load_dataset("json", data_files=args.data_file)["train"]
    # Split into train and test with fixed test size
    ds = ds.train_test_split(test_size=30, seed=42)
    
    # Keep a copy of the test set before formatting
    test_ds = ds["test"]

# Define prompts
SYSTEM = "You are a graph‑reasoning assistant."
PROMPT = "{input}\n\nLet's think step by step:\n"
TARGET = "{label}"

# Analyze token lengths in the dataset (only if processing from scratch)
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

# Only analyze lengths if we have the original data (not cached formatted data)
if 'input' in ds["train"].column_names:
    print("\nAnalyzing dataset token lengths...")
    length_stats = analyze_example_lengths(ds["train"])
    print("\nDataset length statistics:")
    print(f"Input lengths - Max: {length_stats['max_input_length']}, Min: {length_stats['min_input_length']}, Avg: {length_stats['avg_input_length']:.1f}")
    print(f"Label lengths - Max: {length_stats['max_label_length']}, Min: {length_stats['min_label_length']}, Avg: {length_stats['avg_label_length']:.1f}")
    
    # Calculate total sequence length needed
    # Base sequence: max_input + max_label + special_tokens + k_tokens
    k_tokens = args.k_tokens  # This should match the k_tokens parameter in KStepRolloutTrainer
    special_tokens = 4  # bos, begin_reasoning, end_reasoning, eos
    total_sequence_length = length_stats['max_input_length'] + length_stats['max_label_length'] + special_tokens + k_tokens
    
    print(f"\nTotal sequence length needed: {total_sequence_length}")
else:
    print("\nUsing cached formatted dataset - skipping token length analysis")
    # For cached data, we need to determine sequence length from the actual data
    sample_length = len(ds["train"][0]["input_ids"])
    total_sequence_length = sample_length
    print(f"Sequence length from cached data: {total_sequence_length}")

# Add special tokens for our task
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
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
    "pad_token": SPECIAL_TOKENS["pad_token"],
    "additional_special_tokens": [
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
model.config.pad_token_id = tokenizer.pad_token_id  # ensure model knows the new pad token

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
    
    # Construct the full sequence with special tokens (no global padding)
    full_sequence = (
        [bos_id] +                # <bos>
        question_tokens +         # question
        [begin_reasoning_id] +    # <begin_reasoning>
        [tokenizer.pad_token_id] * reasoning_space +  # Space for reasoning tokens
        [end_reasoning_id] +      # <end_reasoning>
        answer_tokens +           # answer
        [eos_id]                  # <eos>
    )

    # Create attention mask (1 for all real tokens, 0 for padding)
    attention_mask = [0 if t == tokenizer.pad_token_id else 1 for t in full_sequence]
    
    return {
        "input_ids": full_sequence,
        "attention_mask": attention_mask
    }

# Format dataset for training (only if not loaded from cache)
cached_dataset_loaded = args.use_cached_data and 'input_ids' in ds['train'].column_names if ds is not None else False

if not cached_dataset_loaded:
    print("Formatting dataset...")
    ds = ds.map(format_example, remove_columns=["input", "label"])
    
    # Save formatted dataset if requested
    if args.save_formatted_data:
        save_formatted_dataset(ds, args.data_file, args.k_tokens)

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
    save_steps=500,
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
    callbacks=[
        ExactMatchCallback(test_ds, tokenizer, k_tokens=args.k_tokens, eval_steps=100)
    ],
    k_tokens=args.k_tokens,  # Number of tokens to generate for reasoning
    tokenizer=tokenizer,  # Pass tokenizer explicitly
    rl_objective=args.train_mode,   
    lambda_pg=0.1               
)
print("Trainer created")  # Debug print

# Start training
trainer.train()

# Save the final model
trainer.save_model(os.path.join(args.output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))