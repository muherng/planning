import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import json

# ───────────────────────────────────────────────────────────────────────────────
# Helper: Exact-match evaluation (adapted from train.py)
# ───────────────────────────────────────────────────────────────────────────────
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "begin_reasoning_token": "<begin_reasoning>",
    "end_reasoning_token": "<end_reasoning>",
    "latent_token": "<latent>",  # NEW – used by latent-reuse baseline
}
SYSTEM = "You are a graph-reasoning assistant."
PROMPT = "{input}\n\nLet's think step by step:\n"
TARGET = "{label}"


def build_prompt_ids(example, tok):
    """<bos> question <begin_reasoning>"""
    text = f"{SYSTEM}\n\n{PROMPT.format(**example)}"
    ids = tok.encode(text, add_special_tokens=False)
    bos = tok.bos_token_id
    begin = tok.encode(SPECIAL_TOKENS["begin_reasoning_token"], add_special_tokens=False)[0]
    return torch.tensor([bos] + ids + [begin], dtype=torch.long)


def calc_answer_ce(model, tok, reasoning_tokens, example):
    """Return CE loss on answer span only (same masking logic as training)."""
    BOS = tok.bos_token_id
    PAD = tok.pad_token_id
    EOS = tok.eos_token_id
    BEGIN = tok.encode(SPECIAL_TOKENS["begin_reasoning_token"], add_special_tokens=False)[0]
    END = tok.encode(SPECIAL_TOKENS["end_reasoning_token"], add_special_tokens=False)[0]

    question = f"{SYSTEM}\n\n{PROMPT.format(**example)}"
    q_ids = tok.encode(question, add_special_tokens=False)
    ans_ids = tok.encode(TARGET.format(**example), add_special_tokens=False)

    seq = ([BOS] + q_ids + [BEGIN] + reasoning_tokens + [END] + ans_ids + [EOS])
    input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(model.device)
    attn = (input_ids != PAD).long()

    labels = torch.full_like(input_ids, -100)
    # supervise answer span + EOS only
    end_pos = 1 + len(q_ids) + 1 + len(reasoning_tokens)
    ans_start = end_pos + 1
    eos_pos = len(seq) - 1
    labels[0, ans_start:eos_pos] = torch.tensor(ans_ids, dtype=torch.long)
    labels[0, eos_pos] = EOS

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    return out.loss.item()


# ───────────────────────────────────────────────────────────────────────────────
# Main evaluation script
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model with exact-match and answer CE")
    p.add_argument("--checkpoint", type=str, required=True, help="Path or HF ID of the model checkpoint")
    p.add_argument("--data_file", type=str, required=True, help="JSONL dataset file (same format as train.py)")
    p.add_argument("--k_tokens", type=int, default=0, help="Number of latent reasoning tokens (K)")
    p.add_argument("--train_mode", type=str, default="wake_sleep",
                   choices=["wake_sleep", "reinforce", "latentreuse"],
                   help="Training mode – needed to decide evaluation logic")
    p.add_argument("--num_examples", type=int, default=None, help="Optional cap on number of examples to evaluate")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Load tokenizer & model ---------------------------------------------------
    print(f"Loading model from: {args.checkpoint}")
    tok = AutoTokenizer.from_pretrained(args.checkpoint)

    # add special tokens if missing
    tok.add_special_tokens(
        {
            "pad_token": SPECIAL_TOKENS["pad_token"],
            "additional_special_tokens": [
                SPECIAL_TOKENS["begin_reasoning_token"],
                SPECIAL_TOKENS["end_reasoning_token"],
                SPECIAL_TOKENS["latent_token"],  # ensure <latent> present
            ],
        }
    )
    if tok.bos_token_id is None or tok.bos_token_id == tok.eos_token_id:
        tok.add_special_tokens({"bos_token": SPECIAL_TOKENS["bos_token"]})

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, device_map="auto")
    model.resize_token_embeddings(len(tok))
    model.eval()

    # Load dataset -------------------------------------------------------------
    print(f"Loading dataset: {args.data_file}")
    ds = load_dataset("json", data_files=args.data_file)["train"]
    if args.num_examples:
        ds = ds.select(range(args.num_examples))

    device = torch.device(args.device)

    # IDs of special tokens (cached)
    EOS_ID = tok.eos_token_id
    PAD_ID = tok.pad_token_id
    LATENT_ID = tok.encode(SPECIAL_TOKENS["latent_token"], add_special_tokens=False)[0]

    exact, total = 0, 0
    ce_sum = 0.0

    for ex in tqdm(ds, desc="Evaluating"):
        # -------- 1) Build prompt & generate K reasoning tokens --------------
        prompt_ids = build_prompt_ids(ex, tok).to(device).unsqueeze(0)

        # ------------------------------------------------------------
        # Obtain reasoning tokens depending on training mode
        # ------------------------------------------------------------
        if args.train_mode == "latentreuse":
            # Insert fixed <latent> placeholders (no generation step)
            reasoning = [LATENT_ID] * args.k_tokens if args.k_tokens > 0 else []
        else:
            # wake_sleep / reinforce: generate reasoning tokens if K>0
            reasoning = []
            if args.k_tokens > 0:
                with torch.no_grad():
                    gen = model.generate(
                        prompt_ids,
                        max_new_tokens=args.k_tokens,
                        pad_token_id=PAD_ID,
                        eos_token_id=EOS_ID,
                        use_cache=True,
                    )
                full = gen[0]
                reasoning_tokens = full[prompt_ids.shape[1]:]
                # strip EOS and pad to k
                eos_positions = (reasoning_tokens == EOS_ID).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    reasoning_tokens = reasoning_tokens[: eos_positions[-1]]
                if reasoning_tokens.numel() < args.k_tokens:
                    pad_len = args.k_tokens - reasoning_tokens.numel()
                    reasoning_tokens = torch.cat([
                        reasoning_tokens,
                        torch.full((pad_len,), PAD_ID, dtype=torch.long, device=device),
                    ])
                reasoning = reasoning_tokens.tolist()

        # -------- 2) Build second input (prompt + reasoning + <end>) ----------
        END_ID = tok.encode(SPECIAL_TOKENS["end_reasoning_token"], add_special_tokens=False)[0]
        second_input = torch.tensor(prompt_ids[0].tolist() + reasoning + [END_ID], device=device).unsqueeze(0)

        gold_answer_ids = tok.encode(ex["label"], add_special_tokens=False)
        gold_answer_len = len(gold_answer_ids)

        # -------- 3) Generate answer -----------------------------------------
        with torch.no_grad():
            ans_gen = model.generate(
                second_input,
                max_new_tokens=gold_answer_len,
                pad_token_id=PAD_ID,
                eos_token_id=EOS_ID,
                use_cache=True,
            )
        answer_tokens = ans_gen[0][second_input.shape[1]:]
        pred_answer = tok.decode(answer_tokens, skip_special_tokens=True).strip()
        gold_answer = ex["label"].strip()

        exact += int(pred_answer == gold_answer)
        total += 1

        # -------- 4) Cross-entropy on answer span ----------------------------
        ce = calc_answer_ce(model, tok, reasoning, ex)
        ce_sum += ce

    acc = exact / total if total else 0.0
    avg_ce = ce_sum / total if total else float("nan")

    print("\n========== Evaluation Results ==========")
    print(f"Exact-match accuracy : {acc:.4%} ({exact}/{total})")
    print(f"Answer cross-entropy : {avg_ce:.4f}")


if __name__ == "__main__":
    main() 