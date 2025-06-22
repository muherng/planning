#!/usr/bin/env python3
"""
diag_denoise_gpt.py
~~~~~~~~~~~~~~~~~~~
Fine-tune a decoder-only transformer so that, given

    <question> <NOISE …> <diag_end> …

where the NOISE spans exactly **one CKY diagonal**, the model restores the
original cell tokens in *parallel*.

Dataset JSON lines must have the keys
    question   : str        (space-separated terminals)
    reasoning  : str        (cell tokens with <diag_end> delimiters)
    answer     : str        (not used here, for future answer-stage)
    n_cells    : int        (# DP cells, delimiter tokens NOT counted)

Example:
{
  "question" : "begin stmt stmt end",
  "reasoning": "[0]_1:BEGIN … <diag_end> …",
  "n_cells"  : 8
}
"""

import json, random, math, argparse, pathlib
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
    Trainer,
    TrainingArguments,
)

# ──────────────────────────────────────────────────────────────────────
# 1  Special tokens
# ──────────────────────────────────────────────────────────────────────
NOISE_TOKEN   = "<NOISE>"
DIAG_END_TOK  = "<diag_end>"

# ──────────────────────────────────────────────────────────────────────
# 2  Utility: slices from delimiter
# ──────────────────────────────────────────────────────────────────────
def diag_slices_from_delim(trace_ids: List[int], delim_id: int) -> List[Tuple[int,int]]:
    """Return [(start,end), …] (end exclusive) for each diagonal."""
    slices, start = [], 0
    for i, tid in enumerate(trace_ids):
        if tid == delim_id:
            slices.append((start, i))
            start = i + 1                  # skip the delimiter
    slices.append((start, len(trace_ids)))
    return slices


# ──────────────────────────────────────────────────────────────────────
# 3  Custom GPT with per-sample diagonal mask
# ──────────────────────────────────────────────────────────────────────
class GPTDiag(GPTNeoForCausalLM):
    def forward(self, input_ids: torch.Tensor, diag_slice: torch.LongTensor | None = None, **kw):
        """
        diag_slice : (B,2) tensor with [d0,d1] (absolute indices *in the batch row*)
        """
        if diag_slice is not None:
            # Build a **4-D** additive attention mask expected by GPT-Neo:
            #   (batch, 1, tgt_len, src_len)
            #   0.0  → attend;   -inf → block

            B, L = input_ids.shape

            # ── base causal mask: lower-triangular (allow past tokens) ──
            causal_bool = torch.tril(torch.ones(L, L, dtype=torch.bool, device=input_ids.device))

            # prepare per-batch boolean mask we will tweak
            mask_bool = causal_bool.unsqueeze(0).expand(B, L, L).clone()

            # ── unblock the chosen CKY diagonal slice for each sample ──
            #    (make it fully bidirectional within the slice)
            for b in range(B):
                d0, d1 = diag_slice[b].tolist()
                mask_bool[b, d0:d1, d0:d1] = True

            # convert to additive fp32 mask
            attn_mask = torch.where(
                mask_bool,
                torch.zeros_like(mask_bool, dtype=torch.float32),
                torch.full_like(mask_bool, float("-inf"), dtype=torch.float32),
            )

            # GPT-Neo expects (B,1,L,L)
            kw["attention_mask"] = attn_mask.unsqueeze(1)
        return super().forward(input_ids=input_ids, **kw)


# ──────────────────────────────────────────────────────────────────────
# 4  Dataset wrapper
# ──────────────────────────────────────────────────────────────────────
class TraceDataset(Dataset):
    """On-the-fly diagonal corruption."""

    def __init__(self, hf_ds, tok):
        self.ds   = hf_ds
        self.tok  = tok
        self.noise_id = tok.convert_tokens_to_ids(NOISE_TOKEN)
        self.delim_id = tok.convert_tokens_to_ids(DIAG_END_TOK)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex      = self.ds[idx]
        q_ids   = self.tok(ex["question"], add_special_tokens=False).input_ids
        trace   = self.tok(ex["reasoning"], add_special_tokens=False).input_ids

        slices  = diag_slices_from_delim(trace, self.delim_id)
        d0, d1  = random.choice(slices)                 # choose a diagonal

        # build input & labels
        inp   = q_ids + trace
        labs  = [-100] * len(inp)
        q_len = len(q_ids)

        for i in range(d0, d1):
            # skip delimiter tokens (never inside a diagonal slice)
            labs[q_len + i] = trace[i]
            inp [q_len + i] = self.noise_id

        return {
            "input_ids" : torch.tensor(inp, dtype=torch.long),
            "labels"    : torch.tensor(labs, dtype=torch.long),
            "diag_slice": torch.tensor((q_len + d0, q_len + d1), dtype=torch.long)
        }

# ──────────────────────────────────────────────────────────────────────
# 5  Collator
# ──────────────────────────────────────────────────────────────────────
def collate(batch, pad_id: int):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B      = len(batch)

    inp  = torch.full((B, maxlen), pad_id, dtype=torch.long)
    lab  = torch.full((B, maxlen), -100, dtype=torch.long)
    slc  = torch.zeros((B, 2),     dtype=torch.long)

    for b, ex in enumerate(batch):
        L = len(ex["input_ids"])
        inp[b, :L] = ex["input_ids"]

        # Some buggy upstream datapoints in older JSONL files may miss the
        # pre-computed "labels" tensor.  Re-derive it on the fly so training
        # can proceed rather than crashing.
        if "labels" in ex:
            lab[b, :L] = ex["labels"]
        else:
            # everything is ignore index except the diagonal slice
            lab[b, :] = -100
            d0, d1 = ex["diag_slice"]
            lab[b, d0:d1] = inp[b, d0:d1]

        slc[b] = ex["diag_slice"]

    return {"input_ids": inp, "labels": lab, "diag_slice": slc}


# ──────────────────────────────────────────────────────────────────────
# 6  Training
# ──────────────────────────────────────────────────────────────────────
def train_cli():
    ap = argparse.ArgumentParser("Diagonal denoising GPT – train")
    ap.add_argument("--model", required=True, help="e.g. EleutherAI/gpt-neo-125M")
    ap.add_argument("--data",  required=True, help="train.jsonl path")
    ap.add_argument("--out",   required=True, help="output dir")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs",type=int, default=3)
    ap.add_argument("--lr",    type=float, default=2e-5)
    ap.add_argument("--fp16",  action="store_true")
    ap.add_argument("--grad_accum", type=int, default=4)
    return ap.parse_args()


def main():
    args = train_cli()

    # ── tokenizer & special tokens ───────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_tokens([NOISE_TOKEN, DIAG_END_TOK])
    pad_id  = tok.pad_token_id or tok.eos_token_id
    noise_id= tok.convert_tokens_to_ids(NOISE_TOKEN)

    # ── dataset ──────────────────────────────────────────────────────
    raw = load_dataset("json", data_files=args.data)["train"]
    dset = TraceDataset(raw, tok)

    # ── model ────────────────────────────────────────────────────────
    model = GPTDiag.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))

    # ── Trainer ──────────────────────────────────────────────────────
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        gradient_accumulation_steps=args.grad_accum,
        save_strategy="steps",
        save_steps=5000,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dset,
        data_collator=lambda b: collate(b, pad_id=pad_id),
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
