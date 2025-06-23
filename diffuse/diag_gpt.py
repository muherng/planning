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

import json, random, math, argparse, pathlib, datetime
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

# ────────────────────────────────────────────────────────────────────
# 7  Evaluation callback: slice-by-slice decoding metrics
# ────────────────────────────────────────────────────────────────────
import torch.nn.functional as F
from transformers import TrainerCallback
import numpy as np
import textwrap, random

class SliceEval(TrainerCallback):
    """
    • Runs every `eval_steps`.
    • For each of N examples:
        – iteratively decodes all diagonals starting from <NOISE>
        – accumulates cross-entropy on each decoded slice
        – counts 0-1 accuracy on the reasoning tokens
    • Prints aggregated CE/token, exact-match %, and 3 random decodes.
    """
    def __init__(self,
                 eval_hfds, tok,
                 noise_id, delim_id,
                 slice_index: int = -1,
                 max_examples=100,
                 eval_steps=100,
                 print_examples=3):
        """If ``slice_index`` >= 0, only that diagonal is evaluated and printed.
        Otherwise the original iterative evaluation is used."""
        self.eval_set = eval_hfds
        self.tok      = tok
        self.noise_id = noise_id
        self.delim_id = delim_id
        self.slice_index = slice_index
        self.max_examples  = max_examples
        self.eval_steps    = eval_steps
        self.print_examples= print_examples
        # accumulate per-token accuracy (fixed-slice mode)
        self._tot_correct = 0
        self._tot_tokens  = 0

        # how often to show running train CE/token from Trainer logs
        self.loss_every = 10

    # --------------------------------------------------------------
    def on_step_end(self, args, state, control, **kw):
        #print(f"on_step_end: {state.global_step}")
        if state.global_step % self.eval_steps == 0:         # only every k steps
            model  = kw["model"].eval()
            device = next(model.parameters()).device

            tot_loss, tot_tok, tot_exact = 0.0, 0, 0
            tot_tok_correct = 0  # token-level accuracy numerator
            samples_to_print = []

            for ex_id in range(min(self.max_examples, len(self.eval_set))):
                ex = self.eval_set[ex_id]                  # raw HF row

                # ---------- tokenise once ----------
                q_ids  = self.tok(ex["question"], add_special_tokens=False).input_ids
                tr_ids = self.tok(ex["reasoning"], add_special_tokens=False).input_ids
                slices = diag_slices_from_delim(tr_ids, self.delim_id)

                if self.slice_index is None or self.slice_index < 0:
                    # ------- original iterative evaluation over all diagonals -------
                    seq = q_ids + [self.noise_id if tid != self.delim_id else tid for tid in tr_ids]
                    example_loss, example_tok = 0.0, 0
                    for d0, d1 in slices:
                        off0, off1 = len(q_ids) + d0, len(q_ids) + d1
                        # ── align with training loss: logits at position t predict token t+1 ──
                        shift0, shift1 = off0 - 1, off1 - 1
                        with torch.no_grad():
                            full_logits = model(
                                input_ids=torch.tensor([seq], device=device),
                                diag_slice=torch.tensor([[off0, off1]], device=device),
                            ).logits[0]

                        logits = full_logits[shift0:shift1]

                        gold = torch.tensor(tr_ids[d0:d1], device=device)
                        loss = F.cross_entropy(logits, gold, reduction="sum")
                        example_loss += loss.item()
                        example_tok += off1 - off0

                        seq[off0:off1] = logits.argmax(-1).tolist()

                    pred_reasoning = self.tok.decode(
                        seq[len(q_ids):], skip_special_tokens=False
                    ).strip()
                    tot_exact += int(pred_reasoning == ex["reasoning"])

                    if len(samples_to_print) < self.print_examples:
                        samples_to_print.append((ex["question"], ex["reasoning"], pred_reasoning))

                else:
                    # ------- fixed slice evaluation (debugging) -------
                    idx = min(self.slice_index, len(slices) - 1)
                    d0, d1 = slices[idx]
                    off0, off1 = len(q_ids) + d0, len(q_ids) + d1
                    # ── align with training loss: logits at position t predict token t+1 ──
                    shift0, shift1 = off0 - 1, off1 - 1

                    # Build sequence with only this slice noised out
                    seq = q_ids + tr_ids
                    for i in range(d0, d1):
                        seq[len(q_ids) + i] = self.noise_id

                    with torch.no_grad():
                        full_logits = model(
                            input_ids=torch.tensor([seq], device=device),
                            diag_slice=torch.tensor([[off0, off1]], device=device),
                        ).logits[0]

                    logits = full_logits[shift0:shift1]

                    gold = torch.tensor(tr_ids[d0:d1], device=device)
                    loss = F.cross_entropy(logits, gold, reduction="sum")
                    example_loss = loss.item()
                    example_tok = off1 - off0

                    seq[off0:off1] = logits.argmax(-1).tolist()

                    # token-level accuracy for this slice
                    pred_ids = logits.argmax(-1)
                    tok_correct = (pred_ids == gold).sum().item()
                    tot_tok_correct += tok_correct

                    # keep raw tokens for printing
                    pred_slice_ids = pred_ids.tolist()
                    gold_slice_ids = gold.tolist()
                    pred_slice_tokens = self.tok.convert_ids_to_tokens(pred_slice_ids)
                    gold_slice_tokens = self.tok.convert_ids_to_tokens(gold_slice_ids)

                    matches = [p == g for p, g in zip(pred_slice_ids, gold_slice_ids)]

                    pred_slice = " ".join(pred_slice_tokens)
                    gold_slice = " ".join(gold_slice_tokens)

                    token_losses_tensor = F.cross_entropy(logits, gold, reduction='none')
                    token_losses_list = [f"{v:.3f}" for v in token_losses_tensor.tolist()]

                    if len(samples_to_print) < self.print_examples:
                        samples_to_print.append((ex["question"], gold_slice_tokens, pred_slice_tokens, matches, token_losses_list))

                    # For exact-match under fixed-slice mode, compare slice only
                    tot_exact += int(pred_slice == gold_slice)

                    pred_reasoning = None  # not used under fixed-slice mode

                    tot_loss += example_loss
                    tot_tok += example_tok

            # ---------- print metrics ----------
            ce_per_tok = tot_loss / tot_tok
            exact_pct = (
                tot_exact / min(self.max_examples, len(self.eval_set)) * 100
            )
            if self.slice_index is None or self.slice_index < 0:
                print(
                    f"\n[EVAL step {state.global_step}] "
                    f"cross-entropy/token={ce_per_tok:.4f} | "
                    f"trace exact-match={exact_pct:.1f}%"
                )
                for q, gold, pred in samples_to_print:
                    print("\nQ:", q)
                    print("Gold reasoning:\n", textwrap.fill(gold, 120))
                    print("Pred reasoning:\n", textwrap.fill(pred, 120))
                    print("-" * 80)
            else:
                print(
                    f"\n[EVAL step {state.global_step}] "
                    f"cross-entropy/token={ce_per_tok:.4f} | "
                    f"slice-{self.slice_index} exact-match={exact_pct:.1f}% | "
                    f"token-acc={(tot_tok_correct / tot_tok * 100):.1f}%"
                )
                for q, gold_toks, pred_toks, matches, token_losses in samples_to_print:
                    gold_line = " ".join(gold_toks)
                    pred_line = " ".join(pred_toks)
                    match_line = " ".join(["✓" if m else "✗" for m in matches])
                    loss_line  = " ".join(token_losses)
                    print("\nQ:", q)
                    print("Gold slice tokens:", gold_line)
                    print("Pred slice tokens:", pred_line)
                    print("Match per token    :", match_line)
                    print("CE per token       :", loss_line)
                    print("-" * 80)

    # --------------------------------------------------------------
    # Short, frequent loss print from Trainer's logging events
    # --------------------------------------------------------------
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Print average cross-entropy per token every ``self.loss_every`` steps."""
        if logs is None or "loss" not in logs:
            return
        if state.global_step % self.loss_every == 0:
            ce = logs["loss"]  # already average per token because labels use -100 mask
            print(f"[step {state.global_step}] train CE/token = {ce:.4f}")

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

    def __init__(self, hf_ds, tok, slice_index: int = -1):
        """If ``slice_index`` >= 0, the *same* diagonal (0-based) will be masked
        on every sample.  Useful for debugging or curriculum learning.  A value
        of -1 keeps the original random-slice behaviour."""

        self.ds   = hf_ds
        self.tok  = tok
        self.noise_id   = tok.convert_tokens_to_ids(NOISE_TOKEN)
        self.delim_id   = tok.convert_tokens_to_ids(DIAG_END_TOK)
        self.slice_index= slice_index

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex      = self.ds[idx]
        q_ids   = self.tok(ex["question"], add_special_tokens=False).input_ids
        trace   = self.tok(ex["reasoning"], add_special_tokens=False).input_ids

        slices = diag_slices_from_delim(trace, self.delim_id)
        if self.slice_index is None or self.slice_index < 0:
            d0, d1 = random.choice(slices)
        else:
            idx = min(self.slice_index, len(slices) - 1)
            d0, d1 = slices[idx]

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
            "diag_slice": torch.tensor((q_len + d0, q_len + d1), dtype=torch.long),
            "question"  : ex["question"],
            "reasoning" : ex["reasoning"],
        }

# ──────────────────────────────────────────────────────────────────────
# 5  Collator
# ──────────────────────────────────────────────────────────────────────
def collate(batch, pad_id: int, tok, slice_index: int = -1):
    # -----------------------------------------------------------------
    #print("\n=== DEBUG: incoming batch snapshot ===")
    #for i, item in enumerate(batch):
    #    print(f"[{i}] type={type(item)} keys={list(item.keys()) if isinstance(item, dict) else 'N/A'}")
    #print("=== END DEBUG ===\n")
    # -----------------------------------------------------------------

    maxlen = 0
    processed = []

    delim_id = tok.convert_tokens_to_ids(DIAG_END_TOK)
    noise_id = tok.convert_tokens_to_ids(NOISE_TOKEN)

    for ex in batch:
        if "labels" in ex and "diag_slice" in ex:
            processed.append(ex)
            maxlen = max(maxlen, len(ex["input_ids"]))
        else:
            # Fallback: raw JSON example – build corruption here
            q_ids = tok(ex["question"], add_special_tokens=False).input_ids
            trace = tok(ex["reasoning"], add_special_tokens=False).input_ids
            slices = diag_slices_from_delim(trace, delim_id)
            if slice_index is None or slice_index < 0:
                d0, d1 = random.choice(slices)
            else:
                idx = min(slice_index, len(slices) - 1)
                d0, d1 = slices[idx]
            inp   = q_ids + trace
            labs  = [-100] * len(inp)
            q_len = len(q_ids)
            for i in range(d0, d1):
                labs[q_len + i] = trace[i]
                inp [q_len + i] = noise_id
            processed.append({
                "input_ids": torch.tensor(inp,  dtype=torch.long),
                "labels"   : torch.tensor(labs, dtype=torch.long),
                "diag_slice": torch.tensor((q_len + d0, q_len + d1), dtype=torch.long),
                "question" : ex["question"],
                "reasoning": ex["reasoning"],
            })
            maxlen = max(maxlen, len(inp))

    B = len(processed)
    inp  = torch.full((B, maxlen), pad_id, dtype=torch.long)
    lab  = torch.full((B, maxlen), -100, dtype=torch.long)
    slc  = torch.zeros((B, 2),     dtype=torch.long)

    for b, ex in enumerate(processed):
        L = len(ex["input_ids"])
        inp[b, :L] = ex["input_ids"]
        lab[b, :L] = ex["labels"]
        slc[b]     = ex["diag_slice"]

    return {"input_ids": inp, "labels": lab, "diag_slice": slc}


# ──────────────────────────────────────────────────────────────────────
# 6  Training
# ──────────────────────────────────────────────────────────────────────
def train_cli():
    ap = argparse.ArgumentParser("Diagonal denoising GPT – train")
    ap.add_argument("--model", required=True, help="e.g. EleutherAI/gpt-neo-125M")
    ap.add_argument("--data",  required=True, help="train.jsonl path")
    ap.add_argument("--out",   required=True, help="output dir", default="diag_gpt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs",type=int, default=3)
    ap.add_argument("--lr",    type=float, default=2e-5)
    ap.add_argument("--fp16",  action="store_true")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a saved checkpoint directory to resume training from. If provided, training will continue from that checkpoint and output will be written back to the same run directory.")
    ap.add_argument("--slice_index", type=int, default=-1,
                    help="If >=0, always corrupt this diagonal slice (0=first). -1=random slice (default)")
    return ap.parse_args()


def main():
    args = train_cli()
    # Determine output directory.  If the user is resuming from a checkpoint
    # keep writing into the SAME run directory so all artefacts live together.
    # A resume path may point either to the run root or to a specific
    # "checkpoint-*" sub-folder.
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if resume_path.name.startswith("checkpoint-"):
            args.out = str(resume_path.parent)
        else:
            args.out = str(resume_path)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"{args.out}_{timestamp}"

    # ── tokenizer & special tokens ───────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_tokens([NOISE_TOKEN, DIAG_END_TOK])
    pad_id  = tok.pad_token_id or tok.eos_token_id
    noise_id= tok.convert_tokens_to_ids(NOISE_TOKEN)

    # after building `tok`, `pad_id`, `noise_id` ...
    raw_train = load_dataset("json", data_files=args.data)["train"]
    train_dset = TraceDataset(raw_train, tok, slice_index=args.slice_index)

    raw_eval  = load_dataset("json", data_files=args.data)["train"].select(range(100))
    # or load a separate dev file
    eval_cb = SliceEval(
        raw_eval,
        tok,
        noise_id,
        tok.convert_tokens_to_ids(DIAG_END_TOK),
        slice_index=args.slice_index,
        max_examples=100,
        eval_steps=20,
    )

    train_mon = TrainMonitor(train_dset, tok, print_every=10, n_examples=2)

    # ── model ────────────────────────────────────────────────────────
    model_name_or_path = args.resume if args.resume else args.model
    model = GPTDiag.from_pretrained(model_name_or_path)
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
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dset,
        data_collator=lambda b: collate(b, pad_id=pad_id, tok=tok, slice_index=args.slice_index),
        callbacks=[eval_cb, train_mon]
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()

# ────────────────────────────────────────────────────────────────────
# 7b  Training-time debug callback (token-level CE print)           
# ────────────────────────────────────────────────────────────────────


class TrainMonitor(TrainerCallback):
    """Print a few training examples with per-token CE to compare with
    evaluation behaviour.  Activated every ``print_every`` steps."""

    def __init__(self, train_dataset, tok, print_every: int = 500, n_examples: int = 2):
        self.train_ds   = train_dataset
        self.tok        = tok
        self.print_every= print_every
        self.n_examples = n_examples

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        if state.global_step == 0 or state.global_step % self.print_every != 0:
            return

        model  = kwargs["model"].eval()
        device = next(model.parameters()).device

        import random

        print("\n[TRAIN DEBUG] Token-level CE on training examples (step", state.global_step, ")")
        for _ in range(self.n_examples):
            ex  = random.choice(self.train_ds)
            inp = ex["input_ids"].unsqueeze(0).to(device)
            slc = ex["diag_slice"].unsqueeze(0).to(device)
            with torch.no_grad():
                full_logits = model(input_ids=inp, diag_slice=slc).logits[0]

            d0, d1 = ex["diag_slice"].tolist()
            gold_ids = ex["labels"][d0:d1].to(device)
            # shift for causal alignment
            shift0, shift1 = d0 - 1, d1 - 1
            pred_ids = full_logits.argmax(-1)[shift0:shift1]
            losses   = F.cross_entropy(full_logits[shift0:shift1], gold_ids, reduction="none")

            gold_toks = self.tok.convert_ids_to_tokens(gold_ids.tolist())
            pred_toks = self.tok.convert_ids_to_tokens(pred_ids.tolist())
            match     = ["✓" if g==p else "✗" for g,p in zip(gold_ids.tolist(), pred_ids.tolist())]
            loss_str  = [f"{l.item():.3f}" for l in losses]

            print("Gold :", " ".join(gold_toks))
            print("Pred :", " ".join(pred_toks))
            print("Match:", " ".join(match))
            print("CE   :", " ".join(loss_str))
            print("-"*80)

if __name__ == "__main__":
    main()