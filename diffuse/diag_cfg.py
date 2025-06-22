#!/usr/bin/env python3
"""
generate_cfg_dataset.py  –  v2  (adds <diag_end> delimiters)

Produces JSON-L lines of the form
    {
      "question" : "tok tok …",
      "reasoning": "cell₀ … cell_k <diag_end> cell_{k+1} … <diag_end> …",
      "answer"   : "rid₀ rid₁ …",
      "n_cells"  : 21          # number of *cells*   (does NOT count delimiters)
    }

The <diag_end> sentinel appears **after every diagonal** of the flattened
DP table except the last one.  Your training script can therefore treat all
tokens between successive <diag_end>s as the “parallel chunk” to predict in
one forward pass.
"""

import argparse, json, random
from pathlib import Path
from collections import defaultdict, namedtuple, deque

# ──────────────────────────────────────────────────────────────────────
# 1.  Constants
# ──────────────────────────────────────────────────────────────────────
DIAG_END = "<diag_end>"

Rule = namedtuple("Rule", ["rid", "lhs", "rhs"])

# ──────────────────────────────────────────────────────────────────────
# 2.  Grammar helper  (unchanged except minor refactor)
# ──────────────────────────────────────────────────────────────────────
class Grammar:
    def __init__(self, rules):
        self.rules = rules
        self.lhs_to_rules = defaultdict(list)
        self.terminals, self.nonterminals = set(), set()
        for r in rules:
            self.lhs_to_rules[r.lhs].append(r)
            if len(r.rhs) == 1:
                self.terminals.add(r.rhs[0])
            self.nonterminals.add(r.lhs)
        # indices
        self.binary_index = defaultdict(list)
        self.unary_index  = defaultdict(list)
        for r in rules:
            if len(r.rhs) == 2:
                self.binary_index[tuple(r.rhs)].append(r)
            elif len(r.rhs) == 1:
                self.unary_index[r.rhs[0]].append(r)
            else:
                raise ValueError("Grammar must be CNF")

        self._min_term_cache: dict[str, tuple[list[str], list[int]]] = {}

    @classmethod
    def from_file(cls, path: Path):
        rules = []
        with path.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                rid, lhs = int(parts[0]), parts[1]
                arrow = parts.index("->")
                rhs = tuple(parts[arrow + 1 :])
                rules.append(Rule(rid, lhs, rhs))
        return cls(rules)

    # ─── sampling helpers ────────────────────────────────────────────
    def sample_rule(self, lhs, rec_prob):
        opts = self.lhs_to_rules[lhs]
        if lhs != "S" or rec_prob is None:
            return random.choice(opts)
        rec = next((r for r in opts if r.rhs == ("S", "S")), None)
        others = [r for r in opts if r is not rec]
        if rec and random.random() < rec_prob:
            return rec
        return random.choice(others or [rec])

    # ─── shortest terminating expansion  (BFS) ───────────────────────
    def get_shortest_terminating(self, nt: str):
        if nt in self._min_term_cache:
            return self._min_term_cache[nt]
        Q = deque()
        Q.append(([nt], []))
        seen = set()
        while Q:
            sent, seq = Q.popleft()
            sig = tuple(sent)
            if sig in seen:
                continue
            seen.add(sig)
            if all(t not in self.nonterminals for t in sent):
                self._min_term_cache[nt] = (sent, seq)
                return sent, seq
            idx = next(i for i, t in enumerate(sent) if t in self.nonterminals)
            cur = sent[idx]
            for r in self.lhs_to_rules[cur]:
                Q.append((sent[:idx] + list(r.rhs) + sent[idx+1:], seq + [r.rid]))
        raise RuntimeError(f"{nt} never terminates")

# ──────────────────────────────────────────────────────────────────────
# 3.  Random derivation  (unchanged)
# ──────────────────────────────────────────────────────────────────────
def sample_derivation(grammar, root, *, max_depth, max_tokens, rec_prob):
    sent, rule_ids, depth = [root], [], 0
    while any(t in grammar.nonterminals for t in sent):
        if depth >= max_depth or len(sent) >= max_tokens:
            # force-terminate with bfs-minimal expansions
            while any(t in grammar.nonterminals for t in sent):
                idx = next(i for i,t in enumerate(sent) if t in grammar.nonterminals)
                term, ids = grammar.get_shortest_terminating(sent[idx])
                sent = sent[:idx] + term + sent[idx+1:]
                rule_ids.extend(ids)
            break
        idx = next(i for i,t in enumerate(sent) if t in grammar.nonterminals)
        nt  = sent[idx]
        r   = grammar.sample_rule(nt, rec_prob)
        sent = sent[:idx] + list(r.rhs) + sent[idx+1:]
        rule_ids.append(r.rid)
        depth += 1
    return sent, rule_ids

# ──────────────────────────────────────────────────────────────────────
# 4.  CKY (NT sets per cell)  +  minimal derivation
# ──────────────────────────────────────────────────────────────────────
def cky_all_sets(tokens, grammar):
    n = len(tokens)
    cell_sets = defaultdict(set)          # (i,j) → set{NT}
    # lexical
    for i,tok in enumerate(tokens):
        for r in grammar.unary_index.get(tok, ()):
            cell_sets[(i,i+1)].add(r.lhs)
    # spans
    for span in range(2, n+1):
        for i in range(0, n-span+1):
            j = i+span
            for k in range(i+1,j):
                for B in cell_sets.get((i,k), ()):
                    for C in cell_sets.get((k,j), ()):
                        for r in grammar.binary_index.get((B,C), ()):
                            cell_sets[(i,j)].add(r.lhs)
    best_seq = cky_minimal(tokens, grammar)
    return cell_sets, best_seq

def cky_minimal(tokens, grammar, root="S"):
    n = len(tokens)
    best = defaultdict(dict)              # (i,j)[A] = tuple[ids]
    for i,tok in enumerate(tokens):
        for r in grammar.unary_index.get(tok, ()):
            best[(i,i+1)][r.lhs]=(r.rid,)
    for span in range(2,n+1):
        for i in range(0,n-span+1):
            j=i+span
            for k in range(i+1,j):
                for B,seqL in best.get((i,k), {}).items():
                    for C,seqR in best.get((k,j), {}).items():
                        for r in grammar.binary_index.get((B,C), ()):
                            cand=(r.rid,)+seqL+seqR
                            cur = best[(i,j)].get(r.lhs)
                            if cur is None or cand<cur:
                                best[(i,j)][r.lhs]=cand
    return list(best.get((0,n), {}).get(root, ()))

# ──────────────────────────────────────────────────────────────────────
# 5.  DP-table flattening WITH delimiter
# ──────────────────────────────────────────────────────────────────────
def flatten_dp(cell_sets, n_tok):
    pieces=[]
    for span in range(1, n_tok+1):
        diag=[]
        for i in range(0, n_tok-span+1):
            j=i+span
            nts=sorted(cell_sets.get((i,j), set()))
            diag.append(f"[{i}]_{span}:{'|'.join(nts) if nts else '_'}")
        pieces.append(" ".join(diag))
    return f" {DIAG_END} ".join(pieces)   # insert delimiter between diagonals

# ──────────────────────────────────────────────────────────────────────
# 6.  Single example
# ──────────────────────────────────────────────────────────────────────
def make_example(grammar, *, max_depth, max_tokens, rec_prob):
    while True:
        try:
            tokens,_=sample_derivation(grammar,"S",
                                       max_depth=max_depth,
                                       max_tokens=max_tokens,
                                       rec_prob=rec_prob)
        except RuntimeError:
            continue
        cell_sets, best=cky_all_sets(tokens, grammar)
        if best:
            break
    reasoning=flatten_dp(cell_sets,len(tokens))
    return {
        "question" : " ".join(tokens),
        "reasoning": reasoning,
        "answer"   : " ".join(map(str,best)),
        "n_cells"  : len(cell_sets)       # diagonals not counted
    }

# ──────────────────────────────────────────────────────────────────────
# 7.  CLI  +  write splits
# ──────────────────────────────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--grammar",required=True,type=Path)
    p.add_argument("--out",default="data",type=Path)
    p.add_argument("--train",type=int,default=100_000)
    p.add_argument("--test", type=int,default=2_000)
    p.add_argument("--max_tokens",type=int,default=30)
    p.add_argument("--max_depth", type=int,default=40)
    p.add_argument("--recursion_prob",type=float,default=None)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

def write_split(n, path, grammar, args):
    with path.open("w") as fh:
        for _ in range(n):
            ex = make_example(grammar,
                              max_depth=args.max_depth,
                              max_tokens=args.max_tokens,
                              rec_prob=args.recursion_prob)
            json.dump(ex, fh); fh.write("\n")

def main():
    args=cli(); random.seed(args.seed)
    grammar=Grammar.from_file(args.grammar)
    args.out.mkdir(parents=True, exist_ok=True)
    write_split(args.train, args.out/f"train_{args.train}_{args.max_tokens}_{args.max_depth}_{args.recursion_prob}.jsonl", grammar, args)
    write_split(args.test,  args.out/f"test_{args.test}_{args.max_tokens}_{args.max_depth}_{args.recursion_prob}.jsonl",  grammar, args)
    print("Done.  Files at", args.out, f"(delimiter token: {DIAG_END})")

if __name__=="__main__":
    main()
