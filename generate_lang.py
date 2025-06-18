#!/usr/bin/env python3
"""
Grammar-agnostic CFG dataset generator.

* Grammar must be in CNF and stored as a plain-text file:
      <id> <lhs> -> <rhs1> <rhs2>
   or <id> <lhs> -> <terminal>
  Tokens are whitespace-separated.  Terminals are any symbols that
  never appear on the LHS of a rule.

* Produces JSONL:
      {"input": "grammar: ...; word: tok tok ...",
       "label": "derivation: id1 id2 id3 ..."}
"""

import argparse, random, json, itertools, string
from pathlib import Path
from collections import defaultdict, namedtuple

Rule = namedtuple("Rule", ["rid", "lhs", "rhs"])        # rhs is tuple[str]

# ---------------------------------------------------------------------
# Grammar utilities
# ---------------------------------------------------------------------
class Grammar:
    def __init__(self, rules):
        self.rules = rules                               # list[Rule]
        self.lhs_to_rules = defaultdict(list)
        self.terminals = set()
        for r in rules:
            self.lhs_to_rules[r.lhs].append(r)
            if len(r.rhs) == 1:
                self.terminals.add(r.rhs[0])
        self.nonterminals = set(self.lhs_to_rules)
        self._index_binary()

    @classmethod
    def from_file(cls, path: Path):
        rules = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                rid, lhs = int(parts[0]), parts[1]
                arrow_idx = parts.index("->")
                rhs = tuple(parts[arrow_idx + 1 :])
                rules.append(Rule(rid, lhs, rhs))
        return cls(rules)

    def _index_binary(self):
        self.binary_index = defaultdict(list)            # (B,C) -> list[Rule]
        self.unary_index = defaultdict(list)             # a -> list[Rule] where rhs = (a,)
        for r in self.rules:
            if len(r.rhs) == 2:
                self.binary_index[tuple(r.rhs)].append(r)
            elif len(r.rhs) == 1:
                self.unary_index[r.rhs[0]].append(r)
            else:
                raise ValueError(f"Grammar not in CNF: {r}")

    # helpers
    def sample_rule(self, lhs):
        rules_list = self.lhs_to_rules[lhs]
        # If user specified a recursion probability and we are expanding the start symbol,
        # bias the selection toward (or away from) the recursive rule S -> S S.
        rec_prob = getattr(self, "recursion_prob", None)
        if rec_prob is not None and lhs == "S":
            rec_rule = next((r for r in rules_list if r.rhs == ("S", "S")), None)
            if rec_rule is None:
                # No explicit recursive rule; fall back to uniform choice
                return random.choice(rules_list)

            # Partition other options (could be empty if grammar only has recursion)
            other_rules = [r for r in rules_list if r is not rec_rule]

            # With probability rec_prob choose the recursive rule; otherwise choose uniformly among others
            if random.random() < rec_prob:
                return rec_rule
            else:
                # if no alternative rules exist, must return the recursive rule
                return random.choice(other_rules) if other_rules else rec_rule
        # default: uniform choice among available rules
        return random.choice(rules_list)

# ---------------------------------------------------------------------
# Random derivation generator (top-down, left-most)
# ---------------------------------------------------------------------
def random_derivation(grammar: Grammar, root: str, max_depth: int, max_tokens: int):
    """
    Returns tokens, rule_seq  (both lists).
    Simple DFS expansion with depth + length limits.
    """
    tokens = [root]
    rules_out = []
    depth = 0
    while any(tok in grammar.nonterminals for tok in tokens):
        if depth >= max_depth or len(tokens) >= max_tokens:
            # force remaining non-terminals into terminals via random unary rules
            nt_idx = next(i for i, t in enumerate(tokens) if t in grammar.nonterminals)
            nt = tokens[nt_idx]
            candidates = [r for r in grammar.lhs_to_rules[nt] if len(r.rhs) == 1]
            if not candidates:
                raise RuntimeError("Cannot terminate derivation under limits")
            r = random.choice(candidates)
        else:
            nt_idx = next(i for i, t in enumerate(tokens) if t in grammar.nonterminals)
            nt = tokens[nt_idx]
            r = grammar.sample_rule(nt)

        # apply rule
        tokens = tokens[:nt_idx] + list(r.rhs) + tokens[nt_idx + 1 :]
        rules_out.append(r.rid)
        depth += 1
    return tokens, rules_out

# ---------------------------------------------------------------------
# CKY with lexicographic-minimal rule sequence
# ---------------------------------------------------------------------
def cky_minimal(tokens, grammar: Grammar, root: str):
    n = len(tokens)
    best = defaultdict(dict)                     # (i,j)[A] = seq (tuple[int])

    # lexical layer
    for i, tok in enumerate(tokens):
        for r in grammar.unary_index.get(tok, ()):
            best[(i, i + 1)][r.lhs] = (r.rid,)
    # spans
    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                left = best.get((i, k), {})
                right = best.get((k, j), {})
                for B, seqL in left.items():
                    for C, seqR in right.items():
                        for r in grammar.binary_index.get((B, C), ()):
                            cand = (r.rid,) + seqL + seqR
                            cur = best[(i, j)].get(r.lhs)
                            if cur is None or cand < cur:
                                best[(i, j)][r.lhs] = cand
    return list(best.get((0, n), {}).get(root, ()))

# ---------------------------------------------------------------------
# Example encoding
# ---------------------------------------------------------------------
def encode_example(tokens, deriv, grammar_str):
    return {
        "input": f"grammar: {grammar_str}; word: {' '.join(tokens)}",
        "label": f"derivation: {' '.join(map(str, deriv))}",
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CFG dataset generator (grammar-agnostic)")
    p.add_argument("--grammar_file", type=str, required=True,
                   help="Path to CNF grammar file")
    p.add_argument("--root", type=str, default="S")
    p.add_argument("--word_len", type=int, default=500,
                   help="Maximum token length of generated sentences")
    p.add_argument("--max_depth", type=int, default=25,
                   help="Maximum derivation depth during sampling")
    p.add_argument("--train_samples", type=int, default=100000)
    p.add_argument("--test_samples", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="data/cfg")
    p.add_argument("--recursion_prob", type=float, default=None,
                   help="Probability of choosing the recursive rule S -> S S when expanding S. If None, use uniform selection.")
    return p.parse_args()

def generate_dataset(n, grammar, root, word_len, max_depth, grammar_str):
    for i in range(n):
        while True:
            try:
                tokens, _ = random_derivation(grammar, root, max_depth, word_len)
            except RuntimeError:
                # Could not terminate within limits, retry sampling
                continue
            if len(tokens) <= word_len:
                break
        deriv = cky_minimal(tokens, grammar, root)
        if not deriv:
            raise RuntimeError("CKY failed—grammar not in CNF or bug.")
        yield encode_example(tokens, deriv, grammar_str)

def main():
    args = parse_args()
    grammar_path = Path(args.grammar_file)
    grammar = Grammar.from_file(grammar_path)
    # store recursion probability (may be None) so Grammar.sample_rule can access it
    grammar.recursion_prob = args.recursion_prob

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grammar_str = "; ".join(f"{r.rid}:{r.lhs}->{ ' '.join(r.rhs)}"
                            for r in grammar.rules)

    train_f = out_path / f"{grammar_path.stem}_train.jsonl"
    test_f  = out_path / f"{grammar_path.stem}_test.jsonl"

    print(f"Generating {args.train_samples} training examples…")
    with train_f.open("w") as f:
        for ex in generate_dataset(args.train_samples, grammar, args.root,
                                   args.word_len, args.max_depth, grammar_str):
            json.dump(ex, f); f.write("\n")

    print(f"Generating {args.test_samples} test examples…")
    with test_f.open("w") as f:
        for ex in generate_dataset(args.test_samples, grammar, args.root,
                                   args.word_len, args.max_depth, grammar_str):
            json.dump(ex, f); f.write("\n")

    print("Done. Files saved to", out_path)

if __name__ == "__main__":
    main()
