# generate_arith_cfg.py
import random, argparse, json
from pathlib import Path
from functools import lru_cache

###############################################
# Fixed ambiguous arithmetic grammar
###############################################
# Rule IDs are stable across all examples
# 1: E -> E + E
# 2: E -> E * E
# 3: E -> ( E )
# 4: E -> id
GRAMMAR_RULES_STR = "1:E->E+E, 2:E->E*E, 3:E->(E), 4:E->id"
RULE_IDS = [1, 2, 3, 4]

TERMINALS = {"+", "*", "(", ")", "id"}

###############################################
# Expression sampling
###############################################

def random_expr(max_depth=4, depth=0):
    """Recursively generate a random arithmetic expression.

    Returns a tuple (tokens, rule_seq) where tokens is a list of terminal
    strings and rule_seq is the sequence of rule IDs *in left-most order*
    used to generate the expression.
    """
    # base case: produce an id with probability depending on depth
    if depth >= max_depth or random.random() < 0.01:
        return ["id"], [4]  # rule 4

    # choose among the three structural rules
    rule = random.choice([1, 2, 3])

    if rule == 1:  # E -> E + E
        left_toks, left_rules = random_expr(max_depth, depth + 1)
        right_toks, right_rules = random_expr(max_depth, depth + 1)
        tokens = left_toks + ["+"] + right_toks
        rules = [1] + left_rules + right_rules  # left-most derivation
        return tokens, rules
    elif rule == 2:  # E -> E * E
        left_toks, left_rules = random_expr(max_depth, depth + 1)
        right_toks, right_rules = random_expr(max_depth, depth + 1)
        tokens = left_toks + ["*"] + right_toks
        rules = [2] + left_rules + right_rules
        return tokens, rules
    else:  # rule == 3, E -> ( E )
        inner_toks, inner_rules = random_expr(max_depth, depth + 1)
        tokens = ["("] + inner_toks + [")"]
        rules = [3] + inner_rules
        return tokens, rules

###############################################
# Lexicographically-minimal derivation via DP
###############################################

def lexicographic_min_derivation(tokens):
    """Return lexicographically smallest rule-ID sequence that derives tokens."""
    n = len(tokens)

    # DP table: best[(i, j)] -> minimal rule sequence deriving span i..j as E
    best = {}

    # Base cases (single token = id)
    for i, tok in enumerate(tokens):
        if tok == "id":
            best[(i, i + 1)] = [4]

    # Handle spans of increasing length
    for span_len in range(2, n + 1):
        for i in range(0, n - span_len + 1):
            j = i + span_len
            candidates = []

            # Rule 3: parenthesis
            if tokens[i] == "(" and tokens[j - 1] == ")":
                inner = best.get((i + 1, j - 1))
                if inner is not None:
                    candidates.append([3] + inner)

            # Rules 1 and 2
            for k in range(i + 1, j - 1):
                token_k = tokens[k]
                if token_k not in {"+", "*"}:
                    continue
                left = best.get((i, k))
                right = best.get((k + 1, j))
                if left is None or right is None:
                    continue
                if token_k == "+":
                    candidates.append([1] + left + right)
                else:  # '*'
                    candidates.append([2] + left + right)

            if candidates:
                best[(i, j)] = min(candidates)  # lexicographic min

    return best.get((0, n))

###############################################
# Example encoding
###############################################

def encode_example(tokens):
    """Return dict with 'input' and 'label' strings for one example."""
    word_str = " ".join(tokens)
    deriv = lexicographic_min_derivation(tokens)
    if deriv is None:
        # Should not happen if tokens generated from grammar
        raise ValueError("Failed to derive generated expression")
    deriv_str = " ".join(str(rid) for rid in deriv)
    return {
        "input": f"grammar: {GRAMMAR_RULES_STR}; word: {word_str}",
        "label": f"derivation: {deriv_str}"
    }

###############################################
# CLI & main loop
###############################################

def parse_args():
    p = argparse.ArgumentParser(description="Generate arithmetic CFG dataset")
    p.add_argument("--word_len", type=int, default=15,
                   help="Maximum length (tokens) of generated expressions")
    p.add_argument("--train_samples", type=int, default=100000,
                   help="Number of training examples to generate")
    p.add_argument("--test_samples", type=int, default=1000,
                   help="Number of test examples to generate")
    p.add_argument("--max_depth", type=int, default=6,
                   help="Maximum recursion depth when sampling expressions")
    return p.parse_args()


def sample_expression(max_len, max_depth):
    """Generate expression whose token length <= max_len."""
    for _ in range(1000):  # try up to 1000 times
        tokens, _ = random_expr(max_depth=max_depth)
        if len(tokens) <= max_len:
            return tokens
        print(f"Failed to generate expression under length constraint")
    raise RuntimeError("Failed to generate expression under length constraint")


def generate_dataset(n_samples, max_len, max_depth):
    for i in range(n_samples):
        print(f"Generating example {i+1} of {n_samples}")
        tokens = sample_expression(max_len, max_depth)
        yield encode_example(tokens)


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = Path("data") / "cfg"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / f"arith_depth{args.max_depth}_{args.train_samples}_train.jsonl"
    test_file = out_dir / f"arith_depth{args.max_depth}_{args.test_samples}_test.jsonl"

    print(f"Generating {args.train_samples} training examples…")
    with train_file.open("w") as f:
        for ex in generate_dataset(args.train_samples, args.word_len, args.max_depth):
            json.dump(ex, f)
            f.write("\n")

    print(f"Generating {args.test_samples} test examples…")
    with test_file.open("w") as f:
        for ex in generate_dataset(args.test_samples, args.word_len, args.max_depth):
            json.dump(ex, f)
            f.write("\n")

    print("Done. Files saved in", out_dir)


if __name__ == "__main__":
    main() 