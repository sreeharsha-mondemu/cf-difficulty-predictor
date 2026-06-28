"""
make_synthetic.py — a fake dataset shaped exactly like what collect.py will
produce (same columns), with a learnable-but-noisy signal. Lets us prove
train.py works before real data exists. NOT for the resume — just a test fixture.

Columns match real output: contestId, index, rating, statement, timeLimit,
memoryLimit, solvedCount.
"""
import csv
import random
import math

random.seed(7)

HARD_WORDS = ["flow", "matching", "convex", "segment tree", "probability",
              "suffix automaton", "centroid", "persistent", "lagrangian"]
EASY_WORDS = ["sum", "count", "print", "array", "maximum", "sort", "pair"]
FILLER = ("You are given a problem. Consider the following setup carefully and "
          "compute the required answer for each test case as described below. ")


def make_statement(difficulty):
    # constraint magnitude loosely tracks difficulty (noisy, non-monotonic)
    exp = max(1, min(18, int(round(random.gauss(2 + difficulty / 250, 2)))))
    bound = f"$1 \\le n \\le 10^{{{exp}}}$"
    parts = [FILLER * random.randint(1, 3), bound + ". "]
    # sprinkle difficulty-correlated vocabulary so TF-IDF has signal
    n_hard = sum(random.random() < difficulty / 3500 for _ in range(4))
    n_easy = sum(random.random() < (3500 - difficulty) / 3500 for _ in range(4))
    for _ in range(n_hard):
        parts.append(random.choice(HARD_WORDS) + " ")
    for _ in range(n_easy):
        parts.append(random.choice(EASY_WORDS) + " ")
    if random.random() < 0.5:
        parts.append("Print the answer modulo $10^9 + 7$. ")
    return "".join(parts)


def main(path="data_synthetic.csv", n=2000, n_contests=200):
    rows = []
    for i in range(n):
        d = random.uniform(800, 3500)                       # latent difficulty
        statement = make_statement(d)
        rating = int(round((d + random.gauss(0, 220)) / 100.0)) * 100
        rating = max(800, min(3500, rating))
        solved = int(max(1, 200000 * math.exp(-d / 700) * random.uniform(0.5, 1.5)))
        rows.append({
            "contestId": 1000 + (i % n_contests),           # contestId ~ time proxy
            "index": "ABCDEF"[i % 6],
            "rating": rating,
            "statement": statement,
            "timeLimit": random.choice([1, 1, 2, 2, 3]),
            "memoryLimit": random.choice([256, 256, 512]),
            "solvedCount": solved,
        })
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic problems -> {path}")


if __name__ == "__main__":
    main()
