"""
features.py — content-based features from a problem statement.

The standout is parse_max_constraint(): the order of magnitude of the input
bound (n <= 10^5 vs 10^18) encodes the intended complexity class, so it's
usually the single strongest predictor of difficulty. Everything here reads the
*statement only* — no Codeforces tags — so the model transfers to other judges.
"""
import re
import math

# common modulo primes — these are NOT size bounds, so exclude them
MOD_CONSTANTS = {1_000_000_007, 998_244_353, 1_000_000_009, 1_000_000_006}

# algorithm-suggestive vocabulary (domain-curated; NOT CF tags)
KEYWORDS = [
    "tree", "graph", "prime", "modulo", "xor", "bitmask", "subsequence",
    "substring", "shortest", "matrix", "permutation", "interval", "query",
    "queries", "minimum", "maximum", "probability", "expected", "gcd",
    "divisor", "palindrome", "flow", "matching", "convex", "segment",
]

_POW10 = re.compile(
    r"(\d+(?:\.\d+)?)?\s*(?:\\cdot|\\times|\*|·)?\s*10\s*\^\s*\{?\s*(\d+)\s*\}?"
)
_PLAIN_INT = re.compile(r"(?<![\d.])\d{1,19}(?![\d^])")


def _candidate_numbers(text):
    """All numeric magnitudes mentioned, expanding 10^k / c*10^k notation."""
    vals = []
    consumed = []
    for m in _POW10.finditer(text):
        coeff = float(m.group(1)) if m.group(1) else 1.0
        exp = int(m.group(2))
        try:
            vals.append(coeff * (10 ** exp))
        except OverflowError:
            vals.append(10.0 ** 18)
        consumed.append((m.start(), m.end()))
    # plain integers NOT already part of a 10^k match
    def overlaps(i):
        return any(a <= i < b for a, b in consumed)
    for m in _PLAIN_INT.finditer(text):
        if overlaps(m.start()):
            continue
        try:
            vals.append(float(int(m.group())))
        except ValueError:
            pass
    return vals


def parse_max_constraint(text):
    """Return log10 of the largest size-like bound (0.0 if none found)."""
    vals = [v for v in _candidate_numbers(text)
            if v >= 1 and int(v) not in MOD_CONSTANTS]
    if not vals:
        return 0.0
    return math.log10(max(vals))


def extract_structured_features(statement, time_limit_s=None, memory_mb=None,
                                solved_count=None):
    """Dense, transferable features computed from statement text + metadata."""
    s = statement or ""
    low = s.lower()
    words = re.findall(r"[a-zA-Z]+", s)
    f = {
        "max_constraint_log10": parse_max_constraint(s),
        "n_chars": len(s),
        "n_words": len(words),
        "n_sentences": s.count(".") + s.count("!") + s.count("?") + 1,
        "n_math_spans": s.count("$") // 2,
        "n_le": len(re.findall(r"\\le\b|\\leq\b|≤|<=", s)),
        "n_latex_ops": len(re.findall(r"\\(?:cdot|times|frac|sum|prod|sqrt)", s)),
        "n_digits": sum(c.isdigit() for c in s),
        "has_mod": 1 if re.search(r"10\s*\^\s*9\s*\+\s*7|998244353|modulo", low) else 0,
    }
    for kw in KEYWORDS:
        f["kw_" + kw] = 1 if kw in low else 0
    if time_limit_s is not None:
        f["time_limit_s"] = float(time_limit_s)
    if memory_mb is not None:
        f["memory_mb"] = float(memory_mb)
    if solved_count is not None:
        # log-scaled popularity; fewer solvers usually => harder
        f["log_solved"] = math.log10(max(1, solved_count))
    return f


STRUCTURED_COLUMNS = list(extract_structured_features("").keys())  # stable order


if __name__ == "__main__":
    samples = [
        ("easy",   r"Given an array of $n$ integers, $1 \le n \le 100$. Print the sum."),
        ("medium", r"$1 \le n \le 2 \cdot 10^5$, $1 \le a_i \le 10^9$. Answer mod $10^9 + 7$."),
        ("hard",   r"$1 \le n \le 10^{18}$. Count valid configurations on the tree."),
        ("unicode","1 ≤ n ≤ 100000, 1 ≤ q ≤ 100000 queries on a segment tree."),
        ("plain",  "Constraints: n <= 1000000 and m <= 500000."),
    ]
    for name, txt in samples:
        print(f"{name:8s} max_constraint_log10 = {parse_max_constraint(txt):4.1f}"
              f"   (max bound ~ 10^{parse_max_constraint(txt):.0f})")
