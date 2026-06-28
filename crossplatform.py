"""
crossplatform.py — Transfer the CF-trained model to LeetCode and AtCoder,
then validate that content-based predictions correlate with independent
difficulty signals (zerotrac Elo for LeetCode, kenkoooo for AtCoder).

Headline metric: Spearman ρ between predicted CF-equivalent rating and each
platform's own difficulty signal.

Usage:
    python crossplatform.py [--lc-only] [--ac-only] [--no-scrape]

Outputs:
    data/leetcode.csv        — slug, title, statement, lc_difficulty, zerotrac_rating
    data/atcoder.csv         — id, statement, ac_difficulty
    figures/lc_scatter.png
    figures/ac_scatter.png
    (Spearman ρ printed to stdout)

Anti-bot note: LeetCode uses a GraphQL API that sometimes 403s without a
session cookie. AtCoder blocks bots on some routes. If either blocks, the
script exits with a clear message and asks the user to intervene.
"""
import argparse
import json
import math
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr

from features import extract_structured_features, STRUCTURED_COLUMNS

DATA_DIR = "data"
CACHE_DIR = "cache"
FIGURES_DIR = "figures"
MODEL_PATH = "model.joblib"

LC_CSV = os.path.join(DATA_DIR, "leetcode.csv")
AC_CSV = os.path.join(DATA_DIR, "atcoder.csv")

ZEROTRAC_URL = (
    "https://raw.githubusercontent.com/"
    "zerotrac/leetcode_problem_rating/main/data.json"
)
AC_PROBLEMS_URL = "https://kenkoooo.com/atcoder/resources/problems.json"
AC_MODELS_URL = "https://kenkoooo.com/atcoder/resources/problem-models.json"

SLEEP = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; cf-difficulty-research/1.0; "
        "academic project github.com/user/cfdiff)"
    )
}


def setup_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"ERROR: {MODEL_PATH} not found. "
            "Run  python train.py data/cf_problems.csv  first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["tfidf"], bundle["structured_columns"]


def predict_ratings(model, tfidf, structured_cols, df_platform):
    """
    Build the same feature matrix as train.py and predict CF-equivalent ratings.
    df_platform must have a 'statement' column; timeLimit/memoryLimit/solvedCount
    are optional (filled with NaN → feature omitted, matching STRUCTURED_COLUMNS).
    """
    feats = []
    for row in df_platform.itertuples():
        stmt = getattr(row, "statement", "") or ""
        tl = getattr(row, "timeLimit", None)
        ml = getattr(row, "memoryLimit", None)
        sc = getattr(row, "solvedCount", None)
        if isinstance(tl, float) and math.isnan(tl):
            tl = None
        if isinstance(ml, float) and math.isnan(ml):
            ml = None
        if isinstance(sc, float) and math.isnan(sc):
            sc = None
        feats.append(extract_structured_features(stmt, tl, ml, sc))
    Xs = pd.DataFrame(feats).reindex(columns=structured_cols, fill_value=0.0)
    Tt = tfidf.transform(df_platform["statement"].fillna(""))
    X = hstack([csr_matrix(Xs.values), Tt]).tocsr()
    return model.predict(X)


# ---------------------------------------------------------------------------
# LeetCode
# ---------------------------------------------------------------------------

LC_GRAPHQL = "https://leetcode.com/graphql"
LC_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    title
    difficulty
    content
  }
}
"""

def _cache_path(prefix, key):
    safe_key = str(key).replace("/", "_")
    return os.path.join(CACHE_DIR, f"{prefix}_{safe_key}.html")


def fetch_lc_statement(slug, session=None):
    path = _cache_path("lc", slug)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    req = session or requests.Session()
    resp = req.post(
        LC_GRAPHQL,
        json={"query": LC_QUERY, "variables": {"titleSlug": slug}},
        headers={**HEADERS, "Content-Type": "application/json",
                 "Referer": f"https://leetcode.com/problems/{slug}/"},
        timeout=20,
    )
    if resp.status_code == 403:
        raise RuntimeError(
            f"LeetCode returned 403 for '{slug}'. "
            "The GraphQL endpoint needs a logged-in session cookie. "
            "Please provide your LeetCode session cookie (LEETCODE_SESSION env var) "
            "or run with --no-scrape to skip statement fetching."
        )
    resp.raise_for_status()
    data = resp.json()
    question = (data.get("data") or {}).get("question") or {}
    content = question.get("content") or ""
    # Strip HTML tags for plain text
    from bs4 import BeautifulSoup
    text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def collect_leetcode(no_scrape=False):
    """
    Download zerotrac ratings, then either load existing LC CSV or scrape statements.
    Returns DataFrame with slug, title, statement, lc_difficulty, zerotrac_rating.
    """
    # 1. zerotrac ratings (the independent validation label)
    print("Downloading zerotrac LeetCode ratings…")
    resp = requests.get(ZEROTRAC_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    zt = resp.json()  # list of {ContestSlug, ProblemIndex, Rating, ...}

    # Build slug -> rating map. The JSON has ContestSlug+ProblemIndex but also
    # a TitleSlug field in newer versions; fall back to constructing from title.
    zt_map = {}
    for entry in zt:
        slug = entry.get("TitleSlug") or entry.get("titleSlug")
        rating = entry.get("Rating") or entry.get("rating")
        if slug and rating:
            zt_map[slug] = float(rating)
    print(f"  zerotrac: {len(zt_map)} rated problems.")

    if no_scrape:
        # Return minimal frame — no statements, predictions will be skipped
        rows = [{"slug": s, "title": s, "statement": "", "lc_difficulty": "",
                 "zerotrac_rating": r} for s, r in zt_map.items()]
        df = pd.DataFrame(rows)
        df.to_csv(LC_CSV, index=False)
        return df

    # 2. Scrape statements for each slug that has a zerotrac rating
    import requests as _req
    session = _req.Session()
    lc_session_cookie = os.environ.get("LEETCODE_SESSION")
    if lc_session_cookie:
        session.cookies.set("LEETCODE_SESSION", lc_session_cookie, domain="leetcode.com")

    rows = []
    slugs = list(zt_map.keys())
    print(f"Scraping {len(slugs)} LeetCode statements (cached if already fetched)…")
    live = 0
    blocked = False
    for i, slug in enumerate(slugs):
        try:
            path = _cache_path("lc", slug)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            else:
                text = fetch_lc_statement(slug, session)
                live += 1
                time.sleep(SLEEP)
            rows.append({
                "slug": slug,
                "title": slug,
                "statement": text,
                "lc_difficulty": "",
                "zerotrac_rating": zt_map[slug],
            })
        except RuntimeError as exc:
            msg = str(exc)
            if "403" in msg:
                print(f"\n{msg}")
                print(
                    "\nHint: export LEETCODE_SESSION=<your cookie value> and re-run, "
                    "or pass --no-scrape to skip LeetCode statement fetching."
                )
                blocked = True
                break
            print(f"  Warning: {slug} — {exc}")
        except Exception as exc:
            print(f"  Warning: {slug} — {exc}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(slugs)}] LC problems scraped…")

    if blocked and not rows:
        sys.exit("LeetCode scraping blocked — see message above.")

    df = pd.DataFrame(rows)
    df.to_csv(LC_CSV, index=False)
    print(f"  Saved {len(df)} rows to {LC_CSV}")
    return df


# ---------------------------------------------------------------------------
# AtCoder
# ---------------------------------------------------------------------------

def _fetch_json(url, label):
    print(f"Fetching {label}…")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_ac_statement(contest, task):
    path = _cache_path("ac", task)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    url = f"https://atcoder.jp/contests/{contest}/tasks/{task}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 403:
        raise RuntimeError(
            f"AtCoder returned 403 for {task}. "
            "AtCoder may require a session cookie. "
            "Set ATCODER_SESSION env var or pass --no-scrape."
        )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {task}")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    # AtCoder statement is in #task-statement
    stmt_div = soup.find("div", id="task-statement")
    if stmt_div is None:
        stmt_div = soup.find("div", class_="problem-statement")
    text = stmt_div.get_text(" ", strip=True) if stmt_div else ""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def collect_atcoder(no_scrape=False):
    """
    Fetch problem list + difficulty models from kenkoooo, scrape statements.
    Returns DataFrame with id, statement, ac_difficulty.
    """
    problems = _fetch_json(AC_PROBLEMS_URL, "AtCoder problem list")
    models = _fetch_json(AC_MODELS_URL, "AtCoder difficulty models")

    # models is {problem_id: {difficulty: float, ...}}
    diff_map = {pid: v.get("difficulty") for pid, v in models.items()
                if isinstance(v, dict) and v.get("difficulty") is not None}

    # Filter to problems that have a difficulty estimate
    rated = [p for p in problems if p["id"] in diff_map]
    print(f"  {len(rated)} AtCoder problems with difficulty estimates.")

    if no_scrape:
        rows = [{"id": p["id"], "statement": "",
                 "ac_difficulty": diff_map[p["id"]]} for p in rated]
        df = pd.DataFrame(rows)
        df.to_csv(AC_CSV, index=False)
        return df

    ac_session_cookie = os.environ.get("ATCODER_SESSION")
    session = requests.Session()
    if ac_session_cookie:
        session.cookies.set("REVEL_SESSION", ac_session_cookie, domain="atcoder.jp")

    rows = []
    print(f"Scraping {len(rated)} AtCoder statements…")
    live = 0
    blocked = False
    for i, p in enumerate(rated):
        pid, contest = p["id"], p["contest_id"]
        try:
            path = _cache_path("ac", pid)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            else:
                text = fetch_ac_statement(contest, pid)
                live += 1
                time.sleep(SLEEP)
            rows.append({
                "id": pid,
                "statement": text,
                "ac_difficulty": diff_map[pid],
            })
        except RuntimeError as exc:
            msg = str(exc)
            if "403" in msg:
                print(f"\n{msg}")
                print(
                    "\nHint: export ATCODER_SESSION=<cookie> and re-run, "
                    "or pass --no-scrape."
                )
                blocked = True
                break
            print(f"  Warning: {pid} — {exc}")
        except Exception as exc:
            print(f"  Warning: {pid} — {exc}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rated)}] AC problems scraped…")

    if blocked and not rows:
        sys.exit("AtCoder scraping blocked — see message above.")

    df = pd.DataFrame(rows)
    df.to_csv(AC_CSV, index=False)
    print(f"  Saved {len(df)} rows to {AC_CSV}")
    return df


# ---------------------------------------------------------------------------
# Correlation + scatter plots
# ---------------------------------------------------------------------------

def spearman_report(label, y_pred, y_true):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    rho, pval = spearmanr(y_pred[mask], y_true[mask])
    print(f"  {label:40s} Spearman rho = {rho:+.3f}  (n={mask.sum()}, p={pval:.2e})")
    return rho


def make_scatter(x, y, xlabel, ylabel, title, path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, alpha=0.25, s=8, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")
    except Exception as exc:
        print(f"  Warning: could not save {path} — {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc-only", action="store_true")
    parser.add_argument("--ac-only", action="store_true")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip statement scraping; use cached CSVs or empty stmts")
    args = parser.parse_args()

    setup_dirs()
    model, tfidf, structured_cols = load_model()

    do_lc = not args.ac_only
    do_ac = not args.lc_only

    results = {}

    # ---- LeetCode ----
    if do_lc:
        print("\n=== LeetCode ===")
        if args.no_scrape and os.path.exists(LC_CSV):
            df_lc = pd.read_csv(LC_CSV)
            print(f"  Loaded {len(df_lc)} rows from {LC_CSV}")
        else:
            df_lc = collect_leetcode(no_scrape=args.no_scrape)

        df_lc = df_lc.dropna(subset=["statement"]).query("statement != ''")
        if len(df_lc) > 0:
            df_lc["pred_rating"] = predict_ratings(model, tfidf, structured_cols, df_lc)
            print("\nLeetCode correlations:")
            rho_zt = spearman_report(
                "predicted vs zerotrac rating",
                df_lc["pred_rating"].values,
                df_lc["zerotrac_rating"].values.astype(float),
            )
            # ordinal: Easy=1, Medium=2, Hard=3 (if available)
            if "lc_difficulty" in df_lc.columns:
                ord_map = {"Easy": 1, "Medium": 2, "Hard": 3}
                ordinal = df_lc["lc_difficulty"].map(ord_map)
                valid = ordinal.notna()
                if valid.sum() > 10:
                    spearman_report(
                        "predicted vs Easy/Med/Hard ordinal",
                        df_lc.loc[valid, "pred_rating"].values,
                        ordinal[valid].values.astype(float),
                    )
            make_scatter(
                df_lc["pred_rating"].values,
                df_lc["zerotrac_rating"].values.astype(float),
                "Predicted CF rating", "zerotrac Elo rating",
                "LeetCode: predicted vs zerotrac",
                os.path.join(FIGURES_DIR, "lc_scatter.png"),
            )
            results["lc_spearman_zerotrac"] = rho_zt
        else:
            print("  No LeetCode rows with statements — skipping prediction.")

    # ---- AtCoder ----
    if do_ac:
        print("\n=== AtCoder ===")
        if args.no_scrape and os.path.exists(AC_CSV):
            df_ac = pd.read_csv(AC_CSV)
            print(f"  Loaded {len(df_ac)} rows from {AC_CSV}")
        else:
            df_ac = collect_atcoder(no_scrape=args.no_scrape)

        df_ac = df_ac.dropna(subset=["statement"]).query("statement != ''")
        if len(df_ac) > 0:
            df_ac["pred_rating"] = predict_ratings(model, tfidf, structured_cols, df_ac)
            print("\nAtCoder correlations:")
            rho_ac = spearman_report(
                "predicted vs kenkoooo difficulty",
                df_ac["pred_rating"].values,
                df_ac["ac_difficulty"].values.astype(float),
            )
            make_scatter(
                df_ac["pred_rating"].values,
                df_ac["ac_difficulty"].values.astype(float),
                "Predicted CF rating", "kenkoooo difficulty",
                "AtCoder: predicted vs kenkoooo",
                os.path.join(FIGURES_DIR, "ac_scatter.png"),
            )
            results["ac_spearman_kenkoooo"] = rho_ac
        else:
            print("  No AtCoder rows with statements — skipping prediction.")

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v:+.3f}")


if __name__ == "__main__":
    main()
