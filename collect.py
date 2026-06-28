"""
collect.py — Codeforces problem collector.

Step A: fetch all rated problems + statistics from the CF API (one call, no scraping).
Step B: scrape each problem page for statement / timeLimit / memoryLimit.
        Uses an on-disk HTML cache so the job is fully resumable.

Output: data/cf_problems.csv  (schema in BUILD_SPEC §4)
        data/cf_tags.csv      (tags, for the baseline only — never used as features)
        data/collect_errors.log

Anti-bot fallback: if Cloudflare blocks scraping, the script prints instructions for
using the HuggingFace deepmind/code_contests dataset instead (see --fallback flag).
"""
import argparse
import csv
import json
import logging
import os
import re
import time

import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
CACHE_DIR = "cache"
PROBLEMS_CSV = os.path.join(DATA_DIR, "cf_problems.csv")
TAGS_CSV = os.path.join(DATA_DIR, "cf_tags.csv")
ERRORS_LOG = os.path.join(DATA_DIR, "collect_errors.log")

API_URL = "https://codeforces.com/api/problemset.problems"
PROBLEM_URL = "https://codeforces.com/problemset/problem/{cid}/{idx}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; cf-difficulty-research/1.0; "
        "academic project github.com/user/cfdiff)"
    )
}

SLEEP_BETWEEN = 1.5  # seconds between live HTTP requests (politeness)


def setup_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def setup_logging():
    logging.basicConfig(
        filename=ERRORS_LOG,
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )


# ---------------------------------------------------------------------------
# Step A — problem list + statistics (single API call)
# ---------------------------------------------------------------------------

def fetch_problem_list():
    print("Fetching problem list from Codeforces API…")
    resp = requests.get(API_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "OK":
        raise RuntimeError(f"API returned status: {data['status']}")

    problems = {
        (p["contestId"], p["index"]): p
        for p in data["result"]["problems"]
        if "rating" in p
    }
    stats = {
        (s["contestId"], s["index"]): s["solvedCount"]
        for s in data["result"]["problemStatistics"]
    }

    rows = []
    for key, p in problems.items():
        rows.append({
            "contestId": p["contestId"],
            "index": p["index"],
            "rating": p["rating"],
            "tags": "|".join(p.get("tags", [])),
            "solvedCount": stats.get(key, 0),
        })

    print(f"  {len(rows)} rated problems found.")
    return rows


def save_tags(rows):
    with open(TAGS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["contestId", "index", "rating", "tags"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["contestId", "index", "rating", "tags"]})
    print(f"  Tags saved to {TAGS_CSV}")


# ---------------------------------------------------------------------------
# Step B — scrape statements (cached, polite)
# ---------------------------------------------------------------------------

def cache_path(contest_id, index):
    return os.path.join(CACHE_DIR, f"{contest_id}_{index}.html")


def fetch_html(contest_id, index):
    path = cache_path(contest_id, index)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(), False  # (html, was_live)
    url = PROBLEM_URL.format(cid=contest_id, idx=index)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html, True


def _strip_label(tag):
    label = tag.find("div", class_="property-title")
    if label:
        label.decompose()
    return tag.get_text(" ", strip=True)


def _parse_seconds(text):
    m = re.search(r"([\d.]+)\s*second", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_mb(text):
    m = re.search(r"([\d.]+)\s*megabyte", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_problem_page(html):
    """Return (statement, time_limit_s, memory_mb) or raise on failure."""
    soup = BeautifulSoup(html, "html.parser")

    # Detect Cloudflare / JS challenge pages
    if "cf-browser-verification" in html or "Just a moment" in html:
        raise RuntimeError("Cloudflare challenge page — scraping blocked")

    ps = soup.find("div", class_="problem-statement")
    if ps is None:
        raise RuntimeError("No .problem-statement div found")

    # time / memory limits
    tl_div = ps.find("div", class_="time-limit")
    ml_div = ps.find("div", class_="memory-limit")
    time_limit = _parse_seconds(_strip_label(tl_div)) if tl_div else None
    memory_mb = _parse_mb(_strip_label(ml_div)) if ml_div else None

    # statement text: header → legend (first child div after header) + input-spec
    # Exclude sample-tests and note
    parts = []
    header = ps.find("div", class_="header")
    if header:
        # legend is the first sibling div after the header
        for sib in header.find_next_siblings("div"):
            cls = sib.get("class", [])
            if "input-specification" in cls or "output-specification" in cls:
                parts.append(sib.get_text(" ", strip=True))
                break
            if "sample-tests" not in cls and "note" not in cls:
                # this is the legend
                parts.append(sib.get_text(" ", strip=True))
    inp = ps.find("div", class_="input-specification")
    if inp and inp.get_text(" ", strip=True) not in parts:
        parts.append(inp.get_text(" ", strip=True))

    statement = " ".join(p for p in parts if p)
    if not statement:
        # fallback: full problem-statement minus samples/note
        for bad in ps.find_all("div", class_=["sample-tests", "note"]):
            bad.decompose()
        statement = ps.get_text(" ", strip=True)

    return statement, time_limit, memory_mb


def scrape_statements(rows, limit=None):
    results = []
    live_count = 0
    blocked_count = 0

    total = len(rows) if limit is None else min(limit, len(rows))
    for i, row in enumerate(rows[:total]):
        cid, idx = row["contestId"], row["index"]
        try:
            html, was_live = fetch_html(cid, idx)
            if was_live:
                live_count += 1
                time.sleep(SLEEP_BETWEEN)
            statement, tl, ml = parse_problem_page(html)
            results.append({
                "contestId": cid,
                "index": idx,
                "rating": row["rating"],
                "statement": statement,
                "timeLimit": tl,
                "memoryLimit": ml,
                "solvedCount": row["solvedCount"],
            })
        except RuntimeError as exc:
            msg = str(exc)
            logging.warning("CF %s/%s — %s", cid, idx, msg)
            if "Cloudflare" in msg:
                blocked_count += 1
                if blocked_count >= 3:
                    print(
                        "\nERROR: Cloudflare is blocking scraping. "
                        "Run with --fallback to use the HuggingFace dataset instead."
                    )
                    break
        except Exception as exc:
            logging.warning("CF %s/%s — unexpected error: %s", cid, idx, exc)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] scraped, {live_count} live requests so far…")

    print(f"  Scraping done: {len(results)} problems, {live_count} live requests.")
    return results


def save_problems(results):
    fieldnames = ["contestId", "index", "rating", "statement",
                  "timeLimit", "memoryLimit", "solvedCount"]
    with open(PROBLEMS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {len(results)} rows to {PROBLEMS_CSV}")


# ---------------------------------------------------------------------------
# Fallback: HuggingFace deepmind/code_contests
# ---------------------------------------------------------------------------

def load_huggingface_fallback():
    """
    Fallback when CF scraping is blocked: use deepmind/code_contests from HuggingFace.
    Maps its schema to the §4 schema. Requires: pip install datasets
    """
    print("Loading deepmind/code_contests from HuggingFace (this may take a while)…")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("Install HuggingFace datasets first:  pip install datasets")
        raise

    ds = load_dataset("deepmind/code_contests", split="train+test+valid",
                      trust_remote_code=True)

    rows = []
    for i, ex in enumerate(ds):
        src = ex.get("source", 0)
        # source 1 = Codeforces (per dataset card)
        if src != 1:
            continue
        diff = ex.get("difficulty", None)
        if diff is None or diff == 0:
            continue
        # difficulty in this dataset is 1-indexed CF rating buckets; map roughly
        # to the same CF scale — they store the raw CF rating in some versions
        rating = int(diff) if diff >= 800 else int(diff) * 100
        if rating < 800 or rating > 3500:
            continue
        statement = (ex.get("description") or "").strip()
        if not statement:
            continue
        rows.append({
            "contestId": ex.get("cf_contest_id", i + 1000),
            "index": ex.get("cf_index", "A"),
            "rating": rating,
            "statement": statement,
            "timeLimit": (ex.get("time_limit") or {}).get("seconds", 2),
            "memoryLimit": (ex.get("memory_limit_bytes", 256 * 1024 * 1024)) / (1024 * 1024),
            "solvedCount": 0,
        })

    print(f"  HuggingFace fallback: {len(rows)} CF problems extracted.")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect Codeforces problems")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Scrape at most this many problems (useful for testing)"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Skip scraping; load statements from HuggingFace deepmind/code_contests"
    )
    args = parser.parse_args()

    setup_dirs()
    setup_logging()

    if args.fallback:
        results = load_huggingface_fallback()
        save_problems(results)
        return

    # Step A
    rows = fetch_problem_list()
    save_tags(rows)

    # Step B
    print(f"\nScraping statements for {len(rows)} problems (cached if already fetched)…")
    results = scrape_statements(rows, limit=args.limit)

    # Drop rows where scraping failed (no statement)
    before = len(results)
    results = [r for r in results if r.get("statement")]
    if before != len(results):
        print(f"  Dropped {before - len(results)} rows with empty statements.")

    save_problems(results)

    # Summary
    import pandas as pd
    df = pd.read_csv(PROBLEMS_CSV)
    null_stmt = df["statement"].isna().sum()
    null_rating = df["rating"].isna().sum()
    print(f"\nFinal: {len(df)} rows | null statement: {null_stmt} | null rating: {null_rating}")


if __name__ == "__main__":
    main()
