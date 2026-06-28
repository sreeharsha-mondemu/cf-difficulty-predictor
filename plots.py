"""
plots.py — generate all figures for the README.

Figures produced (all saved to figures/):
    feature_importance.png  — top structured-feature importances from the CF model
    rating_distribution.png — histogram of CF ratings in the training data
    residual_plot.png       — predicted vs actual scatter on the test set

Usage:
    python plots.py data/cf_problems.csv

Requires model.joblib (produced by train.py).
"""
import math
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import extract_structured_features, STRUCTURED_COLUMNS

FIGURES_DIR = "figures"
MODEL_PATH = "model.joblib"


def setup():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def contest_level_split(df, test_frac=0.15):
    contests = np.sort(df["contestId"].unique())
    n_test = max(1, int(len(contests) * test_frac))
    test_contests = set(contests[-n_test:])
    is_test = df["contestId"].isin(test_contests)
    return df[~is_test].copy(), df[is_test].copy()


def build_structured(df):
    feats = [
        extract_structured_features(
            r.statement,
            getattr(r, "timeLimit", None),
            getattr(r, "memoryLimit", None),
            getattr(r, "solvedCount", None),
        )
        for r in df.itertuples()
    ]
    return pd.DataFrame(feats).reindex(columns=STRUCTURED_COLUMNS, fill_value=0.0)


# ---------------------------------------------------------------------------
# Figure 1 — feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(bundle, top_n=12):
    model_struct = None
    # We only have m_full in the bundle; re-derive importance via the structured
    # feature slice (first len(STRUCTURED_COLUMNS) features of m_full).
    # If a dedicated struct model isn't stored, we load train.py's m_struct
    # by re-fitting — but that's expensive. Instead pull from bundle directly.
    # The bundle stores the full model; we approximate by using raw gain on
    # the first N features (which correspond to structured columns).
    m = bundle["model"]
    sc = bundle["structured_columns"]
    tfidf = bundle["tfidf"]
    n_struct = len(sc)
    gains = m.booster_.feature_importance("gain")
    struct_gains = gains[:n_struct]

    pairs = sorted(zip(sc, struct_gains), key=lambda x: -x[1])[:top_n]
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#4C72B0", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (LightGBM gain)")
    ax.set_title("Top structured features")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — rating distribution histogram
# ---------------------------------------------------------------------------

def plot_rating_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(800, 3600, 100)
    ax.hist(df["rating"], bins=bins, color="#55A868", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Codeforces rating")
    ax.set_ylabel("Number of problems")
    ax.set_title("Rating distribution (training set)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "rating_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — predicted vs actual scatter (test set)
# ---------------------------------------------------------------------------

def plot_residuals(df, bundle):
    _, test = contest_level_split(df)
    m = bundle["model"]
    tfidf = bundle["tfidf"]
    sc = bundle["structured_columns"]
    Xs = build_structured(test)
    Tt = tfidf.transform(test["statement"].fillna(""))
    X = hstack([csr_matrix(Xs.values), Tt]).tocsr()
    pred = m.predict(X)
    actual = test["rating"].values

    mae = mean_absolute_error(actual, pred)
    rmse = math.sqrt(mean_squared_error(actual, pred))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, pred, alpha=0.25, s=8, edgecolors="none", color="#C44E52")
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="perfect")
    ax.set_xlabel("Actual CF rating")
    ax.set_ylabel("Predicted CF rating")
    ax.set_title(f"Test-set predictions\nMAE={mae:.0f}  RMSE={rmse:.0f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "residual_plot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}  (MAE={mae:.0f}, RMSE={rmse:.0f})")
    return mae, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/cf_problems.csv"
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run collect.py first.")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: {MODEL_PATH} not found. Run train.py first.")

    setup()
    df = pd.read_csv(path)
    bundle = joblib.load(MODEL_PATH)

    print(f"Loaded {len(df)} problems from {path}")
    plot_rating_distribution(df)
    plot_feature_importance(bundle)
    plot_residuals(df, bundle)
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
