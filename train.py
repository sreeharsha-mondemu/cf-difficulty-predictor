"""
train.py — baselines, a structured-feature model, and a structured+text model,
all evaluated with a leakage-safe contest-level split.

Usage:
    python make_synthetic.py          # or run collect.py for real data
    python train.py data_synthetic.csv

Reports MAE/RMSE vs three baselines and prints structured-feature importances.
"""
import sys
import math
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

from features import extract_structured_features, STRUCTURED_COLUMNS


def contest_level_split(df, test_frac=0.15):
    """Hold out the NEWEST contests (highest contestId) as test — no random
    splitting, so near-duplicate problems from one round can't leak across."""
    contests = np.sort(df["contestId"].unique())
    n_test = max(1, int(len(contests) * test_frac))
    test_contests = set(contests[-n_test:])
    is_test = df["contestId"].isin(test_contests)
    return df[~is_test].copy(), df[is_test].copy()


def build_structured(df):
    feats = [extract_structured_features(r.statement, r.timeLimit,
                                         r.memoryLimit, r.solvedCount)
             for r in df.itertuples()]
    return pd.DataFrame(feats).reindex(columns=STRUCTURED_COLUMNS, fill_value=0.0)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    within = np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 100) * 100
    print(f"  {name:28s} MAE {mae:6.1f} | RMSE {rmse:6.1f} | within +-100: {within:4.1f}%")
    return mae


def main(path):
    df = pd.read_csv(path)
    train, test = contest_level_split(df)
    print(f"loaded {len(df)} problems | train {len(train)} / test {len(test)} "
          f"(split by contest)\n")
    ytr, yte = train["rating"].values, test["rating"].values

    # ---- baselines ----
    print("BASELINES")
    report("predict global mean", yte, np.full_like(yte, ytr.mean(), dtype=float))
    # constraint-only: bin by max_constraint_log10, predict per-bin mean
    Xtr_s, Xte_s = build_structured(train), build_structured(test)
    cm_tr = Xtr_s["max_constraint_log10"].round().values
    cm_te = Xte_s["max_constraint_log10"].round().values
    binmean = {b: ytr[cm_tr == b].mean() for b in np.unique(cm_tr)}
    pred_cm = np.array([binmean.get(b, ytr.mean()) for b in cm_te])
    report("constraint-bin mean (1 feat)", yte, pred_cm)

    # ---- structured-feature LightGBM (interpretable) ----
    print("\nMODELS")
    params = dict(objective="regression_l1", n_estimators=400, learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, verbose=-1)
    m_struct = lgb.LGBMRegressor(**params).fit(Xtr_s.values, ytr)
    report("structured features", yte, m_struct.predict(Xte_s.values))

    # ---- structured + TF-IDF (fit on TRAIN only) ----
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3,
                            stop_words="english")
    Ttr = tfidf.fit_transform(train["statement"])
    Tte = tfidf.transform(test["statement"])
    Xtr = hstack([csr_matrix(Xtr_s.values), Ttr]).tocsr()
    Xte = hstack([csr_matrix(Xte_s.values), Tte]).tocsr()
    m_full = lgb.LGBMRegressor(**params).fit(Xtr, ytr)
    report("structured + TF-IDF", yte, m_full.predict(Xte))

    # ---- interpretability: top structured features by gain ----
    print("\nTOP STRUCTURED FEATURES (by gain)")
    imp = sorted(zip(STRUCTURED_COLUMNS, m_struct.booster_.feature_importance("gain")),
                 key=lambda kv: -kv[1])
    for name, gain in imp[:8]:
        print(f"  {name:24s} {gain:10.0f}")

    import joblib
    joblib.dump({"model": m_full, "tfidf": tfidf,
                 "structured_columns": STRUCTURED_COLUMNS}, "model.joblib")
    print("\nsaved model.joblib")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic.csv")
