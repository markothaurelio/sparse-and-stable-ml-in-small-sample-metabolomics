import os
import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss


# -----------------------------
# Config (prototype defaults)
# -----------------------------
DATA_FILE = "m_MTBLS161_mecfs_metabolite_profiling.tsv"
SAMPLE_COL_PATTERN = r"^(CFS|CONT)\d+"

MISSINGNESS_THRESHOLD = 0.4

# repeated nested CV settings (keep small for prototyping)
REPEATS = 10
OUTER_SPLITS = 5
INNER_SPLITS = 5
RANDOM_STATE = 42

# tuning grid (small but meaningful)
C_GRID = np.logspace(-2, 2, 9)             # 0.01 ... 100
L1R_GRID = [1.0, 0.8, 0.5]                 # LASSO + Elastic Net mixes

SCORING = "roc_auc"


# -----------------------------
# Helpers
# -----------------------------
def jaccard(a: set, b: set) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0


def ci95_mean(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std(ddof=1) if len(x) > 1 else 0.0
    se = std / np.sqrt(len(x)) if len(x) > 0 else np.nan
    return mean, std, (mean - 1.96 * se, mean + 1.96 * se)


def load_metabolights_matrix(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    sample_cols = [c for c in df.columns if re.match(SAMPLE_COL_PATTERN, str(c))]
    if not sample_cols:
        raise ValueError("No sample columns found. Check SAMPLE_COL_PATTERN.")

    X = df[sample_cols].T
    X.columns = df["metabolite_identification"]

    # fix numeric formatting like "1,071.00" -> 1071.00
    X = X.replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")

    if (X.dtypes == "object").any():
        bad = X.columns[X.dtypes == "object"].tolist()
        raise ValueError(f"Non-numeric columns still present: {bad}")

    return X


def split_serum_urine(X: pd.DataFrame):
    X_serum = X[X.index.str.contains("serum", case=False, na=False)]
    X_urine = X[X.index.str.contains("urine", case=False, na=False)]
    return X_serum, X_urine


def make_labels(index: pd.Index) -> pd.Series:
    # 1 = CFS, 0 = Control
    return pd.Series(index.str.startswith("CFS").astype(int), index=index)


# -----------------------------
# Leakage-safe missingness filter (inside CV)
# -----------------------------
class MissingnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.kept_features_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        keep = X_df.isna().mean() < self.threshold
        self.kept_features_ = keep[keep].index
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.loc[:, self.kept_features_]


def make_pipeline():
    return Pipeline([
        ("miss_filter", MissingnessFilter(threshold=MISSINGNESS_THRESHOLD)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            solver="saga",
            l1_ratio=1.0,          # tuned
            C=1.0,                 # tuned
            max_iter=20000,
            random_state=RANDOM_STATE
        ))
    ])


# -----------------------------
# 1) Naive CV (optimistic) - tunes and reports on same CV
# -----------------------------
def naive_cv_estimate(X, y):
    cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"model__C": C_GRID, "model__l1_ratio": L1R_GRID},
        cv=cv,
        scoring=SCORING,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_score_, search.best_params_


# -----------------------------
# 2) Repeated Nested CV + stability + calibration
# -----------------------------
def repeated_nested_cv(X, y):
    outer_aucs = []
    outer_briers = []
    selected_sets = []
    best_params_list = []

    for r in range(REPEATS):
        outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            inner_cv = StratifiedKFold(
                n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + 10_000 + r
            )

            search = GridSearchCV(
                estimator=make_pipeline(),
                param_grid={"model__C": C_GRID, "model__l1_ratio": L1R_GRID},
                cv=inner_cv,
                scoring=SCORING,
                n_jobs=-1
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params_list.append(search.best_params_)

            y_prob = best_model.predict_proba(X_test)[:, 1]
            outer_aucs.append(roc_auc_score(y_test, y_prob))
            outer_briers.append(brier_score_loss(y_test, y_prob))

            # selected features = non-zero coefficients in the final fitted outer model
            kept = list(best_model.named_steps["miss_filter"].kept_features_)
            coefs = best_model.named_steps["model"].coef_[0]
            selected = set([f for f, c in zip(kept, coefs) if c != 0.0])
            selected_sets.append(selected)

    outer_aucs = np.array(outer_aucs, dtype=float)
    outer_briers = np.array(outer_briers, dtype=float)

    return outer_aucs, outer_briers, selected_sets, best_params_list


def summarize_stability(selected_sets):
    total_models = len(selected_sets)

    # selection frequency
    freq = {}
    for s in selected_sets:
        for f in s:
            freq[f] = freq.get(f, 0) + 1

    # Jaccard distribution across pairs
    j_scores = []
    for i in range(total_models):
        for j in range(i + 1, total_models):
            j_scores.append(jaccard(selected_sets[i], selected_sets[j]))
    j_scores = np.array(j_scores, dtype=float) if j_scores else np.array([np.nan])

    return freq, j_scores


# -----------------------------
# Main
# -----------------------------
print("RUNNING:", os.path.abspath(__file__))

X_all = load_metabolights_matrix(DATA_FILE)
X_serum, X_urine = split_serum_urine(X_all)
y_serum = make_labels(X_serum.index)

print("X (all):", X_all.shape)
print("Serum:", X_serum.shape, "Urine:", X_urine.shape)
print("Serum class counts:\n", y_serum.value_counts())

# ---- Naive (optimistic) ----
naive_auc, naive_params = naive_cv_estimate(X_serum, y_serum)
print("\n=== Naive CV (optimistic) ===")
print(f"Naive CV AUC: {naive_auc:.4f}")
print("Best params:", naive_params)

# ---- Repeated Nested (unbiased) ----
outer_aucs, outer_briers, selected_sets, best_params_list = repeated_nested_cv(X_serum, y_serum)

m_auc, s_auc, ci_auc = ci95_mean(outer_aucs)
m_br, s_br, ci_br = ci95_mean(outer_briers)

print("\n=== Repeated Nested CV (unbiased) ===")
print(f"Outer evaluations: {len(outer_aucs)}  (REPEATS={REPEATS} × OUTER_SPLITS={OUTER_SPLITS})")
print(f"AUC mean={m_auc:.4f}, std={s_auc:.4f}, 95% CI=({ci_auc[0]:.4f}, {ci_auc[1]:.4f})")
print(f"Brier mean={m_br:.4f}, std={s_br:.4f}, 95% CI=({ci_br[0]:.4f}, {ci_br[1]:.4f})")

print("\n=== Optimism gap (naive - nested mean) ===")
print(f"Gap: {(naive_auc - m_auc):.4f}")

# ---- Feature stability ----
freq, j_scores = summarize_stability(selected_sets)
print("\n=== Feature stability ===")
print(f"Mean Jaccard: {np.nanmean(j_scores):.4f}  (pairs={len(j_scores)})")

# print top 10 most frequently selected features
total_models = len(selected_sets)
print("\nTop 10 selection frequencies:")
for f, c in sorted(freq.items(), key=lambda x: -x[1])[:10]:
    print(f"{f}: {c}/{total_models} ({c/total_models:.1%})")