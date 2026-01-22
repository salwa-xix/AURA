from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
IN_SCORES = DATA_DIR / "company_monthly_scores.csv"
OUT = DATA_DIR / "company_risk_predictions.csv"

# 1) Proper time-based split:
#    Train on earlier months, test on the latest N months
N_TEST_MONTHS = 3

# 2) Feature set
#    Option A (recommended) Use KPIs + member_count (drivers), not IVI/U_score (avoid redundancy)
USE_IVI_AS_FEATURE = False  # set True if you want IVI included as a summary feature

# 3) Class imbalance handling
USE_CLASS_WEIGHT_BALANCED = True

# 4) Decision threshold (0.5 is default; you can tune later)
THRESHOLD = 0.5

# Load
df = pd.read_csv(IN_SCORES, dtype={"company_id": str, "period": str})
df = df.sort_values(["company_id", "period"]).reset_index(drop=True)

# next month High Risk
df["risk_band_next"] = df.groupby("company_id")["risk_band"].shift(-1)
df["IVI_next"] = df.groupby("company_id")["IVI"].shift(-1)
df["target_highrisk_next"] = (df["risk_band_next"] == "High Risk").astype(int)

# Drop last month per company (no next label)
model_df = df.dropna(subset=["risk_band_next"]).copy()

# Convert "YYYYMM" to int for sorting
model_df["period_int"] = (
    model_df["period"].astype(str).str.replace("-", "").str[:6].astype(int)
)

# Strong driver-style features --avoid redundancy with IVI if USE_IVI_AS_FEATURE=False---
base_features = [
    "avoidable_er_rate",
    "er_visits_per_100_members",
    "avoidable_er_cost",
    "total_er_visits",
    "member_count",
]

# if you implemented these better normalized KPIs, prefer them
# (If columns exist, use them; otherwise fall back to existing ones)
preferred_if_exist = [
    "avoidable_er_per_100_members",
    "avoidable_cost_per_member",
    "cost_per_member",
]

# Build final features list based on availability
feature_cols = []

# Add preferred normalized columns if present
for c in preferred_if_exist:
    if c in model_df.columns:
        feature_cols.append(c)

# Always include core features if present
for c in base_features:
    if c in model_df.columns and c not in feature_cols:
        feature_cols.append(c)

# Optionally add IVI and U_score as summary signals
if USE_IVI_AS_FEATURE:
    for c in ["IVI", "U_score"]:
        if c in model_df.columns and c not in feature_cols:
            feature_cols.insert(0, c)  # put at front

# Final safety check
missing_feats = [c for c in feature_cols if c not in model_df.columns]
if missing_feats:
    raise ValueError(
        f"Missing feature columns: {missing_feats}\nAvailable columns: {list(model_df.columns)}"
    )

# Build X/y
X_all = model_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_all = model_df["target_highrisk_next"].astype(int)

# Drop rows with NaNs in features ---keep it strict for clean model----------
before = len(model_df)
valid_mask = ~X_all.isna().any(axis=1)
model_df = model_df.loc[valid_mask].copy()
X_all = X_all.loc[valid_mask].copy()
y_all = y_all.loc[valid_mask].copy()
after = len(model_df)
if after < before:
    print(f" Dropped {before - after} rows due to NaNs in features.")

# Time-based split
all_periods = np.sort(model_df["period_int"].unique())
if len(all_periods) <= N_TEST_MONTHS:
    raise ValueError(
        f"Not enough unique months for time split. "
        f"Found {len(all_periods)} months, need > {N_TEST_MONTHS}."
    )

test_periods = all_periods[-N_TEST_MONTHS:]
train_periods = all_periods[:-N_TEST_MONTHS]

train_mask = model_df["period_int"].isin(train_periods)
test_mask = model_df["period_int"].isin(test_periods)

X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

print(" Time split:")
print(
    " - Train months:",
    train_periods[0],
    "→",
    train_periods[-1],
    f"({len(train_periods)} months)",
)
print(
    " - Test months :",
    test_periods[0],
    "→",
    test_periods[-1],
    f"({len(test_periods)} months)",
)
print(" - Train rows:", len(X_train), " Test rows:", len(X_test))
print(
    " - Positive rate (train):",
    round(y_train.mean(), 4),
    " Positive rate (test):",
    round(y_test.mean(), 4),
)

# Model
class_weight = "balanced" if USE_CLASS_WEIGHT_BALANCED else None

clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                max_iter=3000, class_weight=class_weight, solver="lbfgs"
            ),
        ),
    ]
)

clf.fit(X_train, y_train)

# Evaluate
proba_test = clf.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= THRESHOLD).astype(int)

auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else float("nan")
print(
    "\n Time-based Test AUC:",
    round(auc, 4) if not np.isnan(auc) else "N/A (single-class in test)",
)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_test))
print("\nClassification Report:\n", classification_report(y_test, pred_test, digits=4))

# Predict for all labeled rows

model_df["risk_probability_next_month"] = clf.predict_proba(X_all)[:, 1]
model_df["pred_highrisk_next"] = (
    model_df["risk_probability_next_month"] >= THRESHOLD
).astype(int)

# Keep clean output
keep_cols = [
    "company_id",
    "period",
    "IVI",
    "U_score",
    "risk_probability_next_month",
    "pred_highrisk_next",
    "IVI_next",
    "target_highrisk_next",
]

# add driver features
keep_cols += [c for c in feature_cols if c not in keep_cols]

out_df = model_df[keep_cols].copy()
out_df.to_csv(OUT, index=False)

print("\n Generated:", OUT)
print("shape:", out_df.shape)
print(out_df.sort_values(["company_id", "period"]).head(10).to_string(index=False))


# Driver analysis

lr = clf.named_steps["lr"]
coef = pd.DataFrame({"feature": feature_cols, "coef": lr.coef_[0]}).sort_values(
    "coef", ascending=False
)

print("\n Top positive drivers (increase risk):")
print(coef.head(7).to_string(index=False))

print("\n Top negative drivers (decrease risk):")
print(coef.tail(7).to_string(index=False))
