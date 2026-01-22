from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IN_KPIS = DATA_DIR / "company_monthly_kpis.csv"
OUT = DATA_DIR / "company_monthly_scores.csv"

df = pd.read_csv(IN_KPIS, dtype={"company_id": str, "period": str})


def safe_minmax_within_group(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def winsorize(s: pd.Series, p=0.99) -> pd.Series:
    s = s.astype(float)
    cap = s.quantile(p)
    return s.clip(upper=cap)


# Choose "bad" utilization metrics (higher = worse)

bad1 = df["avoidable_er_rate"]  # share within ER (quality issue)

# These two should exist ideally
bad2 = df.get("avoidable_er_per_100_members", df["er_visits_per_100_members"])  # volume
bad3 = df.get("avoidable_cost_per_member", df["avoidable_er_cost"])  # cost impact

# Winsorize to reduce outlier dominance
bad2 = winsorize(bad2, 0.99)
bad3 = winsorize(bad3, 0.99)

# Normalize within each month (period) for fair comparison
n1 = df.groupby("period", group_keys=False)["avoidable_er_rate"].apply(
    safe_minmax_within_group
)
n2 = df.groupby("period", group_keys=False).apply(
    lambda g: safe_minmax_within_group(bad2.loc[g.index])
)
n3 = df.groupby("period", group_keys=False).apply(
    lambda g: safe_minmax_within_group(bad3.loc[g.index])
)

# Convert to "good"
good1, good2, good3 = 1 - n1, 1 - n2, 1 - n3

# U_score (0..100)
df["U_score"] = (0.45 * good1 + 0.35 * good2 + 0.20 * good3) * 100

# Prototype: H/E neutral
df["H_score"] = 50.0
df["E_score"] = 50.0

# IVI
df["IVI"] = 0.60 * df["U_score"] + 0.20 * df["H_score"] + 0.20 * df["E_score"]


def band(x):
    if x < 55:
        return "High Risk"
    if x < 70:
        return "Medium Risk"
    return "Low Risk"


df["risk_band"] = df["IVI"].apply(band)

df.to_csv(OUT, index=False)

print("Generated:", OUT)
print("shape:", df.shape)
print(df.sort_values(["company_id", "period"]).head(10).to_string(index=False))
print("\nRisk band distribution:")
print(df["risk_band"].value_counts())
