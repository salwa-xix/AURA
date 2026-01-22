from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

MEMBERS_PATH = DATA_DIR / "members_synth.csv"
CLAIMS_PATH = DATA_DIR / "claims_synth.csv"

OUT_KPIS = DATA_DIR / "company_monthly_kpis.csv"
OUT_SEG = DATA_DIR / "avoidable_er_segments.csv"

members = pd.read_csv(MEMBERS_PATH, dtype={"company_id": str, "member_id": str})
claims = pd.read_csv(
    CLAIMS_PATH, dtype={"PROV_CODE": str, "member_id": str, "period": str}
)

# 1) Attach company_id
claims = claims.merge(members[["member_id", "company_id"]], on="member_id", how="left")

# Drop unlinked claims (data integrity)
missing_rate = claims["company_id"].isna().mean()
if missing_rate > 0:
    before = len(claims)
    claims = claims.dropna(subset=["company_id"])
    after = len(claims)
    print(
        f" Dropped {before - after} claims with missing company_id (missing rate was {missing_rate:.2%})."
    )

# 2) Avoidable ER = ER with no admission
claims["avoidable_er"] = (
    (claims["is_er"] == 1) & (claims["admission_flag"] == 0)
).astype(int)

# 3) Company-month KPIs
monthly = (
    claims.groupby(["company_id", "period"])
    .agg(
        total_claims=("claim_id", "count"),
        total_cost=("net_bill", "sum"),
        total_er_visits=("is_er", "sum"),
        avoidable_er_visits=("avoidable_er", "sum"),
    )
    .reset_index()
)

# Avoidable ER cost
avoidable_cost = (
    claims.loc[claims["avoidable_er"] == 1]
    .groupby(["company_id", "period"])["net_bill"]
    .sum()
    .reset_index()
    .rename(columns={"net_bill": "avoidable_er_cost"})
)

monthly = monthly.merge(avoidable_cost, on=["company_id", "period"], how="left")
monthly["avoidable_er_cost"] = monthly["avoidable_er_cost"].fillna(0)

# Member count per company
company_members = (
    members.groupby("company_id")["member_id"]
    .nunique()
    .reset_index(name="member_count")
)
monthly = monthly.merge(company_members, on="company_id", how="left")

# Rates (share within ER)
monthly["avoidable_er_rate"] = (
    monthly["avoidable_er_visits"] / monthly["total_er_visits"].replace(0, pd.NA)
).fillna(0)


# Volume-normalized (fair comparison across companies)
monthly["er_visits_per_100_members"] = np.where(
    monthly["member_count"] > 0,
    (monthly["total_er_visits"] / monthly["member_count"]) * 100,
    0,
)

monthly["avoidable_er_per_100_members"] = np.where(
    monthly["member_count"] > 0,
    (monthly["avoidable_er_visits"] / monthly["member_count"]) * 100,
    0,
)

monthly["cost_per_member"] = np.where(
    monthly["member_count"] > 0, monthly["total_cost"] / monthly["member_count"], 0
)

monthly["avoidable_cost_per_member"] = np.where(
    monthly["member_count"] > 0,
    monthly["avoidable_er_cost"] / monthly["member_count"],
    0,
)

monthly.to_csv(OUT_KPIS, index=False)

print(" Generated:", OUT_KPIS, "shape:", monthly.shape)
print(monthly.sort_values(["company_id", "period"]).head(10).to_string(index=False))

# 4) Segmentation
seg_cols = [
    "PROVIDER_NETWORK",
    "PROVIDER_PRACTICE",
    "PROVIDER_REGION",
    "PROVIDER_TOWN",
    "PROV_NAME",
]
for c in seg_cols:
    if c not in claims.columns:
        claims[c] = "Unknown"
    claims[c] = claims[c].astype(str).fillna("Unknown")

avoidable = claims[claims["avoidable_er"] == 1].copy()

segments = (
    avoidable.groupby(["company_id", "period"] + seg_cols)
    .agg(
        avoidable_er_visits=("claim_id", "count"), avoidable_er_cost=("net_bill", "sum")
    )
    .reset_index()
)

segments.to_csv(OUT_SEG, index=False)

print("\n Generated:", OUT_SEG, "shape:", segments.shape)
print(segments.head(10).to_string(index=False))
