from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

# Paths
DATA_DIR = Path("data")
PROV_CLEAN = DATA_DIR / "provider_clean.csv"

OUT_MEMBERS = DATA_DIR / "members_synth.csv"
OUT_CLAIMS = DATA_DIR / "claims_synth.csv"

DATA_DIR.mkdir(exist_ok=True)

N_COMPANIES = 30

# Members per company
MIN_MEMBERS_PER_COMPANY = 350
MAX_MEMBERS_PER_COMPANY = 900  # upper is exclusive in randint

# Claims volume
CLAIMS_PER_MEMBER = 3.2  # ~~3 claims/member/year

# ER & Admission
ER_PROB_IF_HOSPITAL = 0.18  # ER only possible for hospitals
ADMIT_PROB_IF_ER = 0.12  # admission only possible if ER

# Cost multipliers
ER_MULT = 2.2
ADM_MULT = 1.8

#  cap to prevent extreme outliers
BILL_CAP = 45000


# If True sample providers with weights (pharmacy/clinic > hospital)
USE_WEIGHTED_PROVIDER_SAMPLING = True

# Load providers
prov = pd.read_csv(PROV_CLEAN, dtype={"PROV_CODE": str})

# Clean key fields
prov["PROV_CODE"] = prov["PROV_CODE"].astype(str).str.strip()
for col in [
    "PROV_NAME",
    "PROVIDER_NETWORK",
    "PROVIDER_PRACTICE",
    "PROVIDER_REGION",
    "PROVIDER_TOWN",
]:
    if col in prov.columns:
        prov[col] = prov[col].astype(str).str.strip()

prov_codes = prov["PROV_CODE"].tolist()


if USE_WEIGHTED_PROVIDER_SAMPLING:
    practice_lower = prov["PROVIDER_PRACTICE"].fillna("").astype(str).str.lower()

    # Heuristic weights (sum will be normalized)
    raw_weights = np.select(
        [
            practice_lower.str.contains("pharmacy"),
            practice_lower.str.contains("clinic"),
            practice_lower.str.contains("dental"),
            practice_lower.str.contains("hospital"),
        ],
        [
            0.38,  # pharmacy high frequency
            0.35,  # clinic
            0.15,  # dental
            0.12,  # hospital lower frequency than OP services
        ],
        default=0.10,
    ).astype(float)

    weights = raw_weights / raw_weights.sum()
else:
    weights = None  # uniform sampling

# Generate Members
companies = [f"C{str(i).zfill(3)}" for i in range(1, N_COMPANIES + 1)]
company_sizes = np.random.randint(
    MIN_MEMBERS_PER_COMPANY, MAX_MEMBERS_PER_COMPANY, size=N_COMPANIES
)

members_rows = []
member_counter = 1

# Join dates
join_start = pd.Timestamp("2024-01-01")

for cid, size in zip(companies, company_sizes):
    for _ in range(size):
        members_rows.append(
            {
                "member_id": f"M{member_counter:07d}",
                "company_id": cid,
                "age": int(np.random.randint(18, 62)),  # 18..61
                "gender": np.random.choice(["M", "F"]),
                "join_date": (
                    join_start + pd.to_timedelta(np.random.randint(0, 365), unit="D")
                )
                .date()
                .isoformat(),
            }
        )
        member_counter += 1

members = pd.DataFrame(members_rows)

# Generate Claims (synthetic)
n_claims = int(len(members) * CLAIMS_PER_MEMBER)

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
service_dates = pd.Series(np.random.choice(dates, size=n_claims)).dt.date.astype(str)

# Provider codes sampling
if USE_WEIGHTED_PROVIDER_SAMPLING and weights is not None:
    sampled_prov_codes = np.random.choice(
        prov["PROV_CODE"].values, size=n_claims, replace=True, p=weights
    )
else:
    sampled_prov_codes = np.random.choice(prov_codes, size=n_claims, replace=True)

claims = pd.DataFrame(
    {
        "claim_id": [f"CL{i:09d}" for i in range(1, n_claims + 1)],
        "member_id": np.random.choice(
            members["member_id"], size=n_claims, replace=True
        ),
        "PROV_CODE": sampled_prov_codes,
        "service_date": service_dates,
    }
)

claims["period"] = pd.to_datetime(claims["service_date"]).dt.strftime("%Y%m")


# Merge Provider context

prov_cols = [
    "PROV_CODE",
    "PROV_NAME",
    "PROVIDER_NETWORK",
    "PROVIDER_PRACTICE",
    "PROVIDER_REGION",
    "PROVIDER_TOWN",
]
prov_keep = [c for c in prov_cols if c in prov.columns]

claims = claims.merge(prov[prov_keep], on="PROV_CODE", how="left")

# Define ER behavior
practice_lower_claims = claims["PROVIDER_PRACTICE"].fillna("").astype(str).str.lower()

is_hospital = practice_lower_claims.str.contains("hospital").astype(int)

# ER only possible if hospital
claims["is_er"] = np.where(
    is_hospital == 1, np.random.binomial(1, ER_PROB_IF_HOSPITAL, size=len(claims)), 0
)

# Admission only possible if ER
claims["admission_flag"] = np.where(
    claims["is_er"] == 1, np.random.binomial(1, ADMIT_PROB_IF_ER, size=len(claims)), 0
)


# Baseline by practice type Gamma for realistic skew
base = np.select(
    [
        practice_lower_claims.str.contains("hospital"),
        practice_lower_claims.str.contains("pharmacy"),
        practice_lower_claims.str.contains("dental"),
        practice_lower_claims.str.contains("clinic"),
    ],
    [
        np.random.gamma(2.0, 320.0, size=len(claims)),  # hospital baseline
        np.random.gamma(2.0, 55.0, size=len(claims)),  # pharmacy
        np.random.gamma(2.0, 130.0, size=len(claims)),  # dental
        np.random.gamma(2.0, 110.0, size=len(claims)),  # clinic
    ],
    default=np.random.gamma(2.0, 110.0, size=len(claims)),
)

er_mult = np.where(claims["is_er"] == 1, ER_MULT, 1.0)
adm_mult = np.where(claims["admission_flag"] == 1, ADM_MULT, 1.0)

claims["net_bill"] = (base * er_mult * adm_mult).round(2)

# Cap extreme outliers for a cleaner prototype distribution
claims["net_bill"] = claims["net_bill"].clip(upper=BILL_CAP)


members.to_csv(OUT_MEMBERS, index=False)
claims.to_csv(OUT_CLAIMS, index=False)

# Quick sanity prints
print("Generated:")
print(" -", OUT_MEMBERS, "shape=", members.shape)
print(" -", OUT_CLAIMS, "shape=", claims.shape)


er_rate = claims["is_er"].mean()
admit_rate_overall = claims["admission_flag"].mean()
admit_rate_given_er = (
    claims.loc[claims["is_er"] == 1, "admission_flag"].mean()
    if (claims["is_er"] == 1).any()
    else 0.0
)

print("\n Rates:")
print(f" - ER rate (overall): {er_rate:.3%}")
print(f" - Admission rate (overall): {admit_rate_overall:.3%}")
print(f" - Admission | ER: {admit_rate_given_er:.3%}")

print("\n net_bill summary:")
print(claims["net_bill"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

print("\nClaims columns:", list(claims.columns))
print("\nSample claims:")
print(claims.head(5).to_string(index=False))
