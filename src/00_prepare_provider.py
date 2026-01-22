import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
IN_PATH  = DATA_DIR / "Provider_Info.xlsx"
OUT_PATH = DATA_DIR / "provider_clean.csv"

df = pd.read_excel(IN_PATH)

# ---- expected columns ---
needed = [
    "PROV_CODE",
    "PROV_NAME",
    "PROVIDER_NETWORK",
    "PROVIDER_PRACTICE",
    "PROVIDER_REGION",
    "PROVIDER_TOWN",
]

missing = [c for c in needed if c not in df.columns]
if missing:
    raise Exception(f"Missing columns in Provider_Info.xlsx: {missing}\nFound columns: {list(df.columns)}")

# --- keep only needed + clean -----
prov = df[needed].copy()

# ---- strip spaces + fill missing
for c in needed:
    prov[c] = prov[c].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})

# ----- remove duplicates 
prov = prov.drop_duplicates(subset=["PROV_CODE"]).reset_index(drop=True)

prov.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("shape:", prov.shape)
print(prov.head(5).to_string(index=False))
