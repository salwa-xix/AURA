# AURA: Early Warning & Utilization Intelligence System

## Project Overview
**AURA** is an end-to-end Early Warning & Utilization Intelligence System designed to support healthcare payers in identifying, explaining, and predicting avoidable Emergency Room (ER) utilization at the corporate client level.

The system combines:

- Structured KPI engineering
- An explainable Utilization Index (IVI)
- Predictive machine learning
- Driver analysis
- Interactive executive dashboard

All data in this prototype is synthetic, generated to replicate realistic utilization behavior while preserving privacy.

**AURA** introduces a forward-looking early warning framework that enables:

- Monthly monitoring
- Risk stratification
- Predictive escalation
- Action-oriented insights

## Pipeline Explanation 

### 1. Provider Preparation
**File:** `00_prepare_provider.py`  
**Objective:** Clean and standardize the provider registry used across all claims.  
**Output:** `provider_clean.csv`  

**Key Steps:**
- Validate required provider fields
- Remove whitespace and malformed values
- Deduplicate providers using `PROV_CODE`

---

### 2. Synthetic Members & Claims Generation
**File:** `01_generate_synth_claims.py`  
**Outputs:** `members_synth.csv`, `claims_synth.csv`  

**Member Generation:**
- Companies assigned unique IDs
- Realistic membership sizes
- Member attributes: age, gender, join date

**Claims Generation:**
- Claims sampled across providers
- Frequency per member reflects real-world patterns
- ER visits only possible at hospital providers
- Admission only possible following ER visit

**Cost Modeling:**
- Gamma distribution for cost skew
- Multipliers for ER & admissions
- Caps to prevent extreme outliers

---

### 3. Company Monthly KPIs
**File:** `02_build_company_kpis.py`  
**Outputs:** `company_monthly_kpis.csv`, `avoidable_er_segments.csv`  

**Key KPIs:**
- Total claims & cost
- ER visits & avoidable ER visits
- Avoidable ER cost  

**Avoidable ER Definition:**  
ER visit with no hospital admission (proxy for non-urgent usage)  

**Normalized Metrics:**  
- ER visits per 100 members
- Avoidable ER rates
- Cost per member metrics  

**Segmentation:** By region, town, network, practice, provider

---

### 4. IVI Scoring Model
**File:** `03_compute_scores.py`  
**Output:** `company_monthly_scores.csv`  

**IVI — Utilization Intelligence Index:**  
A single interpretable score (0–100) representing utilization health.  


**Risk Bands:**
- IVI < 55 → High Risk
- IVI 55–70 → Medium Risk
- IVI ≥ 70 → Low Risk

---

### 5. Predictive Modeling (AI)
**File:** `04_predict_next_month_risk.py`  
**Output:** `company_risk_predictions.csv`  

**Objective:** Predict whether a company will become High Risk next month.  

**Model:** Logistic Regression (explainable and stable)  
**Features:** Utilization KPIs, normalized intensity indicators, member size  
**Target:** Next-month IVI band (binary: 1 = High Risk, 0 = Otherwise)

---

### 6. Dashboard (Streamlit)
**File:** `app.py`  

**Purpose:** Convert analytics into decision-ready insights.  

**Views:**
- **Executive Overview:** IVI trend, current utilization snapshot, next-month risk probability, recommended actions
- **Company Deep Dive:** ER intensity trends, cost evolution, avoidable vs admitted mix, top providers
- **Portfolio Monitor:** Risk distribution, priority escalation list
- **Prediction Center:** AI-based next-month risk ranking and drivers

**Design Principles:**
- Explainability first
- Fair normalization across company sizes
- Time-aware prediction
- Action-oriented outputs

---

## Technologies Used
- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Plotly
- Synthetic data modeling

---

## Disclaimer
All data used in this prototype is synthetic and does not represent any real individuals or companies.



