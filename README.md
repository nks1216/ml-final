# Topic 1: Predicting HMDA Mortgage Approvals (Kyungsu Noh)

## Project Scope:

- **Develop four classification models** to predict mortgage loan outcomes (Approved vs. Denied) using a comprehensive set of applicant, loan, and neighborhood-level features.
The models include: (It can be changed.)

    (1) Logistic Regression (baseline)

    (2) Random Forest

    (3) XGBoost

    (4) CatBoost (new model)

- **Compare model performance** using appropriate evaluation metrics (e.g., precision, recall, F1-score, ROC-AUC) and address class imbalance if necessary.

- **Identify key predictors** that most strongly influence loan approval outcomes through model-based feature importance and interpretability tools.

- **Assess demographic disparities** by evaluating whether model performance differs across groups defined by race, ethnicity, gender, or age (applied to the best-performing model).

- **Examine geographic variation** by comparing approval patterns and model behavior across the Big Four Texas counties (Travis, Harris, Dallas, Bexar), using the best-performing model.

## Further Feasible Extensions:

- Data quality and missingness analysis

- Fairness metrics (e.g., demographic parity, equal opportunity)

- Explainability using SHAP or permutation importance

- County‑level heterogeneity analysis

- Class imbalance handling

- Train/validation/test split strategy

## Data Scope:

This project utilizes data managed by the Consumer Financial Protection Bureau (CFPB) under the Home Mortgage Disclosure Act (HMDA). HMDA requires nearly all U.S. financial institutions to maintain, report, and publicly disclose loan-level information about mortgages.

Data is programmatically retrieved through the CFPB HMDA API using the Python `requests` library, ensuring a reproducible and automated data pipeline.

The dataset consists of **310,241 HMDA mortgage applications** from Texas’s four largest counties—**Travis (Austin), Harris (Houston), Dallas (Dallas), and Bexar (San Antonio)**.
A total of **109 features** are used to predict a single outcome variable: **loan approval**.

See `data/hmda_loader.py` for reference.

## Data Access

Due to GitHub’s 100MB file size limit, the raw HMDA dataset is stored externally.

Download the dataset from Google Drive:
https://drive.google.com/drive/folders/1y5r9Sv6s_8ITIrgsAzXTee7e0a_xjZtS?usp=drive_link


## Data Preprocessing: Feature Selection & Transformation

To ensure the integrity of the machine learning models and prevent **Data Leakage**, we performed a rigorous feature selection process, reducing the initial **109 variables** to a finalized set of **44 variables (including the target variable)**. This refined dataset prioritizes primary applicant characteristics and economic indicators over internal bank markers or redundant metadata, ensuring the model's predictive power is both robust and ethically sound for fairness analysis.

### 1. Feature Selection: Excluded Variables

The following features were excluded from the training dataset for specific technical and analytical reasons:

### A. Data Leakage (Internal Decision & Post-Decision Markers)

These features contain information determined during or after the loan decision process. Including them would lead to "Data Leakage," where the model "cheats" by looking at the results of internal algorithms rather than applicant qualifications.

Internal Decisions: aus-1 to aus-5 (Automated Underwriting System results) and initially_payable_to_institution. These reflect the bank's internal AI/process conclusions.

Post-Decision Metadata: denial_reason-1 to denial_reason-4 and purchaser_type.

Pricing Metadata: interest_rate, rate_spread, total_loan_costs, total_points_and_fees, and origination_charges. These are typically finalized only for approved loans.

Regulatory Flags: hoepa_status, which is determined based on the final APR.

### B. Identifiers and Constants

Variables that provide no predictive power or possess extremely high cardinality.

Constants: activity_year (fixed at 2023) and state_code (fixed at TX).

Identifiers: lei (Legal Entity Identifier).

High Cardinality: census_tract was excluded to prevent overfitting; regional trends are instead captured through county_code and derived_msa-md (focused on 4 major Texas counties).

### C. Redundant and Observational Features

To reduce multi-collinearity and focus on "Fairness" analysis, redundant features were removed.

Raw Demographics: applicant_race-1~5, applicant_ethnicity-1~5, and applicant_sex were removed in favor of the CFPB's "derived" versions (derived_race, derived_ethnicity, derived_sex).

Observational Metadata: _observed flags (e.g., applicant_race_observed) were removed as they describe the collection method rather than applicant creditworthiness.

Binary Age Flags: applicant_age_above_62 was removed because the binned applicant_age provides more granular information.

### 2. Data Transformation: Numerical & Categorical

We transformed the raw data into a format suitable for both traditional statistical models (Logistic Regression) and gradient boosting models (CatBoost, XGBoost).

### A. Ordinal Mapping (Age)

To capture the chronological trend of aging, we transformed the categorical age ranges into Ordinal Integers.

Mapping: Categories like "25-34", "35-44", etc., were mapped to a scale of 0 to 6. This allows the model to treat age as a ranked numerical value.

### B. Range-to-Midpoint Conversion (DTI & Units)

HMDA often provides data in ranges to protect privacy. We converted these into continuous numerical values.

Method: Range strings (e.g., "30%-36%", "5-24 units") were converted to their mathematical midpoints (e.g., 33.0, 14.5).

Imputation: Missing numerical values were imputed using the Median to maintain distribution robustness.

### C. Categorical Encoding

Remaining string-based features (e.g., loan_purpose, derived_race) were kept as categorical.

Missing categorical values were filled with a dedicated "Unknown" label to treat missingness as a potential signal.

### 3. Target Variable Refinement

The target variable was derived from action_taken to focus exclusively on the institution's credit decision logic:

Class 1 (Approved): action_taken = 1 (Loan originated)

Class 0 (Denied): action_taken = 3 (Application denied)

Exclusions: Applications that were withdrawn by the applicant (4) or closed for incompleteness (5) were removed to ensure the model only learns from definitive approval or denial outcomes.

---

### Target Variable (1)

| Variable Name | Description |
|---------------|-------------|
| `target` | Final outcome of the application (1: Approved, 0: Denied) |

### 1. Institutional & Geographic Metadata (5)

| Variable Name | Description |
|---------------|-------------|
| `derived_msa-md` | Derived Metropolitan Statistical Area or Division code |
| `county_code` | Five-digit FIPS county code |
| `conforming_loan_limit` | Whether the loan amount is within GSE limits |
| `derived_loan_product_type` | Derived categorization of the loan product |
| `derived_dwelling_category` | Derived categorization of the dwelling type |

### 2. Loan Application Details (10)

| Variable Name | Description |
|---------------|-------------|
| `preapproval` | Whether pre-approval was requested |
| `loan_type` | Conventional, FHA, VA, or RHS/FSA |
| `loan_purpose` | Purchase, Refinance, Home Improvement, etc. |
| `lien_status` | First or subordinate lien |
| `reverse_mortgage` | Reverse mortgage flag |
| `open-end_line_of_credit` | HELOC/open-end credit flag |
| `business_or_commercial_purpose` | Whether the loan is for business purposes |
| `loan_amount` | Requested or originated loan amount (Numeric) |
| `loan_to_value_ratio` | Loan-to-Value ratio (Numeric) |
| `loan_term` | Loan maturity in months (Numeric) |

### 3. Pricing & Property Features (6)

| Variable Name | Description |
|---------------|-------------|
| `negative_amortization` | Negative amortization flag |
| `interest_only_payment` | Interest-only payment flag |
| `balloon_payment` | Balloon payment flag |
| `other_nonamortizing_features` | Other non-standard payment features |
| `property_value` | Appraised property value (Numeric) |
| `construction_method` | Site-built or manufactured |

### 4. Property & Occupancy (6)

| Variable Name | Description |
|---------------|-------------|
| `occupancy_type` | Primary, secondary, or investment |
| `manufactured_home_secured_property_type` | Security type for manufactured homes |
| `manufactured_home_land_property_interest` | Land property interest |
| `total_units` | Number of dwelling units (Numeric) |
| `multifamily_affordable_units` | Affordable units (for multifamily) |
| `income` | Applicant(s) gross annual income (Numeric) |

### 5. Credit & Submission Metrics (4)

| Variable Name | Description |
|---------------|-------------|
| `debt_to_income_ratio` | Debt-to-Income ratio (Numeric Midpoint) |
| `applicant_credit_score_type` | Credit score model used for the applicant |
| `co-applicant_credit_score_type` | Credit score model used for the co-applicant |
| `submission_of_application` | Whether submitted directly to the institution |

### 6. Applicant Demographics (Fairness Variables) (5)

| Variable Name | Description |
|---------------|-------------|
| `derived_ethnicity` | Aggregate ethnicity of the applicant |
| `derived_race` | Aggregate race of the applicant |
| `derived_sex` | Aggregate sex of the applicant |
| `applicant_age` | Applicant age range (Mapped to Ordinal 0–6) |
| `co-applicant_age` | Co-applicant age range (Mapped to Ordinal 0–6) |

### 7. Census Tract Demographics (7)

| Variable Name | Description |
|---------------|-------------|
| `tract_population` | Total population in the census tract |
| `tract_minority_population_percent` | Minority population percentage in the tract |
| `ffiec_msa_md_median_family_income` | MSA median family income |
| `tract_to_msa_income_percentage` | Tract income relative to MSA median |
| `tract_owner_occupied_units` | Owner-occupied units in the tract |
| `tract_one_to_four_family_homes` | 1–4 family homes in the tract |
| `tract_median_age_of_housing_units` | Median age of housing units in the tract |

---



(readme draft)

## 1. Project Overview

#### Research Questions

#### Motivation

## 2. Dataset Description

### 2.1. Data Source

### 2.2 Variables

### 2.3. Train / Test Split

### 2.4. Data Limitations

## 3. Modeling Approach and Individual Model Results

### 3.1. Logistic (can be changed)

### 3.2. Random Forest (can be changed)

### 3.3. XGBoost (new model, can be changed)

### 3.4. CatBoost (new model, can be changed)

## 4. Comparative Evaluation of Models

### 4.1. Performance Metrics

- Accuracy  
- Precision  
- Recall  
- F1‑Score  
- ROC‑AUC  
- Confusion Matrix  

### 4.2. Actual vs Predicted

- Confusion Matrix (per model)  
- ROC Curve Comparison  
- Precision–Recall Curve Comparison  

### 4.3. Top 5 Feature Importances

- Logistic Regression: absolute coefficients  
- Random Forest: Gini importance  
- XGBoost / CatBoost: gain or split importance  

### 4.4. Top 20 Feature Importances

- Based on the best-performing model  
- (Optional) SHAP summary plot for interpretability  

### 4.5. Key Takeaways and Recommendations

- Comparative performance summary  
- Most influential predictors  
- Practical and fairness-related implications  

## 5. Reproducibility

### 5.1. Clone the repository  
```
git clone https://github.com/nks1216/ml-final.git
cd ml-final
```

### 5.2. Setting up the Virtual Environment

- Create a virtual environment: `python3 -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install all required packages: `pip install -r requirements.txt`

### 5.3. Download the data 

### 5.4 Run the code
```

```
For convenience, individual model scripts are provided.

---

## 6. Limitations and Future Improvements

**Modeling Limitations**

**Future Improvements**

---

## 7. Collaboration and Workflow

- All team members worked through GitHub Issues and feature branches, following a branch‑per‑issue workflow.
- Each member opened pull requests for their work and merged them after review and testing.
- The repository contains more than 30 commits across multiple contributors.
- All code and documentation were merged into the main branch before submission.

