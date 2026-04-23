# Predicting HMDA Mortgage Approvals 

## 1. Project Overview

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

#### Research Questions (To be updated)

#### Motivation (To be updated)

---

## 2. Dataset Description

### 2.1. Data Source

This project utilizes data managed by the Consumer Financial Protection Bureau (CFPB) under the Home Mortgage Disclosure Act (HMDA). HMDA requires nearly all U.S. financial institutions to maintain, report, and publicly disclose loan-level information about mortgages.

Data is programmatically retrieved through the CFPB HMDA API using the Python `requests` library, ensuring a reproducible and automated data pipeline.

The dataset consists of **310,241 HMDA mortgage applications** from Texas’s four largest counties—**Travis (Austin), Harris (Houston), Dallas (Dallas), and Bexar (San Antonio)**.
A total of **109 features** are used to predict a single outcome variable: **loan approval**. 

Due to GitHub’s 100MB file size limit, the raw HMDA dataset is stored externally.

Download the dataset from Google Drive:
https://drive.google.com/drive/folders/1y5r9Sv6s_8ITIrgsAzXTee7e0a_xjZtS?usp=drive_link

Please refer to `data/data_dictionary.md` for detailed descriptions of the raw data variables.

## 2.2. Data Preprocessing: Feature Selection & Transformation

This section summarizes the preprocessing steps applied to reduce the original 109 HMDA variables to a finalized set of 44 modeling features.
The process emphasizes data **leakage prevention, fairness analysis, and model interpretability**.

### 2.2.1. Feature Selection: Excluded Variables

### A. Data Leakage (Internal Decision & Post-Decision Markers)

These variables reflect internal bank processes or information determined after the credit decision. Including them would allow the model to “peek” at the outcome.

• Internal Decisions
  - aus-1 ~ aus-5 (Automated Underwriting System results)
  - initially_payable_to_institution

• Post‑Decision Metadata
  - denial_reason-1 ~ denial_reason-4
  - purchaser_type

• Pricing Metadata (finalized only for approved loans)
  - interest_rate
  - rate_spread
  - total_loan_costs
  - total_points_and_fees
  - origination_charges

• Regulatory Flags
  - hoepa_status

### B. Identifiers and Constants

Variables with no predictive value or extremely high cardinality.

• Constants
  - activity_year (fixed at 2023)
  - state_code (fixed at TX)

• Identifiers
  - lei (Legal Entity Identifier)

• High Cardinality
  - census_tract (replaced by county_code and derived_msa-md)

### C. Redundant and Observational Features

• Raw Demographics (replaced by derived_* variables)
  - applicant_race-1~5
  - applicant_ethnicity-1~5
  - applicant_sex

• Observational Metadata
  - applicant_race_observed
  - co-applicant_race_observed
  - applicant_ethnicity_observed
  - co-applicant_ethnicity_observed

• Binary Age Flags
  - applicant_age_above_62

### 2.2.2. Data Transformation: Numerical & Categorical

### A. Ordinal Mapping (Age)

Age ranges were converted into ordinal integers (0–6) to preserve chronological ordering.

Example:

"25-34" → 1  
"35-44" → 2  
"75+"   → 6

### B. Range-to-Midpoint Conversion (DTI & Units)

HMDA often reports values as ranges for privacy.
These were converted into continuous numerical midpoints.

Examples:

"30%-36%" → 33.0  
"5-24 units" → 14.5

Missing values were imputed using the median.

### C. Categorical Encoding

• String-based variables retained as categorical

• Missing values replaced with "Unknown"

### 2.2.3. Target Variable Refinement

- The `target` variable was derived from `action_taken` to focus exclusively on the institution's credit decision logic:

- Class 1 (Approved): `action_taken` = 1 (Loan originated)

- Class 0 (Denied): `action_taken` = 3 (Application denied)

- Exclusions: Applications that were withdrawn by the applicant (4) or closed for incompleteness (5) were removed to ensure the model only learns from definitive approval or denial outcomes.

---


### 2.3. Variables 

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


### 2.4. Train / Test Split

The cleaned dataset is split into training (80%) and testing (20%) sets using scikit-learn to ensure reliable model evaluation.

- Stratified Sampling: We applied stratification on the target column to maintain the original distribution of approved and denied loans in both sets.

- Reproducibility: A fixed random_state of 42 was used to ensure consistent results across different runs.

- Outputs: The split files are saved as `train.csv` and `test.csv` in the `data/split/` directory.

### 2.5. Data Limitations

---


## 3. Modeling Approach and Individual Model Results

### 3.1. Logistic Regression

#### 1. Performance Summary

| Metric | Value |
|---|---|
| Accuracy | 0.7533 |
| Precision (Approved) | 0.8632 |
| Recall (Approved) | 0.7731 |
| F1 Score | 0.8156 |
| ROC-AUC | 0.8117 |
| Average Precision | 0.9011 |

Evaluated on 39,095 held-out test samples.

#### 2. Confusion Matrix & ROC Curve

| Confusion Matrix | ROC Curve |
|---|---|
| `logistic_confusion_matrix` | `logistic_roc_curve` |

#### 3. Precision–Recall Curve & Feature Importance

| Precision–Recall Curve | Top 20 Coefficients |
|---|---|
| `logistic_precision_recall_curve` | `logistic_top20_coefficients` |

Top 5 drivers (absolute coefficient magnitude):

- multifamily_affordable_units = Exempt
- reverse_mortgage
- open-end_line_of_credit
- applicant_credit_score_type
- co-applicant_credit_score_type

#### 4. Interpretation

Logistic Regression serves as the interpretable baseline model for the mortgage approval classification task. It performs reasonably well overall, especially in predicting approved loans, with strong precision (0.8632) and a solid F1 score (0.8156). However, its performance is weaker for denied loans, which reflects both class imbalance and the limited flexibility of a linear classifier relative to more complex nonlinear models.

The model’s ROC-AUC of 0.8117 shows that it provides meaningful discrimination between approved and denied applications, but it remains below the XGBoost benchmark currently reported in the project. This is expected, since Logistic Regression assumes a linear relationship between predictors and approval probability, while tree-based methods can capture richer interactions and nonlinearities. Still, Logistic Regression is valuable because it is transparent, fast to estimate, and easy to interpret through coefficient signs and magnitudes.

### 3.2. Random Forest (can be changed)

### 3.3. XGBoost

#### 1. Performance Summary

| Metric               | Value  |
|----------------------|:------:|
| Accuracy             | 0.8471 |
| Precision (Approved) | 0.9087 |
| Recall (Approved)    | 0.8709 |
| F1 Score             | 0.8894 |
| ROC-AUC              | 0.9109 |
| Average Precision    | 0.9529 |

_Evaluated on 39,095 held-out test samples._

#### 2. Confusion Matrix & ROC Curve

| Confusion Matrix | ROC Curve |
|:---:|:---:|
| <img width="420" height="350" alt="xgboost_confusion_matrix" src="https://github.com/user-attachments/assets/03bd43bd-fda5-4f97-8469-61d12387da67" /> | <img width="420" height="350" alt="xgboost_roc_curve" src="https://github.com/user-attachments/assets/91f420aa-971f-4a1a-9346-6b3c9cd7dc84" /> |

#### 3. Precision–Recall Curve & Feature Importance

| Precision–Recall Curve | Top 20 Feature Importances |
|:---:|:---:|
| <img width="420" height="350" alt="xgboost_precision_recall_curve" src="https://github.com/user-attachments/assets/91fd70e4-3500-46b3-8c7e-40d64b824f3e" /> | <img width="420" height="350" alt="xgboost_feature_importance_top20" src="https://github.com/user-attachments/assets/7b199875-898d-47ec-88f5-244c4b8ddf87" />|





**Top 5 drivers (gain):**
1. `construction_method`
2. `derived_dwelling_category`
3. `manufactured_home_secured_property_type`
4. `debt_to_income_ratio`
5. `loan_purpose`


### 3.4. CatBoost (new model, can be changed)

## 4. Comparative Evaluation of Models

### 4.1. Performance Metrics (can be changed)

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

## 5. Fairness check

## 6. Reproducibility

### 6.1. Clone the repository  
```
git clone https://github.com/nks1216/ml-final.git
cd ml-final
```

### 6.2. Setting up the Virtual Environment

- Create a virtual environment: `python3 -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install all required packages: `pip install -r requirements.txt`

### 6.3. Data Preparation

Run the scripts in the following order to collect and prepare the dataset:

1. Download Raw Data: Fetches 2023 HMDA data for Big 4 TX counties.

```
python3 src/data/hmda_loader.py
```

2. Clean Data: Performs preprocessing and prevents data leakage.

```
python3 src/data/clean_hmda.py
```

3. Split Data: Creates train.csv and test.csv in data/split/.

```
python3 src/data/split_data.py
```

### 6.4 Run the code
```

```
For convenience, individual model scripts are provided.

---

## 7. Limitations and Future Improvements

**Modeling Limitations**

**Future Improvements**

---

## 8. Collaboration and Workflow

- All team members worked through GitHub Issues and feature branches, following a branch‑per‑issue workflow.
- Each member opened pull requests for their work and merged them after review and testing.
- The repository contains more than 30 commits across multiple contributors.
- All code and documentation were merged into the main branch before submission.


## Further Feasible Extensions:

- Data quality and missingness analysis

- Fairness metrics (e.g., demographic parity, equal opportunity)

- Explainability using SHAP or permutation importance

- County‑level heterogeneity analysis

- Class imbalance handling

- Train/validation/test split strategy
