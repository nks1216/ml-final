# Topic 1: Predicting HMDA Mortgage Approvals (Kyungsu)

## Introduction

This project utilizes data managed by the Consumer Financial Protection Bureau (CFPB) under the Home Mortgage Disclosure Act (HMDA). HMDA requires nearly all U.S. financial institutions to maintain, report, and publicly disclose loan-level information about mortgages.

## Problem Statement:

Build a classification model to predict loan outcomes (Approved vs. Denied) by leveraging a diverse set of features, including applicant income, loan amount, debt-to-income (DTI) ratio, race, gender, and geographical location.

## Data Scope:

The dataset consists of millions of records annually, providing a robust foundation for large-scale sampling and high-velocity model training.

### Key Variable Groups 

## 1. Financial Variables

| Variable | Definition |
|---------|------------|
| loan_amount | Dollar amount of the loan applied for. |
| loan_to_value_ratio | Ratio of loan amount to property value (LTV). |
| debt_to_income_ratio | Applicant’s monthly debt payments divided by monthly income (DTI). |
| interest_rate | Interest rate assigned to the loan. |
| loan_term | Length of the loan in months. |
| property_value | Estimated market value of the property securing the loan. |

---

## 2. Applicant Characteristics

| Variable | Definition |
|---------|------------|
| income | Applicant’s annual income used for underwriting. |
| applicant_age | Age of the primary applicant. |
| applicant_race | Self‑reported race of the applicant. |
| applicant_ethnicity | Self‑reported ethnicity of the applicant. |
| applicant_sex | Self‑reported sex of the applicant. |
| applicant_credit_score_type | Type of credit score model used (e.g., FICO, VantageScore). |

---

## 3. Geographic & Neighborhood Variables

| Variable | Definition |
|---------|------------|
| census_tract | Census tract identifier for the property location. |
| derived_msa-md | CFPB‑derived Metropolitan Statistical Area / Metropolitan Division code. |
| ffiec_msa_md_median_family_income | Median family income for the MSA/MD. |
| tract_minority_population_percent | Percentage of minority residents in the census tract. |

---

## 4. Loan Type & Purpose

| Variable | Definition |
|---------|------------|
| loan_type | Type of loan (Conventional, FHA, VA, USDA). |
| loan_purpose | Purpose of the loan (Home purchase, Refinance, Home improvement). |
| lien_status | Indicates whether the loan is a first lien, subordinate lien, or unsecured. |

---

## 5. Application Process Variables

| Variable | Definition |
|---------|------------|
| preapproval | Indicates whether the applicant sought preapproval. |
| aus-1 | Result from the primary Automated Underwriting System (e.g., Approve/Eligible). |

---

## Target Variable

| Variable | Definition |
|---------|------------|
| action_taken | Outcome of the loan application, recoded as **Approved** vs. **Denied**. |

## Methodology (API Integration):
 
Data is programmatically retrieved through the CFPB HMDA API using the Python requests library, ensuring a reproducible and automated data pipeline.

## Model:

(1) Logistic

(2) Random Forest

(3) XGBoost (new model)

(4) CatBoost (new model)

## Predictive Inference:

The analysis also examines which financial indicators—such as **DTI, income, and loan amount—most strongly influence approval outcomes**, providing insight into how lending decisions align with economic fundamentals.

## Algorithmic Fairness Analysis

Given HMDA’s regulatory purpose, we additionally evaluate whether **model performance differs across demographic groups** to ensure the model does not inadvertently reflect or amplify existing disparities.

See `hmda_loader.py` and `hmda_raw_2023_TX_48453.csv` for reference.

I only used **Travis County (48453) in Texas** for the initial project setup. (48,254 results)

---

# Topic 2: 

---

# Topic 3: 

---

# Topic 4: 

---


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