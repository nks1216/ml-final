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