# Topic 1: Predicting HMDA Mortgage Approvals (Kyungsu)

## Project Scope:

- **Develop four classification models** to predict mortgage loan outcomes (Approved vs. Denied) using a comprehensive set of applicant, loan, and neighborhood-level features.
The models include:

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


## Key Variable Groups (Representative subset of the 109 features)

### 1. Financial Variables

| Variable | Definition |
|---------|------------|
| `loan_amount` | Dollar amount of the loan applied for. |
| `loan_to_value_ratio` | Ratio of loan amount to property value (LTV). |
| `debt_to_income_ratio` | Applicant’s monthly debt payments divided by monthly income (DTI). |
| `interest_rate` | Interest rate assigned to the loan. |
| `loan_term` | Length of the loan in months. |
| `property_value` | Estimated market value of the property securing the loan. |

### 2. Applicant Characteristics

| Variable | Definition |
|---------|------------|
| `income` | Applicant’s annual income used for underwriting. |
| `applicant_age` | Age of the primary applicant. |
| `applicant_race` | Self‑reported race of the applicant. |
| `applicant_ethnicity` | Self‑reported ethnicity of the applicant. |
| `applicant_sex` | Self‑reported sex of the applicant. |
| `applicant_credit_score_type` | Type of credit score model used (e.g., FICO, VantageScore). |

### 3. Geographic & Neighborhood Variables

| Variable | Definition |
|---------|------------|
| `census_tract` | Census tract identifier for the property location. |
| `derived_msa-md` | CFPB‑derived Metropolitan Statistical Area / Metropolitan Division code. |
| `ffiec_msa_md_median_family_income` | Median family income for the MSA/MD. |
| `tract_minority_population_percent` | Percentage of minority residents in the census tract. |

### 4. Loan Type & Purpose

| Variable | Definition |
|---------|------------|
| `loan_type` | Type of loan (Conventional, FHA, VA, USDA). |
| `loan_purpose` | Purpose of the loan (Home purchase, Refinance, Home improvement). |
| `lien_status` | Indicates whether the loan is a first lien, subordinate lien, or unsecured. |

### 5. Application Process Variables

| Variable | Definition |
|---------|------------|
| `preapproval` | Indicates whether the applicant sought preapproval. |
| `aus-1` | Result from the primary Automated Underwriting System (e.g., Approve/Eligible). |

### Target Variable

| Variable | Definition |
|---------|------------|
| `action_taken` | Outcome of the loan application, recoded as **Approved** vs. **Denied**. |


---

# Topic 2: Predicting U.S. Startup Failure (Razan) 

## Introduction

This project uses two Crunchbase datasets sourced from Kaggle to predict whether a U.S. startup will fail based on its funding profile, sector, and geography. Crunchbase is the leading platform for tracking startup activity, funding rounds, investor relationships, and company outcomes. Both datasets are publicly available, crowdsourced, and editor-verified by Crunchbase. The data covers companies founded between 2000 and 2014.

## Problem Statement

Build a classification model to predict startup failure (closed vs. operating) using funding history, investor type, sector, and geographic features.

## Data Scope

Two Crunchbase datasets joined on `permalink` — a unique company identifier assigned by Crunchbase (e.g. `/organization/uber`). The join matches 98.5% of records with no manual modifications.

| Dataset | Source | Records | Kaggle URL |
|---|---|---|---|
| Startup Success/Fail | Crunchbase via Kaggle | ~66,000 global | `yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase` |
| StartUp Investments | Crunchbase via Kaggle | ~54,000 global | `arindam235/startup-investments-crunchbase` |

**After joining and filtering to U.S. only: 29,696 companies, 2,302 confirmed failures (7.8%)**

### Key Variable Groups

## 1. Funding Variables

| Variable | Definition |
|---|---|
| `funding_total_usd` | Total funding raised across all rounds (USD) |
| `funding_rounds` | Number of distinct funding rounds raised |
| `venture` | Amount raised from venture capital (USD) |
| `angel` | Amount raised from angel investors (USD) |
| `seed` | Amount raised in seed round (USD) |
| `round_A` through `round_D` | Amount raised in each specific round (USD) |
| `debt_financing` | Amount raised through debt (USD) |
| `grant` | Amount raised through grants (USD) |

---

## 2. Timeline Variables

| Variable | Definition |
|---|---|
| `founded_at` | Date the company was founded |
| `first_funding_at` | Date of first external funding round |
| `last_funding_at` | Date of most recent funding round |
| `funding_gap` | Derived: months between first and last funding (runway proxy) |
| `time_to_first_funding` | Derived: months from founding to first funding |

---

## 3. Company Characteristics

| Variable | Definition |
|---|---|
| `category_list` | Startup sector (SaaS, Social Media, Biotech, Games, etc.) |
| `state_code` | U.S. state of incorporation |
| `region` | U.S. region (e.g. SF Bay Area, New York) |

---

## 4. Target Variable

| Variable | Definition |
|---|---|
| `status` | Outcome — recoded as **closed** (failed) vs. **operating** (survived) |

---

## What the Data Tells Us

Before modeling, the data reveals clear patterns:

**Highest failure rates by sector (min. 100 companies):**

| Sector | Failure Rate |
|---|---|
| Curated Web | 21.1% |
| Social Media | 19.1% |
| Public Relations | 17.1% |
| Messaging | 16.4% |
| Games | 15.3% |

**Lowest failure rates:**

| Sector | Failure Rate |
|---|---|
| Medical | 0.6% |
| Real Estate | 1.2% |
| EdTech | 2.4% |
| B2B | 2.6% |

**Funding type signal (failed vs. surviving):**

| Type | Failed avg | Surviving avg |
|---|---|---|
| Seed | $164,212 | $253,789 |
| Venture | $6,727,887 | $7,261,590 |
| Angel | $85,385 | $52,738 |

Notable: failed startups raised *more* from angel investors on average, suggesting angel-only funding may be a risk signal rather than a safety net.

## Methodology

Data is downloaded programmatically via the Kaggle API using `setup.sh`, ensuring a fully reproducible and automated data pipeline. Both datasets are joined in `data_cleaning.py` with no manual modifications.

## Class Imbalance

7.8% failure rate, a model that always predicts "operating" would be 92% accurate and useless. Addressed with:
- Class weights (`scale_pos_weight` in XGBoost/CatBoost)
- SMOTE oversampling
- Evaluated using precision-recall curve, not accuracy

## Models

(1) Logistic Regression (baseline)

(2) XGBoost

(3) CatBoost (new model, handles categorical features natively)

(4) Best model + Optuna Bayesian hyperparameter tuning (new technique)

## Predictive Inference

The analysis examines which signals most strongly influence failure. specifically whether **funding type, number of rounds, and sector** matter more than raw dollar amounts — providing insight into what distinguishes startups that survive from those that close.

## Algorithmic Fairness Analysis

We evaluate whether model predictions differ systematically across **sector, geographic region (coastal vs. non-coastal), and funding type (VC-backed vs. angel-only)** to ensure the model does not amplify existing ecosystem biases.

## Data Limitations

- **Survivorship bias:** Only funded startups appear, bootstrapped companies are invisible
- **Noisy labels:** Many closed startups never updated their Crunchbase profile, some `operating` labels are likely dead companies
- **No timing on failure:** The data captures whether a company failed, not when, we cannot measure failure within a specific time window
- **No founder data:** Founder experience, demographics, and prior exits unavailable
- **No revenue or traction data:** Product-market fit signals absent from both datasets
- **Founded date missing** for 16.5% of records, limits time-based feature engineering for those rows
- **Time period:** Covers 2000–2014, may not reflect post-2020 market dynamics

----

# Topic 3: Predicting Consumer Relief in CFPB Complaint Outcomes (Omar)

## Introduction

This project uses the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database to predict whether a consumer complaint ends with relief. The dataset contains public complaints about consumer financial products and services and includes information on the product, issue, company, geography, submission channel, timing, and company response.

## Problem Statement

Build a classification model to predict whether a consumer complaint results in relief using complaint characteristics such as product type, issue, company, state, submission method, timing, and response-related variables.

## Data Scope

The project uses the CFPB Consumer Complaint Database, an official public dataset of complaint-level observations. The database supports filtering and export, making the project fully reproducible with a programmatic data pipeline.

### Key Variable Groups

## 1. Complaint Characteristics

| Variable | Definition |
|---|---|
| `product` | Consumer financial product involved in the complaint |
| `sub_product` | More specific product category when available |
| `issue` | Main issue identified by the consumer |
| `sub_issue` | More detailed issue category when available |
| `complaint_what_happened` | Narrative text of the complaint, when provided |

---

## 2. Company / Response Variables

| Variable | Definition |
|---|---|
| `company` | Company named in the complaint |
| `company_response_to_consumer` | Company response category |
| `timely_response` | Whether the company responded on time |
| `company_public_response` | Public-facing company response, if available |

---

## 3. Geography / Submission Variables

| Variable | Definition |
|---|---|
| `state` | State of the consumer |
| `submitted_via` | Channel used to submit the complaint |
| `date_received` | Date complaint was received |
| `date_sent_to_company` | Date complaint was sent to the company |

---

## 4. Derived Timing / Text Variables

| Variable | Definition |
|---|---|
| `year_month` | Derived complaint month for time patterns |
| `narrative_length` | Derived word or character length of complaint narrative |
| `has_narrative` | Indicator for whether narrative text is present |

---

## Target Variable

| Variable | Definition |
|---|---|
| `relief` | Outcome recoded as **relief** (closed with monetary relief or non-monetary relief) vs. **no relief** (all other outcomes) |

## What the Data Tells Us

Before modeling, the data can reveal clear patterns such as:

- which financial products are most likely to end with relief  
- whether certain issue categories are more associated with monetary or non-monetary relief  
- whether response patterns differ by company, state, or submission channel  
- whether complaints with narratives behave differently from complaints without narratives  

These patterns help motivate the final feature set and provide context for the classification results.

## Methodology

Data will be downloaded programmatically and cleaned entirely through code, with no manual modifications. The analysis will focus on predicting relief outcomes using structured complaint information and, where available, text-based features from complaint narratives.

## Class Imbalance

Relief outcomes are less common than non-relief outcomes, so a model that predicts only "no relief" could still appear accurate while being useless in practice. This will be addressed with:

- Class weights
- Threshold tuning
- Evaluation using precision-recall metrics, not accuracy alone

## Models

(1) Logistic Regression (baseline)

(2) Random Forest

(3) XGBoost (new model)

(4) Best model + SHAP or threshold tuning (new technique)

## Predictive Inference

The analysis examines which signals most strongly influence whether a complaint ends with relief, specifically whether **product type, issue category, company, submission channel, and timing** matter more than geography alone.

## Algorithmic Fairness / Group Analysis

We evaluate whether model predictions differ systematically across **product groups, states, and submission channels** to ensure the model does not amplify existing disparities in complaint handling outcomes.

## Data Limitations

- **Outcome limitation:** Complaint outcomes do not perfectly measure true consumer harm or true case merit
- **Reporting limitation:** Company response categories may reflect reporting practices as well as complaint severity
- **Missing text:** Not all complaints include narratives, limiting text-based analysis
- **Selection bias:** The data represents submitted complaints, not the full universe of consumer financial problems
- **Class imbalance:** Relief outcomes are less common than non-relief outcomes
- **Unobserved variables:** Important factors such as internal company processes, legal context, or case complexity are not observed in the data

---

# Topic 4:

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

----



# Topic 3 : Predicting Hacker News Virality

Hacker News (HN) is a social news aggregator run by Y Combinator, focused on technology, startups, and programming. Since its launch in 2007, it has become one of the most influential online communities in the tech industry, where a submission reaching the front page can drive substantial traffic, career opportunities, and even funding interest for its author. 

## 1. Research Question

We frame the task as a **binary classification problem**:

> Given only the information available at the moment of submission, can we predict whether a Hacker News story will reach **100 or more points within 24 hours** of being posted?

The 100-point threshold approximates the score typically required to appear on the HN front page, which serves as the practical definition of "going viral" in this community.

- Which features most strongly influence virality (title wording, linked domain, submission time, author reputation)?

## 2. Data Collection

All data will be sourced exclusively from official Hacker News channels to ensure clear provenance and reproducibility.

**Google BigQuery Public Dataset**
- Table: `bigquery-public-data.hacker_news.full`
- Contains every item (story, comment, poll, job) from 2007 to the present, maintained by Google Cloud
- We will extract stories submitted between 2022 and 2025, excluding deleted or dead posts
- Expected sample size: approximately 400,000–600,000 stories

**Fields collected**
- `id`, `title`, `url`, `text`, `by`, `time`, `type`
- `score` and `descendants` (used for labeling, collected 24+ hours after submission)
- Author metadata — `karma` and account creation time — queried via the `/user/` endpoint

## 3. Methodology

### 3.1 Feature Engineering
- **Title features**: length, presence of numbers or special characters, "Show HN:" / "Ask HN:" prefix, and a set of trending keyword flags (e.g., *AI*, *Rust*, *GPT*)
- **URL features**: top-level domain, historical mean score of that domain
- **Temporal features**: hour of day (UTC and US Pacific), day of week, year
- **Author features**: historical karma at submission time, account age, prior submission count and average score

### 3.2 Models

- Logistic Regression with L1 and L2 regularization
- Random Forest
- Gradient Boosting with **LightGBM**, tuned using **Optuna** Bayesian hyperparameter search
- Title embeddings produced by **Sentence-BERT** (`all-MiniLM-L6-v2`), stacked with tabular features
- Probability calibration via **isotonic regression** on a held-out set

### 3.3 Evaluation
- **Temporal split**: training on 2022–2024, validation on early 2025, testing on late 2025. A random split would leak future information and inflate performance
- **Metrics**: ROC-AUC and PR-AUC (given class imbalance), plus F1 at an operating threshold selected through cost-sensitive analysis
- **Calibration quality**: Brier score and calibration curves
