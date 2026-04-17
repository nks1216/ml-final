# final project

# Topic 1: Predicting HMDA Mortgage Approvals (Kyungsu)

## Introduction

This project utilizes data managed by the Consumer Financial Protection Bureau (CFPB) under the Home Mortgage Disclosure Act (HMDA). HMDA requires nearly all U.S. financial institutions to maintain, report, and publicly disclose loan-level information about mortgages.

## Problem Statement:

Build a classification model to predict loan outcomes (Approved vs. Denied) by leveraging a diverse set of features, including applicant income, loan amount, debt-to-income (DTI) ratio, race, gender, and geographical location.

## Data Scope:

The dataset consists of millions of records annually, providing a robust foundation for large-scale sampling and high-velocity model training.

## Methodology (API Integration):
 
Data is programmatically retrieved through the CFPB HMDA API using the Python requests library, ensuring a reproducible and automated data pipeline.

## Model:

Logistic, Random Forest, XGBoost, CatBoost (new model)

## Predictive Inference:

The analysis also examines which financial indicators—such as DTI, income, and loan amount—most strongly influence approval outcomes, providing insight into how lending decisions align with economic fundamentals.

## Algorithmic Fairness Analysis

Given HMDA’s regulatory purpose, we additionally evaluate whether model performance differs across demographic groups to ensure the model does not inadvertently reflect or amplify existing disparities.

See `hmda_loader.py` and `hmda_raw_2023_TX_48453.csv` for reference.
(I used Travis County (48453) for the initial project setup.)

---