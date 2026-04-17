# ml-final
ml-final


Sample Project Topic 1: Predicting HMDA Mortgage Approvals

Introduction

This project utilizes data managed by the Consumer Financial Protection Bureau (CFPB) under the Home Mortgage Disclosure Act (HMDA). HMDA requires nearly all U.S. financial institutions to maintain, report, and publicly disclose loan-level information about mortgages.

Core Components

Problem Statement:

Build a classification model to predict loan outcomes (Approved vs. Denied) by leveraging a diverse set of features, including applicant income, loan amount, debt-to-income (DTI) ratio, race, gender, and geographical location.

Data Scope:

The dataset consists of millions of records annually, providing a robust foundation for large-scale sampling and high-velocity model training.

Methodology (API Integration):

Data is programmatically retrieved through the CFPB HMDA API using the Python requests library, ensuring a reproducible and automated data pipeline.

Analytical Value & Research Objectives

Predictive Inference:

Beyond simple accuracy, this project focuses on identifying which financial indicators—such as the Debt-to-Income (DTI) ratio—exert the most significant influence on approval decisions from a commercial banking perspective.

Algorithmic Fairness Analysis:

Aligned with the "Theory Project" track, we will conduct a rigorous investigation into model fairness. This involves auditing the model for potential predictive biases across demographic groups (e.g., race and gender) to ensure the model does not reinforce systemic disparities.