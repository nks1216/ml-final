# HMDA Full Data Dictionary (109 Variables)

## 1. Institutional & Geographic Metadata (11)

| Variable Name | Description |
|---------------|-------------|
| `activity_year` | The calendar year the data covers |
| `lei` | Legal Entity Identifier of the financial institution |
| `derived_msa-md` | Derived Metropolitan Statistical Area or Division code |
| `state_code` | Two-letter postal abbreviation for the state |
| `county_code` | Five-digit FIPS county code |
| `census_tract` | 11-digit census tract identifier |
| `conforming_loan_limit` | Whether the loan amount is within GSE limits |
| `derived_loan_product_type` | Derived categorization of the loan product |
| `derived_dwelling_category` | Derived categorization of the dwelling type |
| `action_taken` | Final outcome of the application |
| `purchaser_type` | Entity that purchased the loan if sold |

## 2. Loan Application Details (10)

| Variable Name | Description |
|---------------|-------------|
| `preapproval` | Whether pre-approval was requested |
| `loan_type` | Conventional, FHA, VA, or RHS/FSA |
| `loan_purpose` | Purchase, Refinance, Home Improvement, etc. |
| `lien_status` | First or subordinate lien |
| `reverse_mortgage` | Reverse mortgage flag |
| `open-end_line_of_credit` | HELOC/open-end credit flag |
| `business_or_commercial_purpose` | Whether the loan is for business purposes |
| `loan_amount` | Requested or originated loan amount |
| `combined_loan_to_value_ratio` | CLTV ratio |
| `loan_term` | Loan maturity in months |

## 3. Pricing, Costs & Amortization (16)

| Variable Name | Description |
|---------------|-------------|
| `interest_rate` | Annual interest rate |
| `rate_spread` | APR minus benchmark rate |
| `hoepa_status` | High-Cost Mortgage status |
| `total_loan_costs` | Total loan costs (fees + points) |
| `total_points_and_fees` | Total points and fees |
| `origination_charges` | Origination charges |
| `discount_points` | Fees to reduce interest rate |
| `lender_credits` | Lender credits |
| `prepayment_penalty_term` | Prepayment penalty duration |
| `intro_rate_period` | Introductory rate period |
| `negative_amortization` | Negative amortization flag |
| `interest_only_payment` | Interest-only payment flag |
| `balloon_payment` | Balloon payment flag |
| `other_nonamortizing_features` | Other non-standard payment features |
| `property_value` | Appraised property value |
| `construction_method` | Site-built or manufactured |

## 4. Property & Occupancy (6)

| Variable Name | Description |
|---------------|-------------|
| `occupancy_type` | Primary, secondary, or investment |
| `manufactured_home_secured_property_type` | Security type for manufactured homes |
| `manufactured_home_land_property_interest` | Land property interest |
| `total_units` | Number of dwelling units |
| `multifamily_affordable_units` | Affordable units (for multifamily) |
| `income` | Applicant(s) gross annual income |

## 5. Credit & Submission (5)

| Variable Name | Description |
|---------------|-------------|
| `debt_to_income_ratio` | DTI ratio |
| `applicant_credit_score_type` | Credit score model used |
| `co-applicant_credit_score_type` | Co-applicant credit score model |
| `submission_of_application` | Whether submitted directly to the institution |
| `initially_payable_to_institution` | Whether initially payable to the lender |

## 6. Applicant Ethnicity (14)

| Variable Name | Description |
|---------------|-------------|
| `derived_ethnicity` | Aggregate ethnicity |
| `applicant_ethnicity_1–5` | Up to five ethnicities reported |
| `applicant_ethnicity_observed` | Whether observed visually |
| `co-applicant_ethnicity_1–5` | Co-applicant ethnicities |
| `co-applicant_ethnicity_observed` | Whether co-applicant ethnicity was observed |
| `applicant_ethnicity_free_form_text_field` | Free-form ethnicity text |

## 7. Applicant Race (14)

| Variable Name | Description |
|---------------|-------------|
| `derived_race` | Aggregate race |
| `applicant_race_1–5` | Up to five races reported |
| `applicant_race_observed` | Whether observed visually |
| `co-applicant_race_1–5` | Co-applicant races |
| `co-applicant_race_observed` | Whether co-applicant race was observed |
| `applicant_race_free_form_text_field` | Free-form race text |

## 8. Applicant Sex & Age (10)

| Variable Name | Description |
|---------------|-------------|
| `derived_sex` | Aggregate sex |
| `applicant_sex` | Applicant sex |
| `co-applicant_sex` | Co-applicant sex |
| `applicant_sex_observed` | Whether observed visually |
| `co-applicant_sex_observed` | Whether co-applicant sex was observed |
| `applicant_age` | Applicant age range |
| `co-applicant_age` | Co-applicant age range |
| `applicant_age_above_62` | Applicant age ≥ 62 flag |
| `co-applicant_age_above_62` | Co-applicant age ≥ 62 flag |
| `applicant_sex_free_form_text_field` | Free-form sex text |

## 9. Underwriting & Denial (16)

| Variable Name | Description |
|---------------|-------------|
| `aus_1–5` | Automated Underwriting Systems used |
| `aus_result_1–5` | AUS recommendations |
| `aus_free_form_text_field` | Free-form AUS text |
| `denial_reason_1–4` | Reasons for denial |
| `denial_reason_free_form_text_field` | Free-form denial reason |

## 10. Census Tract Demographics (7)

| Variable Name | Description |
|---------------|-------------|
| `tract_population` | Total population |
| `tract_minority_population_percent` | Minority population percentage |
| `ffiec_msa_md_median_family_income` | MSA median family income |
| `tract_to_msa_income_percentage` | Tract income relative to MSA |
| `tract_owner_occupied_units` | Owner-occupied units |
| `tract_one_to_four_family_homes` | 1–4 family homes |
| `tract_median_age_of_housing_units` | Median housing unit age |

