# HMDA Full Data Dictionary (109 Variables)

**Note**: This data dictionary describes the raw HMDA dataset (`hmda_raw_2023_TX_big4.csv`).

It does **not** apply to the cleaned and split datasets (`hmda_cleaned.csv`, `train.csv`, `test.csv`), as feature selection and transformations were applied during preprocessing.

## 1. Institutional & Geographic Metadata (11)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `activity_year` | Calendar year | 2023 |
| `lei` | Legal Entity Identifier | 20-character alphanumeric string |
| `derived_msa-md` | Derived MSA/MD code | 5-digit FIPS code |
| `state_code` | State abbreviation | Two-letter postal code (e.g., TX) |
| `county_code` | FIPS county code | 5-digit code (e.g., 48453) |
| `census_tract` | 11-digit census tract ID | 11-digit numeric string |
| `conforming_loan_limit` | Loan limit status | Conforming, Non-conforming, Exempt |
| `derived_loan_product_type` | Derived loan type | e.g., "Conventional:First Lien" |
| `derived_dwelling_category` | Derived dwelling type | e.g., "Single Family (1-4 Units):Detached" |
| `action_taken` | Final outcome of application | 1: Originated, 2: Approved not accepted, 3: Denied, 4: Withdrawn, 5: Incomplete |
| `purchaser_type` | Type of purchasing entity | 0: N/A, 1: Fannie Mae, 2: Ginnie Mae, 3: Freddie Mac, etc. |

## 2. Loan Application Details (10)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `preapproval` | Pre-approval request | 1: Requested, 2: Not requested |
| `loan_type` | Type of loan | 1: Conventional, 2: FHA, 3: VA, 4: RHS/FSA |
| `loan_purpose` | Purpose of loan | 1: Purchase, 2: Improvement, 31: Refi, 32: Cash-out, 4: Other |
| `lien_status` | Lien priority | 1: First Lien, 2: Subordinate Lien |
| `reverse_mortgage` | Reverse mortgage flag | 1: Yes, 2: No, 1111: Exempt |
| `open-end_line_of_credit` | HELOC/Open-end flag | 1: Yes, 2: No, 1111: Exempt |
| `business_or_commercial_purpose` | Business purpose flag | 1: Yes, 2: No, 1111: Exempt |
| `loan_amount` | Requested amount | Numeric (actual dollar amount) |
| `combined_loan_to_value_ratio` | Combined LTV ratio | Numeric (string format, e.g., "80.0", "Exempt") |
| `loan_term` | Loan maturity | Numeric (in months, e.g., "360", "Exempt") |

## 3. Pricing, Costs & Amortization (16)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `interest_rate` | Annual interest rate | Numeric (e.g., "6.5", "Exempt") |
| `rate_spread` | APR minus benchmark | Numeric (e.g., "1.25", "Exempt") |
| `hoepa_status` | High-cost mortgage status | 1: High-cost, 2: Not high-cost, 3: N/A |
| `total_loan_costs` | Total costs | Numeric (actual dollar amount) |
| `total_points_and_fees` | Total points and fees | Numeric (actual dollar amount) |
| `origination_charges` | Origination charges | Numeric (actual dollar amount) |
| `discount_points` | Fees for lower rate | Numeric (actual dollar amount) |
| `lender_credits` | Credits from lender | Numeric (actual dollar amount) |
| `prepayment_penalty_term` | Penalty duration | Numeric (in months) |
| `intro_rate_period` | Intro rate duration | Numeric (in months) |
| `negative_amortization` | Negative amortization | 1: Yes, 2: No, 1111: Exempt |
| `interest_only_payment` | Interest-only flag | 1: Yes, 2: No, 1111: Exempt |
| `balloon_payment` | Balloon payment flag | 1: Yes, 2: No, 1111: Exempt |
| `other_nonamortizing_features`| Other payment features | 1: Yes, 2: No, 1111: Exempt |
| `property_value` | Appraised value | Numeric (e.g., "450000", "Exempt") |
| `construction_method` | Build method | 1: Site-built, 2: Manufactured |

## 4. Property & Occupancy (6)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `occupancy_type` | Intended use | 1: Primary, 2: Second, 3: Investment |
| `manufactured_home_secured_property_type` | Security type | 1: Real Property, 2: Personal, 3: N/A |
| `manufactured_home_land_property_interest` | Land interest | 1: Direct, 2: Indirect, 3: Lease, 4: Unpaid, 5: N/A |
| `total_units` | Number of units | 1, 2, 3, 4, 5-24, 25-49, 50-99, 100-149, >149 |
| `multifamily_affordable_units` | Affordable unit count | Numeric or "Exempt" |
| `income` | Gross annual income | Numeric (in thousands, e.g., "150") |

## 5. Credit & Submission (5)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `debt_to_income_ratio` | DTI ratio | <20%, 20%-<30%, 30%-36%, 37%, 38%, ..., >60%, Exempt |
| `applicant_credit_score_type` | Credit model used | 1: Equifax, 2: Exp, 3: TU, 4: Vantage, 7: Other, 9: N/A |
| `co-applicant_credit_score_type`| Co-app credit model | 1-9: Same as above, 10: No co-applicant |
| `submission_of_application` | Submission channel | 1: Direct, 2: Indirect, 3: N/A |
| `initially_payable_to_institution` | Payable to lender | 1: Yes, 2: No, 3: N/A |

## 6. Applicant Ethnicity (14)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `derived_ethnicity` | Aggregate ethnicity | Hispanic/Latino, Not Hispanic/Latino, Joint, Unknown |
| `applicant_ethnicity_1–5` | Reported ethnicities | 1: Hispanic/Latino, 2: Not Hispanic/Latino, 3: Not provided, 4: N/A |
| `applicant_ethnicity_observed` | Visual observation | 1: Yes, 2: No, 3: N/A |
| `co-applicant_ethnicity_1–5`| Co-app ethnicities | 1-4: Same as above, 5: No co-applicant |
| `co-applicant_ethnicity_observed`| Co-app visual obs | 1-3: Same as above, 4: No co-applicant |
| `applicant_ethnicity_free_form_text_field` | Free-form text | String (e.g., specific sub-group) |

## 7. Applicant Race (14)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `derived_race` | Aggregate race | White, Black, Asian, Am-Indian, Pacific-Islander, Joint, Unknown |
| `applicant_race_1–5` | Reported races | 1: Am-Indian, 2: Asian, 3: Black, 4: Island-Pac, 5: White, etc. |
| `applicant_race_observed` | Visual observation | 1: Yes, 2: No, 3: N/A |
| `co-applicant_race_1–5` | Co-app races | 1-8: Same as above, 8: No co-applicant |
| `co-applicant_race_observed`| Co-app visual obs | 1-3: Same as above, 4: No co-applicant |
| `applicant_race_free_form_text_field` | Free-form text | String (e.g., specific sub-group) |

## 8. Applicant Sex & Age (10)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `derived_sex` | Aggregate sex | Male, Female, Joint, Unknown |
| `applicant_sex` | Applicant sex | 1: Male, 2: Female, 3: Not provided, 4: N/A, 6: Shared |
| `co-applicant_sex` | Co-applicant sex | 1-4: Same as above, 5: No co-applicant |
| `applicant_sex_observed` | Visual observation | 1: Yes, 2: No, 3: N/A |
| `co-applicant_sex_observed` | Co-app visual obs | 1-3: Same as above, 4: No co-applicant |
| `applicant_age` | Applicant age range | <25, 25-34, 35-44, 45-54, 55-64, 65-74, >74, 8888 (N/A) |
| `co-applicant_age` | Co-applicant age range | Same as above, 9999 (No co-applicant) |
| `applicant_age_above_62` | Age ≥ 62 flag | Yes, No, N/A |
| `co-applicant_age_above_62` | Co-app Age ≥ 62 flag | Yes, No, N/A |
| `applicant_sex_free_form_text_field` | Free-form text | String |

## 9. Underwriting & Denial (16)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `aus_1–5` | AUS systems | 1: DU, 2: LP, 3: GUS, 4: FHA, 5: Other, 6: N/A |
| `aus_result_1–5` | AUS recommendations | 1: Approve/Eligible, 2: Approve/Ineligible, ..., 17: N/A |
| `aus_free_form_text_field` | Free-form AUS | String |
| `denial_reason_1–4` | Reasons for denial | 1: Debt, 2: Employment, 3: Credit, 4: Collateral, ..., 10: N/A |
| `denial_reason_free_form_text_field`| Free-form denial | String |

## 10. Census Tract Demographics (7)

| Variable Name | Description | Values / Range |
|---------------|-------------|----------------|
| `tract_population` | Total population | Numeric (e.g., 5420) |
| `tract_minority_population_percent` | Minority % | Numeric (0 - 100) |
| `ffiec_msa_md_median_family_income` | Median family income | Numeric (e.g., 98000) |
| `tract_to_msa_income_percentage` | Relative income % | Numeric (Percentage) |
| `tract_owner_occupied_units` | Owner-occupied count | Numeric |
| `tract_one_to_four_family_homes` | 1-4 family count | Numeric |
| `tract_median_age_of_housing_units` | Housing age median | Numeric (Years) |