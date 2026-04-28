# HMDA Full Data Dictionary (Raw 99 Variables)

**Note**: This data dictionary describes the raw HMDA dataset containing **99 variables** (`hmda_raw_2023_TX_big4.csv`) as retrieved via the CFPB API. 

It does **not** apply to the cleaned and split datasets (`hmda_cleaned.csv`, `train.csv`, `test.csv`), as feature selection and transformations were applied during preprocessing.

---

## 1. Institutional & Geographic Metadata (11)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `activity_year` | Calendar year of the data | 2023 |
| `lei` | Legal Entity Identifier | 20-character alphanumeric string |
| `derived_msa-md` | Derived MSA/MD code | 5-digit FIPS code |
| `state_code` | State abbreviation | Two-letter code (e.g., TX) |
| `county_code` | FIPS county code | 5-digit code (e.g., 48029) |
| `census_tract` | 11-digit census tract ID | 11-digit numeric string |
| `conforming_loan_limit` | Loan limit status | C: Conforming, NC: Non-conforming, U: Unknown |
| `derived_loan_product_type` | Derived loan and lien type | e.g., "FHA:First Lien" |
| `derived_dwelling_category` | Derived dwelling type | e.g., "Single Family (1-4 Units):Site-Built" |
| `action_taken` | Final outcome of application | 1: Originated, 3: Denied, 6: Purchased, etc. |
| `purchaser_type` | Type of purchasing entity | 0: N/A, 1: Fannie Mae, 2: Ginnie Mae, etc. |

## 2. Loan Application Details (10)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `preapproval` | Pre-approval request | 1: Requested, 2: Not requested |
| `loan_type` | Type of loan | 1: Conventional, 2: FHA, 3: VA, 4: RHS/FSA |
| `loan_purpose` | Purpose of loan | 1: Purchase, 2: Improvement, 31: Refi, etc. |
| `lien_status` | Lien priority | 1: First Lien, 2: Subordinate Lien |
| `reverse_mortgage` | Reverse mortgage flag | 1: Yes, 2: No, 1111: Exempt |
| `open-end_line_of_credit` | HELOC/Open-end flag | 1: Yes, 2: No, 1111: Exempt |
| `business_or_commercial_purpose` | Business purpose flag | 1: Yes, 2: No, 1111: Exempt |
| `loan_amount` | Requested loan amount | Numeric (actual dollar amount) |
| `loan_to_value_ratio` | Loan-to-value ratio | Numeric (e.g., "80.0", "Exempt") |
| `loan_term` | Loan maturity in months | Numeric (e.g., "360", "Exempt") |

## 3. Pricing, Costs & Amortization (16)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `interest_rate` | Annual interest rate | Numeric (e.g., "4.875", "Exempt") |
| `rate_spread` | APR minus benchmark rate | Numeric (e.g., "1.25", "Exempt") |
| `hoepa_status` | High-cost mortgage status | 1: High-cost, 2: Not high-cost, 3: N/A |
| `total_loan_costs` | Total loan costs | Numeric (actual dollar amount) |
| `total_points_and_fees` | Total points and fees | Numeric (actual dollar amount) |
| `origination_charges` | Total origination charges | Numeric (actual dollar amount) |
| `discount_points` | Fees paid for lower rate | Numeric (actual dollar amount) |
| `lender_credits` | Credits provided by lender | Numeric (actual dollar amount) |
| `prepayment_penalty_term` | Penalty duration | Numeric (in months) |
| `intro_rate_period` | Introductory rate duration | Numeric (in months) |
| `negative_amortization` | Negative amortization flag | 1: Yes, 2: No, 1111: Exempt |
| `interest_only_payment` | Interest-only flag | 1: Yes, 2: No, 1111: Exempt |
| `balloon_payment` | Balloon payment flag | 1: Yes, 2: No, 1111: Exempt |
| `other_nonamortizing_features`| Other payment features | 1: Yes, 2: No, 1111: Exempt |
| `property_value` | Appraised property value | Numeric (e.g., "215000", "Exempt") |
| `construction_method` | Build method | 1: Site-built, 2: Manufactured |

## 4. Property, Income & Submission (11)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `occupancy_type` | Intended use of property | 1: Primary, 2: Second, 3: Investment |
| `manufactured_home_secured_property_type` | Security type for MH | 1: Real Property, 2: Personal, 3: N/A |
| `manufactured_home_land_property_interest` | Land interest for MH | 1: Direct, 2: Indirect, 3: Lease, etc. |
| `total_units` | Total number of units | 1, 2, 3, 4, 5-24, 25-49, 50-99, etc. |
| `multifamily_affordable_units` | Affordable unit count | Numeric or "Exempt" |
| `income` | Gross annual income | Numeric (in thousands, e.g., "150") |
| `debt_to_income_ratio` | DTI ratio range | <20%, 20%-<30%, 30%-36%, ..., Exempt |
| `applicant_credit_score_type` | Credit model used | 1: Equifax, 2: Exp, 3: TU, 4: Vantage, etc. |
| `co-applicant_credit_score_type`| Co-app credit model | 1-9: Same as above, 10: No co-applicant |
| `submission_of_application` | Submission channel | 1: Direct, 2: Indirect, 3: N/A |
| `initially_payable_to_institution` | Payable to lender | 1: Yes, 2: No, 3: N/A |

## 5. Applicant Ethnicity (13)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `derived_ethnicity` | Aggregate ethnicity | Hispanic, Not Hispanic, Joint, Unknown |
| `applicant_ethnicity-1` to `-5` | Reported ethnicities | 1: Hispanic, 2: Not Hispanic, 3: Not provided, etc. |
| `co-applicant_ethnicity-1` to `-5`| Co-app ethnicities | 1-4: Same as above, 5: No co-applicant |
| `applicant_ethnicity_observed` | Visual observation flag | 1: Yes, 2: No, 3: N/A |
| `co-applicant_ethnicity_observed`| Co-app visual obs | 1-3: Same as above, 4: No co-applicant |

## 6. Applicant Race (13)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `derived_race` | Aggregate race | White, Black, Asian, Am-Indian, Joint, etc. |
| `applicant_race-1` to `-5` | Reported races | 1: Am-Indian, 2: Asian, 3: Black, 5: White, etc. |
| `co-applicant_race-1` to `-5` | Co-app races | 1-5: Same as above, 8: No co-applicant |
| `applicant_race_observed` | Visual observation flag | 1: Yes, 2: No, 3: N/A |
| `co-applicant_race_observed`| Co-app visual obs | 1-3: Same as above, 4: No co-applicant |

## 7. Applicant Sex & Age (9)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `derived_sex` | Aggregate sex | Male, Female, Joint, Unknown |
| `applicant_sex` | Applicant sex | 1: Male, 2: Female, 3: Not provided, 4: N/A |
| `co-applicant_sex` | Co-applicant sex | 1-4: Same as above, 5: No co-applicant |
| `applicant_sex_observed` | Visual observation flag | 1: Yes, 2: No, 3: N/A |
| `co-applicant_sex_observed` | Co-app visual obs | 1-3: Same as above, 4: No co-applicant |
| `applicant_age` | Applicant age range | <25, 25-34, 35-44, ..., 8888 (N/A) |
| `co-applicant_age` | Co-applicant age range | Same as above, 9999 (No co-applicant) |
| `applicant_age_above_62` | Age ≥ 62 flag | Yes, No, N/A |
| `co-applicant_age_above_62` | Co-app Age ≥ 62 flag | Yes, No, N/A |

## 8. Underwriting & Denial (9)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `aus-1` to `aus-5` | Automated Underwriting Systems | 1: DU, 2: LP, 3: GUS, 4: FHA, 5: Other, 6: N/A |
| `denial_reason-1` to `-4` | Reasons for loan denial | 1: Debt, 2: Employment, 3: Credit, ..., 10: N/A |

## 9. Census Tract Demographics (7)

| Variable Name | Description | Values / Range |
|:---|:---|:---|
| `tract_population` | Total census tract population | Numeric (e.g., 6678) |
| `tract_minority_population_percent` | Minority population % | Numeric (0 - 100) |
| `ffiec_msa_md_median_family_income` | Area median family income | Numeric (e.g., 89100) |
| `tract_to_msa_income_percentage` | Tract income vs. MSA income | Numeric (Percentage) |
| `tract_owner_occupied_units` | Owner-occupied unit count | Numeric |
| `tract_one_to_four_family_homes` | 1-4 family home count | Numeric |
| `tract_median_age_of_housing_units` | Median age of housing | Numeric (Years) |