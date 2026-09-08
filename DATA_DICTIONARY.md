# Data dictionary

Every field in `data/processed/panel.parquet`, one row per local authority per
financial year.

## Identifiers

| Field | Type | Notes |
|---|---|---|
| `la_code` | string | ONS code. `E06` unitary, `E07` non-metropolitan district, `E08` metropolitan district, `E09` London borough |
| `la_name` | string | As published, so pre-merger names appear in early years |
| `financial_year` | string | `2018-19` through `2024-25` |

## Denominator

| Field | Type | Notes |
|---|---|---|
| `households_in_area_000` | float | Resident households in thousands, published in H-CLIC table A1 |
| `households_in_area` | float | The same figure, in households |

Every count below has a companion `rate_<name>` giving that count per 1,000
resident households. Rates are what the models see; raw counts would make any
model a population-size detector.

## Scale of demand (H-CLIC table A1)

| Field | Meaning |
|---|---|
| `assessments_total` | Total initial homelessness assessments |
| `owed_any_duty` | Assessed as owed a prevention or relief duty |
| `owed_prevention` | Threatened with homelessness within 56 days |
| `owed_relief` | Already homeless |
| `s21_notice` | Threatened by service of a valid Section 21 notice. Not collected in 2018-19 |

## Household composition (tables A5P, A5R)

| Field | Meaning |
|---|---|
| `relief_total` | Households owed a relief duty |
| `prevention_total` | Households owed a prevention duty |
| `relief_with_children` | Of those, households containing dependent children |
| `prevention_with_children` | As above, for the prevention duty |

The published tables split each household type by the sex of the main
applicant, so the two `*_with_children` fields are row sums across single
parents, couples and three-or-more-adult households with dependent children.
`relief_with_children` is the project's outcome.

## Reason for loss of the last settled home (table A2R)

`loss_end_ast`, `loss_rent_arrears`, `loss_landlord_selling`,
`loss_family_friends`, `loss_domestic_abuse`, `loss_social_tenancy`,
`loss_supported_housing`, `loss_institution`.

These are recorded for households already homeless. Sub-reasons published
beneath each heading are not carried through.

## Support needs (table A3)

`need_any`, `need_mental_health`, `need_physical_health`,
`need_domestic_abuse`, `need_drug`, `need_alcohol`, `need_offending`,
`need_repeat_homeless`, `need_rough_sleeping`, `need_care_leaver_18_20`,
`need_young_parent`, `need_learning_disab`.

A household can have several needs, so these do not sum to `need_any`.
`need_budgeting` appears only from 2022-23 and is excluded for that reason.

## Accommodation at the time of application (table A4R)

`from_private_rented`, `from_social_rented`, `from_family`, `from_friends`,
`from_temp_accom`, `from_rough_sleeping`.

## Deprivation (Indices of Deprivation 2019)

`imd_score`, `income_score`, `employment_score`, `education_score`,
`health_score`, `crime_score`, `barriers_score`, `living_env_score`,
`idaci_score`.

Average scores across the authority's LSOAs. Rank-based summaries are not
used, because ranks are not comparable once authorities merge. Seven post-2019
successor authorities have these scores rebuilt by population-weighting their
predecessors' LSOAs; fourteen districts abolished in April 2019 inherit their
successor's score.

## Derived fields in `design.parquet`

| Field | Meaning |
|---|---|
| `d1_<rate>` | One-year change in that rate |
| `d2_rate_relief_with_children` | Two-year change in the outcome, to separate a one-off spike from a sustained rise |
| `share_relief_children` | Share of homeless households containing children |
| `share_prevention_children` | The same, for prevention duties |
| `prevention_to_relief` | Ratio of prevention to relief duties, a rough proxy for how early an authority intervenes |
| `next_year` | The outcome rate one year later, for the same authority |
| `outcome_year` | The financial year `next_year` refers to |
| `y_level` | 1 if `next_year` is at or above the worst-quintile cutoff. The cutoff is a quantile of the training years only |
| `y_escalation` | 1 if `next_year` is at least 25% above the current rate. Null where the current rate is zero |

## Missing values

Missing means the authority filed no return, or that the measure did not exist
in that release. It is never recoded to zero. Gradient boosting handles the
gaps natively; the logistic pipeline imputes the median inside the
cross-validation fold, so no test-fold information reaches the imputer.
