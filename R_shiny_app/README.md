# PS-DRS: Population-Specific Dementia Risk Score — Shiny App

An interactive R Shiny web application for estimating an individual's probability of developing dementia over 5, 10, or 15 years. The PS-DRS provides **population-specific scores** for three racial groups using Cox Proportional Hazards models developed on UK Biobank data and externally validated in the All of Us cohort.

## Background

Unlike most existing dementia risk tools built predominantly on White European populations, the PS-DRS was developed using a **mixture-of-experts deep transfer learning** framework that borrows information from the larger White population to improve predictions for smaller minority groups.

| Population | Training N | Dementia cases |
|:-----------|----------:|---------------:|
| White      | 472,363   | 7,745          |
| Black      | 8,048     | 133            |
| Asian      | 9,872     | 152            |

## Features

- **Population-Specific Risk Assessment**: Separate Cox-PH models for White, Black, and Asian populations
- **Age-Stratified Models**: Different model coefficients for age ≤65 and >65
- **Multi-Horizon Prediction**: 5, 10, and 15-year dementia risk probabilities
- **Risk Categorization**: Age-appropriate Low/Medium/High thresholds
- **Sullivan PS-DRS (0–100)**: Interpretable integer-scale score analogous to the Framingham risk score
- **Top Risk Factor Contributions**: Top 15 features ranked by their impact on the 10-year risk
- **Missing Value Imputation**: Auto-fills missing inputs with median/mode from disease-free training participants
- **Data Quality Tab**: Completeness metrics, range validation, and per-feature imputation status
- **Model Info Tab**: Browsable coefficient table by race, age stratum, and timepoint

## Installation

### Prerequisites

- R (≥ 4.1)
- RStudio (recommended)

### Required R Packages

```r
install.packages(c(
  "shiny",
  "shinydashboard",
  "dplyr",
  "readr",
  "DT",
  "plotly",
  "jsonlite"
))
```

## Running the App

### Option 1: RStudio
1. Open `app.R` in RStudio
2. Click the **Run App** button

### Option 2: R Console
```r
library(shiny)
setwd("path/to/R_shiny_app")
runApp("app.R")
```

### Option 3: Command Line
```bash
cd path/to/R_shiny_app
R -e "shiny::runApp('app.R')"
```

## How It Works

### Risk Calculation Pipeline

1. **Feature scaling** — RobustScaler (median-centered, IQR-scaled) applied at prediction time using stored parameters:
   ```
   x_scaled = (x - center) / scale
   ```

2. **Linear predictor** — weighted sum of scaled features:
   ```
   LP = Σ(β_j × x_scaled_j)
   ```

3. **Risk probability** — using the Cox baseline survival S₀(t):
   ```
   Risk(t) = 1 - S₀(t)^exp(LP)
   ```

4. **Risk category** — assigned from age-specific thresholds table below

   | Age Group  | Low     | Medium      | High   |
   |:-----------|:--------|:------------|:-------|
   | ≤ 65 years | < 1%    | 1% – 3%     | ≥ 3%   |
   | > 65 years | < 5%    | 5% – 15%    | ≥ 15%  |


### Sullivan PS-DRS (0–100)

```
raw_score = Σ(point_weight_j × x_scaled_j)
PS-DRS    = 100 × (raw_score − P5) / (P95 − P5)   [clipped to 0–100]
```

Point weights are scaled relative to the age coefficient: `pw_j = 10 × β_j / β_age`.

> **Note:** The Sullivan score is unavailable for **Asian (>65)** at the 10- and 15-year horizons. The small stratum size (N = 996) produced a negative β_age, making point weights clinically misleading. For this group the app shows the Cox risk probability only.

## App Usage

1. **Select Demographics** — choose Race and enter Age in the sidebar
2. **Enter Risk Factors** — fill in available health and lifestyle values. Inputs are grouped by category and show training-data ranges for each feature
3. **Enable Imputation** — check "Impute missing features" to auto-fill blanks with disease-free training medians/modes
4. **Calculate** — click the **Calculate** button
5. **Review Results**:
   - Bar chart of 5 / 10 / 15-year risk (color-coded by category)
   - Table of risk %, risk category, Sullivan score, hazard ratio, and linear predictor
   - Top 15 contributing risk factors (red = increasing risk, green = decreasing risk)
   - **Data Quality** tab: feature completeness, range warnings, provided vs. imputed feature lists
   - **Model Info** tab: full coefficient table for any race × age × timepoint combination

## File Reference

| File | Contents |
|:-----|:---------|
| `app.R` | Main Shiny application |
| `shiny_app_params/shiny_model_params.json` | Master file: all coefficients, scalers, imputation values, Sullivan parameters |
| `shiny_app_params/clinical_thresholds.json` | Risk cut-offs (Low / Medium / High) by age stratum |
| `shiny_app_params/imputation_and_scaling.csv` | Table of imputation + scaling values per feature |
| `shiny_app_params/shiny_master_summary.csv` | Model performance summary across all race × age × timepoint combinations |
| `shiny_app_params/coefficients_[race]_[age]_[t]y.csv` | Per-model coefficient tables |
| `shiny_app_params/feature_summary_race_stratified.csv` | Feature stats |
| `shiny_app_params/formatted_feature_coding.csv` | Dropdown option labels/values for categorical features |
| `PSDRS_ShinyApp_Guide.Rmd` | Detailed user guide with step-by-step worked example |

## Limitations

- Trained on UK Biobank participants aged 40–73. Predictions outside this range should be interpreted with caution.
- UKB is a relatively healthy, well-educated cohort; dementia prevalence in UKB Black participants (~1.7%) is lower than in the general UK population.
- The Asian group in UKB is predominantly South Asian; results may differ for East Asian subgroups.
- Population specific categories are broad within group heterogenetity is not captured.

## Tutorial

To compute risk predictions on an entire dataset, [`PSDRS_ShinyApp_Guide.Rmd`](PSDRS_ShinyApp_Guide.Rmd) provides a complete R function that loops over each row and appends the predicted dementia risk probability, risk category (Low/Medium/High), and Sullivan PS-DRS score as output columns.

<!-- ## Citation

If you use this calculator, please cite:

```
```

## License -->