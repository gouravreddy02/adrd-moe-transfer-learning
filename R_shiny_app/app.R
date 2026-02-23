library(shiny)
library(shinydashboard)
library(dplyr)
library(readr)
library(DT)
library(plotly)
library(jsonlite)

# ------------------------------------------------------------------------------
# GLOBAL SETTINGS & FILE PATHS
# ------------------------------------------------------------------------------

# Root directory: use the directory containing this script for portability
DATA_DIR <- getwd()

# JSON model parameters (Cox-PH coefficients, Sullivan weights, scaler params)
MODEL_PARAMS_FILE <- file.path(DATA_DIR, "shiny_app_params/shiny_model_params.json")

# UI metadata: feature stats for range validation, coded options for dropdowns
FEATURE_FILE <- file.path(DATA_DIR, "shiny_app_params/feature_summary_race_stratified.csv")
FORMATTED_OPTS_FILE <- file.path(DATA_DIR, "shiny_app_params/formatted_feature_coding.csv")
# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: FILE READING & DATA PROCESSING
# ------------------------------------------------------------------------------

# Safely read a CSV file, checking for existence and removing "Unnamed" columns.
safe_read_csv <- function(path) {
  if (!file.exists(path)) stop(paste("Missing file:", path))
  readr::read_csv(path, show_col_types = FALSE) %>%
    dplyr::select(-dplyr::starts_with("Unnamed"))
}

# Convert age to stratum (leq65 or gt65)
age_to_stratum <- function(age) {
  ifelse(age <= 65, "leq65", "gt65")
}

# Get features for a specific race/age/timepoint from JSON model
get_model_features <- function(params, race, age_stratum, timepoint) {
  tryCatch({
    model <- params$models[[race]][[age_stratum]][[as.character(timepoint)]]
    if (is.null(model)) return(character(0))
    unlist(model$features)
  }, error = function(e) {
    character(0)
  })
}

# Impute missing values using imputation values from the JSON model.
# Returns a list containing the filled values and flags indicating which features were imputed.
impute_with_model <- function(values_named, model) {
  imputation_values <- model$imputation_values

  out <- values_named
  imp <- rep(FALSE, length(values_named))
  names(imp) <- names(values_named)

  for (nm in names(values_named)) {
    if (is.na(values_named[[nm]]) && !is.null(imputation_values[[nm]])) {
      out[[nm]] <- imputation_values[[nm]]
      imp[[nm]] <- TRUE
    }
  }
  list(values = out, imputed = imp)
}

# Check input values against min/max ranges from the stats dataframe.
# Returns a dataframe of warnings if any values are outside the expected range.
check_ranges <- function(values_named, stats_df, race) {
  sub <- stats_df %>% filter(race == !!race)

  warn_rows <- list()
  for (nm in names(values_named)) {
    val <- values_named[[nm]]
    if (is.na(val)) next
    row <- sub %>% filter(feature == !!nm) %>% slice(1)
    if (nrow(row) == 0) next
    
    # Ignore binary features for range warnings (they are usually 0 or 1)
    is_bin <- if("is_binary" %in% names(row)) row$is_binary[1] else NA
    if (!is.na(is_bin) && (isTRUE(as.logical(is_bin)) || is_bin == "True")) next

    lo <- row$min[1]
    hi <- row$max[1]
    
    # Check if value is outside [min, max]
    if (!is.na(lo) && !is.na(hi) && is.finite(lo) && is.finite(hi) && (val < lo || val > hi)) {
      
      severity <- "Medium"
      if (val < lo * 0.8 || val > hi * 1.2) {
        severity <- "High"
      }

      warn_rows[[length(warn_rows) + 1]] <- data.frame(
        feature = nm,
        value = val,
        min_value = lo,
        max_value = hi,
        severity = severity,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(warn_rows) == 0) return(data.frame())
  do.call(rbind, warn_rows)
}

# ============================================================================
# FUNCTION 1: Calculate Cox-PH Predicted Probability
# ============================================================================
calculate_cox_probability <- function(user_data, model) {

  # 1. Get baseline survival S0(t)
  S0 <- model$baseline_survival_S0
  if (is.null(S0)) return(NULL)

  # 2. Get features, coefficients, and scaler parameters from model
  features <- unlist(model$features)
  coefficients <- model$coefficients
  scaler_center <- model$scaler_center
  scaler_scale <- model$scaler_scale

  # 3. Calculate linear predictor
  # LP = sum(x_scaled_j * beta_j)
  # x_scaled_j = (x_j - center_j) / scale_j
  linear_predictor <- 0

  for (feat in features) {
    raw_val <- user_data[[feat]]
    if (is.null(raw_val) || is.na(raw_val)) next

    # Get parameters
    beta <- coefficients[[feat]]
    center <- scaler_center[[feat]]
    scale <- scaler_scale[[feat]]

    # Standardize: x_scaled = (x - center) / scale
    if (!is.null(center) && !is.null(scale) && scale != 0) {
      std_val <- (raw_val - center) / scale
    } else {
      std_val <- 0
    }

    # Add to linear predictor
    if (!is.null(beta) && !is.na(beta)) {
      linear_predictor <- linear_predictor + beta * std_val
    }
  }

  # 4. Calculate probability
  # Risk(t) = 1 - S0(t)^exp(LP)
  hazard_ratio <- exp(linear_predictor)
  probability <- 1 - S0^hazard_ratio

  return(list(
    probability = probability,
    probability_pct = probability * 100,
    linear_predictor = linear_predictor,
    hazard_ratio = hazard_ratio,
    baseline_survival = S0
  ))
}

# ============================================================================
# FUNCTION 2: Calculate Sullivan Risk Score (0-100)
# ============================================================================
calculate_sullivan_score <- function(user_data, model) {

  # 1. Get Sullivan parameters from model
  # NOTE: sullivan_params is NULL for models where beta_age was negative
  # (typically small strata with insufficient events, e.g. Asian/gt65).
  # In those cases fall back to showing the risk probability only.
  sullivan <- model$sullivan_params
  if (is.null(sullivan)) return(NULL)

  # 2. Get calibration parameters
  P5 <- sullivan$P5
  P95 <- sullivan$P95
  if (is.null(P5) || is.null(P95)) return(NULL)

  # 3. Get features and parameters
  features <- unlist(model$features)
  point_weights <- sullivan$point_weights
  scaler_center <- model$scaler_center
  scaler_scale <- model$scaler_scale

  # 4. Calculate raw score
  # raw = sum(point_weight_j * x_scaled_j)
  raw_score <- 0

  for (feat in features) {
    # Get user value (already imputed if enabled)
    raw_val <- user_data[[feat]]
    if (is.null(raw_val) || is.na(raw_val)) next

    # Get parameters
    pw <- point_weights[[feat]]
    center <- scaler_center[[feat]]
    scale <- scaler_scale[[feat]]

    # Standardize: x_scaled = (x - center) / scale
    if (!is.null(center) && !is.null(scale) && scale != 0) {
      std_val <- (raw_val - center) / scale
    } else {
      std_val <- 0
    }

    # Add points
    if (!is.null(pw) && !is.na(pw)) {
      raw_score <- raw_score + pw * std_val
    }
  }

  # 5. Calibrate to 0-100 scale
  # PS-DRS = 100*(raw - P5)/(P95 - P5), clipped [0,100]
  calibrated_score <- 100 * (raw_score - P5) / (P95 - P5)
  calibrated_score <- max(0, min(100, calibrated_score))

  return(list(
    raw_score = raw_score,
    calibrated_score = calibrated_score,
    P5 = P5,
    P95 = P95
  ))
}

# ============================================================================
# FUNCTION 3: Categorize Risk (Low/Medium/High)
# ============================================================================
categorize_risk <- function(probability_pct, age_stratum, thresholds) {

  # Get age-specific thresholds from metadata
  thresh <- thresholds[[age_stratum]]
  if (is.null(thresh)) return(list(category = "Unknown", low_upper = NA, medium_upper = NA))

  low_upper <- thresh$Low_upper
  medium_upper <- thresh$Medium_upper

  # Categorize based on percentage thresholds
  category <- ifelse(probability_pct < low_upper, "Low",
                     ifelse(probability_pct < medium_upper, "Medium", "High"))

  return(list(
    category = category,
    low_upper = low_upper,
    medium_upper = medium_upper
  ))
}

# ============================================================================
# MAIN FUNCTION: Complete Risk Assessment
# ============================================================================
calculate_dementia_risk <- function(user_data, model, age_stratum, thresholds) {

  # 1. Calculate Cox probability
  cox_result <- calculate_cox_probability(user_data, model)

  if (is.null(cox_result)) return(NULL)

  # 2. Calculate Sullivan score
  sullivan_result <- calculate_sullivan_score(user_data, model)

  # 3. Categorize risk based on Cox probability percentage
  risk_cat <- categorize_risk(cox_result$probability_pct, age_stratum, thresholds)

  # 4. Return complete results
  return(list(
    # Cox-PH outputs (for risk categorization)
    predicted_probability = cox_result$probability,
    predicted_probability_pct = cox_result$probability_pct,
    hazard_ratio = cox_result$hazard_ratio,
    linear_predictor = cox_result$linear_predictor,

    # Sullivan score (for clinical communication)
    sullivan_score = if(!is.null(sullivan_result)) sullivan_result$calibrated_score else NA,
    sullivan_raw = if(!is.null(sullivan_result)) sullivan_result$raw_score else NA,

    # Risk category (based on Cox probability)
    risk_category = risk_cat$category,
    category_thresholds = list(
      low_upper = risk_cat$low_upper,
      medium_upper = risk_cat$medium_upper
    )
  ))
}

# ------------------------------------------------------------------------------
# UI DEFINITION
# ------------------------------------------------------------------------------
ui <- dashboardPage(
  dashboardHeader(title = "Dementia Risk Calculator", titleWidth = 300),
  
  # Sidebar: Navigation and global inputs (Race, Age, Imputation Toggle)
  dashboardSidebar(
    width = 300,
    sidebarMenu(
      id = "tabs",
      menuItem("Calculator", tabName = "calc", icon = icon("calculator")),
      menuItem("Data Quality", tabName = "quality", icon = icon("check-circle")),
      menuItem("Model Info", tabName = "info", icon = icon("info-circle"))
    ),
    selectInput("race", "Race", choices = c("White", "Black", "Asian"), selected = "White"),
    numericInput("age", "Age (years)", value = 60, min = 40, max = 90, step = 1),
    tags$div(
      style = "padding: 0 15px; margin-bottom: 10px; font-size: 12px; color: #666;",
      textOutput("age_stratum_display")
    ),
    conditionalPanel(
      condition = "input.tabs === 'calc'",
      checkboxInput("use_impute", "Impute missing features", value = FALSE),
      actionButton("calc_btn", "Calculate", class = "btn-primary", style = "width: 90%; margin: 5%;")
    )
  ),

  # Body: Tab content for Calculator, Quality, and Info
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .warning-box-high {
          background-color: #f8d7da;
          border-left: 5px solid #dc3545;
          padding: 15px;
          margin: 10px 0;
          border-radius: 5px;
        }
        .warning-box-medium {
          background-color: #fff3cd;
          border-left: 5px solid #ffc107;
          padding: 15px;
          margin: 10px 0;
          border-radius: 5px;
        }
        .info-box-good {
          background-color: #d4edda;
          border-left: 5px solid #28a745;
          padding: 15px;
          margin: 10px 0;
          border-radius: 5px;
        }
        .imputed-feature {
          background-color: #e7f3ff;
          border-left: 3px solid #2196F3;
          padding: 5px;
          margin: 3px 0;
          border-radius: 3px;
          font-size: 11px;
        }
        .provided-feature {
          background-color: #e8f5e8;
          border-left: 3px solid #28a745;
          padding: 5px;
          margin: 3px 0;
          border-radius: 3px;
          font-size: 11px;
        }
      "))
    ),
    tabItems(
      # Tab 1: Calculator
      tabItem(
        tabName = "calc",
        tags$div(
          style = "display: flex; gap: 10px;",
          tags$div(
            style = "flex: 1;",
            box(
              width = NULL,
              title = "Risk percent (5, 10, 15 years)",
              status = "primary",
              solidHeader = TRUE,
              div(style = "display: flex; flex-direction: column; min-height: 450px;",
                  plotlyOutput("risk_plot", height = "250px"),
                  div(style = "margin-top: auto;",
                      uiOutput("risk_table"))
              )
            )
          ),
          tags$div(
            style = "flex: 1;",
            box(
              width = NULL,
              title = "Top Risk Factor Contributions",
              status = "warning",
              solidHeader = TRUE,
              collapsible = TRUE,
              style = "height: 100%;",
              p("Top 15 factors impacting the risk ranked by size of impact (10-year model). RED = this factor is increasing your risk and GREEN = this factor is decreasing your risk."),
              div(style = "height: 400px; overflow-y: auto;",
                  uiOutput("feature_contribution_list"))
            )
          )
        ),
        fluidRow(
          box(
            width = 12,
            title = "Feature inputs (race specific top features)",
            status = "info",
            solidHeader = TRUE,
            p("Enter values for as many features as you have available. Unknown features will be imputed with training means if imputation is enabled."),
            uiOutput("feature_inputs") # Dynamic feature input fields
          )
        )
      ),
      
      # Tab 2: Data Quality (Warnings & Imputation)
      tabItem(
        tabName = "quality",
        fluidRow(
          box(
            title = "Data Quality Assessment",
            status = "primary",
            solidHeader = TRUE,
            width = 12,

            conditionalPanel(
              condition = "input.calc_btn > 0",

              h4(strong("Completeness Analysis")),
              uiOutput("completeness_display"),

              br(),
              h4(strong("Range Validation")),
              uiOutput("range_validation_display"),

              br(),
              h4(strong("Feature Status")),
              uiOutput("feature_status_display")
            ),

            conditionalPanel(
              condition = "input.calc_btn == 0",
              p("Calculate risk to see data quality assessment.")
            )
          )
        )
      ),
      
      # Tab 3: Model Information
      tabItem(
        tabName = "info",
        fluidRow(
          box(
            title = "Model Information",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            
            div(class = "info-box-good",
                h5("Model Specifications:"),
                uiOutput("model_specs_display"),
                
                br(),
                h5("Key Features:"),
                tags$ol(
                  tags$li(strong("Top Features:"), 
                          "Uses race-specific top features based on importance"),
                  tags$li(strong("Time-Dependent Risk:"), 
                          "Calculates 5, 10, and 15-year risk probabilities"),
                  tags$li(strong("Stratified Models:"), 
                          "Coefficients are specific to Race, Age Group, and Timepoint"),
                  tags$li(strong("Range Checking:"), 
                          "Detects when feature values are outside training data ranges"),
                  tags$li(strong("Mean Imputation:"), 
                          "Missing features imputed with disease free population training data")
                )
            ),
            
            br(),
            h4("Model Coefficients by Race/Age/Time"),
            fluidRow(
              column(4, selectInput("info_race", "Select Race:", choices = c("White", "Black", "Asian"), selected = "White")),
              column(4, selectInput("info_age_stratum", "Select Age Stratum:",
                                    choices = c("≤65" = "leq65", ">65" = "gt65"),
                                    selected = "leq65")),
              column(4, selectInput("info_timepoint", "Select Timepoint:", choices = c(5, 10, 15), selected = 5))
            ),
            
            DT::dataTableOutput("feature_info_table")
          )
        )
      )
    )
  )
)

# ------------------------------------------------------------------------------
# SERVER LOGIC
# ------------------------------------------------------------------------------
server <- function(input, output, session) {

  # Load JSON model parameters (replaces multiple CSV files)
  model_params <- jsonlite::fromJSON(MODEL_PARAMS_FILE, simplifyVector = FALSE)

  # Load UI metadata files
  stats_df <- safe_read_csv(FEATURE_FILE)
  feature_options_df <- safe_read_csv(FORMATTED_OPTS_FILE)

  # ----------------------------------------------------------------------------
  # Initialization & Dynamic UI
  # ----------------------------------------------------------------------------

  # Reactive: Convert age to stratum
  age_stratum <- reactive({
    req(input$age)
    age_to_stratum(input$age)
  })

  # Display age stratum to user
  output$age_stratum_display <- renderText({
    stratum <- age_stratum()
    label <- if (stratum == "leq65") "age ≤65 model" else "age >65 model"
    paste("Using", label)
  })

  # Reactive: Get feature list from the 10-year model (available for all race/age combos)
  current_features <- reactive({
    req(input$race, input$age)
    stratum <- age_stratum()
    get_model_features(model_params, input$race, stratum, 10)
  })

  # Filter stats_df by race and model features (used for UI input generation & imputation)
  filtered_stats <- reactive({
    req(input$race, input$age)
    model_feats <- current_features()

    stats_df %>%
      filter(race == !!input$race, feature %in% model_feats) %>%
      arrange(category, feature)
  })

  feature_list <- reactive({
    current_features()
  })

  # Reset state when race or age changes to prevent conflicts
  observeEvent(c(input$race, input$age), {
    # Clear imputation checkbox
    updateCheckboxInput(session, "use_impute", value = FALSE)

    # Clear calculation results
    calc_state(NULL)

    # Clear imputed features tracker
    imputed_ui_features(list())
  })

  # Dynamically generate input fields (numeric or dropdown) for each feature.
  # Grouped by category.
  output$feature_inputs <- renderUI({
    sub <- filtered_stats()
    feats <- unique(sub$feature)
    
    if (length(feats) == 0)
      return(p("No features found matching criteria."))
    
    # Ensure columns exist for display
    sub <- sub %>% select(feature, feature_description, category, min, max, mean, is_binary)
    ui_blocks <- list()
    for (cat in unique(sub$category)) {
      cat_df <- sub %>% filter(category == !!cat)
      cat_ui <- list()
      for (i in seq_len(nrow(cat_df))) {
        f <- cat_df$feature[i]
        
        # Define vars
        lo <- cat_df$min[i]
        hi <- cat_df$max[i]
        mu <- cat_df$mean[i]
        is_bin <- cat_df$is_binary[i]
        desc <- cat_df$feature_description[i]
        
        # Check binary status
        is_binary_feat <- !is.na(is_bin) && (isTRUE(as.logical(is_bin)) || is_bin == "True")
        
        # Check if we have specific options for this feature in the CSV
        opts <- feature_options_df %>% filter(colname_match_data == !!f)
        
        # Use description from CSV if available, otherwise from stats_df
        
        if (nrow(opts) > 0) {
          # CASE 1: Categorical/Ordinal/Binary defined in CSV
          # Create a named vector for choices: "Label" = "value"
          # Ensure order matches the CSV (which is sorted by value now)
          
          # Use unique in case of duplicates, preserving order
          opts <- opts %>% distinct(option_label, option_value)
          
          choices_vec <- setNames(opts$option_value, opts$option_label)
          
          cat_ui[[i]] <- selectInput(
            inputId = paste0("feat__", f),
            label = desc,
            choices = c("Select..." = "", choices_vec),
            selected = ""
          )
          
        } else {
          # CASE 2: No specific options found. Check if Binary or Continuous.
          
          if (is_binary_feat) {
            # Binary: No range in label, use SelectInput default Yes/No
            cat_ui[[i]] <- selectInput(
              inputId = paste0("feat__", f),
              label = desc,
              choices = c("Select..." = "", "Yes" = 1, "No" = 0),
              selected = ""
            )
          } else {
            # Continuous: Use numericInput with min/max hints
            if (!is.na(lo) && !is.na(hi) && is.finite(lo) && is.finite(hi)) {
               lab <- paste0(desc, " (", lo, "-", hi, ")")
            } else {
               lab <- paste0(desc, " (", f, ")")
            }
            
            cat_ui[[i]] <- numericInput(
              inputId = paste0("feat__", f),
              label = lab,
              value = NA_real_, 
              min = if(!is.na(lo)) lo else NA, 
              max = if(!is.na(hi)) hi else NA,
              step = 0.1
            )
          }
        }
      }
      
      # Create a collapsible detail block for the category
      ui_blocks[[cat]] <- tags$details(
        name = "feature_categories_group",
        style = "margin-bottom: 5px; border: 1px solid #d2d6de; border-radius: 3px;",
        tags$summary(
          style = "background-color: #f4f4f4; padding: 10px; cursor: pointer; font-weight: bold; outline: none;",
          paste0(cat, " (", nrow(cat_df), ")")
        ),
        div(
          style = "padding: 15px; background-color: white; border-top: 1px solid #d2d6de; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;",
          do.call(tagList, cat_ui)
        )
      )
    }
    
    tagList(
      p(paste0("Using ", length(feats),
               " race-specific top features for ",
               input$race, " (Age: ", if(age_stratum() == "leq65") "≤65" else ">65", ").")),
      div(
        style = "display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; align-items: start;",
        do.call(tagList, ui_blocks)
      )
    )
  })
  
  # ----------------------------------------------------------------------------
  # Calculation Logic
  # ----------------------------------------------------------------------------
  
  # Reactive value to store calculation results 
  calc_state <- reactiveVal(NULL)

  # Main Calculation Trigger
  observeEvent(input$calc_btn, {
    req(input$race, input$age)

    feats <- feature_list()
    if (length(feats) == 0) return()

    stratum <- age_stratum()

    # Harvest values from UI inputs (id = "feat__<feature_name>")
    values <- rep(NA_real_, length(feats))
    names(values) <- feats

    for (f in feats) {
      id <- paste0("feat__", f)
      if (!is.null(input[[id]])) values[[f]] <- as.numeric(input[[id]])
    }

    # Range validation (disabled — can enhance later)
    warn_df <- data.frame()

    # Calculate Completeness Metrics
    n_total <- length(feats)

    # Identify which features were auto-filled by the UI and NOT subsequently edited
    # imputed_ui_features() stores feature -> original_imputed_value pairs
    imputed_snapshot <- imputed_ui_features()
    auto_filled_features <- character(0)
    for (f in names(imputed_snapshot)) {
      current_val <- as.numeric(input[[paste0("feat__", f)]])
      original_val <- as.numeric(imputed_snapshot[[f]])
      # Round to 6 decimals to avoid float artifacts from JSON 
      if (!is.na(current_val) && !is.na(original_val) && round(current_val, 6) == round(original_val, 6)) {
        auto_filled_features <- c(auto_filled_features, f)
      }
    }

    n_provided <- sum(!is.na(values) & !(names(values) %in% auto_filled_features))
    initial_completeness <- if (n_total > 0) n_provided / n_total else 0
    imputed_flags <- rep(FALSE, length(values))
    names(imputed_flags) <- names(values)

    if (input$use_impute) {
      # Get the 10-year model for imputation values
      model_ref <- model_params$models[[input$race]][[stratum]][["10"]]

      if (!is.null(model_ref)) {
        imp <- impute_with_model(values, model_ref)
        values_used <- imp$values

        # Combined imputed flags
        backend_imputed <- imp$imputed
        ui_imputed <- names(values) %in% auto_filled_features

        imputed_flags <- backend_imputed | ui_imputed
      } else {
        values_used <- values
      }
    } else {
      values_used <- values
    }

    n_imputed <- sum(imputed_flags)
    final_completeness <- if (n_total > 0) sum(!is.na(values_used)) / n_total else 0

    # Get clinical thresholds from metadata
    thresholds <- model_params$metadata$clinical_thresholds

    # Calculate Risk for 5, 10, 15 years
    risk_results <- list()

    for (tp in c(5, 10, 15)) {
      # Get model for this timepoint
      model <- model_params$models[[input$race]][[stratum]][[as.character(tp)]]

      if (!is.null(model)) {
        res <- calculate_dementia_risk(
          user_data = values_used,
          model = model,
          age_stratum = stratum,
          thresholds = thresholds
        )

        if (!is.null(res)) {
          risk_results[[length(risk_results) + 1]] <- data.frame(
            time_years = tp,
            risk_percent = res$predicted_probability_pct,
            hr = res$hazard_ratio,
            lp = res$linear_predictor,
            sullivan = res$sullivan_score,
            risk_category = res$risk_category
          )
        }
      }
    }

    risks_df <- if(length(risk_results) > 0) do.call(rbind, risk_results) else data.frame()

    calc_state(list(
      values_raw = values,
      values_used = values_used,
      imputed_flags = imputed_flags,
      range_warnings = warn_df,
      risks = risks_df,

      # Completeness Metadata
      n_total_features = n_total,
      initial_completeness = initial_completeness,
      final_completeness = final_completeness,
      n_imputed = n_imputed,
      used_features = names(values_used[!is.na(values_used)])
    ))
  })

  # ----------------------------------------------------------------------------
  # Imputation Control (Auto-fill UI)
  # ----------------------------------------------------------------------------
  
  # When the "Impute missing" checkbox is toggled, either fill or clear UI inputs.
  # Note: Imputation values are the same across all timepoints for a given race/age stratum
  # Stores named list: feature_name -> value_set_in_UI
  imputed_ui_features <- reactiveVal(list())

  observeEvent(input$use_impute, {
    req(input$race, input$age)

    if (isTRUE(input$use_impute)) {
      # CHECKED: Impute missing values in UI immediately using JSON model
      feats <- feature_list()
      newly_imputed <- list()  # feature_name -> value set in UI

      # Get imputation values from the 10-year model
      stratum <- age_stratum()
      model_ref <- model_params$models[[input$race]][[stratum]][["10"]]

      if (is.null(model_ref)) return()

      impute_map <- model_ref$imputation_values

      # Get binary flags from stats_df
      sub <- filtered_stats()
      is_bin_map <- setNames(sub$is_binary, sub$feature)

      for (f in feats) {
        id <- paste0("feat__", f)
        curr_val <- input[[id]]

        # Check if empty (robust to different NA types)
        is_blank <- is.null(curr_val) || (length(curr_val) == 1 && is.na(curr_val)) || (is.character(curr_val) && curr_val == "")

        if (is_blank) {
           impute_val <- if (!is.null(impute_map[[f]])) impute_map[[f]] else NA
           is_bin <- if (!is.null(is_bin_map[[f]])) is_bin_map[[f]] else NA
           is_binary_feat <- !is.na(is_bin) && (isTRUE(as.logical(is_bin)) || is_bin == "True")

           # Check if it's a dropdown (has options) or numeric
           opts <- feature_options_df %>% filter(colname_match_data == !!f)

           if (nrow(opts) > 0) {
             # It's a dropdown. Find the option nearest to the imputation value.
             if(!is.na(impute_val)) {
               # Get all possible numeric values for this feature
               valid_vals <- suppressWarnings(as.numeric(opts$option_value))
               valid_vals <- valid_vals[!is.na(valid_vals)]

               if(length(valid_vals) > 0) {
                 # Find closest value
                 closest <- valid_vals[which.min(abs(valid_vals - impute_val))]
                 updateSelectInput(session, id, selected = as.character(closest))
                 newly_imputed[[f]] <- closest
               }
             }
           } else if (is_binary_feat) {
             # Binary fallback
             val <- ifelse(!is.na(impute_val) && impute_val > 0.5, 1, 0)
             updateSelectInput(session, id, selected = as.character(val))
             newly_imputed[[f]] <- val
           } else {
             # Numeric
             val <- if (is.na(impute_val)) NA_real_ else impute_val
             updateNumericInput(session, id, value = val)
             newly_imputed[[f]] <- val
           }
        }
      }
      # Track what we touched
      imputed_ui_features(newly_imputed)

    } else {
      # UNCHECKED: Clear values we previously imputed
      to_clear <- names(imputed_ui_features())

      # Need binary map for clearing logic too
      sub <- filtered_stats()
      is_bin_map <- setNames(sub$is_binary, sub$feature)

      for (f in to_clear) {
        id <- paste0("feat__", f)

        is_bin <- if (!is.null(is_bin_map[[f]])) is_bin_map[[f]] else NA
        is_binary_feat <- !is.na(is_bin) && (isTRUE(as.logical(is_bin)) || is_bin == "True")

        opts <- feature_options_df %>% filter(colname_match_data == !!f)

        if (nrow(opts) > 0 || is_binary_feat) {
          updateSelectInput(session, id, selected = "")
        } else {
          updateNumericInput(session, id, value = NA_real_)
        }
      }
      imputed_ui_features(list())
    }
  })

  # Auto-populate Age feature from global age input
  # Trigger on both age and race changes (race changes feature list)
  observeEvent(c(input$age, input$race), {
    req(input$age)
    # Update Age_p21003_i0 feature if it exists in the feature list
    feats <- feature_list()
    if ("Age_p21003_i0" %in% feats) {
      updateNumericInput(session, "feat__Age_p21003_i0", value = input$age)
    }
  })

  output$risk_table <- renderUI({
    st <- calc_state()
    if (is.null(st)) return(NULL)

    sullivan_unavailable <- all(is.na(st$risks$sullivan))

    df <- st$risks %>%
      mutate(
        risk_percent = round(risk_percent, 1),
        hr = round(hr, 2),
        lp = round(lp, 2),
        sullivan = ifelse(is.na(sullivan), "N/A", round(sullivan, 0))
      ) %>%
      select(time_years, risk_percent, risk_category, sullivan, hr, lp) %>%
      rename(
        `Time (Years)` = time_years,
        `Dementia Risk (%)` = risk_percent,
        `Risk Category` = risk_category,
        `Sullivan Risk Score` = sullivan,
        `Hazard Ratio` = hr,
        `Linear Predictor` = lp
      )

    table_out <- DT::datatable(df, options = list(dom = "t"), rownames = FALSE)

    if (sullivan_unavailable) {
      tagList(
        table_out,
        p(style = "font-size: 11px; color: #888; margin-top: 6px;",
          em("Note: Sullivan Risk Score is unavailable for this race/age group ",
             "(model instability due to insufficient events). ",
             "Risk is based on Cox probability only."))
      )
    } else {
      table_out
    }
  })

  output$risk_plot <- renderPlotly({
    st <- calc_state()
    if (is.null(st)) {
      # Return empty plot without NA values to avoid warnings
      df <- data.frame(time_years = factor(c(5,10,15)), risk_percent = c(0,0,0))
      return(plot_ly(df, x = ~time_years, y = ~risk_percent, type = "bar", marker = list(color = '#e0e0e0')) %>%
              layout(xaxis = list(title="Years"),
                     yaxis = list(title="Risk Percent (%)", range = c(0, 100)),
                     showlegend=FALSE,
                     annotations = list(
                       list(x = 0.5, y = 0.5, text = "Enter data and click Calculate",
                            xref = "paper", yref = "paper", showarrow = FALSE,
                            font = list(size = 14, color = "#999"))
                     )))
    }
  
    df <- st$risks %>% arrange(time_years)
    
    # Define colors
    df$color <- dplyr::case_when(
        df$risk_category == "Low" ~ "#28a745",    # Green
        df$risk_category == "Medium" ~ "#ffc107", # Orange
        df$risk_category == "High" ~ "#dc3545",   # Red
        TRUE ~ "#007bff"                          # Blue (default)
    )
    
    plot_ly(df, x = ~factor(time_years), y = ~risk_percent, type = "bar", marker = list(color = ~color)) %>%
      layout(xaxis = list(title="Years"), yaxis = list(title="Risk Percent (%)"), showlegend=FALSE)
  })

  # Feature Contribution List
  output$feature_contribution_list <- renderUI({
    st <- calc_state()
    if (is.null(st)) {
      return(p("Enter data and click Calculate to see risk factors",
               style = "text-align: center; color: #999; padding: 20px;"))
    }

    # Use 10-year model for contribution analysis
    stratum <- age_stratum()
    model <- model_params$models[[input$race]][[stratum]][["10"]]

    if (is.null(model)) return(NULL)

    # Get features and calculate contributions
    features <- unlist(model$features)
    coefficients <- model$coefficients

    # Calculate contribution = coefficient * standardized_value
    contributions <- sapply(features, function(f) {
      raw_val <- st$values_used[[f]]
      if (is.null(raw_val) || is.na(raw_val)) return(0)

      coef <- coefficients[[f]]
      center <- model$scaler_center[[f]]
      scale <- model$scaler_scale[[f]]

      if (is.null(coef) || is.null(center) || is.null(scale) || scale == 0) return(0)

      std_val <- (raw_val - center) / scale
      coef * std_val
    })

    # Get feature descriptions
    desc_map <- stats_df %>% select(feature, feature_description) %>% distinct()
    contrib_df <- data.frame(
      feature = features,
      contribution = contributions,
      stringsAsFactors = FALSE
    ) %>%
      left_join(desc_map, by = "feature") %>%
      mutate(feature_description = ifelse(is.na(feature_description), feature, feature_description)) %>%
      arrange(desc(abs(contribution))) %>%
      head(15)

    # Create colored list items
    list_items <- lapply(1:nrow(contrib_df), function(i) {
      color <- if (contrib_df$contribution[i] > 0) "#dc3545" else "#28a745"  # Red for increase, Green for decrease
      direction <- if (contrib_df$contribution[i] > 0) "↑" else "↓"

      tags$div(
        style = paste0("padding: 8px; margin: 4px 0; border-left: 4px solid ", color, "; background-color: #f8f9fa;"),
        tags$span(
          style = paste0("color: ", color, "; font-weight: bold; font-size: 16px;"),
          paste0(i, ". ", direction, " ", contrib_df$feature_description[i])
        )
      )
    })

    do.call(tagList, list_items)
  })

  # ----------------------------------------------------------------------------
  # Data Quality Renderers
  # ----------------------------------------------------------------------------

  output$completeness_display <- renderUI({
    st <- calc_state()
    if (is.null(st)) return(NULL)

    div(
      fluidRow(
        column(3, div(class = "info-box-good",
                      h5("Initial:"),
                      h3(paste0(round(st$initial_completeness * 100), "%")))),
        column(3, div(class = "info-box-good",
                      h5("Final:"),
                      h3(paste0(round(st$final_completeness * 100), "%")))),
        column(3, div(class = "info-box-good",
                      h5("Provided:"),
                      h3(length(st$used_features) - st$n_imputed))),
        column(3, div(class = "info-box-good",
                      h5("Imputed:"),
                      h3(st$n_imputed)))
      ),
      br(),
      p(paste0("Total features in model: ", st$n_total_features))
    )
  })

  output$range_validation_display <- renderUI({
    st <- calc_state()
    if (is.null(st)) return(NULL)

    if (nrow(st$range_warnings) > 0) {
      div(p(paste(nrow(st$range_warnings), "feature(s) outside training range")))
    } else {
      div(class = "info-box-good",
          p("All feature values within training data ranges"))
    }
  })

  output$feature_status_display <- renderUI({
    st <- calc_state()
    if (is.null(st)) return(NULL)

    imp <- st$imputed_flags
    desc_map <- stats_df %>% select(feature, feature_description) %>% distinct()

    # Split features into provided and imputed lists
    provided_rows <- list()
    imputed_rows <- list()

    for (f in names(imp)) {
      val_used <- st$values_used[[f]]
      if (is.na(val_used)) next

      desc_row <- desc_map %>% filter(feature == !!f)
      label <- if (nrow(desc_row) > 0) desc_row$feature_description[1] else f

      if (as.logical(imp[[f]])) {
        imputed_rows[[length(imputed_rows) + 1]] <- tags$tr(
          tags$td(style = "padding: 4px 8px;", label),
          tags$td(style = "padding: 4px 8px; text-align: right;", round(val_used, 2))
        )
      } else {
        provided_rows[[length(provided_rows) + 1]] <- tags$tr(
          tags$td(style = "padding: 4px 8px;", label),
          tags$td(style = "padding: 4px 8px; text-align: right;", round(val_used, 2))
        )
      }
    }

    # Build provided table
    provided_table <- if (length(provided_rows) > 0) {
      div(style = "background-color: #e8f5e8; border-left: 3px solid #28a745; border-radius: 3px; padding: 8px; max-height: 400px; overflow-y: auto;",
        tags$table(style = "width: 100%; border-collapse: collapse;",
          tags$thead(tags$tr(
            tags$th(style = "padding: 4px 8px; text-align: left; border-bottom: 1px solid #28a745;", "Feature"),
            tags$th(style = "padding: 4px 8px; text-align: right; border-bottom: 1px solid #28a745;", "Provided Value")
          )),
          tags$tbody(provided_rows)
        )
      )
    } else {
      div(style = "background-color: #e8f5e8; border-left: 3px solid #28a745; border-radius: 3px; padding: 8px;",
          p(style = "margin: 0; color: #666;", "No provided features"))
    }

    # Build imputed table
    imputed_table <- if (length(imputed_rows) > 0) {
      div(style = "background-color: #e7f3ff; border-left: 3px solid #2196F3; border-radius: 3px; padding: 8px; max-height: 400px; overflow-y: auto;",
        tags$table(style = "width: 100%; border-collapse: collapse;",
          tags$thead(tags$tr(
            tags$th(style = "padding: 4px 8px; text-align: left; border-bottom: 1px solid #2196F3;", "Feature"),
            tags$th(style = "padding: 4px 8px; text-align: right; border-bottom: 1px solid #2196F3;", "Imputed Value")
          )),
          tags$tbody(imputed_rows)
        )
      )
    } else {
      div(style = "background-color: #e7f3ff; border-left: 3px solid #2196F3; border-radius: 3px; padding: 8px;",
          p(style = "margin: 0; color: #666;", "No imputed features"))
    }

    fluidRow(
      column(6, h4("Provided Features"), provided_table),
      column(6, h4("Imputed Features"), imputed_table,
        p(style = "font-size: 11px; color: #888; margin-top: 6px; font-style: italic;",
          "Note: Missing values are filled using the median (continuous) or mode (binary) of disease-free participants from the training data.")
      )
    )
  })

  # ----------------------------------------------------------------------------
  # Model Info Renderers
  # ----------------------------------------------------------------------------

  # Model specs display
  output$model_specs_display <- renderUI({
    specs <- tagList()
    for (race in c("White", "Black", "Asian")) {
      model_ref <- model_params$models[[race]][["leq65"]][["10"]]
      n_features <- if (!is.null(model_ref)) length(unlist(model_ref$features)) else 0

      age_strata <- model_params$metadata$age_strata
      age_labels <- sapply(age_strata, function(s) model_params$metadata$age_strata_labels[[s]])
      age_str <- paste(age_labels, collapse = ", ")

      specs <- tagList(
        specs,
        p(strong(paste0(race, ":")),
          paste0(n_features, " top features | Age Groups: ", age_str))
      )
    }
    specs
  })

  # Feature info table
  output$feature_info_table <- DT::renderDataTable({
    req(input$info_race, input$info_age_stratum, input$info_timepoint)

    # Get model from JSON
    model <- model_params$models[[input$info_race]][[input$info_age_stratum]][[as.character(input$info_timepoint)]]

    if (is.null(model)) {
      return(datatable(data.frame(Note = "No model available for this combination"),
                      options = list(dom = "t"), rownames = FALSE))
    }

    # Extract features, coefficients, and Sullivan points (NULL if sullivan_params unavailable)
    features <- unlist(model$features)
    coefficients <- model$coefficients
    point_weights <- if (!is.null(model$sullivan_params)) model$sullivan_params$point_weights else NULL

    # Build dataframe
    df_rows <- lapply(features, function(f) {
      coef <- coefficients[[f]]
      hr <- if (!is.null(coef)) exp(coef) else NA
      points <- point_weights[[f]]

      data.frame(
        feature = f,
        coefficient = if (!is.null(coef)) coef else NA,
        hr = hr,
        sullivan_points = if (!is.null(points)) points else NA,
        stringsAsFactors = FALSE
      )
    })

    df <- do.call(rbind, df_rows)

    # Get descriptions
    desc_map <- stats_df %>% select(feature, feature_description) %>% distinct()

    # Join and format
    joined <- df %>%
      left_join(desc_map, by = "feature") %>%
      select(feature_description, coefficient, hr, sullivan_points) %>%
      rename(
        `Feature` = feature_description,
        `Coefficient (β)` = coefficient,
        `Hazard Ratio` = hr,
        `Sullivan Points` = sullivan_points
      ) %>%
      arrange(desc(abs(ifelse(is.na(`Sullivan Points`), `Coefficient (β)`, `Sullivan Points`))))

    DT::datatable(joined,
                  options = list(pageLength = 15, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(c("Coefficient (β)", "Hazard Ratio", "Sullivan Points"), 3)
  })

}

shinyApp(ui = ui, server = server)
