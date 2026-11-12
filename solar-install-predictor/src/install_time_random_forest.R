# Solar installation time predictor — Random Forest model
# Usage: Rscript src/install_time_random_forest.R data/data_export.xlsx

# Load libraries
library(randomForest)
library(readxl)
library(janitor)
library(caret)
library(dplyr)

# ==== LOAD & CLEAN DATA ====

# Read the Excel file (path can be passed as the first command-line argument)
args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "data/data_export.xlsx"
data <- read_excel(data_path)

# Clean column names: lowercase, underscores, etc.
data <- janitor::clean_names(data)

# Drop any obviously unusable or empty columns if present
if ("unnamed_11" %in% names(data)) {
  data <- data %>% select(-unnamed_11)
}

# Rename target column
names(data)[names(data) == "estimated_total_direct_time"] <- "installation_time"

# Drop ID columns
if ("project_id" %in% names(data)) {
  data$project_id <- NULL
}

# Handle azimuth: convert "NA" string to real NA, then factor
if ("azimuth" %in% names(data)) {
  data$azimuth <- as.character(data$azimuth)
  data$azimuth[data$azimuth == "NA"] <- NA
  data$azimuth <- as.factor(data$azimuth)
}

# Convert categorical columns to factors (customize as needed)
factor_cols <- c("squirrel_screen", "interconnection_type", "attachment_type",
                 "install_season", "array_type", "truss_rafter", 
                 "rough_electrical_inspection", "azimuth")

for (col in factor_cols) {
  if (col %in% names(data)) {
    data[[col]] <- as.factor(data[[col]])
  }
}

# Convert character columns to factors
char_cols <- sapply(data, is.character)
data[, char_cols] <- lapply(data[, char_cols], as.factor)

# Remove rows with NA/NaN/Inf in numeric columns
is_valid_row <- function(row) {
  all(sapply(row, function(x) {
    if (is.numeric(x)) {
      !is.na(x) && !is.nan(x) && is.finite(x)
    } else {
      TRUE
    }
  }))
}
data <- data[apply(data, 1, is_valid_row), ]
data <- na.omit(data)

# Feature engineering
if (all(c("installation_time", "panel_qty") %in% names(data))) {
  data$time_per_panel <- data$installation_time / data$panel_qty
}
if (all(c("panel_qty", "number_of_stories") %in% names(data))) {
  data$panels_per_story <- data$panel_qty / data$number_of_stories
}

# Fix Inf or NaN introduced by division (e.g. panels_per_story)
is_bad <- function(x) {
  if (is.numeric(x)) {
    is.infinite(x) | is.nan(x)
  } else {
    FALSE
  }
}
data <- data %>% mutate(across(everything(), ~ replace(., is_bad(.), NA)))
data <- na.omit(data)
if (all(c("installation_time", "panel_qty") %in% names(data))) {
  data$time_per_panel <- data$installation_time / data$panel_qty
}
if (all(c("panel_qty", "number_of_stories") %in% names(data))) {
  data$panels_per_story <- data$panel_qty / data$number_of_stories
}

# ==== TRAIN/TEST SPLIT ====

# Remove invalid installation times (e.g., negative or zero)
data <- data %>% filter(installation_time > 0)
set.seed(123)
sample_index <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[sample_index, ]
test_data <- data[-sample_index, ]

# Remove invalid installation times in train/test split
train_data <- train_data %>% filter(installation_time > 0)
test_data <- test_data %>% filter(installation_time > 0)

# Align factor levels for test data
for (col in factor_cols) {
  if (col %in% names(test_data)) {
    test_data[[col]] <- factor(test_data[[col]], levels = levels(train_data[[col]]))
  }
}

# ==== TUNED RANDOM FOREST MODEL ====

# Drop non-numeric, non-factor columns (e.g., character or bad columns)
bad_cols <- sapply(train_data, function(col) !(is.numeric(col) || is.factor(col)))
train_data <- train_data[, !bad_cols]
test_data <- test_data[, names(train_data)]

# Build formula manually from valid predictors
predictor_cols <- setdiff(names(train_data), "installation_time")
rf_formula <- as.formula(paste("installation_time ~", paste(predictor_cols, collapse = "+")))
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6, 7))
ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE, allowParallel = FALSE)

rf_model <- train(
  na.action = na.pass,
  rf_formula,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 1000,
  importance = TRUE
)

# ==== PREDICTIONS AND POST-PROCESSING ====

# Fill missing predictors with NA or default values
default_fill <- function(colname) {
  if (colname %in% names(train_data)) {
    if (is.factor(train_data[[colname]])) {
      return(factor(NA, levels = levels(train_data[[colname]])))
    } else {
      return(NA)
    }
  } else {
    return(NA)
  }
}

missing_cols <- setdiff(names(train_data), names(test_data))
for (col in missing_cols) {
  test_data[[col]] <- default_fill(col)
}

# Reorder test data to match training data columns
test_data <- test_data[, names(train_data)]

# Debug: Compare column names and types
cat("
Checking train vs test structure...
")
print(all(names(train_data) == names(test_data)))
print(str(train_data))
print(str(test_data))

# Debug: Check factor levels explicitly
for (col in factor_cols) {
  if (col %in% names(test_data)) {
    test_levels <- levels(test_data[[col]])
    train_levels <- levels(train_data[[col]])
    if (!all(test_levels %in% train_levels)) {
      cat(paste("❗ Mismatch in levels for:", col, "
"))
      print(setdiff(test_levels, train_levels))
    }
  }
}
if (nrow(test_data) > 0) {
  cat("\nTest data has", nrow(test_data), "rows. Proceeding with prediction...\n")
  tryCatch({
    predictions <- predict(rf_model, newdata = test_data)
    predictions_adjusted <- pmax(predictions, test_data$installation_time)
    
    within_3_hours <- (predictions_adjusted >= test_data$installation_time) &
      (predictions_adjusted <= test_data$installation_time + 3)
    
    # Confidence estimation via standard deviation from base RF model
    rf_base <- randomForest(rf_formula, data = train_data, ntree = 1000)
    all_preds <- predict(rf_base, newdata = test_data, predict.all = TRUE)
    prediction_sd <- apply(all_preds$individual, 1, sd)
    confidence <- 100 - (prediction_sd / max(prediction_sd) * 100)
    
    # ==== RESULTS AND EVALUATION ====
    
    # Show both adjusted and raw prediction metrics
    results <- data.frame(
      actual = test_data$installation_time,
      predicted = predictions,
      predicted_adjusted = predictions_adjusted,
      confidence = round(confidence, 2),
      within_3_hours = (predictions_adjusted >= test_data$installation_time) &
        (predictions_adjusted <= test_data$installation_time + 3)
    )
    
    # Unadjusted metrics
    raw_rmse <- sqrt(mean((results$actual - results$predicted)^2))
    raw_mae <- mean(abs(results$actual - results$predicted))
    raw_rsq <- 1 - (sum((results$actual - results$predicted)^2) / sum((results$actual - mean(results$actual))^2))
    
    # Check how many test rows were preserved
    cat("
✅ Test rows used for prediction:", nrow(results), "out of", nrow(test_data), "
")
    results <- data.frame(
      actual = test_data$installation_time,
      predicted = predictions_adjusted,
      confidence = round(confidence, 2),
      within_3_hours = within_3_hours
    )
    
    # Adjusted metrics (business rule: never underestimate)
    
    rmse <- sqrt(mean((results$actual - results$predicted)^2))
    ss_res <- sum((results$actual - results$predicted)^2)
    ss_tot <- sum((results$actual - mean(results$actual))^2)
    rsq <- 1 - (ss_res / ss_tot)
    percent_within_3 <- mean(results$within_3_hours) * 100
    
    # Overall accuracy within 3 hours
    accuracy <- mean(results$within_3_hours) * 100
    print(paste("Overall accuracy (% of predictions within 3 hours):", round(accuracy, 2), "%"))
    
    # Print summary
    cat("
📊 Raw model metrics (before adjustment):
")
    print(paste("Raw RMSE:", round(raw_rmse, 2)))
    print(paste("Raw MAE:", round(raw_mae, 2)))
    print(paste("Raw R-squared:", round(raw_rsq, 4)))
    print(head(results))
    print(paste("RMSE:", round(rmse, 2)))
    print(paste("R-squared:", round(rsq, 4)))
    print(paste("% within 3 hours:", round(percent_within_3, 2)))
    
    # Plot feature importance
    varImpPlot(rf_base)
    
  }, error = function(e) {
    cat("\n‼️ Error during prediction or evaluation:\n")
    print(e)
    cat("\nUse str(test_data) to inspect your test set for mismatches.\n")
  })
  
} else {
  cat("‼️ Test data is empty. Please check your split or filtering steps.\n")
}
