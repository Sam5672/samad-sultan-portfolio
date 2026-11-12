# Solar Installation Time Predictor

**ENGR 1020, University of Virginia · Team project for Midwest Solar LLC · Jan–May 2025**
**My role:** I built the backend prediction model, a Random Forest in R. Teammates built the JavaFX desktop interface that calls it.

## The problem

Midwest Solar schedules installation crews from estimated job lengths. When an estimate is too short, a crew runs over, later jobs slip and customers are kept waiting. The company wanted a tool that predicts how long an installation will take from the job's details, and that leans toward overestimating rather than underestimating.

## What I built

`src/install_time_random_forest.R` takes historical job records and:

1. **Cleans the data:** standardizes column names, drops ID and empty columns, converts categorical fields (attachment type, array type, interconnection type, install season, azimuth and others) to factors, and removes rows with missing or invalid values.
2. **Engineers features:** for example, panels per story, from panel count and building height.
3. **Trains a tuned Random Forest:** 1,000 trees, with `mtry` tuned over 2–7 using 10-fold cross-validation (`caret`).
4. **Biases toward overestimating:** follows the business rule that an underestimate costs more than an overestimate.
5. **Adds a confidence score:** each prediction gets a confidence value based on how much the individual trees disagree. A wide spread between trees means lower confidence.
6. **Reports results:** RMSE, MAE and R², the share of predictions within 3 hours, and a feature-importance plot.

## Tech stack

R · randomForest · caret · dplyr · janitor · readxl

## Running it

```r
install.packages(c("randomForest", "readxl", "janitor", "caret", "dplyr"))
```
```bash
Rscript src/install_time_random_forest.R path/to/data_export.xlsx
```

**Data is not included.** The job records belong to Midwest Solar LLC and can't be shared publicly. The script expects an Excel export with one row per job. The target column is `estimated_total_direct_time`, and the predictors include fields such as `panel_qty`, `number_of_stories`, `array_type`, `attachment_type`, `interconnection_type`, `install_season`, `truss_rafter` and `azimuth`.

## Known issues and next steps

On reviewing this code later, I found two problems that make the reported accuracy look better than it really is. I'm leaving them documented here rather than hiding them:

- **Target leakage in a feature.** `time_per_panel` is calculated as `installation_time / panel_qty` and then used as a predictor. Combined with `panel_qty`, it gives the model the answer. **Fix:** remove it from the predictors.
- **The "never underestimate" adjustment uses the true value.** `pmax(predictions, actual)` raises each prediction using the real installation time, which isn't known when making a real prediction. This also inflates the "% within 3 hours" figure. **Fix:** add a safety margin learned from the model's errors on the training data (for example, a high percentile of the residuals), or use quantile regression forests to predict an upper bound directly.

Other improvements I'd make: hold out the most recent jobs as the test set instead of using a random split, and test whether more job details improve accuracy.
