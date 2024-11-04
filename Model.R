# Install required packages if they are not already installed
required_packages <- c("tidyverse", "caret", "glmnet", "ranger", "kernlab", "gbm")
installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Install necessary packages
install.packages("randomForest")
install.packages("gbm")
install.packages("doParallel")

# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(corrplot)
library(randomForest)
library(e1071)
library(glmnet)
library(gbm)
library(doParallel)

# Load the dataset
data <- read.csv("C:/Users/rsd20/OneDrive/Documents/Masters/Unit 3 Statistical Data Analytics and Databases/Assignment 2/city_day dataset.csv")

# View the structure and summary of the dataset
str(data)
summary(data)

# Check for missing values
colSums(is.na(data))

# Remove rows with missing values
data <- na.omit(data) 

# Set seed for reproducibility
set.seed(123)

# Split data into training (60%), validation (20%), and test (20%) sets
trainIndex <- createDataPartition(data$AQI, p = 0.6, list = FALSE)
train_data <- data[trainIndex, ]
temp_data <- data[-trainIndex, ]

valIndex <- createDataPartition(temp_data$AQI, p = 0.5, list = FALSE)
validation_data <- temp_data[valIndex, ]
test_data <- temp_data[-valIndex, ]

# Assuming your data has already been split into train_data, validation_data, and test_data

# Remove the Date column from all datasets
train_data <- train_data %>% select(-Date)
validation_data <- validation_data %>% select(-Date)
test_data <- test_data %>% select(-Date)

# Proceed with feature engineering, modeling, and training as usual

# Feature Selection: Correlation Analysis
# Select numeric columns for correlation analysis
numeric_train_data <- train_data %>% select(where(is.numeric))

# Check if there are at least two numeric columns before calculating correlation
if (ncol(numeric_train_data) >= 2) {
  cor_matrix <- cor(numeric_train_data)
  corrplot(cor_matrix, method = "circle")
} else {
  print("Not enough numeric columns for correlation analysis.")
}

# Feature Selection: Recursive Feature Elimination (RFE)
predictors <- train_data %>% select(-AQI) %>% select(where(is.numeric))
target <- train_data$AQI

# Check for any NA or infinite values in predictors
predictors <- predictors %>% na.omit()
predictors <- predictors[is.finite(rowSums(predictors)), ]

# Define control function for RFE using lmFuncs
control <- rfeControl(functions = lmFuncs, method = "cv", number = 10)

# Run RFE
results_rfe <- tryCatch(
  rfe(predictors, target, sizes = c(1:10), rfeControl = control),
  error = function(e) { message("RFE failed: ", e); NULL }
)

if (!is.null(results_rfe)) {
  selected_features_rfe <- predictors(results_rfe)
  print("RFE successful")
} else {
  print("RFE could not complete.")
}

# Convert predictors to matrix format for glmnet
x <- as.matrix(train_data %>% select(-AQI))
y <- as.numeric(train_data$AQI)

# Fit lasso model with cross-validation
lasso_model <- cv.glmnet(x, y, alpha = 1)

# Extract coefficients at the optimal lambda and convert to matrix
lasso_coefficients <- as.matrix(coef(lasso_model, s = "lambda.min"))

# Identify non-zero coefficients, excluding the intercept
selected_features_lasso <- rownames(lasso_coefficients)[lasso_coefficients[, 1] != 0]
selected_features_lasso <- selected_features_lasso[selected_features_lasso != "(Intercept)"]

# Output the selected features
print(selected_features_lasso)

# Parallel Processing Setup
cores <- detectCores()
cl <- makeCluster(cores - 1)  # Reserve one core for system processes
registerDoParallel(cl)

# Benchmark Model (Linear Regression)
system.time({
  benchmark_model <- train(AQI ~ ., data = train_data, method = "lm", trControl = trainControl(allowParallel = TRUE))
  preds_benchmark <- predict(benchmark_model, validation_data)
  benchmark_performance <- postResample(preds_benchmark, validation_data$AQI)
  print(benchmark_performance)
})

# Random Forest Model
rf_grid <- expand.grid(mtry = seq(1, ncol(train_data) - 1, by = 1), splitrule = "variance", min.node.size = c(1, 5, 10))
rf_control <- trainControl(method = "cv", number = 5, search = "random")

system.time({
  rf_model <- train(AQI ~ ., data = train_data, method = "ranger", trControl = rf_control, tuneGrid = rf_grid, num.trees = 500)
  rf_preds <- predict(rf_model, validation_data)
  rf_performance <- postResample(rf_preds, validation_data$AQI)
  print(rf_performance)
})

# SVM Model
svm_grid <- expand.grid(C = c(0.01, 0.1, 1, 10), sigma = c(0.01, 0.1, 1))
svm_control <- trainControl(method = "cv", number = 5)

system.time({
  svm_model <- train(AQI ~ ., data = train_data, method = "svmRadial", trControl = svm_control, tuneGrid = svm_grid)
  svm_preds <- predict(svm_model, validation_data)
  svm_performance <- postResample(svm_preds, validation_data$AQI)
  print(svm_performance)
})

# GBM Model with Correct Tuning Grid
gbm_grid <- expand.grid(
  interaction.depth = c(1, 5, 10), 
  n.trees = seq(50, 500, by = 50), 
  shrinkage = c(0.01, 0.1, 0.2),
  n.minobsinnode = c(5, 10)  # Add this line for minimum observations in node
)

gbm_control <- trainControl(method = "cv", number = 5, search = "random")

# Train the GBM model
system.time({
  gbm_model <- train(
    AQI ~ ., 
    data = train_data, 
    method = "gbm", 
    trControl = gbm_control, 
    tuneGrid = gbm_grid, 
    verbose = FALSE
  )
  
  # Predictions and performance for GBM
  gbm_preds <- predict(gbm_model, validation_data)
  gbm_performance <- postResample(gbm_preds, validation_data$AQI)
  print(gbm_performance)
})

# Stop Parallel Processing
stopCluster(cl)
registerDoSEQ()

# Final Predictions on Test Data
test_preds_benchmark <- predict(benchmark_model, test_data)
test_preds_rf <- predict(rf_model, test_data)
test_preds_svm <- predict(svm_model, test_data)
test_preds_gbm <- predict(gbm_model, test_data)

# Calculate Performance on Test Data
benchmark_test_performance <- postResample(test_preds_benchmark, test_data$AQI)
rf_test_performance <- postResample(test_preds_rf, test_data$AQI)
svm_test_performance <- postResample(test_preds_svm, test_data$AQI)
gbm_test_performance <- postResample(test_preds_gbm, test_data$AQI)

# Compile performance metrics from each model
results <- data.frame(
  Model = c("Benchmark", "Random Forest", "SVM", "GBM"),
  RMSE = c(benchmark_test_performance[1], rf_test_performance[1], svm_test_performance[1], gbm_test_performance[1]),
  R2 = c(benchmark_test_performance[2], rf_test_performance[2], svm_test_performance[2], gbm_test_performance[2])
)

# Display results
print(results)