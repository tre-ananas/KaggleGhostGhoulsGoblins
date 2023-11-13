####################################################################################################
####################################################################################################
# Ghosts, Ghouls, Goblins (Kaggle)                                       ###########################
# Ryan Wolff                                                             ###########################
# 11 November 2023                                                       ###########################
# Data Location and Description:                                         ###########################
# https://www.kaggle.com/competitions/ghouls-goblins-and-ghosts-boo/data ###########################
####################################################################################################
####################################################################################################


########################################################################
########################################################################
#############################
# Start run in parallel
# cl <- makePSOCKcluster(3)
# registerDoParallel(cl)

# End run in parallel
# stopCluster(cl)
#############################
########################################################################
########################################################################

#################################################################
#################################################################
# Impute Missing Data from Special Missing Data Set #############
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)

# Load Data
ggg_train_na <- vroom("trainWithmissingValues.csv")
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# EDA for NAs
na_count <-sapply(ggg_train_na, function(y) sum(length(which(is.na(y)))))
data.frame(na_count)

# Impute Missing Values
# Create Recipe
imp_rec <- recipe(type ~ ., data = ggg_train_na) %>%
  step_impute_mean(hair_length) %>%
  step_impute_mean(rotting_flesh) %>%
  step_impute_mean(bone_length)

# Prep, Bake, and View Recipe
imp_prep <- prep(imp_rec)
imp_train <- bake(imp_prep, ggg_train_na)

# Calculate RMSE
rmse_vec(ggg_train[is.na(ggg_train_na)], imp_train[is.na(ggg_train_na)]) # RMSE = .1526155

#################################################################
#################################################################
# Gradient Boosted Trees                            #############
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(xgboost)
library(embed)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Recipe (leave out 'id')
xgb_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_lencode_glm(color, outcome = vars(type))# glm allows you to target encode a factor on a factor

xgb_prep <- prep(xgb_rec)
bake(xgb_prep, ggg_train)

# Create an XGBopst model specification
xgb_spec <- boost_tree(trees = tune(), tree_depth = tune(), min_n = tune(), learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Create an XGBoost Workflow
xgb_wf <- workflow() %>%
  add_recipe(xgb_rec) %>%
  add_model(xgb_spec)

xgb_grid <- grid_regular(
  trees(range = c(100, 1000)),
  tree_depth(range = c(1, 10)),
  min_n(range = c(1, 10)),
  learn_rate(range = c(0.01, 0.1)),
  levels = 10
)

# Split data for cross-validation (CV)
xgb_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation
xgb_cv_results <- xgb_wf %>%
  tune_grid(resamples = xgb_folds,
            grid = xgb_grid,
            metrics = metric_set(accuracy))

# Find best tuning parameters
xgb_best_tune <- xgb_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it
xgb_final_wf <- xgb_wf %>%
  finalize_workflow(xgb_best_tune) %>%
  fit(data = ggg_train)

# Predict class
xgb_preds <- predict(xgb_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions
vroom_write(x=xgb_preds, file="xgb_preds_3.csv", delim = ",")

#################################################################
#################################################################
# Random Forest                                     #############
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(parsnip)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Recipe (leave out 'id')
rf_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_dummy(color) %>%
  step_zv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

rf_prep <- prep(rf_rec)
bake(rf_prep, ggg_train)

# Create Random Forest model specification
rf_spec <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create classification forest workflow
rf_wf <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_spec)

# Grid of values to tune over
rf_tg <- grid_regular(mtry(range = c(1, 5)),
                         min_n(),
                         levels = 5)

# Split data for cross-validation (CV)
rf_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation
rf_cv_results <- rf_wf %>%
  tune_grid(resamples = rf_folds,
            grid = rf_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters
rf_best_tune <- rf_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it
rf_final_wf <- rf_wf %>%
  finalize_workflow(rf_best_tune) %>%
  fit(data = ggg_train)

# Predict class
rf_preds <- predict(rf_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions
vroom_write(x=rf_preds, file="rf_preds_2.csv", delim = ",")

#################################################################
#################################################################
# Single-Layer Neural Network                       #############
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Recipe (leave out 'id')
nn_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_dummy(color) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1) # Scale Xs to [0, 1]

nn_prep <- prep(nn_rec)
bake(nn_prep, ggg_train)

# Create Neural Network model specification
nn_spec <- mlp(hidden_units = tune(),
               epochs = 200) %>%
  set_engine("nnet") %>%
  set_mode('classification')

# Create classification NN workflow
nn_wf <- workflow() %>%
  add_recipe(nn_rec) %>%
  add_model(nn_spec)

# Grid of values to tune over
nn_tg <- grid_regular(hidden_units(range = c(1, 10)),
                      levels = 10)

# Split data for cross-validation (CV)
nn_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation
nn_cv_results <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters
nn_best_tune <- nn_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it
nn_final_wf <- nn_wf %>%
  finalize_workflow(nn_best_tune) %>%
  fit(data = ggg_train)

# Plot cross-validation with hidden_units on the x-axis and mean(accuracy) on the y-axis for class
nn_cv_results %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

# Predict class
nn_preds <- predict(nn_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions
vroom_write(x=nn_preds, file="nn_preds_2.csv", delim = ",")


#################################################################
#################################################################
# Support Vector Machines                      ##################
#################################################################
#################################################################

# DATA CLEANING -------------------------------------------------

# Load Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(kernlab)
library(ggplot2)
library(patchwork)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Box plot for bone_length
bl_bw <- ggplot(ggg_train, aes(x = type, y = bone_length, fill = type)) +
  geom_boxplot() +
  labs(title = "Box Plot for Bone Length by Type")

# Box plot for rotting_flesh
rf_bw <- ggplot(ggg_train, aes(x = type, y = rotting_flesh, fill = type)) +
  geom_boxplot() +
  labs(title = "Box Plot for Rotting Flesh by Type")

# Box plot for hair_length
hl_bw <- ggplot(ggg_train, aes(x = type, y = hair_length, fill = type)) +
  geom_boxplot() +
  labs(title = "Box Plot for Hair Length by Type")

# Box plot for has_soul
hs_bw <- ggplot(ggg_train, aes(x = type, y = has_soul, fill = type)) +
  geom_boxplot() +
  labs(title = "Box Plot for Has Soul by Type")

# Create a 4-Way Plot of box and whisker plots
fourway_bw <- (bl_bw + rf_bw) / (hl_bw + hs_bw)
fourway_bw

# Scatter plot for id vs. bone_length
bl_sp <- ggplot(ggg_train, aes(x = id, y = bone_length, color = type)) +
  geom_point() +
  labs(title = "Scatter Plot for id vs. Bone Length", x = "ID", y = "Bone Length")

# Scatter plot for id vs. rotting_flesh
rf_sp <- ggplot(ggg_train, aes(x = id, y = rotting_flesh, color = type)) +
  geom_point() +
  labs(title = "Scatter Plot for id vs. Rotting Flesh", x = "ID", y = "Rotting Flesh")

# Scatter plot for id vs. hair_length
hl_sp <- ggplot(ggg_train, aes(x = id, y = hair_length, color = type)) +
  geom_point() +
  labs(title = "Scatter Plot for id vs. Hair Length", x = "ID", y = "Hair Length")

# Scatter plot for id vs. has_soul
hs_sp <- ggplot(ggg_train, aes(x = id, y = has_soul, color = type)) +
  geom_point() +
  labs(title = "Scatter Plot for id vs. Has Soul", x = "ID", y = "Has Soul")

# Create a 4-Way Plot of box and whisker plots
fourway_sp <- (bl_sp + rf_sp) / (hl_sp + hs_sp)
fourway_sp

# Recipe (leave out 'id')
svms_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_normalize(all_predictors())

svms_prep <- prep(svms_rec)
bake(svms_prep, ggg_train)

# Create linear SVMS model
svms_lin_mod <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create radial SVMS model
svms_rad_mod <- svm_rbf(rbf_sigma = tune(),
                        cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create poly SVMS model
svms_poly_mod <- svm_poly(degree = tune(),
                          cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create linear SVMS workflow
svms_lin_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_lin_mod)

# Create radial SVMS workflow
svms_rad_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_rad_mod)

# Create poly SVMS workflow
svms_poly_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_poly_mod)

# Grid of values to tune over (linear)
svms_lin_tg <- grid_regular(cost(),
                            levels = 10)

# Grid of values to tune over (radial)
svms_rad_tg <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 10)

# Grid of values to tune over (poly)
svms_poly_tg <- grid_regular(degree(),
                            cost(),
                            levels = 15)

# Split data for cross-validation (CV) (linear)
svms_lin_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Split data for cross-validation (CV) (radial)
svms_rad_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Split data for cross-validation (CV) (poly)
svms_poly_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation (linear)
svms_lin_cv_results <- svms_lin_wf %>%
  tune_grid(resamples = svms_lin_folds,
            grid = svms_lin_tg,
            metrics = metric_set(accuracy))

# Run cross-validation (radial)
svms_rad_cv_results <- svms_rad_wf %>%
  tune_grid(resamples = svms_rad_folds,
            grid = svms_rad_tg,
            metrics = metric_set(accuracy))

# Run cross-validation (poly)
svms_poly_cv_results <- svms_poly_wf %>%
  tune_grid(resamples = svms_poly_folds,
            grid = svms_poly_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters (linear)
svms_lin_best_tune <- svms_lin_cv_results %>%
  select_best("accuracy")

# Find best tuning parameters (radial)
svms_rad_best_tune <- svms_rad_cv_results %>%
  select_best("accuracy")

# Find best tuning parameters (poly)
svms_poly_best_tune <- svms_poly_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it (linear)
svms_lin_final_wf <- svms_lin_wf %>%
  finalize_workflow(svms_lin_best_tune) %>%
  fit(data = ggg_train)

# Finalize workflow and fit it (radial)
svms_rad_final_wf <- svms_rad_wf %>%
  finalize_workflow(svms_rad_best_tune) %>%
  fit(data = ggg_train)

# Finalize workflow and fit it (poly)
svms_poly_final_wf <- svms_poly_wf %>%
  finalize_workflow(svms_poly_best_tune) %>%
  fit(data = ggg_train)

# Predict class (linear)
svms_lin_preds <- predict(svms_lin_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Predict class (radial)
svms_rad_preds <- predict(svms_rad_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Predict class (poly)
svms_poly_preds <- predict(svms_poly_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions (linear)
vroom_write(x=svms_lin_preds, file="svms_lin_preds_2.csv", delim = ",")

# Create a CSV with the predictions (radial)
vroom_write(x=svms_rad_preds, file="svms_rad_preds_3.csv", delim = ",")

# Create a CSV with the predictions (poly)
vroom_write(x=svms_poly_preds, file="svms_poly_preds_3.csv", delim = ",")

#################################################################
#################################################################
# Support Vector Machines with Recursive Feature Elimination ####
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(kernlab)
library(caret)
library(ggplot2)
library(e1071)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" and "color" into factors
ggg_train$type <- as.factor(ggg_train$type)
ggg_train$color <- as.factor(ggg_train$color)

# Define the control parameters for the RFE
ctrl <- rfeControl(functions=rfFuncs, 
                   method="cv", 
                   number=10)

# Specify the features and the target variable
features <- c("bone_length", "rotting_flesh", "hair_length", "has_soul", "color")
target <- "type"  # Replace with your target variable name

# Create the RFE model
# Random Forest
rfe_model <- rfe(x = ggg_train[features], y = ggg_train[[target]], sizes=c(1:length(features)),
                 rfeControl=ctrl, metric = "Accuracy")
# Support Vector Machine - Linear
rfe_model <- rfe(x = ggg_train[features], y = ggg_train[[target]], sizes=c(1:length(features)),
                 rfeControl=rfeControl(functions=rfFuncs, method="cv", number=10),
                 method="svmLinear", metric = "Accuracy")
# Support Vector Machine - Radial
rfe_model <- rfe(x = ggg_train[features], y = ggg_train[[target]], sizes=c(1:length(features)),
                 rfeControl=rfeControl(functions=rfFuncs, method="cv", number=10),
                 method="svmRadial", metric = "Accuracy")
# Gradient Boosted Machines
rfe_model <- rfe(x = ggg_train[features], y = ggg_train[[target]], sizes=c(1:length(features)),
                 rfeControl=rfeControl(functions=rfFuncs, method="cv", number=10),
                 method="gbm", metric = "Accuracy")
# Linear Regression
rfe_model <- rfe(x = ggg_train[features], y = ggg_train[[target]], sizes=c(1:length(features)),
                 rfeControl=rfeControl(functions=rfFuncs, method="cv", number=10),
                 method="lm", metric = "Accuracy")

# Print the results
print(rfe_model)

# Get the selected features
selected_features <- rfe_model$optVariables
selected_features

# Random Forest:
# 1 hair_length
# 2 has_soul
# 3 bone_length
# 4 rotting_flesh

# SVM Linear:
# 1 hair_length
# 2 has_soul
# 3 bone_length
# 4 rotting_flesh

# SVM Radial:
# 1 hair_length
# 2 has_soul
# 3 bone_length
# 4 rotting_flesh

# GBM:
# 1 hair_length
# 2 has_soul
# 3 rotting_flesh
# 4 bone_length

# Linear Regression:
# 1 hair_length
# 2 has_soul
# 3 rotting_flesh
# 4 bone_length

# Recipe (leave out 'id')
svms_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul, data = ggg_train) %>%
  step_normalize(all_predictors())

svms_prep <- prep(svms_rec)
bake(svms_prep, ggg_train)

# Create linear SVMS model
svms_lin_mod <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create radial SVMS model
svms_rad_mod <- svm_rbf(rbf_sigma = tune(),
                        cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create poly SVMS model
svms_poly_mod <- svm_poly(degree = tune(),
                          cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create linear SVMS workflow
svms_lin_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_lin_mod)

# Create radial SVMS workflow
svms_rad_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_rad_mod)

# Create poly SVMS workflow
svms_poly_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_poly_mod)

# Grid of values to tune over (linear)
svms_lin_tg <- grid_regular(cost(),
                            levels = 10)

# Grid of values to tune over (radial)
svms_rad_tg <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 10)

# Grid of values to tune over (poly)
svms_poly_tg <- grid_regular(degree(),
                            cost(),
                            levels = 10)

# Split data for cross-validation (CV) (linear)
svms_lin_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Split data for cross-validation (CV) (radial)
svms_rad_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Split data for cross-validation (CV) (poly)
svms_poly_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation (linear)
svms_lin_cv_results <- svms_lin_wf %>%
  tune_grid(resamples = svms_lin_folds,
            grid = svms_lin_tg,
            metrics = metric_set(accuracy))

# Run cross-validation (radial)
svms_rad_cv_results <- svms_rad_wf %>%
  tune_grid(resamples = svms_rad_folds,
            grid = svms_rad_tg,
            metrics = metric_set(accuracy))

# Run cross-validation (poly)
svms_poly_cv_results <- svms_poly_wf %>%
  tune_grid(resamples = svms_poly_folds,
            grid = svms_poly_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters (linear)
svms_lin_best_tune <- svms_lin_cv_results %>%
  select_best("accuracy")

# Find best tuning parameters (radial)
svms_rad_best_tune <- svms_rad_cv_results %>%
  select_best("accuracy")

# Find best tuning parameters (poly)
svms_poly_best_tune <- svms_poly_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it (linear)
svms_lin_final_wf <- svms_lin_wf %>%
  finalize_workflow(svms_lin_best_tune) %>%
  fit(data = ggg_train)

# Finalize workflow and fit it (radial)
svms_rad_final_wf <- svms_rad_wf %>%
  finalize_workflow(svms_rad_best_tune) %>%
  fit(data = ggg_train)

# Finalize workflow and fit it (poly)
svms_poly_final_wf <- svms_poly_wf %>%
  finalize_workflow(svms_poly_best_tune) %>%
  fit(data = ggg_train)

# Predict class (linear)
svms_lin_preds <- predict(svms_lin_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Predict class (radial)
svms_rad_preds <- predict(svms_rad_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Predict class (poly)
svms_poly_preds <- predict(svms_poly_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions (linear)
vroom_write(x=svms_lin_preds, file="svms_lin_preds_4.csv", delim = ",")

# Create a CSV with the predictions (radial)
vroom_write(x=svms_rad_preds, file="svms_rad_preds_5.csv", delim = ",")

# Create a CSV with the predictions (poly)
vroom_write(x=svms_poly_preds, file="svms_poly_preds_5.csv", delim = ",")

#################################################################
#################################################################
# K Nearest Neighbors                     #######################
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(kknn)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" and "color" into factors
ggg_train$type <- as.factor(ggg_train$type)

# Recipe (leave out 'id')
knn_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_dummy(color) %>%
  step_normalize(all_numeric_predictors())

knn_prep <- prep(knn_rec)
bake(knn_prep, ggg_train)

# Create KNN model specification
knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# Create KNN workflow
knn_wf <- workflow() %>%
  add_recipe(knn_rec) %>%
  add_model(knn_mod)

# Grid of values to tune over
knn_tg <- grid_regular(neighbors(),
                      levels = 10)

# Split data for cross-validation (CV)
knn_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation
knn_cv_results <- knn_wf %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters
knn_best_tune <- knn_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it
knn_final_wf <- knn_wf %>%
  finalize_workflow(knn_best_tune) %>%
  fit(data = ggg_train)

# Predict class
knn_preds <- predict(knn_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions
vroom_write(x=knn_preds, file="knn_preds_1.csv", delim = ",")

#################################################################
#################################################################
# Model Stacking - Linear and Radial SVMs        ################
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(lme4)
library(stacks)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" and "color" into factors
ggg_train$type <- as.factor(ggg_train$type)

# Recipe (leave out 'id')
svms_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_normalize(all_predictors())

svms_prep <- prep(svms_rec)
bake(svms_prep, ggg_train)

# Set up cross validation
stack_folds <- vfold_cv(ggg_train,
                  v = 10,
                  repeats = 1) # Split data for CV

stack_untuned_model <- control_stack_grid() # Control grid for tuning over a grid
stack_tuned_model <- control_stack_resamples() # Control grid for models we aren't tuning

# Create linear SVMS model
svms_lin_mod <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create radial SVMS model
svms_rad_mod <- svm_rbf(rbf_sigma = tune(),
                        cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create linear SVMS workflow
svms_lin_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_lin_mod)

# Create radial SVMS workflow
svms_rad_wf <- workflow() %>%
  add_recipe(svms_rec) %>%
  add_model(svms_rad_mod)

# Grid of values to tune over (linear)
svms_lin_tg <- grid_regular(cost(),
                            levels = 5)

# Grid of values to tune over (radial)
svms_rad_tg <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 10)

# Run cross-validation (linear)
svms_lin_cv_results <- svms_lin_wf %>%
  tune_grid(resamples = stack_folds,
            grid = svms_lin_tg,
            metrics = metric_set(roc_auc),
            control = stack_untuned_model)

# Run cross-validation (radial)
svms_rad_cv_results <- svms_rad_wf %>%
  tune_grid(resamples = stack_folds,
            grid = svms_rad_tg,
            metrics = metric_set(roc_auc),
            control = stack_untuned_model)

# Specify models to include in stacked model
stack_stack <- stacks() %>%
  add_candidates(svms_lin_cv_results) %>%
  add_candidates(svms_rad_cv_results)

# Fit model w/ LASSO penalized regression meta-learner
stacked_model <- stack_stack %>%
  blend_predictions() %>%
  fit_members()

# Predict class (stacked)
stacked_preds <- predict(stacked_model,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions (linear)
vroom_write(x=stacked_preds, file="stacked_preds.csv", delim = ",")

#################################################################
#################################################################
# Boosted Trees and BART                         ################
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(bonsai)
library(lightgbm)
library(embed)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Recipe (leave out 'id')
bst_brt_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_lencode_glm(color, outcome = vars(type)) %>% # glm allows you to target encode a factor on a factor
  step_zv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

bst_brt_rec <- prep(bst_brt_rec)
bake(bst_brt_rec, ggg_train)

# Create a Boost model specification
bst_spec <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# Create a BART model specification
bart_spec <- bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")

# Create a Boost Workflow
bst_wf <- workflow() %>%
  add_recipe(bst_brt_rec) %>%
  add_model(bst_spec)

# BART workflow
bart_wf <- workflow() %>%
  add_recipe(bst_brt_rec) %>%
  add_model(bart_spec)

# Set up Boost tuning grid
bst_grid <- grid_regular(
  trees(range = c(500, 1000)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(0.01, 0.1)),
  levels = 5
)

# Set up BART tuning grid
bart_grid <- grid_regular(
  trees(range = c(500, 1000)),
        levels = 5)

# Split data for cross-validation (CV) for boost
bst_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Split data for cross-validation (CV) for bart
bart_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation for boost
bst_cv_results <- bst_wf %>%
  tune_grid(resamples = bst_folds,
            grid = bst_grid,
            metrics = metric_set(accuracy))

# Run cross-validation for bart
bart_cv_results <- bart_wf %>%
  tune_grid(resamples = bart_folds,
            grid = bart_grid,
            metrics = metric_set(accuracy))

# Find best tuning parameters (boost)
bst_best_tune <- bst_cv_results %>%
  select_best("accuracy")

# Find best tuning parameters (bart)
bart_best_tune <- bart_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it (boost)
bst_final_wf <- bst_wf %>%
  finalize_workflow(bst_best_tune) %>%
  fit(data = ggg_train)

# Finalize workflow and fit it (bart)
bart_final_wf <- bart_wf %>%
  finalize_workflow(bart_best_tune) %>%
  fit(data = ggg_train)

# Predict class (Boost)
bst_preds <- predict(bst_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Predict class (Bart)
bart_preds <- predict(bart_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions (boost)
vroom_write(x=bst_preds, file="bst_preds.csv", delim = ",")

# Create a CSV with the predictions (bart)
vroom_write(x=bart_preds, file="bart_preds.csv", delim = ",")


#################################################################
#################################################################
# NAIVE BAYES                                     ################
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(naivebayes)
library(discrim)
library(embed)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Create Recipe
nb_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul, data = ggg_train) %>%
  step_normalize(all_numeric_predictors())

# Prep, Bake, and View Recipe
nb_prep <- prep(nb_rec)
bake(nb_prep, ggg_train) %>%
  slice(1:10)

# Create Naive Bayes model
nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# Create Naive Bayes workflow
nb_wf <- workflow() %>%
  add_recipe(nb_rec) %>%
  add_model(nb_mod)

# Grid of values to tune over
nb_tg <- grid_regular(Laplace(),
                      smoothness(),
                      levels = 10)

# Split data for cross-validation (CV)
nb_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation
nb_cv_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters
nb_best_tune <- nb_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it
nb_final_wf <- nb_wf %>%
  finalize_workflow(nb_best_tune) %>%
  fit(data = ggg_train)

# Predict class (Naive Bayes)
nb_preds <- predict(nb_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions
vroom_write(x=nb_preds, file="nb_preds_4.csv", delim = ",")

#################################################################
#################################################################
# Model Stacking - Radial SVM, RF, and NB        ################
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(lme4)
library(stacks)
library(naivebayes)
library(discrim)
library(kernlab)
library(parsnip)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" and "color" into factors
ggg_train$type <- as.factor(ggg_train$type)

# --------------------------------------------------------------------------------------------

# RF Recipe
rf_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul, data = ggg_train) %>%
  step_zv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())
rf_prep <- prep(rf_rec)
bake(rf_prep, ggg_train)

# NB Recipe
nb_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul, data = ggg_train) %>%
  step_normalize(all_numeric_predictors())
nb_prep <- prep(nb_rec)
bake(nb_prep, ggg_train)

# SVMS Recipe
svms_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_normalize(all_predictors())
svms_prep <- prep(svms_rec)
bake(svms_prep, ggg_train)


#################################################################
#################################################################
# NAIVE BAYES AGAIN                              ################
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(naivebayes)
library(discrim)
library(embed)

# Load Data
ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

# Turn "type" into factor
ggg_train$type <- as.factor(ggg_train$type)

# Create Recipe
nb_rec <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = ggg_train) %>%
  step_mutate_at(color, fn = factor) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_normalize(all_numeric_predictors())

# Prep, Bake, and View Recipe
nb_prep <- prep(nb_rec)
bake(nb_prep, ggg_train) %>%
  slice(1:10)

# Create Naive Bayes model
nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# Create Naive Bayes workflow
nb_wf <- workflow() %>%
  add_recipe(nb_rec) %>%
  add_model(nb_mod)

# Grid of values to tune over
nb_tg <- grid_regular(Laplace(),
                      smoothness(),
                      levels = 5)

# Split data for cross-validation (CV)
nb_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

# Run cross-validation
nb_cv_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tg,
            metrics = metric_set(accuracy))

# Find best tuning parameters
nb_best_tune <- nb_cv_results %>%
  select_best("accuracy")

# Finalize workflow and fit it
nb_final_wf <- nb_wf %>%
  finalize_workflow(nb_best_tune) %>%
  fit(data = ggg_train)

# Predict class (Naive Bayes)
nb_preds <- predict(nb_final_wf,
                     new_data = ggg_test,
                     type = "class") %>%
  bind_cols(ggg_test$id, .) %>%
  rename(type = .pred_class) %>%
  rename(id = ...1) %>%
  select(id, type)

# Create a CSV with the predictions
vroom_write(x=nb_preds, file="nb_preds_5.csv", delim = ",")

