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

# #################################################################
# #################################################################
# # Impute Missing Data from Special Missing Data Set #############
# #################################################################
# #################################################################

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
  step_lencode_glm(color, outcome = vars(type)) %>% # glm allows you to target encode a factor on a factor
  step_zv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

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
  trees(range = c(500, 1000)),
  tree_depth(range = c(3, 10)),
  min_n(range = c(1, 10)),
  learn_rate(range = c(0.01, 0.1)),
  levels = 5
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
vroom_write(x=xgb_preds, file="xgb_preds_2.csv", delim = ",")

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
  step_dummy(color)
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1) # Scale Xs to [0, 1]

nn_prep <- prep(nn_rec)
bake(nn_prep, ggg_train)

# Create Neural Network model specification
nn_spec <- mlp(hidden_units = tune(),
               epochs = 100) %>%
  set_engine("nnet") %>%
  set_mode('classification')

# Create classification NN workflow
nn_wf <- workflow() %>%
  add_recipe(nn_rec) %>%
  add_model(nn_spec)

# Grid of values to tune over
nn_tg <- grid_regular(hidden_units(range = c(1, 5)),
                      levels = 5)

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
vroom_write(x=nn_preds, file="nn_preds.csv", delim = ",")
