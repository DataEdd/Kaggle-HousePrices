# gbm_clean_updated.R
# Full pipeline: preprocessing flag combos + gbm tuning in R

# 0) Libraries
if (!requireNamespace("crayon", quietly = TRUE)) install.packages("crayon")
library(crayon)      # dependency for caret
library(data.table)  # fast data handling
library(purrr)       # list-map utilities
library(recipes)     # tidy preprocessing
library(caret)       # modeling & CV
library(gbm)         # original GBM implementation
library(doParallel)  # parallel backend

# 1) Set seed and working directory
set.seed(42)
setwd("~/Documents/VisualStudioCode/VSC/KaggleComps/HousingPrices")

# 2) Load data & define y, X
df <- fread("data/raw/train.csv")
if ("Id" %in% names(df)) df[, Id := NULL]
y <- log1p(df$SalePrice)
X <- df[, !"SalePrice", with = FALSE]

# 3) Identify column types
time_cols        <- intersect(names(X), c("YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"))
numeric_cols     <- names(which(sapply(X, is.numeric)))
cat_cols         <- setdiff(names(X)[sapply(X, is.character)], time_cols)
numeric_non_time <- setdiff(numeric_cols, time_cols)
cat_non_time     <- setdiff(cat_cols,    time_cols)

# 4) Force categorical columns to factors
X[, (cat_non_time) := lapply(.SD, as.factor), .SDcols = cat_non_time]

# 5) Best-cat list from EDA
ONEHOT_CAT_COLS <- c(
  'MSZoning','Alley','LotShape','LotConfig','Neighborhood','Condition1',
  'BldgType','HouseStyle','RoofMatl','Foundation','HeatingQC',
  'CentralAir','KitchenQual','Functional','FireplaceQu','GarageFinish',
  'MiscFeature'
)
select_best_cols <- function(cols) intersect(cols, ONEHOT_CAT_COLS)

# 6) Flags and all combos
flags <- c("use_winsor","use_log1p","use_var_thresh","use_rare_group","do_drop_sparse","use_best_cat")
all_flag_combos <- expand.grid(rep(list(c(FALSE, TRUE)), length(flags)))
names(all_flag_combos) <- flags

# 7) Helper to drop sparse columns (NA fraction > 50%)
drop_sparse_cols <- function(dat, threshold = 0.5) {
  na_frac <- sapply(dat, function(col) mean(is.na(col)))
  keep    <- names(na_frac)[na_frac <= threshold]
  dat[, ..keep]
}

# 8) Parallel backend setup
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# 9) Loop over preprocessing combos and tune GBM
results <- pmap(all_flag_combos, function(use_winsor, use_log1p, use_var_thresh,
                                          use_rare_group, do_drop_sparse, use_best_cat) {
  # 9a) Prepare non-time data
  X_nt <- X[, c(numeric_non_time, cat_non_time), with = FALSE]
  if (do_drop_sparse) X_nt <- drop_sparse_cols(X_nt)
  
  # 9b) Build recipe
  rec <- recipe(~ ., data = X_nt) %>%
    # numeric imputation + NA indicator
    step_impute_median(all_numeric(), -all_outcomes()) %>%
    step_indicate_na(all_numeric()) %>%
    # optional winsorization
    { if (use_winsor) step_mutate_at(., all_numeric(), fn = function(x) {
      q <- quantile(x, c(0.005, 0.995), na.rm = TRUE)
      pmin(pmax(x, q[1]), q[2])
    }) else . } %>%
    # optional log1p
    { if (use_log1p) step_log(., all_numeric(), offset = 1) else . } %>% # <-- FIXED
    # optional near-zero variance removal
    { if (use_var_thresh) step_nzv(., all_numeric()) else . } %>% # <-- FIXED
    # categorical: treat NAs/unseen as "__MISSING__"
    step_unknown(all_nominal(), new_level = "__MISSING__") %>%
    # optional rare-level grouping
    { if (use_rare_group) step_other(., all_nominal(), threshold = 30, other = "__OTHER__") else . } %>%
    # dummy encoding
    { if (use_best_cat) {
      step_dummy(., select_best_cols(names(X_nt)), one_hot = FALSE)
    } else {
      step_dummy(., all_nominal(), one_hot = FALSE)
    }
    }
  
  # 9c) Prep & bake
  dat_baked <- bake(prep(rec, training = X_nt), new_data = X_nt)
  
  # 9d) Engineered time features
  T <- X[, ..time_cols]
  T2 <- data.table(
    AgeAtSale        = T$YrSold - T$YearBuilt,
    YearsSinceRemodel= T$YrSold - T$YearRemodAdd,
    GarageAge        = T$YrSold - T$GarageYrBlt,
    GarageMissing    = as.integer(is.na(T$GarageYrBlt)),
    MoSin            = sin(2 * pi * T$MoSold / 12),
    MoCos            = cos(2 * pi * T$MoSold / 12),
    SaleYear         = T$YrSold
  )
  T2[is.na(T2)] <- 0
  
  train_dat <- cbind(dat_baked, T2)
  
  # 9e) Caret tuning grid & reproducible seeds
  tunegrid <- expand.grid(
    n.trees           = c(100, 300, 500),
    interaction.depth = c(1, 3, 5),
    shrinkage         = c(0.01, 0.05, 0.1),
    n.minobsinnode    = c(5, 10, 20)
  )
  # seeds list: length = number of folds + 1
  n_folds <- 5
  seeds_list <- vector("list", length = n_folds + 1)
  # first n_folds: one vector per tuning candidate count
  for (i in seq_len(n_folds)) {
    seeds_list[[i]] <- sample.int(1e5, nrow(tunegrid))
  }
  # final model
  seeds_list[[n_folds + 1]] <- sample.int(1e5, 1)
  
  ctrl <- trainControl(
    method      = "cv",
    number      = n_folds,
    seeds       = seeds_list,
    verboseIter = FALSE
  )
  
  # 9f) Train GBM
  tr <- train(
    x         = train_dat,
    y         = y,
    method    = "gbm",
    tuneGrid  = tunegrid,
    metric    = "RMSE",
    trControl = ctrl,
    verbose   = FALSE
  )
  
  # 9g) Return flags, best tune, RMSE
  list(
    flags       = list(use_winsor, use_log1p, use_var_thresh,
                       use_rare_group, do_drop_sparse, use_best_cat),
    best_params = tr$bestTune,
    best_RMSE   = min(tr$results$RMSE)
  )
})

# 10) Stop parallel cluster
stopCluster(cl)

# 11) Summarize results
data_out <- data.table::rbindlist(
  purrr::map2(results, seq_along(results), ~{
    data.table::data.table(
      combo_id       = .y,
      use_winsor     = .x$flags[[1]],
      use_log1p      = .x$flags[[2]],
      use_var_thresh = .x$flags[[3]],
      use_rare_group = .x$flags[[4]],
      do_drop_sparse = .x$flags[[5]],
      use_best_cat   = .x$flags[[6]],
      RMSE           = .x$best_RMSE,
      .x$best_params
    )
  })
)

# 12) Print summary
#print(data_out)
data_out_sorted <- data_out[order(RMSE)]
print(data_out_sorted)