# RF_pipeline.py

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold

# -----------------------------------------------------------------------------
# 1. Read raw data from data/raw/
# -----------------------------------------------------------------------------
train = pd.read_csv('data/raw/train.csv')
test  = pd.read_csv('data/raw/test.csv')

# Preserve the 'Id' column from test for output
if 'Id' in test.columns:
    test_ids = test[['Id']].copy()
else:
    test_ids = pd.DataFrame({'Id': []})

# Drop 'Id' from train & test so they're not treated as features
if 'Id' in train.columns:
    train = train.drop(columns=['Id'])
if 'Id' in test.columns:
    test = test.drop(columns=['Id'])

# Separate target from train
y = train['SalePrice']
X_train_full = train.drop(columns=['SalePrice'])
X_test_full  = test.copy()

# -----------------------------------------------------------------------------
# 2. Identify “time” vs. non‐time columns
# -----------------------------------------------------------------------------
time_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']

numeric_cols_all     = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_all = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_non_time     = [c for c in numeric_cols_all     if c not in time_cols]
categorical_non_time = [c for c in categorical_cols_all if c not in time_cols]

# -----------------------------------------------------------------------------
# 3. Drop columns with >50% missing (based on TRAIN only)
# -----------------------------------------------------------------------------
def drop_sparse_columns(df, threshold=0.50):
    """
    Identify columns where > threshold fraction of values are missing.
    Returns a list of those column names.
    """
    miss_frac = df.isna().mean()
    return miss_frac[miss_frac > threshold].index.tolist()

# Compute which non‐time columns to drop (on training data)
to_drop_non_time = drop_sparse_columns(
    X_train_full[numeric_non_time + categorical_non_time],
    threshold=0.50
)

# Drop those columns from both train & test
X_train = X_train_full.drop(columns=to_drop_non_time)
X_test  = X_test_full.drop(columns=to_drop_non_time)

# Update the non‐time lists after dropping
numeric_non_time     = [c for c in numeric_non_time     if c not in to_drop_non_time]
categorical_non_time = [c for c in categorical_non_time if c not in to_drop_non_time]

# -----------------------------------------------------------------------------
# 4. Define custom transformers for numeric & categorical (no scaling here)
# -----------------------------------------------------------------------------
def winsorize_numeric(df_num):
    """
    Clip each numeric column at [0.5%, 99.5%] quantiles.
    """
    dfw = pd.DataFrame(df_num).copy()
    for col in dfw.columns:
        lo, hi = dfw[col].quantile([0.005, 0.995])
        dfw[col] = dfw[col].clip(lower=lo, upper=hi)
    return dfw

winsorizer = FunctionTransformer(winsorize_numeric, validate=False)

def log1p_numeric(df_num):
    """
    Apply np.log1p to every numeric column (clipping negatives to 0 first).
    """
    dfn = pd.DataFrame(df_num).clip(lower=0).copy()
    return np.log1p(dfn)

log1p_transformer = FunctionTransformer(log1p_numeric, validate=False)

def rare_group_cat(X_df, threshold=30):
    """
    Any categorical level with < threshold rows → "__OTHER__".
    """
    Xr = pd.DataFrame(X_df).copy()
    for col in Xr.columns:
        freqs = Xr[col].value_counts(dropna=False)
        rare_vals = freqs[freqs < threshold].index
        Xr[col] = Xr[col].where(~Xr[col].isin(rare_vals), other="__OTHER__")
    return Xr

rare_group_transformer = FunctionTransformer(rare_group_cat, validate=False)

# -----------------------------------------------------------------------------
# 5. Curated “Best‐cat” list (17 columns)
# -----------------------------------------------------------------------------
BEST_CAT_LIST = [
    'MSZoning', 'Alley', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1',
    'BldgType', 'HouseStyle', 'RoofMatl', 'Foundation', 'HeatingQC',
    'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'MiscFeature'
]

def select_best_cat_cols(X_df):
    """
    Return the subset of BEST_CAT_LIST present in X_df.columns.
    Used by ColumnTransformer at fit() time.
    """
    return [c for c in BEST_CAT_LIST if c in X_df.columns]

# -----------------------------------------------------------------------------
# 6. Build a ColumnTransformer for non‐time features (no StandardScaler here)
#    Numeric pipeline: median‐impute → winsorize → log1p → variance threshold
#    Categorical pipeline: impute "__MISSING__" → rare‐group → one‐hot (BEST_CAT_LIST)
# -----------------------------------------------------------------------------
def build_non_time_preprocessor(numeric_cols, categorical_cols):
    # Numeric sub‐pipeline (no scaling yet)
    num_pipeline = Pipeline([
        ("impute_num", SimpleImputer(strategy="median", add_indicator=True)),
        ("winsor", winsorizer),
        ("log1p", log1p_transformer),
        ("var_thresh", VarianceThreshold(threshold=0.01))
    ])

    # Categorical sub‐pipeline (BEST_CAT_LIST only)
    cat_pipeline = Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("rare_group", rare_group_transformer),
        ("onehot_best", OneHotEncoder(
            drop="first", 
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    coltrans = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, select_best_cat_cols)
    ], remainder="drop")

    return coltrans

# Instantiate & fit the non‐time preprocessor
non_time_preproc = build_non_time_preprocessor(numeric_non_time, categorical_non_time)
non_time_preproc.fit(X_train[numeric_non_time + categorical_non_time])

# -----------------------------------------------------------------------------
# 7. Transform non‐time features for train & test
# -----------------------------------------------------------------------------
X_nt_train = non_time_preproc.transform(X_train[numeric_non_time + categorical_non_time])
X_nt_test  = non_time_preproc.transform(X_test[numeric_non_time + categorical_non_time])

# Count number of non‐time features after preprocessing
n_non_time_feats = X_nt_train.shape[1]

# -----------------------------------------------------------------------------
# 8. Engineer time features (no scaling yet)
# -----------------------------------------------------------------------------
def process_time_features(df_time):
    """
    Create these seven features from the raw time columns:
      - AgeAtSale, YearsSinceRemodel, GarageAge, GarageMissing, MoSin, MoCos, SaleYear
    """
    df_time_eng = pd.DataFrame(index=df_time.index)
    df_time_eng['AgeAtSale']         = df_time['YrSold'] - df_time['YearBuilt']
    df_time_eng['YearsSinceRemodel'] = df_time['YrSold'] - df_time['YearRemodAdd']
    df_time_eng['GarageAge']         = df_time['YrSold'] - df_time['GarageYrBlt']
    df_time_eng['GarageMissing']     = df_time['GarageYrBlt'].isna().astype(int)
    df_time_eng['MoSin']             = np.sin(2 * np.pi * df_time['MoSold'] / 12)
    df_time_eng['MoCos']             = np.cos(2 * np.pi * df_time['MoSold'] / 12)
    df_time_eng['SaleYear']          = df_time['YrSold']
    return df_time_eng

# 8a) Compute & gather raw time features for TRAIN
df_time_train = X_train[time_cols]
time_feats_train = process_time_features(df_time_train).values
time_feats_train[np.isnan(time_feats_train)] = 0.0  # fill any NaN

# 8b) Compute raw time features for TEST
df_time_test = X_test[time_cols]
time_feats_test = process_time_features(df_time_test).values
time_feats_test[np.isnan(time_feats_test)] = 0.0

# -----------------------------------------------------------------------------
# 9. Concatenate non‐time + time into a single un‐scaled feature matrix
# -----------------------------------------------------------------------------
X_train_unscaled = np.hstack([X_nt_train, time_feats_train])  # type: ignore
X_test_unscaled  = np.hstack([X_nt_test,  time_feats_test])   # type: ignore

# -----------------------------------------------------------------------------
# 10. Fit a SINGLE StandardScaler on the ENTIRE TRAIN matrix (all features)
# -----------------------------------------------------------------------------
full_scaler = StandardScaler().fit(X_train_unscaled)
X_train_scaled = full_scaler.transform(X_train_unscaled)
X_test_scaled  = full_scaler.transform(X_test_unscaled)

# -----------------------------------------------------------------------------
# 11. BUILD LIST OF ORIGINAL FEATURE NAMES (NOT NT1/NT2…) 
#
#     We now recover the “real” names for each non‐time column (numeric or one‐hot dummy),
#     and then append the 7 time‐feature names.  At the end we will have exactly
#     len(all_feature_names) == X_train_scaled.shape[1].
# -----------------------------------------------------------------------------

# 11a) Numeric feature names after imputation+indicator:
num_pipe = non_time_preproc.named_transformers_['num']
imputer = num_pipe.named_steps['impute_num']
imp_feature_names = imputer.get_feature_names_out(numeric_non_time).tolist()
#    e.g. ['LotFrontage', 'LotArea', ..., 'LotFrontage_missing', 'LotArea_missing', ...]

# 11b) Determine which of those survived VarianceThreshold:
var_thresh = num_pipe.named_steps['var_thresh']
mask = var_thresh.get_support()  # boolean mask, length == len(imp_feature_names)
numeric_kept_names = [
    name for name, keep in zip(imp_feature_names, mask) if keep
]

# 11c) Categorical feature names after one‐hot:
cat_pipe = non_time_preproc.named_transformers_['cat']
onehot = cat_pipe.named_steps['onehot_best']
best_present = select_best_cat_cols(X_train[categorical_non_time])
cat_onehot_names = onehot.get_feature_names_out(best_present).tolist()
#    e.g. ['MSZoning_RH', 'MSZoning_RL', ..., 'Alley__MISSING__', 'Alley_Grvl', ...]

# 11d) Time feature names (engineered):
time_colnames = ['AgeAtSale', 'YearsSinceRemodel', 'GarageAge',
                 'GarageMissing', 'MoSin', 'MoCos', 'SaleYear']

# 11e) Concatenate all names in the same left‐to‐right order as X_train_scaled:
all_feature_names = numeric_kept_names + cat_onehot_names + time_colnames

# Verify the total length matches
assert len(all_feature_names) == X_train_scaled.shape[1], (
    f"Column count mismatch! {len(all_feature_names)} vs. {X_train_scaled.shape[1]}"
)

# -----------------------------------------------------------------------------
# 12. Build DataFrames and save cleaned CSVs using real column names
# -----------------------------------------------------------------------------
df_clean_train = pd.DataFrame(
    X_train_scaled,
    columns=all_feature_names,
    index=X_train.index
)
df_clean_train["SalePrice"] = y.values

df_clean_test = pd.DataFrame(
    X_test_scaled,
    columns=all_feature_names,
    index=X_test.index
)

# Save to CSV under data/processed/
df_clean_train.to_csv("data/processed/RF_clean_train_rf.csv", index=False)
df_clean_test.to_csv("data/processed/RF_clean_test_rf.csv",   index=False)
test_ids.to_csv("data/processed/RF_test_id.csv", index=False)

print("Finished. Wrote RF_clean_train_rf.csv, RF_clean_test_rf.csv, and RF_test_id.csv to data/processed/")
