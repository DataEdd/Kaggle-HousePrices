# PLS_pipeline.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# -----------------------------------------------------------------------------
# 1. Read raw data (from data/raw/ directory)
# -----------------------------------------------------------------------------
train = pd.read_csv('data/raw/train.csv')
test  = pd.read_csv('data/raw/test.csv')

# Preserve 'Id' column from test
if 'Id' in test.columns:
    test_ids = test[['Id']].copy()
else:
    test_ids = pd.DataFrame({'Id': []})

# Drop 'Id' column if present
if 'Id' in train.columns:
    train = train.drop(columns=['Id'])
if 'Id' in test.columns:
    test = test.drop(columns=['Id'])

# Separate target from train
y = train['SalePrice']
X_train_full = train.drop(columns=['SalePrice'])
X_test_full  = test.copy()

# -----------------------------------------------------------------------------
# 2. Identify time vs. non-time columns
# -----------------------------------------------------------------------------
time_cols = [c for c in X_train_full.columns if c in [
    'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'
]]

numeric_cols     = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_non_time     = [c for c in numeric_cols     if c not in time_cols]
categorical_non_time = [c for c in categorical_cols if c not in time_cols]

# -----------------------------------------------------------------------------
# 3. Determine which non-time columns to drop due to >50% missing (based on TRAIN only)
# -----------------------------------------------------------------------------
def drop_sparse_columns(df, threshold=0.50):
    miss_frac = df.isna().mean()
    to_drop = miss_frac[miss_frac > threshold].index.tolist()
    return to_drop

# Compute columns to drop (train only)
to_drop_non_time = drop_sparse_columns(
    X_train_full[numeric_non_time + categorical_non_time],
    threshold=0.50
)

# Apply the same drop to both train & test
X_train = X_train_full.drop(columns=to_drop_non_time)
X_test  = X_test_full.drop(columns=to_drop_non_time)

# Recompute non-time lists after dropping
numeric_non_time     = [c for c in numeric_non_time     if c not in to_drop_non_time]
categorical_non_time = [c for c in categorical_non_time if c not in to_drop_non_time]

# -----------------------------------------------------------------------------
# 4. Define custom transformers for non-time features
# -----------------------------------------------------------------------------
def winsorize_numeric(X_df):
    """
    Clip each numeric column at [0.5%, 99.5%] quantiles.
    """
    Xw = pd.DataFrame(X_df).copy()
    for col in Xw.columns:
        lo, hi = Xw[col].quantile([0.005, 0.995])
        Xw[col] = Xw[col].clip(lower=lo, upper=hi)
    return Xw

def log1p_all(X_df):
    """
    Apply np.log1p to all passed columns (assuming non-negative).
    """
    return np.log1p(pd.DataFrame(X_df).clip(lower=0))

def rare_group_cat(X_df, threshold=30):
    """
    Any categorical level with fewer than threshold rows is replaced with "__OTHER__".
    """
    Xr = pd.DataFrame(X_df).copy()
    for col in Xr.columns:
        freqs = Xr[col].value_counts(dropna=False)
        rare_labels = freqs[freqs < threshold].index
        Xr[col] = Xr[col].where(~Xr[col].isin(rare_labels), other="__OTHER__")
    return Xr

winsorize_transformer  = FunctionTransformer(winsorize_numeric)
log1p_transformer      = FunctionTransformer(log1p_all)
rare_group_transformer = FunctionTransformer(rare_group_cat)

# -----------------------------------------------------------------------------
# 5. "Best-cat" list (17 columns)
# -----------------------------------------------------------------------------
ONEHOT_CAT_COLS = [
    'MSZoning', 'Alley', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1',
    'BldgType', 'HouseStyle', 'RoofMatl', 'Foundation', 'HeatingQC',
    'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'MiscFeature'
]

def select_best_cols(X_df):
    """
    Return all categories from ONEHOT_CAT_COLS that remain in X_df.
    """
    return [c for c in ONEHOT_CAT_COLS if c in X_df.columns]

# -----------------------------------------------------------------------------
# 6. Build a pipeline for non-time features using the recommended PLS flags:
#    - use_winsor = True
#    - use_log1p = True
#    - skip var_thresh
#    - use_rare_group = True
#    - drop_sparse_flag (already applied above)
#    - use_best_cat = True
# -----------------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols):
    # (A) Numeric sub-pipeline: impute -> winsorize -> log1p -> scale
    num_steps = [
        ("impute_num", SimpleImputer(strategy="median", add_indicator=True)),
        ("winsorize", winsorize_transformer),
        ("log1p", log1p_transformer),
        ("scale_num", StandardScaler())
    ]
    num_pipe = Pipeline(num_steps)

    # (B) Categorical sub-pipeline: impute -> rare_group -> onehot(best-cat)
    cat_pipe = Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("rare_group", rare_group_transformer),
        ("onehot_best", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    cat_features = select_best_cols

    coltrans = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_features),
    ], remainder="drop")

    # Wrap only the ColumnTransformer in a pipeline to name it "preproc_nt"
    return Pipeline([
        ("preproc_nt", coltrans)
    ])

# Instantiate the pipeline
non_time_preproc = build_preprocessor(numeric_non_time, categorical_non_time)
# Fit on training non-time columns
non_time_preproc.fit(X_train[numeric_non_time + categorical_non_time])

# -----------------------------------------------------------------------------
# 7. Transform non-time features for train & test
# -----------------------------------------------------------------------------
X_nt_train = non_time_preproc.transform(X_train[numeric_non_time + categorical_non_time])
X_nt_test  = non_time_preproc.transform(X_test[numeric_non_time + categorical_non_time])

# Instead of deriving exact original names, use generic names for non-time dims
n_non_time_feats = X_nt_train.shape[1]
generic_non_time_names = [f"NT_{i}" for i in range(n_non_time_feats)]

# -----------------------------------------------------------------------------
# 8. Engineer & scale time features for train & test
# -----------------------------------------------------------------------------
def process_time_features(df_time):
    df_time_eng = pd.DataFrame(index=df_time.index)
    df_time_eng['AgeAtSale']        = df_time['YrSold'] - df_time['YearBuilt']
    df_time_eng['YearsSinceRemodel']= df_time['YrSold'] - df_time['YearRemodAdd']
    df_time_eng['GarageAge']        = df_time['YrSold'] - df_time['GarageYrBlt']
    df_time_eng['GarageMissing']    = df_time['GarageYrBlt'].isna().astype(int)
    df_time_eng['MoSin']            = np.sin(2 * np.pi * df_time['MoSold'] / 12)
    df_time_eng['MoCos']            = np.cos(2 * np.pi * df_time['MoSold'] / 12)
    df_time_eng['SaleYear']         = df_time['YrSold']
    return df_time_eng

# Train time features
df_time_train = X_train[time_cols]
time_eng_train = process_time_features(df_time_train).values
time_eng_train[np.isnan(time_eng_train)] = 0.0
time_scaler = StandardScaler().fit(time_eng_train)
X_time_train_scaled = time_scaler.transform(time_eng_train)

# Test time features
df_time_test = X_test[time_cols]
time_eng_test = process_time_features(df_time_test).values
time_eng_test[np.isnan(time_eng_test)] = 0.0
X_time_test_scaled = time_scaler.transform(time_eng_test)

time_feature_names = [
    "AgeAtSale", "YearsSinceRemodel", "GarageAge",
    "GarageMissing", "MoSin", "MoCos", "SaleYear"
]

# -----------------------------------------------------------------------------
# 9. Concatenate non-time + time arrays into DataFrames
# -----------------------------------------------------------------------------
X_train_full = np.hstack([X_nt_train, X_time_train_scaled])
X_test_full  = np.hstack([X_nt_test,  X_time_test_scaled])

# Build DataFrames with column names
train_cols = generic_non_time_names + time_feature_names
df_clean_train = pd.DataFrame(X_train_full, columns=train_cols, index=X_train.index)
df_clean_test  = pd.DataFrame(X_test_full,  columns=train_cols, index=X_test.index)

# Add target back to cleaned train DataFrame
df_clean_train["SalePrice"] = y.values

# -----------------------------------------------------------------------------
# 10. Save cleaned CSVs
# -----------------------------------------------------------------------------
df_clean_train.to_csv("data/processed/PLS_clean_train_final.csv", index=False)
df_clean_test.to_csv("data/processed/PLS_clean_test_final.csv",   index=False)
test_ids.to_csv("data/processed/PLS_test_id.csv", index=False)

print("Finished. Wrote PLS_clean_train_final.csv, PLS_clean_test_final.csv and PLS_test_id.csv to data/processed/")
