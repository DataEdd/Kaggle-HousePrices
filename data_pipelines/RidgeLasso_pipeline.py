# RidgeLasso_pipeline.py

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

winsorize_transformer  = FunctionTransformer(winsorize_numeric)
log1p_transformer      = FunctionTransformer(log1p_all)

# -----------------------------------------------------------------------------
# 5. Categorical pipeline: impute + one‐hot (all categories after drop_sparse_flag)
# -----------------------------------------------------------------------------
def build_non_time_preprocessor(numeric_cols, categorical_cols):
    num_pipeline = Pipeline([
        ("impute_num", SimpleImputer(strategy="median", add_indicator=True)),
        ("winsor", winsorize_transformer),
        ("log1p", log1p_transformer),
        ("scale_num", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot_all", OneHotEncoder(
            drop="if_binary", 
            handle_unknown="ignore", 
            sparse_output=False
        ))
    ])

    coltrans = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ], remainder="drop")

    return coltrans

# Instantiate and fit the non‐time preprocessor
non_time_preproc = build_non_time_preprocessor(numeric_non_time, categorical_non_time)
non_time_preproc.fit(X_train[numeric_non_time + categorical_non_time])

# -----------------------------------------------------------------------------
# 6. Transform non‐time features for train & test
# -----------------------------------------------------------------------------
X_nt_train = non_time_preproc.transform(X_train[numeric_non_time + categorical_non_time])
X_nt_test  = non_time_preproc.transform(X_test[numeric_non_time + categorical_non_time])

# Record how many non‐time features
n_non_time_feats = X_nt_train.shape[1]

# -----------------------------------------------------------------------------
# 7. Time‐feature engineering: same as PCR but scale without imputation flag
# -----------------------------------------------------------------------------
def process_time_features(df_time):
    df_time_eng = pd.DataFrame(index=df_time.index)
    df_time_eng['AgeAtSale']         = df_time['YrSold'] - df_time['YearBuilt']
    df_time_eng['YearsSinceRemodel'] = df_time['YrSold'] - df_time['YearRemodAdd']
    df_time_eng['GarageAge']         = df_time['YrSold'] - df_time['GarageYrBlt']
    df_time_eng['GarageMissing']     = df_time['GarageYrBlt'].isna().astype(int)
    df_time_eng['MoSin']             = np.sin(2 * np.pi * df_time['MoSold'] / 12)
    df_time_eng['MoCos']             = np.cos(2 * np.pi * df_time['MoSold'] / 12)
    df_time_eng['SaleYear']          = df_time['YrSold']
    return df_time_eng

# Fit scaler on train time features
df_time_train = X_train[time_cols]
time_feats_train = process_time_features(df_time_train).values
time_feats_train[np.isnan(time_feats_train)] = 0.0

time_scaler = StandardScaler().fit(time_feats_train)
X_time_train_scaled = time_scaler.transform(time_feats_train)

# Transform TEST time features
df_time_test = X_test[time_cols]
time_feats_test = process_time_features(df_time_test).values
time_feats_test[np.isnan(time_feats_test)] = 0.0

X_time_test_scaled = time_scaler.transform(time_feats_test)

# -----------------------------------------------------------------------------
# 8. Concatenate non‐time + time arrays into final feature matrices
# -----------------------------------------------------------------------------
X_train_feat = np.hstack([X_nt_train, X_time_train_scaled]) # type: ignore
X_test_feat  = np.hstack([X_nt_test,  X_time_test_scaled]) # type: ignore

# -----------------------------------------------------------------------------
# 9. Fit final Ridge with best alpha = 8.43101393208247 (from 5‐fold CV grid-search)
# -----------------------------------------------------------------------------
#best_alpha = 8.43101393208247
#ridge_model = Ridge(alpha=best_alpha)
#ridge_model.fit(X_train_feat, y)

# -----------------------------------------------------------------------------
# 10. Build DataFrames and save cleaned CSVs + test IDs file
# -----------------------------------------------------------------------------
# Column names for non-time + onehot + time:
# We'll save raw feature arrays as CSV with generic column names Feature1, Feature2, ...
n_features = X_train_feat.shape[1]
feature_cols = [f"Feature{i+1}" for i in range(n_features)]

# TRAIN DataFrame
df_clean_train = pd.DataFrame(
    X_train_feat, 
    columns=feature_cols, 
    index=X_train.index
)
df_clean_train["SalePrice"] = y.values

# TEST DataFrame
df_clean_test = pd.DataFrame(
    X_test_feat, 
    columns=feature_cols, 
    index=X_test.index
)

# Save cleaned features and test IDs
df_clean_train.to_csv("data/processed/RidgeLasso_clean_train_ridge.csv", index=False)
df_clean_test.to_csv("data/processed/RidgeLasso_clean_test_ridge.csv",   index=False)
test_ids.to_csv("data/processed/RidgeLasso_test_id.csv", index=False)

print("Finished. Wrote clean_train_ridge.csv, clean_test_ridge.csv, and Ridge_test_id.csv.")
