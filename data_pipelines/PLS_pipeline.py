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
# 3. Determine which non-time columns to drop due to >50% missing (train only)
# -----------------------------------------------------------------------------
def drop_sparse_columns(df, threshold=0.50):
    miss_frac = df.isna().mean()
    return miss_frac[miss_frac > threshold].index.tolist()

to_drop_non_time = drop_sparse_columns(
    X_train_full[numeric_non_time + categorical_non_time],
    threshold=0.50
)

# Apply identical drop to both train & test
X_train = X_train_full.drop(columns=to_drop_non_time)
X_test  = X_test_full.drop(columns=to_drop_non_time)

# Update non-time lists after dropping
numeric_non_time     = [c for c in numeric_non_time     if c not in to_drop_non_time]
categorical_non_time = [c for c in categorical_non_time if c not in to_drop_non_time]

# -----------------------------------------------------------------------------
# 4. Define custom transformers for non-time features
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

winsorize_transformer = FunctionTransformer(winsorize_numeric, validate=False)

def log1p_all(df_num):
    """
    Apply np.log1p to every numeric column (clipping negatives to 0 first).
    """
    dfn = pd.DataFrame(df_num).clip(lower=0).copy()
    return np.log1p(dfn)

log1p_transformer = FunctionTransformer(log1p_all, validate=False)

def rare_group_cat(df_cat, threshold=30):
    """
    Any categorical level with < threshold rows → "__OTHER__".
    """
    Xr = pd.DataFrame(df_cat).copy()
    for col in Xr.columns:
        freqs = Xr[col].value_counts(dropna=False)
        rare_vals = freqs[freqs < threshold].index
        Xr[col] = Xr[col].where(~Xr[col].isin(rare_vals), other="__OTHER__")
    return Xr

rare_group_transformer = FunctionTransformer(rare_group_cat, validate=False)

# -----------------------------------------------------------------------------
# 5. "Best-cat" list (17 columns)
# -----------------------------------------------------------------------------
ONEHOT_CAT_COLS = [
    'MSZoning', 'Alley', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1',
    'BldgType', 'HouseStyle', 'RoofMatl', 'Foundation', 'HeatingQC',
    'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'MiscFeature'
]

def select_best_cols(df):
    """
    Return the subset of ONEHOT_CAT_COLS present in df.columns.
    """
    return [c for c in ONEHOT_CAT_COLS if c in df.columns]

# -----------------------------------------------------------------------------
# 6. Build a ColumnTransformer for non-time features (no scaling inside)
#    Numeric pipeline: impute → winsorize → log1p
#    Categorical pipeline: impute "__MISSING__" → rare-group → one-hot (BEST_CAT_LIST)
# -----------------------------------------------------------------------------
def build_non_time_preprocessor(numeric_cols, categorical_cols):
    num_pipeline = Pipeline([
        ("impute_num", SimpleImputer(strategy="median", add_indicator=True)),
        ("winsorize", winsorize_transformer),
        ("log1p", log1p_transformer)
        # no StandardScaler here
    ])

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
        ("cat", cat_pipeline, select_best_cols)
    ], remainder="drop")

    return coltrans

non_time_preproc = build_non_time_preprocessor(numeric_non_time, categorical_non_time)
non_time_preproc.fit(X_train[numeric_non_time + categorical_non_time])

# -----------------------------------------------------------------------------
# 7. Transform non-time features for train & test
# -----------------------------------------------------------------------------
X_nt_train = non_time_preproc.transform(X_train[numeric_non_time + categorical_non_time])
X_nt_test  = non_time_preproc.transform(X_test[numeric_non_time + categorical_non_time])

# -----------------------------------------------------------------------------
# Helper: Extract original feature names from non-time preprocessor
# -----------------------------------------------------------------------------
def get_non_time_feature_names(preproc, numeric_cols, categorical_cols):
    """
    Returns a list of original feature names corresponding to the output columns
    of the non-time ColumnTransformer (imputed+indicator numeric + one-hot dummies).
    """
    feature_names = []

    # Numeric side:
    num_imputer = preproc.named_transformers_['num'].named_steps['impute_num']
    imp_out = num_imputer.get_feature_names_out(numeric_cols).tolist()
    # ['LotFrontage', 'LotArea', ..., 'LotFrontage_missing', 'LotArea_missing', ...]
    feature_names += imp_out

    # Categorical side:
    ohe = preproc.named_transformers_['cat'].named_steps['onehot_best']
    best_present = select_best_cols(X_train[categorical_non_time])
    ohe_out = ohe.get_feature_names_out(best_present).tolist()
    # e.g. ['MSZoning_FV', 'MSZoning_RH', ..., 'Alley__MISSING__', ...]
    feature_names += ohe_out

    return feature_names

non_time_feature_names = get_non_time_feature_names(
    non_time_preproc,
    numeric_non_time,
    categorical_non_time
)

# -----------------------------------------------------------------------------
# 8. Engineer time features (no scaling yet)
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

df_time_train = X_train[time_cols]
time_feats_train = process_time_features(df_time_train).values
time_feats_train[np.isnan(time_feats_train)] = 0.0

df_time_test = X_test[time_cols]
time_feats_test = process_time_features(df_time_test).values
time_feats_test[np.isnan(time_feats_test)] = 0.0

time_feature_names = [
    'AgeAtSale', 'YearsSinceRemodel', 'GarageAge',
    'GarageMissing', 'MoSin', 'MoCos', 'SaleYear'
]

# -----------------------------------------------------------------------------
# 9. Concatenate non-time + time into a single unscaled feature matrix
# -----------------------------------------------------------------------------
X_train_unscaled = np.hstack([X_nt_train, time_feats_train])  # type: ignore
X_test_unscaled  = np.hstack([X_nt_test,  time_feats_test])   # type: ignore

all_feature_names = non_time_feature_names + time_feature_names

# -----------------------------------------------------------------------------
# 10. Fit a SINGLE StandardScaler on the ENTIRE TRAIN matrix (all features)
# -----------------------------------------------------------------------------
full_scaler = StandardScaler().fit(X_train_unscaled)
X_train_scaled = full_scaler.transform(X_train_unscaled)
X_test_scaled  = full_scaler.transform(X_test_unscaled)

# -----------------------------------------------------------------------------
# 11. Build DataFrames and save cleaned CSVs with original feature names
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

df_clean_train.to_csv("data/processed/PLS_clean_train_final.csv", index=False)
df_clean_test.to_csv("data/processed/PLS_clean_test_final.csv",  index=False)
test_ids.to_csv("data/processed/PLS_test_id.csv", index=False)

print(
    "Finished. Wrote:\n"
    "  • data/processed/PLS_clean_train_final.csv\n"
    "  • data/processed/PLS_clean_test_final.csv\n"
    "  • data/processed/PLS_test_id.csv"
)
