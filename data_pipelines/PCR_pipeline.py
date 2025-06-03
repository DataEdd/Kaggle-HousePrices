# PCR_pipeline.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA

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
    miss_frac = df.isna().mean()
    to_drop = miss_frac[miss_frac > threshold].index.tolist()
    return to_drop

# Compute which non-time columns to drop
to_drop_non_time = drop_sparse_columns(
    X_train_full[numeric_non_time + categorical_non_time],
    threshold=0.50
)

# Apply identical drop to both train & test
X_train = X_train_full.drop(columns=to_drop_non_time)
X_test  = X_test_full.drop(columns=to_drop_non_time)

# Update our non-time lists
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

winsorizer = FunctionTransformer(winsorize_numeric, validate=False)

# -----------------------------------------------------------------------------
# 5. “Best‐cat” list (17 curated columns)
# -----------------------------------------------------------------------------
BEST_CAT_LIST = [
    'MSZoning', 'Alley', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1',
    'BldgType', 'HouseStyle', 'RoofMatl', 'Foundation', 'HeatingQC',
    'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'MiscFeature'
]

def select_best_cat_cols(X_df):
    return [c for c in BEST_CAT_LIST if c in X_df.columns]

# -----------------------------------------------------------------------------
# 6. Build a preprocessor for non‐time features using “best_cat = True”
#     (Winsorize + scale numerics; impute + one‐hot only the 17 best‐cat columns)
# -----------------------------------------------------------------------------
def build_non_time_preprocessor(numeric_cols, categorical_cols):
    # Numeric pipeline: median‐impute → winsorize → scale
    num_pipeline = Pipeline([
        ("impute_num", SimpleImputer(strategy="median", add_indicator=True)),
        ("winsor", winsorizer),
        ("scale_num", StandardScaler())
    ])

    # Categorical pipeline: impute → one‐hot (only best_cat columns)
    cat_pipeline = Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot_best", OneHotEncoder(
            drop="first",           # drop one dummy to avoid collinearity
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # ColumnTransformer will call select_best_cat_cols at fit time to pick existing best_cat columns
    coltrans = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, select_best_cat_cols),
    ], remainder="drop")

    return coltrans

# Instantiate and fit the non‐time preprocessor
non_time_preproc = build_non_time_preprocessor(numeric_non_time, categorical_non_time)
non_time_preproc.fit(X_train[numeric_non_time + categorical_non_time])

# -----------------------------------------------------------------------------
# 7. Transform non‐time features for train & test
# -----------------------------------------------------------------------------
X_nt_train = non_time_preproc.transform(X_train[numeric_non_time + categorical_non_time])
X_nt_test  = non_time_preproc.transform(X_test[numeric_non_time + categorical_non_time])

# Determine number of resulting non‐time features
n_non_time_feats = X_nt_train.shape[1]

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

# Build and fit a scaler for these time features on TRAIN
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
# 9. Concatenate non‐time + time arrays into a single feature matrix
# -----------------------------------------------------------------------------
X_train_feat = np.hstack([X_nt_train, X_time_train_scaled]) # type: ignore
X_test_feat  = np.hstack([X_nt_test,  X_time_test_scaled]) # type: ignore

# -----------------------------------------------------------------------------
# 10. Apply PCA with the best‐found n_components = 116
#      (from your 5‐fold CV tuning)
# -----------------------------------------------------------------------------
n_components = 116
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_feat)
X_test_pca  = pca.transform(X_test_feat)

# -----------------------------------------------------------------------------
# 11. Build DataFrames and save cleaned CSVs
# -----------------------------------------------------------------------------
# Column names for PCA components: PC1, PC2, ..., PC116
pca_colnames = [f"PC{i+1}" for i in range(n_components)]

# TRAIN DataFrame
df_clean_train = pd.DataFrame(
    X_train_pca,
    columns=pca_colnames,
    index=X_train.index
)
# Append target back
df_clean_train["SalePrice"] = y.values

# TEST DataFrame
df_clean_test = pd.DataFrame(
    X_test_pca,
    columns=pca_colnames,
    index=X_test.index
)

# Save to CSV
df_clean_train.to_csv("data/processed/PCR_clean_train_pcr.csv", index=False)
df_clean_test.to_csv("data/processed/PCR_clean_test_pcr.csv",   index=False)
test_ids.to_csv("data/processed/PCR_test_id.csv", index=False)

print("Finished. Wrote PCR_clean_train_pcr.csv, PCR_clean_test_pcr.csv and PCR_test_id.csv to data/processed/")
