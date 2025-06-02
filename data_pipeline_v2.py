import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def winsorize_series(s, lower_pct=0.005, upper_pct=0.995):
    low, high = s.quantile([lower_pct, upper_pct])
    return s.clip(lower=low, upper=high)

def compute_skewed_columns(df, threshold=1.0):
    skew = df.skew().abs().sort_values(ascending=False)
    return skew[skew > threshold].index.tolist()

def fit_pipeline_for_maps(df: pd.DataFrame) -> dict:
    """
    Fit‐mode on TRAIN → build mean_maps containing:
      • drop50_cols
      • winsor_bounds
      • impute_median
      • skew_cols
      • rare_group_map
      • oh_cols
      • pre_vt_cols      ← list of columns just before VarianceThreshold
      • vt_selector
      • scaler (or None)
    """
    df = df.copy()
    df = df.drop(columns=["Id"])       
    y = df["SalePrice"].copy()         
    df = df.drop(columns=["SalePrice"])

    mean_maps: dict = {}

    # 1) Drop columns with >50% missing
    missing_frac = df.isna().mean()
    drop50 = missing_frac[missing_frac > 0.5].index.tolist()
    df = df.drop(columns=drop50)
    mean_maps["drop50_cols"] = drop50

    # 2) Identify numeric vs. categorical
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object"]).columns.tolist()

    # 3) Winsorize each numeric at [0.5%, 99.5%]
    winsor_bounds: dict[str, tuple[float, float]] = {}
    for c in numeric_cols:
        low, high = df[c].quantile([0.005, 0.995])
        winsor_bounds[c] = (low, high)
        df[c] = df[c].clip(lower=low, upper=high)
    mean_maps["winsor_bounds"] = winsor_bounds

    # 4) Missing‐flag + median‐impute for numeric NAs
    impute_median: dict[str, float] = {}
    for c in numeric_cols:
        if df[c].isna().any():
            impute_median[c] = df[c].median()
            df[f"{c}_miss"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(impute_median[c])
    mean_maps["impute_median"] = impute_median

    # 5) Recompute numeric_cols (now includes any "_miss" flags)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 6) Log1p‐transform any numeric with |skew| > 1
    skew_cols = compute_skewed_columns(df[numeric_cols], threshold=1.0)
    mean_maps["skew_cols"] = skew_cols
    for c in skew_cols:
        df[f"{c}_log"] = np.log1p(df[c].values)
        df = df.drop(columns=[c])

    # 7) Rare‐group each categorical (<30 → "Other"), then one‐hot small‐cardinal
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    rare_group_map: dict[str, list[str]] = {}
    for c in cat_cols:
        vc = df[c].value_counts()
        rare_levels = vc[vc < 30].index.tolist()
        rare_group_map[c] = rare_levels
        df[c] = df[c].replace(rare_levels, "Other")
    mean_maps["rare_group_map"] = rare_group_map

    oh_cols = [c for c in cat_cols if c in df.columns and df[c].nunique() <= 5]
    mean_maps["oh_cols"] = oh_cols
    df_oh = pd.get_dummies(df[oh_cols], drop_first=True)

    # Drop all original categorical columns, then append the one‐hot dummies
    df = df.drop(columns=cat_cols)
    df = pd.concat([df.reset_index(drop=True), df_oh.reset_index(drop=True)], axis=1)

    # ──────────────────────────────────────────────────────────────────────────
    # ↳ **New step: convert any remaining bool columns to int before scaling**
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
    # ──────────────────────────────────────────────────────────────────────────

    # 8) Record the order of columns just before VarianceThreshold
    pre_vt_cols = df.columns.tolist()
    mean_maps["pre_vt_cols"] = pre_vt_cols

    # 9) Drop near‐constant features via VarianceThreshold(threshold=0.01)
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df.values)
    mean_maps["vt_selector"] = selector

    df_reduced = pd.DataFrame(
        selector.transform(df.values), # type: ignore
        columns=[col for i, col in enumerate(df.columns) if selector.get_support()[i]],
        index=df.index
    ) # type: ignore

    # 10) Standardize all numeric columns (if any remain)
    final_numeric_cols = df_reduced.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(final_numeric_cols) > 0:
        scaler = StandardScaler().fit(df_reduced[final_numeric_cols].values)
        mean_maps["scaler"] = scaler
    else:
        mean_maps["scaler"] = None

    return mean_maps


def clean_pipeline_transform(df: pd.DataFrame, mean_maps: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform‐mode cleaning for TEST data:
      1) Drop same >50%‐missing columns
      2) Winsorize using winsor_bounds
      3) Create "_miss" flags + median‐impute
      4) Log1p‐transform skewed columns
      5) Rare‐group + one‐hot only the stored oh_cols
      6) **Convert any bool → int here as well**
      7) REINDEX to mean_maps['pre_vt_cols'], filling any missing columns with 0
      8) VarianceThreshold and StandardScaler as before
    Returns (df_cleaned, ids).
    """
    df = df.copy()
    ids = df["Id"].copy()
    df = df.drop(columns=["Id"])

    # 1) Drop >50%‐missing columns
    drop50 = mean_maps["drop50_cols"]
    df = df.drop(columns=[c for c in drop50 if c in df.columns], errors="ignore")

    # 2) Winsorize numeric columns
    winsor_bounds = mean_maps["winsor_bounds"]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for c in numeric_cols:
        if c in df.columns:
            low, high = winsor_bounds[c]
            df[c] = df[c].clip(lower=low, upper=high)

    # 3) Missing‐flag + median‐impute for numeric
    impute_median = mean_maps["impute_median"]
    for c, med in impute_median.items():
        if c in df.columns:
            df[f"{c}_miss"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(med)

    # 4) Log1p‐transform skewed columns
    skew_cols = mean_maps["skew_cols"]
    for c in skew_cols:
        if c in df.columns:
            df[f"{c}_log"] = np.log1p(df[c].values)
            df = df.drop(columns=[c])

    # 5) Rare‐group + one‐hot small‐cardinal categorical
    rare_group_map = mean_maps["rare_group_map"]
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].replace(rare_group_map[c], "Other")

    oh_cols = mean_maps["oh_cols"]
    df_oh = pd.get_dummies(df[oh_cols], drop_first=True)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df.reset_index(drop=True), df_oh.reset_index(drop=True)], axis=1)

    # ──────────────────────────────────────────────────────────────────────────
    # ↳ **Convert any bool columns → int before scaling**
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
    # ──────────────────────────────────────────────────────────────────────────

    # 6) Fill any remaining NaNs with 0 (just in case)
    df = df.fillna(0)

    # 7) Reindex to pre-VT column order, filling missing with 0
    pre_vt_cols = mean_maps["pre_vt_cols"]
    df = df.reindex(columns=pre_vt_cols, fill_value=0)

    # 8) Apply VarianceThreshold
    selector = mean_maps["vt_selector"]
    df_reduced = pd.DataFrame(
        selector.transform(df.values),
        columns=[col for i, col in enumerate(df.columns) if selector.get_support()[i]],
        index=df.index
    )

    # 9) Standardize numeric columns if scaler exists
    scaler = mean_maps["scaler"]
    final_numeric_cols = df_reduced.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if (scaler is not None) and len(final_numeric_cols) > 0:
        df_reduced[final_numeric_cols] = scaler.transform(df_reduced[final_numeric_cols].values)

    return df_reduced, ids


# ──────────────────────────────────────────────────────────────────────────────
# 3) SAVE clean_train_v2.csv → apply fit_pipeline_for_maps to train.csv
# ──────────────────────────────────────────────────────────────────────────────
raw_train = pd.read_csv('data/raw/train.csv')
mean_maps = fit_pipeline_for_maps(raw_train)

# Now re‐apply those maps to produce the cleaned training matrix
df_train_noid = raw_train.drop(columns=['Id'])
y_train = df_train_noid['SalePrice'].copy()
df_train_noid = df_train_noid.drop(columns=['SalePrice'])

# Step‐by‐step replicate transform on train:
#    pass exactly “Id + all feature cols (no SalePrice)” into transform
train_input = raw_train[['Id'] + [c for c in raw_train.columns if c not in ('Id','SalePrice')]]
X_train_transformed, ids_train = clean_pipeline_transform(train_input, mean_maps)

df_clean_train_v2 = X_train_transformed.copy()
df_clean_train_v2.insert(0, 'Id', ids_train.values) # type: ignore
df_clean_train_v2['SalePrice'] = y_train.values

os.makedirs('data/processed', exist_ok=True)
df_clean_train_v2.to_csv('data/processed/clean_train_v3.csv', index=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4) SAVE clean_test_v2.csv → apply clean_pipeline_transform to test.csv
# ──────────────────────────────────────────────────────────────────────────────
raw_test = pd.read_csv('data/raw/test.csv')
test_input = raw_test[['Id'] + [c for c in raw_test.columns if c != 'Id']]

X_test_transformed, ids_test = clean_pipeline_transform(test_input, mean_maps)
df_clean_test_v2 = X_test_transformed.copy()
df_clean_test_v2.insert(0, 'Id', ids_test.values) # type: ignore

df_clean_test_v2.to_csv('data/processed/clean_test_v3.csv', index=False)

print("Saved → data/processed/clean_train_v3.csv\n      → data/processed/clean_test_v3.csv")
