# model_pipeline.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error

# ────────────────────────────────────────────────────────────────────────────────
#  LowRankImputer for matrix completion
# ────────────────────────────────────────────────────────────────────────────────

class LowRankImputer:
    """
    Iterative low-rank SVD imputer

    Parameters
    ----------
    n_components : int   (rank M)
    tol          : float (relative improvement stop)
    max_iter     : int   (safety cap)
    """

    def __init__(self, n_components: int = 2, tol: float = 1e-7, max_iter: int = 50):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    # sklearn-compatible API ----------------------------------------------------
    def get_params(self, deep=True):
        return {
            "n_components": self.n_components,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # --------------------------------------------------------------------------
    def fit(self, X, y=None):
        X = X.copy().to_numpy(float)
        self.miss_mask_ = np.isnan(X)
        # init with column means
        col_mean = np.nanmean(X, axis=0)
        X[self.miss_mask_] = np.take(col_mean, np.where(self.miss_mask_)[1])
        prev = np.inf
        for _ in range(self.max_iter):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            L = U[:, : self.n_components] * s[: self.n_components]
            X_hat = L @ Vt[: self.n_components]
            X[self.miss_mask_] = X_hat[self.miss_mask_]
            obj = np.nanmean((X_hat[~self.miss_mask_] - X[~self.miss_mask_]) ** 2)
            if prev - obj < self.tol * obj:
                break
            prev = obj
        self.X_completed_ = X
        return self

    def transform(self, X):
        Xc = X.copy().to_numpy(float)
        mask = np.isnan(Xc)
        Xc[mask] = np.take(np.nanmean(self.X_completed_, axis=0),np.where(mask)[1])
        return Xc

    # needed so GridSearchCV can call estimator.score when no y is supplied
    def score(self, X, y=None):
        mask = ~np.isnan(X.to_numpy())
        mse = np.mean(
            (self.transform(X)[mask] - X.to_numpy()[mask]) ** 2
        )
        # negate because GridSearchCV maximises score
        return -mse
    
# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame."""
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)



# ────────────────────────────────────────────────────────────────────────────────
# Main cleaning / encoding function
# ────────────────────────────────────────────────────────────────────────────────

def _imputer_scorer(estimator, X, y=None):
    """Custom scorer that only needs X (unsupervised)."""
    mask = ~np.isnan(X.to_numpy())
    mse = mean_squared_error(X.to_numpy()[mask], estimator.transform(X)[mask])
    return -mse

def handle_missing_values(df: pd.DataFrame, mean_maps: dict | None = None):
    """
    Clean & encode the Ames Housing dataset.

    Returns
    -------
    df_clean  : cleaned feature frame (SalePrice retained if it came in)
    ids       : original Id column (Series, row-aligned)
    mean_maps : dict (only in FIT mode)
    """
    # 0 ── pull Id, detect mode ──────────────────────────────────────────────────
    ids         = df['Id'].copy()
    df          = df.drop(columns=['Id'])
    has_target  = 'SalePrice' in df.columns
    is_fit      = mean_maps is None
    if is_fit:
        mean_maps = {}

    # ── helper for target-mean encoding ────────────────────────────────────────
    def mean_encode(column: str):
        nonlocal mean_maps, df
        if is_fit:
            # require SalePrice to exist in fit mode
            mean_maps[column] = df.groupby(column)['SalePrice'].mean()
        df[f'{column}_te'] = df[column].map(mean_maps[column]).fillna(mean_maps[column].mean())
        df.drop(columns=[column], inplace=True)

    # 1 ── Electrical (drop) ────────────────────────────────────────────────────
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df.drop(columns=['Electrical'], inplace=True)

    # 2 ── Masonry Veneer ───────────────────────────────────────────────────────
    mean_encode('MasVnrType')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    # 3 ── Basement (ordinal) ───────────────────────────────────────────────────
    bsmt_map = {
        'BsmtQual':     {'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4},
        'BsmtCond':     {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
        'BsmtExposure': {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4},
        'BsmtFinType1': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
        'BsmtFinType2': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
    }
    for col, mp in bsmt_map.items():
        df[col] = df[col].fillna('None')
        df[f'{col}_ord'] = df[col].map(mp)
    df['HasBsmt'] = (df['BsmtQual'] != 'None').astype(int)
    df.drop(columns=list(bsmt_map.keys()), inplace=True)

    # 4 ── Garage (ordinal) ─────────────────────────────────────────────────────
    for c in ['GarageType','GarageFinish','GarageQual','GarageCond']:
        df[c] = df[c].fillna('None')
    gt_map = {'None':0,'CarPort':1,'Detchd':2,'Basment':3,'2Types':4,'Attchd':5,'BuiltIn':6}
    fin_map = {'None':0,'Unf':1,'RFn':2,'Fin':3}
    qc_map  = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    df['GarageType_ord']   = df['GarageType'].map(gt_map)
    df['GarageFinish_ord'] = df['GarageFinish'].map(fin_map)
    df['GarageQual_ord']   = df['GarageQual'].map(qc_map)
    df['GarageCond_ord']   = df['GarageCond'].map(qc_map)
    df['GarageCars']       = df['GarageCars'].fillna(0).astype(int)
    df['GarageArea']       = df['GarageArea'].fillna(0)
    df['GarageAge']        = np.where(df['GarageYrBlt'].isna(), 0,
                                      df['YrSold'] - df['GarageYrBlt'])
    df['HasGarage']        = (df['GarageAge'] > 0).astype(int)
    df.drop(columns=['GarageType','GarageFinish','GarageQual',
                     'GarageCond','GarageYrBlt'], inplace=True)

    # 5 ── Time features ────────────────────────────────────────────────────────
    df['HouseAge']   = df['YrSold'] - df['YearBuilt']
    df['RemodAge']   = df['YrSold'] - df['YearRemodAdd']
    df['MoSold_sin'] = np.sin(2*np.pi*df['MoSold']/12)
    df['MoSold_cos'] = np.cos(2*np.pi*df['MoSold']/12)
    df.drop(columns=['YearBuilt','YearRemodAdd','MoSold'], inplace=True)

    # 6 ── MiscFeature flag ─────────────────────────────────────────────────────
    df['HasMisc'] = df['MiscFeature'].notna().astype(int)
    df.drop(columns=['MiscFeature'], inplace=True)

    # 7 ── Fireplace ordinal ────────────────────────────────────────────────────
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    fire_map = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    df['FireplaceQu_ord'] = df['FireplaceQu'].map(fire_map)
    df.drop(columns=['FireplaceQu'], inplace=True)

    # 8 ── Pool info ────────────────────────────────────────────────────────────
    df['PoolArea'] = df['PoolArea'].fillna(0)
    cap = df['PoolArea'].quantile(0.95)
    df['PoolHighOutlier'] = (df['PoolArea'] > cap).astype(int)
    drop_cols = [c for c in df.columns if c.startswith('Pool') and
                 c not in ['PoolArea','PoolHighOutlier']]
    df.drop(columns=drop_cols, inplace=True)

    # 9 ── Alley drop ───────────────────────────────────────────────────────────
    df.drop(columns=[c for c in ['Alley'] if c in df.columns], inplace=True)

    # 10 ── Fence flag ──────────────────────────────────────────────────────────
    df['Fence'] = df['Fence'].fillna('None')
    df['HasFence'] = (df['Fence'] != 'None').astype(int)
    df.drop(columns=['Fence'], inplace=True)

    # 11 ── LotFrontage ─────────────────────────────────────────────────────────
    df['LotFrontage'] = df['LotFrontage'].fillna(0)

    # 12 ── MSZoning mean ───────────────────────────────────────────────────────
    mean_encode('MSZoning')

    # 13 ── Street flag ─────────────────────────────────────────────────────────
    df['Street_flag'] = (df['Street'] == 'Pave').astype(int)
    df.drop(columns=['Street'], inplace=True)

    # 14 ── LotShape mean ───────────────────────────────────────────────────────
    mean_encode('LotShape')

    # 15 / 16  LandContour & Utilities drop ─────────────────────────────────────
    df.drop(columns=[c for c in ['LandContour','Utilities'] if c in df.columns],
            inplace=True)

    # 17 ── LotConfig mean ──────────────────────────────────────────────────────
    mean_encode('LotConfig')

    # 18 ── LandSlope ordinal ───────────────────────────────────────────────────
    df['LandSlope'] = df['LandSlope'].fillna('Gtl')
    df['LandSlope_ord'] = df['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})
    df.drop(columns=['LandSlope'], inplace=True)

    # 19 ── Neighborhood mean ───────────────────────────────────────────────────
    mean_encode('Neighborhood')

    # 20 ── Condition flags ─────────────────────────────────────────────────────
    df['Condition1'] = df['Condition1'].fillna('Norm')
    df['Condition2'] = df['Condition2'].fillna('Norm')
    df['MultipleConditions'] = (df['Condition2'] != 'Norm').astype(int)
    df['RoadProximity'] = ((df['Condition1'].isin(['Artery','Feedr'])) |
                           (df['Condition2'].isin(['Artery','Feedr']))).astype(int)
    df['RailroadProximity'] = (df['Condition1'].str.startswith('RR') |
                               df['Condition2'].str.startswith('RR')).astype(int)
    df['PositiveOffsite'] = ((df['Condition1'].isin(['PosN','PosA'])) |
                             (df['Condition2'].isin(['PosN','PosA']))).astype(int)
    df.drop(columns=['Condition1','Condition2'], inplace=True)

    # 21 ── BldgType mean ───────────────────────────────────────────────────────
    mean_encode('BldgType')

    # 22 ── HouseStyle ordinal ──────────────────────────────────────────────────
    df['HouseStyle_ord'] = df['HouseStyle'].map(
        {'1Story':1,'1.5Fin':2,'SFoyer':2,'2Story':3,'SLvl':4})
    df.drop(columns=['HouseStyle'], inplace=True)

    # 23 ── RoofStyle flag ──────────────────────────────────────────────────────
    df['RoofStyle_Common'] = df['RoofStyle'].isin(['Gable','Hip']).astype(int)
    df.drop(columns=['RoofStyle'], inplace=True)

    # 24 ── RoofMatl drop ───────────────────────────────────────────────────────
    df.drop(columns=[c for c in ['RoofMatl'] if c in df.columns], inplace=True)

    # 25 / 26 ── Exterior & ExterQual/Cond means ───────────────────────────────
    for col in ['Exterior1st','Exterior2nd','ExterQual','ExterCond']:
        mean_encode(col)

    # 27 ── Foundation one-hot ──────────────────────────────────────────────────
    fd = pd.get_dummies(df['Foundation'], prefix='Foundation', drop_first=True)
    df = pd.concat([df, fd], axis=1)
    df.drop(columns=['Foundation'], inplace=True)

    # 28 ── Heating drop ────────────────────────────────────────────────────────
    df.drop(columns=['Heating'], inplace=True, errors='ignore')

    # 29 ── HeatingQC ordinal ───────────────────────────────────────────────────
    df['HeatingQC_ord'] = df['HeatingQC'].map({'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
    df.drop(columns=['HeatingQC'], inplace=True)

    # 30 ── CentralAir flag ─────────────────────────────────────────────────────
    df['CentralAir_flag'] = (df['CentralAir']=='Y').astype(int)
    df.drop(columns=['CentralAir'], inplace=True)

    # 31-35 ── Remaining mean-encoded categoricals ─────────────────────────────
    for col in ['KitchenQual','Functional','PavedDrive','SaleType','SaleCondition']:
        mean_encode(col)

    # 36 ── Standardize Features ─────────────────────────────────────────────────
    std_cols = df.select_dtypes("number").columns.tolist()
    if "SalePrice" in std_cols:
        std_cols.remove("SalePrice")
    if is_fit:
        scaler = StandardScaler().fit(df[std_cols])
        mean_maps["_scaler"] = scaler
    else:
        scaler = mean_maps["_scaler"]
    df[std_cols] = scaler.transform(df[std_cols])

    # 37 ── Boolean → int ───────────────────────────────────────────────────────
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 38 ── Matrix Completion for remaining Na ──────────────────────────────────
    numeric_cols = df.select_dtypes("number").columns.tolist()
    if "SalePrice" in numeric_cols:
        numeric_cols.remove("SalePrice")

    if is_fit:
        grid = GridSearchCV(
            estimator=LowRankImputer(),
            param_grid={"n_components": list(range(1, 8))},
            scoring=_imputer_scorer,  # custom X-only scorer
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        grid.fit(df[numeric_cols])
        best_imp = grid.best_estimator_
        mean_maps["_imputer"] = best_imp
    else:
        best_imp = mean_maps["_imputer"]

    df[numeric_cols] = best_imp.transform(df[numeric_cols])


    # ── return ────────────────────────────────────────────────────────────────
    if is_fit:
        return df, ids, mean_maps
    else:
        return df, ids
