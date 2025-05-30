# model_pipeline.py
import numpy as np
import pandas as pd

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
def handle_missing_values(df: pd.DataFrame, mean_maps: dict | None = None):
    """
    Clean & encode the Ames Housing dataset.

    Parameters
    ----------
    df : DataFrame
        Raw train or test data (train includes SalePrice).
    mean_maps : dict | None
        • None  →  FIT mode: compute and return mean-encoding maps  
        • dict  →  TRANSFORM mode: apply the provided maps (no SalePrice needed)

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

    # 36 ── Boolean → int ───────────────────────────────────────────────────────
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # ── return ────────────────────────────────────────────────────────────────
    if is_fit:
        return df, ids, mean_maps
    else:
        return df, ids
