import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, mean_squared_error
import warnings
import matplotlib.pyplot as plt

# ── 1.  Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/clean_train.csv')
y  = np.log(df['SalePrice'].values)        
X  = df.drop(columns=['SalePrice']).values

# ── 2.  PCR pipeline ───────────────────────────────────────────────────────────
pcr = Pipeline([
    ('nzv',   VarianceThreshold()),              
    #('scale', StandardScaler()),                   
    ('pca',   PCA(svd_solver='randomized', random_state=42)),
    ('lin',   LinearRegression())
])

# ── 3.  CV setup ──────────────────────────────────────────────────────────────
def neg_rmse(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))

scorer     = make_scorer(neg_rmse, greater_is_better=True)
max_pc     = min(100, X.shape[1])                   
param_grid = {'pca__n_components': list(range(1, max_pc + 1, 5))}

cv  = KFold(n_splits=10, shuffle=True, random_state=42)

# ── 4.  Grid-search with warnings suppressed ─────────────────────────────────
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                            message=".*encountered in matmul",
                            category=RuntimeWarning)
    gcv = GridSearchCV(
        estimator=pcr,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2            
    ).fit(X, y)

# ── 5.  Results ───────────────────────────────────────────────────────────────
best_k   = gcv.best_params_['pca__n_components']
best_rmse = -gcv.best_score_
print(f'Best #PCs: {best_k:>3d}   CV log-RMSE: {best_rmse:0.5f}')

ks   = [p['pca__n_components'] for p in gcv.cv_results_['params']]
rmse = [-s for s in gcv.cv_results_['mean_test_score']]

plt.plot(ks, rmse, marker='o')
plt.axvline(best_k, color='r', ls='--', lw=1)
plt.xlabel('# Principal Components')
plt.ylabel('CV log-RMSE ↓')
plt.title('PCR Cross-Validation Curve')
plt.show()
