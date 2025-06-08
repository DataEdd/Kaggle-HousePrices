import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_predict
from scipy import stats
import seaborn as sns

#def neg_rmse(y_true, y_pred):
#    """Negative RMSE for log‐target (so that higher is better)."""
#    return -np.sqrt(mean_squared_error(y_true, y_pred))

def make_dirs_if_needed(path: str):
    """Helper to create a directory if it doesn't exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def compute_coef_path(X_std: np.ndarray, y_log: np.ndarray, m_max: int):
    """
    Compute a (m_max × p) array of PLS coefficients, where each row k
    corresponds to the standardized coefficients when using k components.
    Assumes that X_std is already standardized (n × p) and y_log is (n,).
    """
    n, p = X_std.shape
    coef_mat = np.zeros((m_max, p))
    for k in range(1, m_max + 1):
        pls_k = PLSRegression(n_components=k, scale=True)
        pls_k.fit(X_std, y_log)
        coef_mat[k - 1, :] = pls_k.coef_.ravel()
    return coef_mat

def compute_response_variance(X_std: np.ndarray, y_log: np.ndarray, m_max: int):
    """
    For a PLS fit with m_max components on (X_std, y_log), compute:
      - per_component_frac[i] = fraction of Var(y) explained by the i-th PLS component
      - cum_explained[k] = cumulative fraction of Var(y) explained up to component k
    """
    pls_full = PLSRegression(n_components=m_max, scale=True)
    pls_full.fit(X_std, y_log)
    T = pls_full.x_scores_           # shape (n, m_max)
    Q = pls_full.y_loadings_.ravel() # shape (m_max,)
    y_centered = y_log - y_log.mean()
    ss_total = np.sum(y_centered ** 2)

    ss_explained = []
    for i in range(m_max):
        ss_i = (T[:, i] ** 2).sum() * (Q[i] ** 2)
        ss_explained.append(ss_i)
    per_component_frac = np.array(ss_explained) / ss_total
    cum_explained = np.cumsum(ss_explained) / ss_total

    return per_component_frac, cum_explained

def main() -> None:
    # 1) ─── Load cleaned training matrix ───────────────────────────────────────
    train_path = os.path.join("data", "processed", "PLS_clean_train.csv")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Could not find training file at: {train_path}")

    df_train = pd.read_csv(train_path).fillna(0)
    if "SalePrice" not in df_train.columns:
        raise KeyError("'SalePrice' column missing from training set")

    X_train_df = df_train.drop(columns=["SalePrice"])
    y_train_log = np.log1p(df_train["SalePrice"].values) # type: ignore
    n_samples, n_features = X_train_df.shape

    print(f"[INFO] Loaded training matrix: {n_samples}×{n_features}")

    # 2) ─── Grid‑search for optimal # of PLS components ────────────────────────
    pipe = Pipeline([("pls", PLSRegression(scale=True))])
    param_grid = {"pls__n_components": list(range(1, n_features + 1))}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error", 
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    print(f"[INFO] Starting GridSearchCV over 1…{n_features} components (5‑fold).")
    grid.fit(X_train_df, y_train_log)
    print("[INFO] GridSearchCV complete.\n")

    best_k   = grid.best_params_["pls__n_components"]
    best_rmse = -grid.best_score_
    print(f"[RESULT] Best n_components: {best_k}")
    print(f"[RESULT] CV RMSE (log target): {best_rmse:.6f}\n")

    # 3) ─── Fit final PLS on full training data ────────────────────────────────
    final_pls = PLSRegression(n_components=best_k, scale=True)
    final_pls.fit(X_train_df, y_train_log)
    print(f"[INFO] Final PLS model (k={best_k}) fitted on full training set.\n")

    # 4) ─── Predict on cleaned test matrix ─────────────────────────────────────
    test_feats_path = os.path.join("data", "processed", "PLS_clean_test.csv")
    if not os.path.isfile(test_feats_path):
        raise FileNotFoundError(f"Could not find test features at: {test_feats_path}")

    X_test_df = pd.read_csv(test_feats_path).fillna(0)
    if X_test_df.shape[1] != n_features:
        raise ValueError("Column mismatch between train and test matrices")

    y_test_log_pred = final_pls.predict(X_test_df).ravel()
    y_test_pred     = np.expm1(y_test_log_pred)  # back‑transform to dollars
    print("[INFO] Test predictions complete (exponentiated from log‑space).\n")

    # 5) ─── Assemble submission CSV ────────────────────────────────────────────
    id_test_path = os.path.join("data", "processed", "PLS_test_id.csv")
    if not os.path.isfile(id_test_path):
        raise FileNotFoundError(f"Could not find test IDs at: {id_test_path}")

    test_ids = pd.read_csv(id_test_path)["Id"].values
    if len(test_ids) != len(y_test_pred):
        raise ValueError("Row mismatch between test IDs and feature matrix")

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_test_pred})
    out_dir = os.path.join("data", "submission")
    make_dirs_if_needed(out_dir)
    sub_path = os.path.join(out_dir, "pls_submission_final.csv")
    submission.to_csv(sub_path, index=False)
    print(f"[SUCCESS] Submission written to: {sub_path}\n")

    # ── 6) DIAGNOSTIC PLOTS ─────────────────────────────────────────
    figs_dir = os.path.join("data", "figs")
    make_dirs_if_needed(figs_dir)

# (a) CV curve 
    sns.set_style('whitegrid') 
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    main_color = palette[1]       # tab:orange
    highlight_color = palette[3]  # tab:red
    scatter_color = palette[4]    # tab:purple
    ORANGE = "#E69F00"         
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    max_k  = 99              
    rmse   = [-cross_val_score(PLSRegression(n_components=k, scale=True),X_train_df, y_train_log, cv=cv,scoring="neg_root_mean_squared_error").mean()
    for k in range(1, max_k + 1)
    ]

    best_k   = int(np.argmin(rmse) + 1)
    best_err = rmse[best_k - 1]

    plt.figure(figsize=(8,4))
    plt.plot(range(1, max_k+1), rmse, color=ORANGE, lw=2, label="CV RMSE")
    plt.axvline(best_k, linestyle="--", color="red", lw=2, label=f"Selected k = {best_k}")
    plt.scatter([best_k], [best_err], color="red", zorder=3, marker="x", s=60)

    plt.xlabel("# PLS Components"); plt.ylabel("CV RMSE (log SalePrice)")
    plt.title("PLS – CV RMSE vs Number of Components")
    plt.ylim(best_err*0.97, best_err*1.20)      # zoom  ±20 %
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend(); plt.tight_layout()
    #plt.show()
    plt.close()

    # (b) Coefficient path with top‑5 highlighted  
    coef_mat   = compute_coef_path(X_train_df, y_train_log, best_k) # type: ignore
    abs_final  = np.abs(coef_mat[best_k-1])
    top_idx    = abs_final.argsort()[-5:][::-1]
    feature_names = X_train_df.columns.tolist()

    plt.figure(figsize=(6, 4))
    for j in range(coef_mat.shape[1]):
        color = ORANGE if j in top_idx else "lightgray"
        lw    = 2.2 if j in top_idx else 0.6
        alpha = 1.0 if j in top_idx else 0.7
        plt.plot(range(1, best_k + 1), coef_mat[:, j], color=color, lw=lw, alpha=alpha)
    # annotate names at path end for top 5
    for j in top_idx:
        plt.text(best_k + 0.5, coef_mat[-1, j], feature_names[j], fontsize=8, va="center")

    plt.xlabel("# Components"); plt.ylabel("Standardised Coefficient")
    plt.title("PLS Coefficient Path – Top 5 Features (Orange)"); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pls_coef_path_2.png"), dpi=150)
    plt.close()

    # (c) QQ‑plot of out‑of‑fold residuals with top‑5 highlighted
    coef_mat   = compute_coef_path(X_train_df, y_train_log, best_k) # type: ignore
    abs_final  = np.abs(coef_mat[best_k-1])
    top_idx    = abs_final.argsort()[-5:][::-1]
    feature_names = X_train_df.columns.tolist()

    plt.figure(figsize=(6, 4))
    for j in range(coef_mat.shape[1]):
        color = "tab:orange" if j in top_idx else "lightgray"
        lw    = 2.0 if j in top_idx else 0.6
        plt.plot(range(1, best_k + 1), coef_mat[:, j], color=color, lw=lw)
    # annotate names at path end
    for j in top_idx:
        plt.text(best_k + 0.3, coef_mat[-1, j], feature_names[j], fontsize=9, va="center")

    plt.xlabel("# Components"); plt.ylabel("Standardised Coefficient")
    plt.title("PLS Coefficient Path – Top 5 Features Highlighted"); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pls_coef_path.png"), dpi=150)
    plt.close()

    # (c) QQ‑plot of out‑of‑fold residuals
    from sklearn.model_selection import cross_val_predict
    from scipy import stats

    oof_pred = cross_val_predict(PLSRegression(n_components=best_k, scale=True),
                                 X_train_df, y_train_log, cv=cv)
    residuals = y_train_log - oof_pred

    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ plot of residuals")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pls_residuals_qq.png"), bbox_inches='tight')
    plt.close()

    # (d) Out-of-Fold predictions vs Actual
    plt.figure(figsize=(6, 4))
    plt.scatter(y_train_log, oof_pred, alpha=0.6, label="OOF predictions", color=main_color, marker='x') 
    min_val, max_val = y_train_log.min(), y_train_log.max()
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label="Ideal")
    plt.xlabel('Actual log SalePrice')
    plt.ylabel('Predicted log SalePrice')
    plt.title(f'PLS Predicted vs Actual (OOF, {best_k} components)')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    plt.savefig('data/figs/PLS_pred_vs_actual.png', bbox_inches='tight')
    plt.close()
   # print(f"[INFO] Diagnostic plots saved to {figs_dir}


if __name__ == "__main__":
    main()