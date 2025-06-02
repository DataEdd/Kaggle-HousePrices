#!/usr/bin/env python3
"""
PLS.py

This script performs Partial Least Squares (PLS) regression on the cleaned Ames Housing dataset
(`clean_train_v2.csv`), using log-transformed SalePrice as the target. It performs an extensive
grid search over the number of PLS components (from 1 up to the total number of features) with
5-fold CV to find the optimal number of components, generates a collection of diagnostic plots
(saved under data/figs with a “pls_” prefix), then refits on the full training set with the
best number of components, and finally predicts on `clean_test_v2.csv`. The submission CSV
(`pls_submission_v2.csv`) is written under data/submissions/ and contains only “Id” and “SalePrice”.

Assumptions:
- The training file is at: data/processed/clean_train_v2.csv
  (contains columns: “Id”, all cleaned features, and “SalePrice”).
- The test-features file is at: data/processed/clean_test_v2.csv
  (contains columns: “Id” and exactly the same feature set as clean_train_v2.csv.drop("SalePrice")).
- The test-ID file is at: data/processed/id_test.csv
  (contains exactly one column “Id” with the same number of rows as clean_test_v2.csv).
- All features in clean_train_v2.csv and clean_test_v2.csv are already encoded and standardized
  appropriately (no further scaling needed in PLSRegression).
- There are no missing values in either file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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
        pls_k = PLSRegression(n_components=k, scale=False)
        pls_k.fit(X_std, y_log)
        coef_mat[k - 1, :] = pls_k.coef_.ravel()
    return coef_mat

def compute_response_variance(X_std: np.ndarray, y_log: np.ndarray, m_max: int):
    """
    For a PLS fit with m_max components on (X_std, y_log), compute:
      - per_component_frac[i] = fraction of Var(y) explained by the i-th PLS component
      - cum_explained[k] = cumulative fraction of Var(y) explained up to component k
    """
    pls_full = PLSRegression(n_components=m_max, scale=False)
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

def main():
    # ─────────────────────────────────────────────────────────────────────────────
    # 1) LOAD AND PREPARE TRAINING DATA
    # ─────────────────────────────────────────────────────────────────────────────
    train_path = os.path.join('data', 'processed', 'clean_train.csv')
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Could not find training file at: {train_path}")

    df_train = pd.read_csv(train_path)
    df_train = df_train.fillna(0)  # Fill any NaNs with 0 (assumed no missing values)
    df_train = df_train.fillna(0)  # Fill any NaNs with 0 (assumed no missing values)
    if 'SalePrice' not in df_train.columns:
        raise KeyError(f"'SalePrice' column not found in {train_path}")

    # Separate features (X_train) and log-target (y_train_log)
    X_train_df = df_train.drop(columns=['SalePrice'])
    y_train = df_train['SalePrice'].values
    y_train_log = np.log(y_train)

    # Convert to numpy arrays (X_train is assumed already standardized)
    X_train = X_train_df.drop(columns=['Id']).values
    n_samples, n_features = X_train.shape

    print(f"[INFO] Loaded training data: {n_samples} rows, {n_features} features.")
    print(f"[INFO] Target will be log-transformed, shape: {y_train_log.shape}.\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) SET UP PIPELINE AND GRID SEARCH FOR OPTIMAL N_COMPONENTS
    # ─────────────────────────────────────────────────────────────────────────────
    pls = PLSRegression(scale=False)
    pipe = Pipeline([('pls', pls)])

    param_grid = {'pls__n_components': list(range(1, n_features + 1))}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    print(f"[INFO] Starting GridSearchCV over n_components=1..{n_features} (5-fold CV).")
    grid.fit(X_train, y_train_log)
    print("[INFO] GridSearchCV complete.\n")

    # Extract best number of components
    best_n_comp = grid.best_params_['pls__n_components']
    best_neg_mse = grid.best_score_
    best_rmse_log = np.sqrt(-best_neg_mse)
    print(f"[RESULT] Best n_components: {best_n_comp}")
    print(f"[RESULT] CV RMSE (log-target) at best n_components: {best_rmse_log:.6f}\n")

    cv_results = grid.cv_results_

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) PREPARE DIRECTORIES FOR PLOTTING
    # ─────────────────────────────────────────────────────────────────────────────
    figs_dir = os.path.join('data', 'figs')
    make_dirs_if_needed(figs_dir)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) DIAGNOSTIC PLOTS FOCUSED ON PLS
    # ─────────────────────────────────────────────────────────────────────────────

    # 4a) CV Curve: RMSE_log vs. Number of Components
    n_comps = cv_results['param_pls__n_components'].data.astype(int)  # type: ignore
    mean_mse = -cv_results['mean_test_score']            # mean MSE (log-space)
    se_mse = cv_results['std_test_score'] / np.sqrt(kf.get_n_splits())
    rmse_vals = np.sqrt(mean_mse)                        # CV RMSE (log-space)
    se_rmse = se_mse / (2 * np.sqrt(mean_mse))           # approximate se via delta

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        n_comps,
        rmse_vals,
        yerr=se_rmse,
        fmt='o-',
        capsize=3,
        label='CV $\\mathrm{RMSE}_{\\log}$'
    )
    plt.axvline(best_n_comp, color='red', linestyle='--', label=f'Best m = {best_n_comp}')
    plt.xlabel("Number of PLS Components", fontsize=12)
    plt.ylabel("CV $\\mathrm{RMSE}_{\\log}$ ↓", fontsize=12)
    plt.title("PLS CV Curve", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'pls_cv_curve.png'), dpi=150)
    plt.close()
    print(f"[PLOT] Saved PLS CV curve to data/figs/pls_cv_curve.png")

    # 4b) Coefficient Path: how each feature’s coefficient changes as m = 1..best_n_comp
    X_std = X_train  # already standardized per assumption
    coef_mat = compute_coef_path(X_std, y_train_log, best_n_comp)

    plt.figure(figsize=(7, 4))
    for j in range(coef_mat.shape[1]):
        plt.plot(
            np.arange(1, best_n_comp + 1),
            coef_mat[:, j],
            color='lightgray',
            linewidth=0.7
        )
    plt.xlabel("Number of PLS Components", fontsize=12)
    plt.ylabel("Standardized Coefficient", fontsize=12)
    plt.title("PLS Coefficient Path", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'pls_coef_path.png'), dpi=150)
    plt.close()
    print(f"[PLOT] Saved PLS coefficient path to data/figs/pls_coef_path.png")

    # 4c) Response Variance Explained by Each Component
    per_comp_frac, cum_explained = compute_response_variance(X_std, y_train_log, best_n_comp)

    plt.figure(figsize=(6, 4))
    plt.bar(
        np.arange(1, best_n_comp + 1),
        per_comp_frac,
        alpha=0.5,
        label='Per-Component'
    )
    plt.plot(
        np.arange(1, best_n_comp + 1),
        cum_explained,
        '-o',
        color='green',
        label='Cumulative'
    )
    plt.axvline(best_n_comp, color='red', linestyle='--', label=f'Best m = {best_n_comp}')
    plt.xlabel("# PLS Components", fontsize=12)
    plt.ylabel("Fraction of $\\mathrm{Var}(\\log Y)$", fontsize=12)
    plt.title("PLS Response Variance Explained", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'pls_response_variance.png'), dpi=150)
    plt.close()
    print(f"[PLOT] Saved PLS response variance explained to data/figs/pls_response_variance.png")

    # 4d) Bias–Variance–MSE Breakdown on Full Train (log-space)
    final_pls = PLSRegression(n_components=best_n_comp, scale=False)
    final_pls.fit(X_std, y_train_log)
    y_hat_log = final_pls.predict(X_std).ravel()

    mse_log = mean_squared_error(y_train_log, y_hat_log)
    var_log = np.var(y_hat_log)
    bias2_log = (np.mean(y_hat_log) - np.mean(y_train_log)) ** 2

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Squared Bias", "Variance", "Train MSE"],
        [bias2_log, var_log, mse_log],
        color=['black', 'green', 'purple']
    )
    plt.title(f"PLS (m={best_n_comp})\nBias² / Variance / Train MSE (log)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'pls_bias_variance_mse.png'), dpi=150)
    plt.close()
    print(f"[PLOT] Saved PLS bias-variance-MSE breakdown to data/figs/pls_bias_variance_mse.png")

    # 4e) First Two PLS Scores vs. log(SalePrice) (if best_n_comp ≥ 2)
    if best_n_comp >= 2:
        T_full = final_pls.x_scores_
        plt.figure(figsize=(5, 5))
        sc = plt.scatter(
            T_full[:, 0],
            T_full[:, 1],
            c=y_train_log,
            cmap='viridis',
            s=20
        )
        plt.colorbar(sc, label="log(SalePrice)")
        plt.xlabel("PLS Score 1", fontsize=12)
        plt.ylabel("PLS Score 2", fontsize=12)
        plt.title("First Two PLS Scores vs log(SalePrice)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, 'pls_score1_vs_score2.png'), dpi=150)
        plt.close()
        print(f"[PLOT] Saved PLS score scatter to data/figs/pls_score1_vs_score2.png")
    else:
        print("[SKIP] best_n_comp < 2; skipping PLS score scatter plot")

    # ─────────────────────────────────────────────────────────────────────────────
    # 5) Final PLS model is already fitted (final_pls)
    print(f"[INFO] Final PLS model with m={best_n_comp} fitted on full train set.\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # 6) LOAD TEST FEATURES AND PREDICT
    # ─────────────────────────────────────────────────────────────────────────────
    test_feats_path = os.path.join('data', 'processed', 'clean_test.csv')
    if not os.path.isfile(test_feats_path):
        raise FileNotFoundError(f"Could not find test-features file at: {test_feats_path}")

    df_test_feats = pd.read_csv(test_feats_path)
    df_test_feats = df_test_feats.fillna(0)  # Fill any NaNs with 0 (assumed no missing values)
    # Drop the “Id” column for features matching
    X_test = df_test_feats.drop(columns=['Id']).values
    n_test_samples, n_test_features = X_test.shape
    if n_test_features != n_features:
        raise ValueError(
            "❌ Column mismatch between clean_train_v2 and clean_test_v2:\n"
            f"  train features: {n_features}\n"
            f"  test features : {n_test_features}"
        )
    print(f"[INFO] Loaded test features: {n_test_samples} rows, {n_test_features} features.\n")

    y_test_log_pred = final_pls.predict(X_test).ravel()
    y_test_pred = np.exp(y_test_log_pred)
    print("[INFO] Completed predictions on test set (exponentiated from log-space).\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # 7) LOAD TEST IDS AND SAVE SUBMISSION
    # ─────────────────────────────────────────────────────────────────────────────
    id_test_path = os.path.join('data', 'processed', 'id_test.csv')
    if not os.path.isfile(id_test_path):
        raise FileNotFoundError(f"Could not find id_test.csv at: {id_test_path}")

    df_id_test = pd.read_csv(id_test_path)
    if 'Id' not in df_id_test.columns or df_id_test.shape[1] != 1:
        raise ValueError(
            "❌ id_test.csv should contain exactly one column named 'Id'.\n"
            f"   Found columns: {df_id_test.columns.tolist()}"
        )

    test_ids = df_id_test['Id'].values
    if len(test_ids) != n_test_samples:
        raise ValueError(
            "❌ Number of rows in id_test.csv does not match number of rows in clean_test_v2.csv.\n"
            f"   len(test_ids) = {len(test_ids)},  clean_test_v2 rows = {n_test_samples}"
        )

    submission = pd.DataFrame({
        'Id':        test_ids,
        'SalePrice': y_test_pred
    })

    output_dir = os.path.join('data', 'submission')
    make_dirs_if_needed(output_dir)
    submission_path = os.path.join(output_dir, 'pls_submission_final.csv')
    submission.to_csv(submission_path, index=False)
    print(f"[SUCCESS] Wrote submission to: {submission_path}\n")


if __name__ == '__main__':
    main()
