
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error

def neg_rmse(y_true, y_pred):
    """Negative RMSE for log‐target (so that higher is better)."""
    return -np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 1) Load cleaned training data
    # ──────────────────────────────────────────────────────────────────────────
    train_path = os.path.join("data", "processed", "clean_train.csv")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Could not find cleaned train file at: {train_path}")
    df_train = pd.read_csv(train_path)
    df_train = df_train.fillna(0)
    if "SalePrice" not in df_train.columns:
        raise KeyError(f"'SalePrice' column not found in {train_path}")

    # Separate features (X_train) and log‐target (y_train_log)
    ids_train = df_train["Id"].values
    y_train = df_train["SalePrice"].values
    y_train_log = np.log(y_train) # type: ignore

    X_train = df_train.drop(columns=["Id", "SalePrice"])
    n_samples, n_features = X_train.shape
    print(f"[INFO] Loaded training: {n_samples} rows × {n_features} features (no Id/SalePrice).")
    print(f"[INFO] Using log‐SalePrice as target (shape: {y_train_log.shape}).\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 2) GridSearchCV to find best n_components (1..n_features) via 5‐fold CV
    # ──────────────────────────────────────────────────────────────────────────
    pipe = Pipeline([
        ("pca", PCA()),
        ("lin", LinearRegression())
    ])

    scorer = make_scorer(neg_rmse, greater_is_better=True)

    param_grid = {
        "pca__n_components": list(range(1, n_features + 1))
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"[INFO] Starting GridSearchCV over n_components=1..{n_features} (5‐fold CV).\n")
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    gcv.fit(X_train.values, y_train_log)
    print("[INFO] GridSearchCV complete.\n")

    best_n_comp = gcv.best_params_["pca__n_components"]
    best_rmse_log = -gcv.best_score_
    print(f"[RESULT] Best #Components: {best_n_comp}")
    print(f"[RESULT] CV RMSE (log‐SalePrice): {best_rmse_log:.6f}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Re‐fit PCA(n_components=best_n_comp) + OLS on full training set
    # ──────────────────────────────────────────────────────────────────────────
    pca_final = PCA(n_components=best_n_comp)
    Z_train = pca_final.fit_transform(X_train.values)

    lin_final = LinearRegression()
    lin_final.fit(Z_train, y_train_log)
    print(f"[INFO] Re‐fitted PCA(n_components={best_n_comp}) + LinearRegression on full train.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Load cleaned test data & predict
    # ──────────────────────────────────────────────────────────────────────────
    test_path = os.path.join("data", "processed", "clean_test.csv")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Could not find cleaned test file at: {test_path}")
    df_test = pd.read_csv(test_path)
    df_test = df_test.fillna(0)
    if "Id" not in df_test.columns:
        raise KeyError(f"'Id' column not found in {test_path}")

    test_ids = df_test["Id"].values
    X_test = df_test.drop(columns=["Id"])
    if X_test.shape[1] != n_features:
        raise ValueError(
            f"Column mismatch: train had {n_features} features, test has {X_test.shape[1]}."
        )
    print(f"[INFO] Loaded test: {X_test.shape[0]} rows × {X_test.shape[1]} features.\n")

    # Project test into PCA space and predict
    Z_test = pca_final.transform(X_test.values)
    y_test_log_pred = lin_final.predict(Z_test)
    y_test_pred = np.exp(y_test_log_pred)
    print("[INFO] Completed predictions on test set (exponentiated out of log‐space).\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Build submission DataFrame and save to CSV
    # ──────────────────────────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Id":        test_ids,
        "SalePrice": y_test_pred
    })

    output_dir = os.path.join("data", "submission")
    os.makedirs(output_dir, exist_ok=True)

    submission_path = os.path.join(output_dir, "pca_submission_final.csv")
    submission.to_csv(submission_path, index=False)
    print(f"[SUCCESS] Wrote submission to: {submission_path}\n")


if __name__ == "__main__":
    main()