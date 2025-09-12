import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")


def load_site_embeddings(path: Path) -> pd.DataFrame:
    """Load site embeddings from parquet or CSV file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)


def load_coordinates(path: Path) -> pd.DataFrame:
    """Load coordinate data from CSV or Excel file."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Ensure required columns exist
    required_cols = {"site_id", "latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Coordinate file must contain columns: {required_cols}")

    return df[["site_id", "latitude", "longitude"]]


def prepare_data(
    embeddings_df: pd.DataFrame, coords_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Merge embeddings with coordinates and prepare X, y matrices.

    Returns:
        X: Feature matrix (embeddings)
        y: Target matrix (lat, lon)
        merged_df: Merged dataframe for reference (includes base_site)
    """
    # Merge on site_id
    merged = embeddings_df.merge(coords_df, on="site_id", how="inner")

    if merged.empty:
        raise ValueError("No matching site_ids between embeddings and coordinates")

    # Derive base_site by stripping replicate suffix like _1.._5
    merged["base_site"] = (
        merged["site_id"].astype(str).str.replace(r"_\d+$", "", regex=True)
    )

    print(
        f"[Data] Merged {len(merged)} rows (replicates) with both embeddings and coordinates"
    )
    n_sites = merged["base_site"].nunique()
    print(f"[Data] Unique base sites: {n_sites}")

    # Extract embedding columns (those starting with 'dim_')
    dim_cols = [col for col in merged.columns if col.startswith("dim_")]
    if not dim_cols:
        raise ValueError(
            "No embedding columns found (expected columns starting with 'dim_')"
        )

    X = merged[dim_cols].values
    y = merged[["latitude", "longitude"]].values

    print(
        f"[Data] Features: {X.shape[1]} dimensions, Targets: {len(merged)} replicate rows across {n_sites} base sites"
    )

    return X, y, merged


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate model performance and return metrics."""
    y_pred = model.predict(X_test)

    # Overall metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Per-target metrics
    lat_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    lon_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    lat_r2 = r2_score(y_test[:, 0], y_pred[:, 0])
    lon_r2 = r2_score(y_test[:, 1], y_pred[:, 1])

    return {
        "mse_overall": mse,
        "mae_overall": mae,
        "r2_overall": r2,
        "lat_mse": lat_mse,
        "lon_mse": lon_mse,
        "lat_r2": lat_r2,
        "lon_r2": lon_r2,
        "predictions": y_pred,
    }


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """Create scatter plots of predicted vs actual coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Latitude plot
    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6)
    axes[0].plot(
        [y_test[:, 0].min(), y_test[:, 0].max()],
        [y_test[:, 0].min(), y_test[:, 0].max()],
        "r--",
        lw=2,
    )
    axes[0].set_xlabel("Actual Latitude")
    axes[0].set_ylabel("Predicted Latitude")
    axes[0].set_title("Latitude Prediction")
    axes[0].grid(True, alpha=0.3)

    # Longitude plot
    axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6)
    axes[1].plot(
        [y_test[:, 1].min(), y_test[:, 1].max()],
        [y_test[:, 1].min(), y_test[:, 1].max()],
        "r--",
        lw=2,
    )
    axes[1].set_xlabel("Actual Longitude")
    axes[1].set_ylabel("Predicted Longitude")
    axes[1].set_title("Longitude Prediction")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "coordinate_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Prediction plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict lat/lon from site embeddings with grouped CV by base_site"
    )
    parser.add_argument(
        "--site-embeddings",
        required=True,
        type=Path,
        help="Path to site embeddings parquet/CSV file",
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        type=Path,
        help="Path to coordinates CSV/Excel file (site_id, latitude, longitude)",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output directory for results"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "rf", "xgb"],
        default="ridge",
        help="Regression model: ridge, rf (random forest), or xgb (XGBoost)",
    )
    parser.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        help="Enable hyperparameter optimization with GridSearchCV (uses GroupKFold)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for GridSearchCV (default: -1, use all CPUs)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5, grouped by base_site)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    print(f"[Loading] Site embeddings: {args.site_embeddings}")
    embeddings_df = load_site_embeddings(args.site_embeddings)

    print(f"[Loading] Coordinates: {args.coordinates}")
    coords_df = load_coordinates(args.coordinates)

    # Prepare data
    X, y, merged_df = prepare_data(embeddings_df, coords_df)

    # Build groups from base_site
    groups_all = merged_df["base_site"].values
    indices = np.arange(len(X))

    # Grouped train/test split (keeps replicates from same base_site together)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups_all[train_idx]
    groups_test = groups_all[test_idx]
    idx_train, idx_test = train_idx, test_idx

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"[Split] Training replicates: {len(X_train)} across {pd.unique(groups_train).size} base sites"
    )
    print(
        f"[Split] Testing replicates:  {len(X_test)} across {pd.unique(groups_test).size} base sites"
    )

    # Choose and train model with optional hyperparameter optimization
    gkf = GroupKFold(n_splits=args.cv_folds)

    if args.model == "ridge":
        base_model = Ridge()
        if args.optimize_hyperparams:
            print("[Optimizing] Ridge hyperparameters with GroupKFold...")
            wrapped = MultiOutputRegressor(base_model)
            param_grid = {"estimator__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
            model = GridSearchCV(
                wrapped,
                param_grid,
                cv=gkf,  # grouped CV
                scoring="r2",
                n_jobs=args.n_jobs,
            )
            model.fit(X_train_scaled, y_train, groups=groups_train)
            print(f"[Best params] {model.best_params_}")
            model_name = (
                f"Ridge Regression (alpha={model.best_estimator_.estimator.alpha})"
            )
        else:
            model = MultiOutputRegressor(Ridge(alpha=1.0))
            model.fit(X_train_scaled, y_train)
            model_name = "Ridge Regression"

    elif args.model == "rf":
        if args.optimize_hyperparams:
            print("[Optimizing] Random Forest hyperparameters with GroupKFold...")
            base_model = RandomForestRegressor(random_state=args.seed)
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            }
            model = GridSearchCV(
                base_model,
                param_grid,
                cv=gkf,  # grouped CV
                scoring="r2",
                n_jobs=args.n_jobs,
            )
            model.fit(X_train_scaled, y_train, groups=groups_train)
            print(f"[Best params] {model.best_params_}")
            model_name = "Random Forest (optimized)"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=args.seed)
            model.fit(X_train_scaled, y_train)
            model_name = "Random Forest"

    elif args.model == "xgb":
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        if args.optimize_hyperparams:
            print("[Optimizing] XGBoost hyperparameters with GroupKFold...")
            wrapped = MultiOutputRegressor(xgb.XGBRegressor(random_state=args.seed))
            param_grid = {
                "estimator__n_estimators": [100, 200, 300],
                "estimator__max_depth": [3, 6, 9],
                "estimator__learning_rate": [0.01, 0.1, 0.2],
                "estimator__subsample": [0.8, 1.0],
            }
            model = GridSearchCV(
                wrapped,
                param_grid,
                cv=gkf,  # grouped CV
                scoring="r2",
                n_jobs=args.n_jobs,
            )
            model.fit(X_train_scaled, y_train, groups=groups_train)
            print(f"[Best params] {model.best_params_}")
            model_name = "XGBoost (optimized)"
        else:
            model = MultiOutputRegressor(xgb.XGBRegressor(random_state=args.seed))
            model.fit(X_train_scaled, y_train)
            model_name = "XGBoost"

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print(f"[Training] {model_name} completed.")

    # Cross-validation (skip if already done during hyperparameter optimization)
    if args.optimize_hyperparams:
        # Use best CV score from GridSearch (already grouped)
        cv_scores = np.array([model.best_score_])
        print(f"[CV] Best cross-validation R² (GroupKFold): {model.best_score_:.4f}")
    else:
        print(f"[CV] {args.cv_folds}-fold GroupKFold cross-validation...")
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=gkf,  # grouped CV
            scoring="r2",
            groups=groups_train,  # critical to respect grouping
        )

    # Evaluate on test set
    metrics = evaluate_model(model, X_test_scaled, y_test)

    # Print results
    print("\n" + "=" * 50)
    print(f"REGRESSION RESULTS ({model_name})")
    print("=" * 50)
    print(
        f"Cross-validation R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})"
    )
    print(f"Test R² (overall):          {metrics['r2_overall']:.4f}")
    print(f"Test MSE (overall):         {metrics['mse_overall']:.4f}")
    print(f"Test MAE (overall):         {metrics['mae_overall']:.4f}")
    print()
    print("Per-coordinate performance:")
    print(f"  Latitude  - R²: {metrics['lat_r2']:.4f}, MSE: {metrics['lat_mse']:.4f}")
    print(f"  Longitude - R²: {metrics['lon_r2']:.4f}, MSE: {metrics['lon_mse']:.4f}")

    # Save detailed results
    test_site_ids = merged_df.iloc[idx_test]["site_id"].values
    test_base_sites = merged_df.iloc[idx_test]["base_site"].values
    results_df = pd.DataFrame(
        {
            "site_id": test_site_ids,
            "base_site": test_base_sites,
            "actual_lat": y_test[:, 0],
            "actual_lon": y_test[:, 1],
            "pred_lat": metrics["predictions"][:, 0],
            "pred_lon": metrics["predictions"][:, 1],
            "lat_error": np.abs(y_test[:, 0] - metrics["predictions"][:, 0]),
            "lon_error": np.abs(y_test[:, 1] - metrics["predictions"][:, 1]),
        }
    )

    results_path = args.output / "regression_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[Saved] Detailed results: {results_path}")

    # Save summary metrics
    summary = {
        "model": model_name,
        "grouped_splitting": True,
        "cv_strategy": "GroupKFold",
        "cv_folds": args.cv_folds,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2_overall": float(metrics["r2_overall"]),
        "test_mse_overall": float(metrics["mse_overall"]),
        "test_mae_overall": float(metrics["mae_overall"]),
        "test_r2_lat": float(metrics["lat_r2"]),
        "test_r2_lon": float(metrics["lon_r2"]),
        "n_features": X.shape[1],
        "n_train_replicates": len(X_train),
        "n_test_replicates": len(X_test),
        "n_train_base_sites": int(pd.unique(groups_train).size),
        "n_test_base_sites": int(pd.unique(groups_test).size),
        "n_total_base_sites": int(pd.unique(groups_all).size),
        "hyperparameter_optimization": bool(args.optimize_hyperparams),
    }

    if args.optimize_hyperparams:
        summary["best_params"] = model.best_params_

    summary_path = args.output / "regression_summary.json"
    import json

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary metrics: {summary_path}")

    # Create plots
    plot_results(y_test, metrics["predictions"], args.output)

    print(f"\n[Done] All results saved to: {args.output}")


if __name__ == "__main__":
    main()
