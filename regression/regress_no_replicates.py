#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# -----------------------------
# Utilities (shared logic)
# -----------------------------
def derive_base_site(sample_id: pd.Series) -> pd.Series:
    """
    Derive base site by stripping phase suffix like _T1/_T2 etc., and trailing replicate numbers like _1, _2.
    """
    s = sample_id.astype(str)
    s = s.str.replace(r"_T\d+$", "", regex=True)   # remove _T1, _T2, ...
    s = s.str.replace(r"_\d+$", "", regex=True)    # remove trailing _<number> replicates
    return s


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in kilometers.
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def unwrap_estimator(fitted):
    """
    Unwrap GridSearchCV -> Pipeline -> final regressor step ('reg').
    """
    est = fitted
    if hasattr(est, "best_estimator_"):
        est = est.best_estimator_
    if hasattr(est, "named_steps"):  # Pipeline
        est = est.named_steps.get("reg", est)
    return est


def extract_feature_importances(fitted, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importances/coefficients from a fitted estimator or pipeline.
    Supports:
      - RandomForestRegressor (feature_importances_)
      - XGBRegressor via MultiOutputRegressor (aggregate mean across targets)
      - Ridge via MultiOutputRegressor (aggregate mean |coef_| across targets)
    Returns a dataframe [feature, importance], sorted desc.
    """
    est = unwrap_estimator(fitted)

    importances = None

    # MultiOutputRegressor case
    if hasattr(est, "estimators_"):
        ests = est.estimators_
        # Trees
        if all(hasattr(e, "feature_importances_") for e in ests):
            imps = np.vstack([e.feature_importances_ for e in ests])
            importances = imps.mean(axis=0)
        # Linear (Ridge)
        elif all(hasattr(e, "coef_") for e in ests):
            coefs = np.vstack([np.abs(getattr(e, "coef_")).ravel() for e in ests])
            importances = coefs.mean(axis=0)
    else:
        # Single estimator path
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            coef = getattr(est, "coef_")
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef).ravel()

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["feature_type"] = np.where(df["feature"] == "samp_collect_device_encoded", "collection_device", "embedding")
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate model performance and return metrics, including haversine distance stats.
    """
    y_pred = model.predict(X_test)

    # Vector metrics (multi-output)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Per-dimension
    lat_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    lon_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    lat_r2 = r2_score(y_test[:, 0], y_pred[:, 0])
    lon_r2 = r2_score(y_test[:, 1], y_pred[:, 1])

    # Haversine
    km_errors = haversine_km(y_test[:, 0], y_test[:, 1], y_pred[:, 0], y_pred[:, 1])
    km_mean = float(np.mean(km_errors))
    km_median = float(np.median(km_errors))
    km_p90 = float(np.percentile(km_errors, 90))

    return {
        "mse_overall": float(mse),
        "mae_overall": float(mae),
        "r2_overall": float(r2),
        "lat_mse": float(lat_mse),
        "lon_mse": float(lon_mse),
        "lat_r2": float(lat_r2),
        "lon_r2": float(lon_r2),
        "predictions": y_pred,
        "haversine_km": km_errors,
        "haversine_km_mean": km_mean,
        "haversine_km_median": km_median,
        "haversine_km_p90": km_p90,
    }


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """Create scatter plots of predicted vs actual coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Latitude
    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6)
    mn, mx = y_test[:, 0].min(), y_test[:, 0].max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2)
    axes[0].set_xlabel("Actual Latitude")
    axes[0].set_ylabel("Predicted Latitude")
    axes[0].set_title("Latitude Prediction")
    axes[0].grid(True, alpha=0.3)

    # Longitude
    axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6)
    mn, mx = y_test[:, 1].min(), y_test[:, 1].max()
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=2)
    axes[1].set_xlabel("Actual Longitude")
    axes[1].set_ylabel("Predicted Longitude")
    axes[1].set_title("Longitude Prediction")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "coordinate_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Prediction plot: {plot_path}")


# -----------------------------
# Embeddings-specific I/O & fusion
# -----------------------------
def load_site_embeddings(path: Path) -> pd.DataFrame:
    """Load site embeddings from parquet or CSV file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the ID column is named 'samp_name'."""
    if "samp_name" in df.columns:
        return df
    if "site_id" in df.columns:
        return df.rename(columns={"site_id": "samp_name"})
    raise ValueError("Embeddings file must contain 'samp_name' or 'site_id' column")


def numeric_sort_dim_cols(cols: List[str]) -> List[str]:
    """Sort dim_* columns numerically: dim_0, dim_1, ..., dim_10 (not lexicographic)."""
    def key(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return float("inf")
    return sorted(cols, key=key)


def fuse_embeddings_concat(df12: pd.DataFrame, df16: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate 12S and 16S embedding vectors for samples present in both.
    - Align on 'samp_name'
    - Order each assay's dim_* numerically before concatenation
    - Reindex fused dims from dim_0..dim_(n12+n16-1)
    """
    df12 = normalize_id_column(df12).copy()
    df16 = normalize_id_column(df16).copy()

    dims12 = numeric_sort_dim_cols([c for c in df12.columns if c.startswith("dim_")])
    dims16 = numeric_sort_dim_cols([c for c in df16.columns if c.startswith("dim_")])
    if not dims12 or not dims16:
        raise ValueError("Both 12S and 16S embeddings must have 'dim_*' columns")

    s12 = df12[["samp_name"] + dims12].set_index("samp_name")
    s16 = df16[["samp_name"] + dims16].set_index("samp_name")
    common = s12.index.intersection(s16.index)

    if len(common) == 0:
        raise ValueError("No overlapping samp_name between 12S and 16S embeddings for fusion")

    # Concatenate vectors
    fused_records = []
    for sid in common:
        v12 = s12.loc[sid].to_numpy(dtype=np.float32)
        v16 = s16.loc[sid].to_numpy(dtype=np.float32)
        vec = np.concatenate([v12, v16], axis=0)
        rec = {"samp_name": sid}
        for i, val in enumerate(vec):
            rec[f"dim_{i}"] = float(val)
        fused_records.append(rec)

    fused_df = pd.DataFrame(fused_records)
    return fused_df


def load_coordinates(path: Path) -> pd.DataFrame:
    """Load coordinate data from CSV, TSV or Excel."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_excel(path)

    required_cols = {"samp_name", "decimalLatitude", "decimalLongitude"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Coordinate file must contain columns: {required_cols}")

    if "samp_collect_device" in df.columns:
        print(f"[Data] Found collection device with {df['samp_collect_device'].nunique()} unique devices")
        print(f"[Data] Collection devices: {df['samp_collect_device'].value_counts().to_dict()}")
    else:
        print("[Warning] 'samp_collect_device' not found; filling with 'unknown'")
        df["samp_collect_device"] = "unknown"

    return df[["samp_name", "decimalLatitude", "decimalLongitude", "samp_collect_device"]]


def prepare_data(
    embeddings_df: pd.DataFrame, coords_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Merge embeddings with coordinates and prepare X, y matrices and feature names.
    """
    embeddings_df = normalize_id_column(embeddings_df)

    merged = embeddings_df.merge(coords_df, on="samp_name", how="inner")

    if merged.empty:
        raise ValueError("No matching samp_name between embeddings and coordinates")

    merged = merged.dropna(subset=["decimalLongitude", "decimalLatitude"]).copy()

    # Group id
    merged["base_site"] = merged['samp_name']#derive_base_site(merged["samp_name"])
    # TEMPORARY TODO : just playing with not merging by site

    print(f"[Data] Merged {len(merged)} rows with embeddings and coordinates")
    print(f"[Data] Unique base sites: {merged['base_site'].nunique()}")

    # Label-encode collection device
    device_encoder = LabelEncoder()
    merged["samp_collect_device_encoded"] = device_encoder.fit_transform(
        merged["samp_collect_device"].fillna("unknown")
    )

    # Feature columns: all 'dim_*' + device
    dim_cols = numeric_sort_dim_cols([c for c in merged.columns if c.startswith("dim_")])
    if not dim_cols:
        raise ValueError("No embedding columns found (expected columns starting with 'dim_')")
    feature_cols = dim_cols + ["samp_collect_device_encoded"]

    X = merged[feature_cols].values
    y = merged[["decimalLatitude", "decimalLongitude"]].values
    return X, y, merged, feature_cols


def build_pipeline(model_name: str, seed: int, optimize: bool) -> (Pipeline, Dict[str, List[Any]]):
    """
    Build a Pipeline and (optionally) a param grid.
    - Ridge: StandardScaler + MultiOutput(Ridge)
    - RF/XGB: just the estimator (trees don't need scaling)
    """
    if model_name == "ridge":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(Ridge()))
        ])
        param_grid = {"reg__estimator__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]} if optimize else {}
        return pipe, param_grid

    elif model_name == "rf":
        reg = RandomForestRegressor(n_estimators=100, random_state=seed)
        pipe = Pipeline([("reg", reg)])
        param_grid = {
            "reg__n_estimators": [50, 100, 200],
            "reg__max_depth": [None, 10, 20],
            "reg__min_samples_split": [2, 5, 10],
        } if optimize else {}
        return pipe, param_grid

    elif model_name == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        reg = MultiOutputRegressor(xgb.XGBRegressor(random_state=seed))
        pipe = Pipeline([("reg", reg)])
        param_grid = {
            "reg__estimator__n_estimators": [100, 200, 300],
            "reg__estimator__max_depth": [3, 6, 9],
            "reg__estimator__learning_rate": [0.01, 0.1, 0.2],
            "reg__estimator__subsample": [0.8, 1.0],
        } if optimize else {}
        return pipe, param_grid

    else:
        raise ValueError(f"Unknown model type: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict lat/lon from site embeddings with grouped CV by base_site (canonical, supports 12S+16S fusion)"
    )
    parser.add_argument("--site-embeddings-12s", required=True, type=Path,
                        help="Path to primary assay site embeddings parquet/CSV (e.g., 12S)")
    parser.add_argument("--site-embeddings-16s", type=Path, default=None,
                        help="Optional path to 16S site embeddings parquet/CSV for fusion")
    parser.add_argument("--fuse", choices=["none", "concat"], default="none",
                        help="How to fuse multi-assay embeddings (default: none). 'concat' requires 16S.")
    parser.add_argument("--coordinates", required=True, type=Path,
                        help="Path to coordinates file (CSV/TSV/Excel) with samp_name, decimalLatitude, decimalLongitude")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--model", choices=["ridge", "rf", "xgb"], default="ridge",
                        help="Regression model: ridge, rf (random forest), or xgb (XGBoost)")
    parser.add_argument("--optimize-hyperparams", action="store_true",
                        help="Enable hyperparameter optimization with GridSearchCV (GroupKFold)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (default: -1)")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds (default: 5, grouped by base_site)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # Load embeddings (12S primary)
    print(f"[Loading] Primary embeddings: {args.site_embeddings_12s}")
    emb_primary = load_site_embeddings(args.site_embeddings_12s)

    # Optional fusion with 16S
    if args.fuse == "concat":
        if args.site_embeddings_16s is None:
            raise ValueError("--fuse=concat requires --site-embeddings-16s")
        print(f"[Loading] 16S embeddings: {args.site_embeddings_16s}")
        emb_16s = load_site_embeddings(args.site_embeddings_16s)

        print("[Fusion] Concatenating 12S + 16S embeddings on common samp_name...")
        embeddings_df = fuse_embeddings_concat(emb_primary, emb_16s)

    else:
        embeddings_df = normalize_id_column(emb_primary)

    # Coordinates
    print(f"[Loading] Coordinates: {args.coordinates}")
    coords_df = load_coordinates(args.coordinates)

    # Prepare data
    X, y, merged_df, feature_cols = prepare_data(embeddings_df, coords_df)

    # Groups
    groups_all = merged_df["base_site"].values

    # Grouped train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups_all[train_idx]
    groups_test = groups_all[test_idx]

    print(f"[Split] Training: {len(X_train)} samples across {pd.unique(groups_train).size} base sites")
    print(f"[Split] Testing:  {len(X_test)} samples across {pd.unique(groups_test).size} base sites")

    # Pipeline + (optional) hyperparameter optimization
    pipe, param_grid = build_pipeline(args.model, args.seed, args.optimize_hyperparams)
    gkf = GroupKFold(n_splits=args.cv_folds)

    if args.optimize_hyperparams and param_grid:
        print("[Optimizing] Hyperparameters with GroupKFold...")
        model = GridSearchCV(pipe, param_grid=param_grid, cv=gkf, scoring="r2", n_jobs=args.n_jobs)
        # Always pass groups for GroupKFold
        model.fit(X_train, y_train, groups=groups_train)
        print(f"[Best params] {model.best_params_}")
        model_name = f"{args.model.upper()} (optimized)"
        cv_scores = np.array([model.best_score_])
        print(f"[CV] Best cross-validation R² (GroupKFold): {model.best_score_:.4f}")
    else:
        model = pipe.fit(X_train, y_train)
        model_name = args.model.upper()
        print(f"[CV] {args.cv_folds}-fold GroupKFold cross-validation...")
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=gkf, scoring="r2", groups=groups_train, n_jobs=args.n_jobs
        )

    # Evaluate on test set (pipeline handles any scaling)
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importance
    importance_df = extract_feature_importances(model, feature_cols)
    importance_path = args.output / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"[Saved] Feature importance: {importance_path}")

    # Print results
    print("\n" + "=" * 50)
    print(f"REGRESSION RESULTS ({model_name})")
    print("=" * 50)
    print(f"Cross-validation R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
    print(f"Test R² (overall):          {metrics['r2_overall']:.4f}")
    print(f"Test MSE (overall):         {metrics['mse_overall']:.4f}")
    print(f"Test MAE (overall):         {metrics['mae_overall']:.4f}")
    print(f"Haversine km — mean: {metrics['haversine_km_mean']:.3f}, "
          f"median: {metrics['haversine_km_median']:.3f}, p90: {metrics['haversine_km_p90']:.3f}")

    # Save detailed results
    test_ids = merged_df.iloc[test_idx]["samp_name"].values
    test_base = merged_df.iloc[test_idx]["base_site"].values

    results_df = pd.DataFrame({
        "sample_id": test_ids,
        "base_site": test_base,
        "actual_lat": y_test[:, 0],
        "actual_lon": y_test[:, 1],
        "pred_lat": metrics["predictions"][:, 0],
        "pred_lon": metrics["predictions"][:, 1],
        "lat_error_deg": np.abs(y_test[:, 0] - metrics["predictions"][:, 0]),
        "lon_error_deg": np.abs(y_test[:, 1] - metrics["predictions"][:, 1]),
        "haversine_km": metrics["haversine_km"],
    })
    results_path = args.output / "regression_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[Saved] Detailed results: {results_path}")

    # Summary
    summary = {
        "model": model_name,
        "grouped_splitting": True,
        "cv_strategy": "GroupKFold",
        "cv_folds": args.cv_folds,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2_overall": metrics["r2_overall"],
        "test_mse_overall": metrics["mse_overall"],
        "test_mae_overall": metrics["mae_overall"],
        "test_haversine_km_mean": metrics["haversine_km_mean"],
        "test_haversine_km_median": metrics["haversine_km_median"],
        "test_haversine_km_p90": metrics["haversine_km_p90"],
        "n_features": int(X.shape[1]),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_train_base_sites": int(pd.unique(groups_train).size),
        "n_test_base_sites": int(pd.unique(groups_test).size),
        "n_total_base_sites": int(pd.unique(groups_all).size),
        "hyperparameter_optimization": bool(args.optimize_hyperparams),
        "fusion": args.fuse,
        "has_16s": bool(args.site_embeddings_16s is not None),
    }
    if hasattr(model, "best_params_"):
        summary["best_params"] = model.best_params_

    summary_path = args.output / "regression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary metrics: {summary_path}")

    # Plot
    plot_results(y_test, metrics["predictions"], args.output)

    print(f"\n[Done] All results saved to: {args.output}")


if __name__ == "__main__":
    main()

