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
# Utilities (shared with embeddings script)
# -----------------------------
def derive_base_site(sample_id: pd.Series) -> pd.Series:
    s = sample_id.astype(str)
    s = s.str.replace(r"_T\d+$", "", regex=True)   # _T1, _T2, ...
    s = s.str.replace(r"_\d+$", "", regex=True)    # trailing _<number>
    return s


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def unwrap_estimator(fitted):
    est = fitted
    if hasattr(est, "best_estimator_"):
        est = est.best_estimator_
    if hasattr(est, "named_steps"):
        est = est.named_steps.get("reg", est)
    return est


def extract_feature_importances(fitted, feature_names: List[str]) -> pd.DataFrame:
    est = unwrap_estimator(fitted)
    importances = None

    if hasattr(est, "estimators_"):
        ests = est.estimators_
        if all(hasattr(e, "feature_importances_") for e in ests):
            imps = np.vstack([e.feature_importances_ for e in ests])
            importances = imps.mean(axis=0)
        elif all(hasattr(e, "coef_") for e in ests):
            coefs = np.vstack([np.abs(getattr(e, "coef_")).ravel() for e in ests])
            importances = coefs.mean(axis=0)
    else:
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            coef = getattr(est, "coef_")
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef).ravel()

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["feature_type"] = np.where(df["feature"] == "samp_collect_device_encoded", "collection_device", "taxonomic")
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    lat_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    lon_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    lat_r2 = r2_score(y_test[:, 0], y_pred[:, 0])
    lon_r2 = r2_score(y_test[:, 1], y_pred[:, 1])

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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6)
    mn, mx = y_test[:, 0].min(), y_test[:, 0].max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2)
    axes[0].set_xlabel("Actual Latitude")
    axes[0].set_ylabel("Predicted Latitude")
    axes[0].set_title("Latitude Prediction")
    axes[0].grid(True, alpha=0.3)

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
# Taxonomic-specific I/O
# -----------------------------
def load_asv_data(path: Path, id_cutoff: float = 97.0) -> pd.DataFrame:
    """Load ASV taxonomic and abundance data (TSV) and filter based on %ID cutoff."""
    df = pd.read_csv(path, sep="\t")

    if '%ID' in df.columns:
        mask = df['%ID'] >= id_cutoff
        taxonomy_columns = ["species", "genus", "family", "order", "class", "phylum"]
        df.loc[~mask, taxonomy_columns] = "unknown"
        print(f"[Filter] Applied %ID cutoff: {id_cutoff}. {(~mask).sum()} rows set to 'unknown'.")
    else:
        print("[Warning] '%ID' column not found in ASV data; skipping filtering step.")

    return df



def load_sample_metadata(path: Path) -> pd.DataFrame:
    """
    Expect TSV with at least: samp_name, decimalLatitude, decimalLongitude, optional samp_collect_device.
    Returns standardized columns: samp_name, decimalLatitude, decimalLongitude, samp_collect_device
    """
    df = pd.read_csv(path, sep="\t")
    df = df[df["decimalLongitude"].notna() & df["decimalLatitude"].notna()].copy()

    if "samp_collect_device" not in df.columns:
        print("[Warning] 'samp_collect_device' not found; filling with 'unknown'")
        df["samp_collect_device"] = "unknown"
    else:
        print(f"[Data] Found collection device with {df['samp_collect_device'].nunique()} unique devices")
        print(f"[Data] Collection devices: {df['samp_collect_device'].value_counts().to_dict()}")

    # Normalize key; for downstream unification we use 'samp_name'
    if "samp_name" not in df.columns and "site_id" in df.columns:
        df = df.rename(columns={"site_id": "samp_name"})

    required = {"samp_name", "decimalLatitude", "decimalLongitude", "samp_collect_device"}
    if not required.issubset(df.columns):
        raise ValueError(f"Sample metadata must contain: {required}")

    return df[["samp_name", "decimalLatitude", "decimalLongitude", "samp_collect_device"]]


def get_best_taxonomic_level(row: pd.Series) -> str:
    """
    Get the most specific non-dropped taxonomic level for an ASV.
    """
    for level in ["species", "genus", "family", "order", "class", "phylum"]:
        v = row.get(level)
        if pd.notna(v) and v not in ("dropped", "Unknown"):
            return f"{level}:{v}"
    return "unknown:unknown"


def prepare_taxonomic_features(asv_df: pd.DataFrame, sample_cols: List[str]) -> pd.DataFrame:
    """
    Convert ASV taxonomic data into sample-level taxonomic composition features.
    Returns dataframe with columns: samp_name + taxonomic features (relative abundance).
    """
    print(f"[Data] Processing {len(asv_df)} ASVs across {len(sample_cols)} samples")
    asv_df = asv_df.copy()
    asv_df["best_taxonomy"] = asv_df.apply(get_best_taxonomic_level, axis=1)
    print(f"[Data] Found {asv_df['best_taxonomy'].nunique()} unique taxonomic groups")

    site_features = {}
    for sample_col in sample_cols:
        if sample_col not in asv_df.columns:
            continue
        comp = asv_df.groupby("best_taxonomy")[sample_col].sum()
        total = comp.sum()
        if total > 0:
            comp = comp / total
        site_features[sample_col] = comp

    features_df = pd.DataFrame(site_features).T.fillna(0)  # rows = samples
    features_df.index.name = "samp_name"
    features_df = features_df.loc[:, (features_df != 0).any(axis=0)]  # drop all-zero taxa
    features_df = features_df.reset_index()
    print(f"[Data] Feature matrix: {features_df.shape[0]} samples × {features_df.shape[1]-1} taxonomic features")
    return features_df


def prepare_data(asv_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Prepare (X, y, merged_df, feature_names) for regression.
    - Taxonomic composition features + encoded collection device
    - Coords unified as decimalLatitude/decimalLongitude
    - samp_name as sample id for consistent outputs
    """
    # Sample columns (exclude controls)
    sample_cols = [c for c in asv_df.columns if c.startswith("V10_")
                   and not c.endswith(("_WC_T1", "_DI_T1", "_BC_1", "_NTC_1"))]
    print(f"[Data] Found {len(sample_cols)} sample columns (excluding controls)")

    taxo_features = prepare_taxonomic_features(asv_df, sample_cols)

    # Merge with metadata on samp_name
    merged = taxo_features.merge(metadata_df, on="samp_name", how="inner")
    if merged.empty:
        raise ValueError("No matching samp_name between taxonomic features and sample metadata")

    #merged["base_site"] = #derive_base_site(merged["samp_name"])
    # TODO
    merged['base_site']  = merged['samp_name']
    print(f"[Data] Merged {len(merged)} samples with features and coordinates")
    print(f"[Data] Unique base sites: {merged['base_site'].nunique()}")

    # Encode device
    le = LabelEncoder()
    merged["samp_collect_device_encoded"] = le.fit_transform(merged["samp_collect_device"].fillna("unknown"))
    print(f"[Data] Encoded devices: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Features = taxonomic columns + device
    exclude = {"samp_name", "decimalLatitude", "decimalLongitude", "base_site", "samp_collect_device"}
    feature_cols = [c for c in merged.columns if c not in exclude]
    taxonomic_cols = [c for c in feature_cols if c != "samp_collect_device_encoded"]
    print(f"[Data] Features: {len(taxonomic_cols)} taxonomic + 1 device = {len(feature_cols)} total")

    X = merged[feature_cols].values
    y = merged[["decimalLatitude", "decimalLongitude"]].values
    return X, y, merged, feature_cols


def build_pipeline(model_name: str, seed: int, optimize: bool) -> (Pipeline, Dict[str, List[Any]]):
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
        description="Predict lat/lon from taxonomic composition + collection device (baseline, unified outputs)"
    )
    parser.add_argument("--asv-data", required=True, type=Path,
                        help="Path to ASV taxonomic data TSV (e.g., asv_lca_with_fishbase_output.tsv)")
    parser.add_argument("--sample-metadata", required=True, type=Path,
                        help="Path to sample metadata TSV (with samp_name, decimalLatitude, decimalLongitude)")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--model", choices=["ridge", "rf", "xgb"], default="ridge",
                        help="Regression model: ridge, rf (random forest), or xgb (XGBoost)")
    parser.add_argument("--optimize-hyperparams", action="store_true",
                        help="Enable hyperparameter optimization with GridSearchCV (GroupKFold)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (default: -1)")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds (default: 5, grouped by base_site)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--id-cutoff",
        type=float,
        default=97.0,
        help="Cutoff for %ID filtering (default: 97.0)"
    )


    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print(f"[Loading] ASV data: {args.asv_data}")
    asv_df = load_asv_data(args.asv_data, args.id_cutoff)


    print(f"[Loading] Sample metadata: {args.sample_metadata}")
    metadata_df = load_sample_metadata(args.sample_metadata)

    # Prepare data
    X, y, merged_df, feature_cols = prepare_data(asv_df, metadata_df)

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

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importance
    importance_df = extract_feature_importances(model, feature_cols)
    importance_path = args.output / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"[Saved] Feature importance: {importance_path}")

    # Print results
    print("\n" + "=" * 70)
    print(f"TAXONOMIC + DEVICE BASELINE RESULTS ({model_name})")
    print("=" * 70)
    print(f"Cross-validation R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
    print(f"Test R² (overall):          {metrics['r2_overall']:.4f}")
    print(f"Test MSE (overall):         {metrics['mse_overall']:.4f}")
    print(f"Test MAE (overall):         {metrics['mae_overall']:.4f}")
    print(f"Haversine km — mean: {metrics['haversine_km_mean']:.3f}, "
          f"median: {metrics['haversine_km_median']:.3f}, p90: {metrics['haversine_km_p90']:.3f}")

    # Save detailed results (unified columns)
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

    # Summary (unified schema; add baseline tag)
    summary = {
        "model": model_name,
        "baseline_type": "taxonomic_composition_plus_collection_device",
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
    }
    if hasattr(model, "best_params_"):
        summary["best_params"] = model.best_params_

    summary_path = args.output / "regression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary metrics: {summary_path}")

    # Plot
    plot_results(y_test, metrics["predictions"], args.output)

    print(f"\n[Done] Baseline results saved to: {args.output}")


if __name__ == "__main__":
    main()

