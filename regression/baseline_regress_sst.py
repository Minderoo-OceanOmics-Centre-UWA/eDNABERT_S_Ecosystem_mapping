#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict SST (°C) from taxonomic composition + collection device (baseline, unified outputs)

Usage example:
--------------
python regress_taxonomic_baseline_sst.py \
  --asv-data asv_lca_with_fishbase_output.tsv \
  --sample-metadata samplemetadata.tsv \
  --sst-table sst_out.csv \
  --sst-id-col samp_name \            # set this to the ID column in your SST table (e.g., 'samp_name' or 'sample_id')
  --sst-col sst \                     # name of the SST column in the SST table
  --output out_taxo_sst \
  --model ridge --cv-folds 5 --seed 42
"""

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
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# -----------------------------
# Utilities
# -----------------------------
def derive_base_site(sample_id: pd.Series) -> pd.Series:
    """Strip trailing _T<digit> and trailing _<replicate_num> to form grouping key."""
    s = sample_id.astype(str)
    s = s.str.replace(r"_T\d+$", "", regex=True)
    s = s.str.replace(r"_\d+$", "", regex=True)
    return s


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
      - XGBRegressor (feature_importances_)
      - Ridge (coef_)
    Returns a dataframe [feature, importance], sorted desc.
    """
    est = unwrap_estimator(fitted)
    importances = None

    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = getattr(est, "coef_")
        # Single-target regression => 1D vector
        importances = np.abs(coef).ravel()
    else:
        # Unknown model type
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["feature_type"] = np.where(df["feature"] == "samp_collect_device_encoded", "collection_device", "taxonomic")
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate model performance for single-target regression (SST in °C).
    """
    y_pred = model.predict(X_test).reshape(-1)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    bias = float(np.mean(y_pred - y_test))
    rmse = float(np.sqrt(mse))

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "bias": bias,
        "predictions": y_pred,
    }


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """Scatter plot: Actual vs Predicted SST."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    mn, mx = float(np.min(y_test)), float(np.max(y_test))
    pad = (mx - mn) * 0.05 if mx > mn else 1.0
    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "r--", lw=2)
    plt.xlabel("Actual SST (°C)")
    plt.ylabel("Predicted SST (°C)")
    plt.title("SST Prediction (Taxonomic + Device)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "coordinate_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Prediction plot: {plot_path}")


# -----------------------------
# I/O helpers
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
    Expect TSV with at least: samp_name, optional samp_collect_device.
    (decimalLatitude/decimalLongitude optional here since SST is our target.)
    """
    df = pd.read_csv(path, sep="\t")

    # Device
    if "samp_collect_device" not in df.columns:
        print("[Warning] 'samp_collect_device' not found in metadata; filling with 'unknown'")
        df["samp_collect_device"] = "unknown"
    else:
        print(f"[Data] Collection devices: {df['samp_collect_device'].value_counts().to_dict()}")


    if "samp_name" not in df.columns:
        raise ValueError("Sample metadata must contain 'samp_name' or 'site_id'.")

    # Keep only relevant columns
    keep = ["samp_name", "samp_collect_device"]
    # (lat/lon can be present but not required for SST)
    for opt in ["decimalLatitude", "decimalLongitude"]:
        if opt in df.columns:
            keep.append(opt)

    return df[keep]


def load_sst_table(path: Path, sst_id_col: str, sst_col: str) -> pd.DataFrame:
    """
    Load SST table CSV produced by your downloader.
    Must contain:
      - an ID column (passed via --sst-id-col), e.g. 'samp_name' or 'sample_id'
      - an SST column (passed via --sst-col), default 'sst'
    """
    df = pd.read_csv(path)
    if sst_id_col not in df.columns:
        raise ValueError(f"SST table missing ID column '{sst_id_col}'")
    if sst_col not in df.columns:
        raise ValueError(f"SST table missing SST column '{sst_col}'")

    # Normalize ID column to 'samp_name' for merging
    if sst_id_col != "samp_name":
        df = df.rename(columns={sst_id_col: "samp_name"})

    # Keep only required columns to avoid duplicate names later
    return df[["samp_name", sst_col]].rename(columns={sst_col: "sst"})


def get_best_taxonomic_level(row: pd.Series) -> str:
    """Pick most specific taxonomy available (species → phylum), excluding 'dropped'/'Unknown'."""
    for level in ["species", "genus", "family", "order", "class", "phylum"]:
        v = row.get(level)
        if pd.notna(v) and v not in ("dropped", "Unknown"):
            return f"{level}:{v}"
    return "unknown:unknown"


def prepare_taxonomic_features(asv_df: pd.DataFrame, sample_cols: List[str]) -> pd.DataFrame:
    """
    Convert ASV taxonomic data into sample-level taxonomic composition features (relative abundances).
    Returns dataframe with columns: samp_name + taxonomic features.
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
    # drop all-zero taxa
    features_df = features_df.loc[:, (features_df != 0).any(axis=0)]
    features_df = features_df.reset_index()
    print(f"[Data] Feature matrix: {features_df.shape[0]} samples × {features_df.shape[1]-1} taxonomic features")
    return features_df


def prepare_data(
    asv_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sst_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Prepare (X, y, merged_df, feature_names) for SST regression.
    - Taxonomic features + encoded device
    - Target = 'sst' from sst_df
    """
    # Sample columns (exclude controls)
    sample_cols = [c for c in asv_df.columns if c.startswith("V10_")
                   and not c.endswith(("_WC_T1", "_DI_T1", "_BC_1", "_NTC_1"))]
    print(f"[Data] Found {len(sample_cols)} sample columns (excluding controls)")

    taxo_features = prepare_taxonomic_features(asv_df, sample_cols)

    # Merge taxonomic features + metadata (for device) + SST
    merged = taxo_features.merge(metadata_df, on="samp_name", how="inner")
    merged = merged.merge(sst_df, on="samp_name", how="inner")

    # remove low temp sites (<0)
    merged['sst'] = merged['sst'].values.astype(float)
    merged = merged[merged['sst'] > 0]



    if merged.empty:
        raise ValueError("No overlapping samp_name across taxonomic features, metadata, and SST table")

    # Grouping key
    merged["base_site"] = derive_base_site(merged["samp_name"])
    print(f"[Data] Merged {len(merged)} samples with features, device and SST")
    print(f"[Data] Unique base sites: {merged['base_site'].nunique()}")

    # Encode device
    le = LabelEncoder()
    merged["samp_collect_device_encoded"] = le.fit_transform(
        merged["samp_collect_device"].fillna("unknown")
    )
    print(f"[Data] Encoded devices: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Build feature set: all taxonomic columns + device
    exclude = {"samp_name", "base_site", "samp_collect_device", "sst"}
    # Optional metadata cols we shouldn't include as features if present
    exclude.update({"decimalLatitude", "decimalLongitude"})

    feature_cols = [c for c in merged.columns if c not in exclude]
    taxonomic_cols = [c for c in feature_cols if c != "samp_collect_device_encoded"]

    X = merged[feature_cols].values
    y = merged["sst"].values.astype(float)

    print(f"[Data] Features: {len(taxonomic_cols)} taxonomic + 1 device = {len(feature_cols)} total")
    return X, y, merged, feature_cols


def build_pipeline(model_name: str, seed: int, optimize: bool) -> (Pipeline, Dict[str, List[Any]]):
    """
    Build a Pipeline and (optionally) a param grid.
    - Ridge: StandardScaler + Ridge
    - RF/XGB: just the estimator (trees don't need scaling)
    """
    if model_name == "ridge":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge())
        ])
        param_grid = {"reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]} if optimize else {}
        return pipe, param_grid

    elif model_name == "rf":
        reg = RandomForestRegressor(n_estimators=200, random_state=seed)
        pipe = Pipeline([("reg", reg)])
        param_grid = {
            "reg__n_estimators": [100, 200, 400],
            "reg__max_depth": [None, 10, 20],
            "reg__min_samples_split": [2, 5, 10],
        } if optimize else {}
        return pipe, param_grid

    elif model_name == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        reg = xgb.XGBRegressor(random_state=seed)
        pipe = Pipeline([("reg", reg)])
        param_grid = {
            "reg__n_estimators": [200, 400, 600],
            "reg__max_depth": [3, 6, 9],
            "reg__learning_rate": [0.01, 0.05, 0.1],
            "reg__subsample": [0.8, 1.0],
            "reg__colsample_bytree": [0.7, 1.0],
        } if optimize else {}
        return pipe, param_grid

    else:
        raise ValueError(f"Unknown model type: {model_name}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict SST (°C) from taxonomic composition + collection device (baseline, unified outputs)"
    )
    parser.add_argument("--asv-data", required=True, type=Path,
                        help="Path to ASV taxonomic data TSV (e.g., asv_lca_with_fishbase_output.tsv)")
    parser.add_argument("--sample-metadata", required=True, type=Path,
                        help="Path to sample metadata TSV (with samp_name and optional samp_collect_device)")
    parser.add_argument("--sst-table", required=True, type=Path,
                        help="Path to CSV containing sample IDs and an 'sst' column (from your OISST downloader)")
    parser.add_argument("--sst-id-col", default="samp_name",
                        help="Column in the SST table to join on (e.g., 'samp_name' or 'sample_id'). Default: sample_id")
    parser.add_argument("--sst-col", default="sst",
                        help="Name of the SST column in the SST table. Default: sst")

    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split (default: 0.2)")
    parser.add_argument("--model", choices=["ridge", "rf", "xgb"], default="ridge",
                        help="Regression model: ridge, rf (random forest), or xgb (XGBoost)")
    parser.add_argument("--optimize-hyperparams", action="store_true",
                        help="Enable hyperparameter optimization with GridSearchCV (GroupKFold)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for GridSearchCV / cross_val_score (default: -1)")
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

    print(f"[Loading] SST table: {args.sst_table}")
    sst_df = load_sst_table(args.sst_table, args.sst_id_col, args.sst_col)

    # Prepare data
    X, y, merged_df, feature_cols = prepare_data(asv_df, metadata_df, sst_df)

    # Grouped train/test split
    groups_all = merged_df["base_site"].values
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
    print(f"TAXONOMIC + DEVICE → SST RESULTS ({model_name})")
    print("=" * 70)
    print(f"Cross-validation R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
    print(f"Test R²:    {metrics['r2']:.4f}")
    print(f"Test RMSE:  {metrics['rmse']:.4f} °C")
    print(f"Test MAE:   {metrics['mae']:.4f} °C")
    print(f"Bias (Δ):   {metrics['bias']:.4f} °C")

    # Save detailed results
    test_ids = merged_df.iloc[test_idx]["samp_name"].values
    test_base = merged_df.iloc[test_idx]["base_site"].values

    results_df = pd.DataFrame({
        "sample_id": test_ids,
        "base_site": test_base,
        "actual_sst": y_test,
        "pred_sst": metrics["predictions"],
        "abs_error_c": np.abs(y_test - metrics["predictions"]),
    })
    results_path = args.output / "regression_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[Saved] Detailed results: {results_path}")

    # Summary (unified schema; includes model + CV info)
    summary = {
        "model": model_name,
        "baseline_type": "taxonomic_composition_plus_collection_device",
        "target": "sst_celsius",
        "grouped_splitting": True,
        "cv_strategy": "GroupKFold",
        "cv_folds": args.cv_folds,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2": metrics["r2"],
        "test_rmse_c": metrics["rmse"],
        "test_mae_c": metrics["mae"],
        "test_bias_c": metrics["bias"],
        "n_features": int(X.shape[1]),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_train_base_sites": int(pd.unique(groups_train).size),
        "n_test_base_sites": int(pd.unique(groups_test).size),
        "n_total_base_sites": int(pd.unique(groups_all).size),
        "hyperparameter_optimization": bool(args.optimize_hyperparams),
        "sst_table": str(args.sst_table),
        "sst_col": str(args.sst_col),
        "sst_id_col": str(args.sst_id_col),
    }
    if hasattr(model, "best_params_"):
        summary["best_params"] = model.best_params_

    summary_path = args.output / "regression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary metrics: {summary_path}")

    # Plot
    plot_results(y_test, metrics["predictions"], args.output)

    print(f"\n[Done] Baseline SST results saved to: {args.output}")


if __name__ == "__main__":
    main()

