import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List
import re

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")


def load_asv_data(path: Path) -> pd.DataFrame:
    """Load ASV taxonomic and abundance data from TSV file."""
    return pd.read_csv(path, sep='\t')


def load_sample_metadata(path: Path) -> pd.DataFrame:
    """Load sample metadata with coordinates and collection device."""
    df = pd.read_csv(path, sep='\t')

    # Keep only samples with coordinates (exclude controls)
    df = df[df['decimalLongitude'].notna() & df['decimalLatitude'].notna()].copy()

    # Create site_id column to match with ASV data (sample names)
    df['site_id'] = df['samp_name']

    # Select relevant columns including collection device
    columns_to_keep = ['site_id', 'decimalLatitude', 'decimalLongitude']
    
    # Add samp_collect_device if it exists in the dataframe
    if 'samp_collect_device' in df.columns:
        columns_to_keep.append('samp_collect_device')
        print(f"[Data] Found collection device data with {df['samp_collect_device'].nunique()} unique devices")
        print(f"[Data] Collection devices: {df['samp_collect_device'].value_counts().to_dict()}")
    else:
        print("[Warning] Column 'samp_collect_device' not found in metadata file")
        df['samp_collect_device'] = 'unknown'  # Add default value
        columns_to_keep.append('samp_collect_device')

    return df[columns_to_keep].rename(columns={
        'decimalLatitude': 'latitude',
        'decimalLongitude': 'longitude'
    })


def get_best_taxonomic_level(row: pd.Series) -> str:
    """
    Get the most specific non-dropped taxonomic level for an ASV.
    Searches from species back to phylum.
    """
    taxonomic_levels = ['species', 'genus', 'family', 'order', 'class', 'phylum']

    for level in taxonomic_levels:
        value = row[level]
        if pd.notna(value) and value != 'dropped' and value != 'Unknown':
            return f"{level}:{value}"

    return "unknown:unknown"


def prepare_taxonomic_features(asv_df: pd.DataFrame, sample_cols: List[str]) -> pd.DataFrame:
    """
    Convert ASV taxonomic data into site-level taxonomic composition features.

    Args:
        asv_df: ASV dataframe with taxonomic info and abundance columns
        sample_cols: List of sample column names (starting with V10_)

    Returns:
        DataFrame with site_id as index and taxonomic features as columns
    """
    print(f"[Data] Processing {len(asv_df)} ASVs across {len(sample_cols)} samples")

    # Get best taxonomic assignment for each ASV
    asv_df['best_taxonomy'] = asv_df.apply(get_best_taxonomic_level, axis=1)

    print(f"[Data] Found {asv_df['best_taxonomy'].nunique()} unique taxonomic groups")

    # Create site-level taxonomic composition
    site_features = {}

    for sample_col in sample_cols:
        if sample_col not in asv_df.columns:
            continue

        # Sum abundances by taxonomic group for this sample
        sample_composition = asv_df.groupby('best_taxonomy')[sample_col].sum()

        # Convert to relative abundance (optional - can help with standardization)
        total_abundance = sample_composition.sum()
        if total_abundance > 0:
            sample_composition = sample_composition / total_abundance

        site_features[sample_col] = sample_composition

    # Convert to dataframe with samples as rows, taxonomic groups as columns
    features_df = pd.DataFrame(site_features).T.fillna(0)
    features_df.index.name = 'site_id'

    print(f"[Data] Created feature matrix: {features_df.shape[0]} samples × {features_df.shape[1]} taxonomic features")

    # Remove taxonomic groups that are completely absent (all zeros)
    features_df = features_df.loc[:, (features_df != 0).any(axis=0)]

    print(f"[Data] After removing absent taxa: {features_df.shape[0]} samples × {features_df.shape[1]} features")

    return features_df.reset_index()


def prepare_data(
    asv_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, LabelEncoder]:
    """
    Prepare taxonomic features, collection device features, and coordinates for regression.

    Returns:
        X: Feature matrix (taxonomic composition + collection device)
        y: Target matrix (lat, lon)
        merged_df: Merged dataframe for reference
        device_encoder: LabelEncoder for collection device (for inverse transform if needed)
    """
    # Get sample columns (those starting with V10_ and not controls)
    sample_cols = [col for col in asv_df.columns if col.startswith('V10_')
                   and not col.endswith(('_WC_T1', '_DI_T1', '_BC_1', '_NTC_1'))]

    print(f"[Data] Found {len(sample_cols)} sample columns (excluding controls)")

    # Create taxonomic features
    features_df = prepare_taxonomic_features(asv_df, sample_cols)

    # Merge with coordinates and collection device
    merged = features_df.merge(metadata_df, on='site_id', how='inner')

    if merged.empty:
        raise ValueError("No matching site_ids between taxonomic features and coordinates")

    # Derive base_site by stripping replicate suffix
    merged['base_site'] = merged['site_id'].astype(str).str.replace(r'_T1$', '', regex=True)
    merged['base_site'] = merged['base_site'].str.replace(r'_\d+$', '', regex=True)

    print(f"[Data] Merged {len(merged)} samples with both taxonomic data and coordinates")
    n_sites = merged['base_site'].nunique()
    print(f"[Data] Unique base sites: {n_sites}")

    # Encode collection device as numerical feature
    device_encoder = LabelEncoder()
    merged['samp_collect_device_encoded'] = device_encoder.fit_transform(
        merged['samp_collect_device'].fillna('unknown')
    )
    
    print(f"[Data] Encoded collection devices: {dict(zip(device_encoder.classes_, range(len(device_encoder.classes_))))}")

    # Extract feature columns (taxonomic composition + collection device)
    feature_cols = [col for col in merged.columns
                   if col not in ['site_id', 'latitude', 'longitude', 'base_site', 
                                 'samp_collect_device']]  # Exclude original device column, keep encoded

    if not feature_cols:
        raise ValueError("No features found after merging")

    # Separate taxonomic and device features for reporting
    taxonomic_cols = [col for col in feature_cols if col != 'samp_collect_device_encoded']
    device_cols = [col for col in feature_cols if col == 'samp_collect_device_encoded']

    X = merged[feature_cols].values
    y = merged[['latitude', 'longitude']].values

    print(f"[Data] Features: {len(taxonomic_cols)} taxonomic groups + {len(device_cols)} collection device feature")
    print(f"[Data] Total features: {X.shape[1]}, Targets: {len(merged)} samples across {n_sites} base sites")

    return X, y, merged, device_encoder


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


def analyze_feature_importance(model, feature_names: List[str], output_dir: Path, device_encoder: LabelEncoder):
    """Analyze and save feature importance if the model supports it."""
    if hasattr(model, 'feature_importances_'):
        # For RandomForest and XGBoost
        importances = model.feature_importances_
    elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
        # For MultiOutputRegressor wrapping RandomForest/XGBoost
        importances = model.estimator.feature_importances_
    elif hasattr(model, 'coef_'):
        # For Ridge regression - use absolute values of coefficients as importance
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        print("[Info] Model does not support feature importance analysis")
        return

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'feature_type': ['collection_device' if 'device_encoded' in name else 'taxonomic' for name in feature_names]
    }).sort_values('importance', ascending=False)

    # Decode device feature name for readability
    importance_df.loc[importance_df['feature'] == 'samp_collect_device_encoded', 'feature'] = 'collection_device'

    # Save top features
    top_features_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(top_features_path, index=False)
    
    # Print top features
    print(f"\n[Feature Importance] Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:50} | {row['importance']:.6f} | {row['feature_type']}")
    
    print(f"[Saved] Feature importance: {top_features_path}")

    # Summary of feature type importance
    type_importance = importance_df.groupby('feature_type')['importance'].agg(['sum', 'mean', 'count'])
    print(f"\n[Feature Type Summary]:")
    for feat_type, stats in type_importance.iterrows():
        print(f"  {feat_type:15} | Total: {stats['sum']:.6f} | Mean: {stats['mean']:.6f} | Count: {stats['count']}")


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
    axes[0].set_title("Latitude Prediction (Taxonomic + Device)")
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
    axes[1].set_title("Longitude Prediction (Taxonomic + Device)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "taxonomic_device_baseline_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Prediction plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict lat/lon from taxonomic composition and collection device (baseline for ecosystem embedding comparison)"
    )
    parser.add_argument(
        "--asv-data",
        required=True,
        type=Path,
        help="Path to ASV taxonomic data TSV file (asv_lca_with_fishbase_output.tsv)",
    )
    parser.add_argument(
        "--sample-metadata",
        required=True,
        type=Path,
        help="Path to sample metadata TSV file (samplemetadata.tsv)",
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

    print(f"[Loading] ASV taxonomic data: {args.asv_data}")
    asv_df = load_asv_data(args.asv_data)

    print(f"[Loading] Sample metadata: {args.sample_metadata}")
    metadata_df = load_sample_metadata(args.sample_metadata)

    # Prepare data
    X, y, merged_df, device_encoder = prepare_data(asv_df, metadata_df)

    # Get feature names for importance analysis
    feature_cols = [col for col in merged_df.columns
                   if col not in ['site_id', 'latitude', 'longitude', 'base_site', 
                                 'samp_collect_device']]

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
        f"[Split] Training samples: {len(X_train)} across {pd.unique(groups_train).size} base sites"
    )
    print(
        f"[Split] Testing samples:  {len(X_test)} across {pd.unique(groups_test).size} base sites"
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
                cv=gkf,
                scoring="r2",
                n_jobs=args.n_jobs,
            )
            model.fit(X_train_scaled, y_train, groups=groups_train)
            print(f"[Best params] {model.best_params_}")
            model_name = f"Taxonomic+Device Ridge Regression (alpha={model.best_estimator_.estimator.alpha})"
        else:
            model = MultiOutputRegressor(Ridge(alpha=1.0))
            model.fit(X_train_scaled, y_train)
            model_name = "Taxonomic+Device Ridge Regression"

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
                cv=gkf,
                scoring="r2",
                n_jobs=args.n_jobs,
            )
            model.fit(X_train_scaled, y_train, groups=groups_train)
            print(f"[Best params] {model.best_params_}")
            model_name = "Taxonomic+Device Random Forest (optimized)"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=args.seed)
            model.fit(X_train_scaled, y_train)
            model_name = "Taxonomic+Device Random Forest"

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
                cv=gkf,
                scoring="r2",
                n_jobs=args.n_jobs,
            )
            model.fit(X_train_scaled, y_train, groups=groups_train)
            print(f"[Best params] {model.best_params_}")
            model_name = "Taxonomic+Device XGBoost (optimized)"
        else:
            model = MultiOutputRegressor(xgb.XGBRegressor(random_state=args.seed))
            model.fit(X_train_scaled, y_train)
            model_name = "Taxonomic+Device XGBoost"

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
            cv=gkf,
            scoring="r2",
            groups=groups_train,
        )

    # Evaluate on test set
    metrics = evaluate_model(model, X_test_scaled, y_test)

    # Analyze feature importance
    analyze_feature_importance(model, feature_cols, args.output, device_encoder)

    # Print results
    print("\n" + "=" * 70)
    print(f"TAXONOMIC + COLLECTION DEVICE BASELINE RESULTS ({model_name})")
    print("=" * 70)
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
    test_devices = merged_df.iloc[idx_test]["samp_collect_device"].values
    
    results_df = pd.DataFrame(
        {
            "site_id": test_site_ids,
            "base_site": test_base_sites,
            "collection_device": test_devices,
            "actual_lat": y_test[:, 0],
            "actual_lon": y_test[:, 1],
            "pred_lat": metrics["predictions"][:, 0],
            "pred_lon": metrics["predictions"][:, 1],
            "lat_error": np.abs(y_test[:, 0] - metrics["predictions"][:, 0]),
            "lon_error": np.abs(y_test[:, 1] - metrics["predictions"][:, 1]),
        }
    )

    results_path = args.output / "taxonomic_device_baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[Saved] Detailed results: {results_path}")

    # Save summary metrics
    summary = {
        "model": model_name,
        "baseline_type": "taxonomic_composition_plus_collection_device",
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
        "n_taxonomic_features": X.shape[1] - 1,  # All except collection device
        "n_device_features": 1,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_train_base_sites": int(pd.unique(groups_train).size),
        "n_test_base_sites": int(pd.unique(groups_test).size),
        "n_total_base_sites": int(pd.unique(groups_all).size),
        "collection_devices": list(device_encoder.classes_),
        "hyperparameter_optimization": bool(args.optimize_hyperparams),
    }

    if args.optimize_hyperparams:
        summary["best_params"] = model.best_params_

    summary_path = args.output / "taxonomic_device_baseline_summary.json"
    import json

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary metrics: {summary_path}")

    # Create plots
    plot_results(y_test, metrics["predictions"], args.output)

    print(f"\n[Done] Taxonomic + Collection Device baseline results saved to: {args.output}")


if __name__ == "__main__":
    main()
