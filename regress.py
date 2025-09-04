#!/usr/bin/env python3
"""
regress.py
Multi-output regression to predict latitude and longitude from site embeddings.

Usage:
    python regress.py --site-embeddings results/site_embeddings_12S.parquet --coordinates coordinates.csv --output results/

Requirements:
    - Site embeddings from calculatePerSite.py
    - Coordinate data (CSV with columns: site_id, latitude, longitude)
"""

import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_site_embeddings(path: Path) -> pd.DataFrame:
    """Load site embeddings from parquet file."""
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)


def load_coordinates(path: Path) -> pd.DataFrame:
    """Load coordinate data from CSV file."""
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    
    # Ensure required columns exist
    required_cols = {'site_id', 'latitude', 'longitude'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Coordinate file must contain columns: {required_cols}")
    
    return df[['site_id', 'latitude', 'longitude']]


def prepare_data(embeddings_df: pd.DataFrame, coords_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Merge embeddings with coordinates and prepare X, y matrices.
    
    Returns:
        X: Feature matrix (embeddings)
        y: Target matrix (lat, lon)
        merged_df: Merged dataframe for reference
    """
    # Merge on site_id
    merged = embeddings_df.merge(coords_df, on='site_id', how='inner')
    
    if merged.empty:
        raise ValueError("No matching site_ids between embeddings and coordinates")
    
    print(f"[Data] Merged {len(merged)} sites with both embeddings and coordinates")
    
    # Extract embedding columns (those starting with 'dim_')
    dim_cols = [col for col in merged.columns if col.startswith('dim_')]
    if not dim_cols:
        raise ValueError("No embedding columns found (expected columns starting with 'dim_')")
    
    X = merged[dim_cols].values
    y = merged[['latitude', 'longitude']].values
    
    print(f"[Data] Features: {X.shape[1]} dimensions, Targets: {len(merged)} sites")
    
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
        'mse_overall': mse,
        'mae_overall': mae,
        'r2_overall': r2,
        'lat_mse': lat_mse,
        'lon_mse': lon_mse,
        'lat_r2': lat_r2,
        'lon_r2': lon_r2,
        'predictions': y_pred
    }


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """Create scatter plots of predicted vs actual coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Latitude plot
    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6)
    axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                 [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Latitude')
    axes[0].set_ylabel('Predicted Latitude')
    axes[0].set_title('Latitude Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # Longitude plot
    axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6)
    axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()], 
                 [y_test[:, 1].min(), y_test[:, 1].max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Longitude')
    axes[1].set_ylabel('Predicted Longitude')
    axes[1].set_title('Longitude Prediction')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'coordinate_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Prediction plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict lat/lon from site embeddings")
    parser.add_argument("--site-embeddings", required=True, type=Path,
                       help="Path to site embeddings parquet file")
    parser.add_argument("--coordinates", required=True, type=Path,
                       help="Path to coordinates CSV/Excel file (site_id, latitude, longitude)")
    parser.add_argument("--output", required=True, type=Path,
                       help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--model", choices=["ridge", "rf"], default="ridge",
                       help="Regression model: ridge or rf (random forest)")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Cross-validation folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
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
    
    # Split data with indices to track site_ids
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=args.test_size, random_state=args.seed
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[Split] Training: {len(X_train)} sites, Testing: {len(X_test)} sites")
    
    # Choose and train model
    if args.model == "ridge":
        base_model = Ridge(alpha=1.0)
        model = MultiOutputRegressor(base_model)
        model_name = "Ridge Regression"
    else:
        base_model = RandomForestRegressor(n_estimators=100, random_state=args.seed)
        model = base_model  # RandomForest handles multi-output natively
        model_name = "Random Forest"
    
    print(f"[Training] {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    print(f"[CV] {args.cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                               cv=args.cv_folds, scoring='r2')
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Print results
    print("\n" + "="*50)
    print(f"REGRESSION RESULTS ({model_name})")
    print("="*50)
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"Test R² (overall):   {metrics['r2_overall']:.4f}")
    print(f"Test MSE (overall):  {metrics['mse_overall']:.4f}")
    print(f"Test MAE (overall):  {metrics['mae_overall']:.4f}")
    print()
    print("Per-coordinate performance:")
    print(f"  Latitude  - R²: {metrics['lat_r2']:.4f}, MSE: {metrics['lat_mse']:.4f}")
    print(f"  Longitude - R²: {metrics['lon_r2']:.4f}, MSE: {metrics['lon_mse']:.4f}")
    
    # Save detailed results
    test_site_ids = merged_df.iloc[idx_test]['site_id'].values
    results_df = pd.DataFrame({
        'site_id': test_site_ids,
        'actual_lat': y_test[:, 0],
        'actual_lon': y_test[:, 1], 
        'pred_lat': metrics['predictions'][:, 0],
        'pred_lon': metrics['predictions'][:, 1],
        'lat_error': np.abs(y_test[:, 0] - metrics['predictions'][:, 0]),
        'lon_error': np.abs(y_test[:, 1] - metrics['predictions'][:, 1])
    })
    
    results_path = args.output / 'regression_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n[Saved] Detailed results: {results_path}")
    
    # Save summary metrics
    summary = {
        'model': model_name,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'test_r2_overall': metrics['r2_overall'],
        'test_mse_overall': metrics['mse_overall'],
        'test_mae_overall': metrics['mae_overall'],
        'test_r2_lat': metrics['lat_r2'],
        'test_r2_lon': metrics['lon_r2'],
        'n_features': X.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    summary_path = args.output / 'regression_summary.json'
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary metrics: {summary_path}")
    
    # Create plots
    plot_results(y_test, metrics['predictions'], args.output)
    
    print(f"\n[Done] All results saved to: {args.output}")


if __name__ == "__main__":
    main()