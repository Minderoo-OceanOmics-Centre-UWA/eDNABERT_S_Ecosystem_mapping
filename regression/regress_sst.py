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
from sklearn.pipeline import Pipeline
import importlib

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# -----------------------------
# Utilities
# -----------------------------
def derive_base_site(sample_id: pd.Series) -> pd.Series:
    s = sample_id.astype(str)
    s = s.str.replace(r"_T\d+$", "", regex=True)   # remove _T1, _T2, ...
    s = s.str.replace(r"_\d+$", "", regex=True)    # remove trailing _<number>
    return s


def unwrap_estimator(fitted):
    """Unwrap GridSearchCV -> Pipeline -> final regressor step ('reg')."""
    est = fitted
    if hasattr(est, "best_estimator_"):
        est = est.best_estimator_
    if hasattr(est, "named_steps"):
        # pipeline with last step 'reg'
        est = est.named_steps.get("reg", est)
    return est


def extract_feature_importances(fitted, feature_names: List[str]) -> pd.DataFrame:
    """Ridge: abs(coef_), RF/XGB: feature_importances_. Handles Pipeline/GridSearchCV."""
    est = unwrap_estimator(fitted)
    importances = None

    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = getattr(est, "coef_")
        importances = np.abs(coef).ravel()

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["feature_type"] = np.where(df["feature"] == "samp_collect_device_encoded", "collection_device", "embedding")
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "predictions": y_pred.reshape(-1),
    }


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    mn, mx = float(np.min(y_test)), float(np.max(y_test))
    plt.plot([mn, mx], [mn, mx], "r--", lw=2)
    plt.xlabel("Actual SST (°C)")
    plt.ylabel("Predicted SST (°C)")
    plt.title("SST Prediction")
    plt.grid(True, alpha=0.3)
    outp = output_dir / "sst_predictions.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Prediction plot: {outp}")


# -----------------------------
# Embeddings I/O & fusion
# -----------------------------
def load_site_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "samp_name" in df.columns:
        return df
    if "site_id" in df.columns:
        return df.rename(columns={"site_id": "samp_name"})
    raise ValueError("Embeddings file must contain 'samp_name' or 'site_id' column")


def numeric_sort_dim_cols(cols: List[str]) -> List[str]:
    def key(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return float("inf")
    return sorted(cols, key=key)


def fuse_embeddings_concat(df12: pd.DataFrame, df16: pd.DataFrame) -> pd.DataFrame:
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

    recs = []
    for sid in common:
        v12 = s12.loc[sid].to_numpy(dtype=np.float32)
        v16 = s16.loc[sid].to_numpy(dtype=np.float32)
        vec = np.concatenate([v12, v16], axis=0)
        rec = {"samp_name": sid}
        for i, val in enumerate(vec):
            rec[f"dim_{i}"] = float(val)
        recs.append(rec)
    return pd.DataFrame(recs)


def load_metadata(path: Path) -> pd.DataFrame:
    """Read metadata/coordinates to obtain samp_collect_device (lat/lon ignored for modeling)."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_excel(path)

    required = {"samp_name"}
    if not required.issubset(df.columns):
        # allow site_id and rename
        if "site_id" in df.columns:
            df = df.rename(columns={"site_id": "samp_name"})
        else:
            raise ValueError("Metadata file must contain 'samp_name' (or 'site_id') column")

    if "samp_collect_device" not in df.columns:
        print("[Warning] 'samp_collect_device' not found; filling with 'unknown'")
        df["samp_collect_device"] = "unknown"

    return df[["samp_name", "samp_collect_device"]]


def load_sst_table(path: Path, id_col: str, sst_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize id column name to 'samp_name'
    if id_col not in df.columns:
        raise ValueError(f"SST table must contain identifier column '{id_col}' (or 'samp_name')")
    if sst_col not in df.columns:
        raise ValueError(f"SST table must contain column '{sst_col}'")

    out = df[[id_col, sst_col]].rename(columns={id_col: "samp_name", sst_col: "sst"})
    out = out.dropna(subset=["samp_name", "sst"])
    return out


def prepare_data(
    embeddings_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sst_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    emb = normalize_id_column(embeddings_df).copy()
    merged = emb.merge(metadata_df, on="samp_name", how="inner").merge(sst_df, on="samp_name", how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples across embeddings, metadata, and SST table")

    merged["base_site"] = derive_base_site(merged["samp_name"])

    # remove low temp sites (<0)
    merged['sst'] = merged['sst'].values.astype(float)
    merged = merged[merged['sst'] > 0]


    # device encoding
    le = LabelEncoder()
    merged["samp_collect_device_encoded"] = le.fit_transform(merged["samp_collect_device"].fillna("unknown"))

    dim_cols = numeric_sort_dim_cols([c for c in merged.columns if c.startswith("dim_")])
    if not dim_cols:
        raise ValueError("No embedding columns found (expected 'dim_*')")
    feature_cols = dim_cols + ["samp_collect_device_encoded"]

    X = merged[feature_cols].values
    y = merged["sst"].values.astype(float)
    return X, y, merged, feature_cols


def dask_chunks():
    """Return chunk dict if dask exists, else None."""
    return {"time": 1} if importlib.util.find_spec("dask") is not None else None


def build_pipeline(model_name: str, seed: int, optimize: bool) -> (Pipeline, Dict[str, List[Any]]):
    if model_name == "ridge":
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
        grid = {"reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]} if optimize else {}
        return pipe, grid
    elif model_name == "rf":
        reg = RandomForestRegressor(n_estimators=300, random_state=seed)
        pipe = Pipeline([("reg", reg)])
        grid = {
            "reg__n_estimators": [100, 300, 600],
            "reg__max_depth": [None, 10, 20],
            "reg__min_samples_split": [2, 5, 10],
        } if optimize else {}
        return pipe, grid
    elif model_name == "xgb":
        if not HAS_XGB:
            raise ImportError("XGBoost not installed. pip install xgboost")
        reg = xgb.XGBRegressor(random_state=seed, n_estimators=400)
        pipe = Pipeline([("reg", reg)])
        grid = {
            "reg__n_estimators": [200, 400, 800],
            "reg__max_depth": [3, 6, 9],
            "reg__learning_rate": [0.01, 0.1, 0.2],
            "reg__subsample": [0.8, 1.0],
            "reg__colsample_bytree": [0.8, 1.0],
        } if optimize else {}
        return pipe, grid
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict SST from site embeddings (+device), grouped CV by base_site; supports 12S+16S fusion."
    )
    # Embeddings (12S primary; optional 16S fusion)
    parser.add_argument("--site-embeddings", required=True, type=Path, help="Path to primary (e.g., 12S) embeddings (CSV/Parquet)")
    parser.add_argument("--site-embeddings-16s", type=Path, default=None, help="Optional path to 16S embeddings (CSV/Parquet)")
    parser.add_argument("--fuse", choices=["none", "concat"], default="none", help="Fusion strategy when 16S is provided (default: none)")
    parser.add_argument("--save-fused", action="store_true", help="Save fused embeddings to output dir if --fuse=concat")

    # Metadata & SST
    parser.add_argument("--metadata", required=True, type=Path, help="Path to sample metadata (CSV/TSV/Excel) with samp_name, samp_collect_device")
    parser.add_argument("--sst-table", required=True, type=Path, help="Path to SST CSV produced by downloader")
    parser.add_argument("--sst-id-col", default="samp_name", help="Identifier column in SST table (default: sample_id; 'samp_name' also accepted)")
    parser.add_argument("--sst-col", default="sst", help="SST column name in SST table (default: sst)")

    # General
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    parser.add_argument("--model", choices=["ridge", "rf", "xgb"], default="ridge", help="Model type")
    parser.add_argument("--optimize-hyperparams", action="store_true", help="GridSearchCV over GroupKFold")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (default: -1)")
    parser.add_argument("--cv-folds", type=int, default=5, help="GroupKFold folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # Load embeddings
    print(f"[Loading] Primary embeddings: {args.site_embeddings}")
    emb_primary = load_site_embeddings(args.site_embeddings)

    if args.fuse == "concat":
        if args.site_embeddings_16s is None:
            raise ValueError("--fuse=concat requires --site-embeddings-16s")
        print(f"[Loading] 16S embeddings: {args.site_embeddings_16s}")
        emb_16s = load_site_embeddings(args.site_embeddings_16s)
        print("[Fusion] Concatenating 12S + 16S on common samp_name...")
        embeddings_df = fuse_embeddings_concat(emb_primary, emb_16s)
        if args.save_fused:
            outp = args.output / "site_embeddings_fused.parquet"
            embeddings_df.to_parquet(outp, index=False)
            print(f"[Saved] Fused embeddings: {outp}")
    else:
        embeddings_df = normalize_id_column(emb_primary)

    # Metadata & SST
    print(f"[Loading] Metadata: {args.metadata}")
    metadata_df = load_metadata(args.metadata)

    print(f"[Loading] SST table: {args.sst_table}")
    sst_df = load_sst_table(args.sst_table, args.sst_id_col, args.sst_col)

    # Prepare data
    X, y, merged_df, feature_cols = prepare_data(embeddings_df, metadata_df, sst_df)
    groups_all = merged_df["base_site"].values

    # Grouped split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups_all[train_idx]
    groups_test = groups_all[test_idx]

    print(f"[Split] Training: {len(X_train)} samples across {pd.unique(groups_train).size} base sites")
    print(f"[Split] Testing:  {len(X_test)} samples across {pd.unique(groups_test).size} base sites")

    # Pipeline + CV
    pipe, grid = build_pipeline(args.model, args.seed, args.optimize_hyperparams)
    gkf = GroupKFold(n_splits=args.cv_folds)

    if args.optimize_hyperparams and grid:
        print("[Optimizing] Hyperparameters with GroupKFold...")
        model = GridSearchCV(pipe, param_grid=grid, cv=gkf, scoring="r2", n_jobs=args.n_jobs)
        model.fit(X_train, y_train, groups=groups_train)
        print(f"[Best params] {model.best_params_}")
        model_name = f"{args.model.upper()} (optimized)"
        cv_scores = np.array([model.best_score_])
        print(f"[CV] Best GroupKFold R²: {model.best_score_:.4f}")
    else:
        model = pipe.fit(X_train, y_train)
        model_name = args.model.upper()
        print(f"[CV] {args.cv_folds}-fold GroupKFold cross-validation...")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=gkf, scoring="r2", groups=groups_train, n_jobs=args.n_jobs)

    # Test evaluation
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importance
    imp_df = extract_feature_importances(model, feature_cols)
    imp_path = args.output / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"[Saved] Feature importance: {imp_path}")

    # Print
    print("\n" + "=" * 50)
    print(f"SST REGRESSION RESULTS ({model_name})")
    print("=" * 50)
    print(f"CV R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"Test R²:       {metrics['r2']:.4f}")
    print(f"Test MSE:      {metrics['mse']:.4f}")
    print(f"Test MAE:      {metrics['mae']:.4f}")

    # Save per-sample predictions
    test_ids = merged_df.iloc[test_idx]["samp_name"].values
    test_base = merged_df.iloc[test_idx]["base_site"].values
    results_df = pd.DataFrame({
        "sample_id": test_ids,
        "base_site": test_base,
        "actual_sst": y_test,
        "pred_sst": metrics["predictions"],
        "abs_error": np.abs(y_test - metrics["predictions"]),
    })
    res_path = args.output / "regression_results_sst.csv"
    results_df.to_csv(res_path, index=False)
    print(f"[Saved] Detailed results: {res_path}")

    # Summary
    summary = {
        "model": model_name,
        "grouped_splitting": True,
        "cv_strategy": "GroupKFold",
        "cv_folds": args.cv_folds,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2": metrics["r2"],
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
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
    sum_path = args.output / "regression_summary_sst.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary: {sum_path}")

    # Plot
    plot_results(y_test, metrics["predictions"], args.output)
    print(f"\n[Done] All outputs saved to: {args.output}")


if __name__ == "__main__":
    main()

