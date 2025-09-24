#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ERDDAP OPeNDAP (.dods) endpoints (NOAA/NCEI)
# NOTE: datasetID does NOT include a file extension; .dods selects OPeNDAP service.
FINAL_DS  = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.dods"
PRELIM_DS = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_prelim_by_time_zlev_lat_lon.dods"

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample NOAA OISST v2.1 daily SST at given lat/lon/date rows."
    )
    p.add_argument("--input", required=True, type=Path, help="Input CSV with lat, lon, date (and optional sample_id)")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path")

    p.add_argument("--lat-col", default="decimalLatitude",
                   help="Latitude column name (default: decimalLatitude)")
    p.add_argument("--lon-col", default="decimalLongitude",
                   help="Longitude column name (default: decimalLongitude)")
    p.add_argument("--date-col", default="eventDate",
                   help="Date column name (default: 'date'; set to 'eventDate' if that's your field). "
                        "Accepts ISO 8601 datetimes like 2022-11-19T19:00:00 and dates like 2022-11-19.")
    p.add_argument("--id-col", default="sample_id",
                   help="Sample ID column name (default: sample_id; auto-created if missing)")

    p.add_argument("--dataset", choices=["final", "prelim"], default="final",
                   help="Which OISST stream to use: final (default) or prelim (latest few days, subject to revision)")
    p.add_argument("--vars", nargs="+", default=["sst"],
                   choices=["sst", "anom", "err", "ice"],
                   help="Variables to extract (default: sst). You can add: anom err ice")
    p.add_argument("--interp", choices=["nearest", "linear"], default="nearest",
                   help="Spatial interpolation (default: nearest). 'linear' is bilinear on the 0.25° grid.")
    p.add_argument("--chunksize", type=int, default=5000,
                   help="Process rows in batches to keep remote requests small (default: 5000)")
    p.add_argument("--timeout", type=int, default=120,
                   help="xarray network timeout seconds (default: 120)")

    # NEW: insecure flag
    p.add_argument("--insecure", action="store_true",
                   help="Disable SSL verification (uses Pydap + requests with verify=False). "
                        "Install pydap if you use this flag: pip install pydap")

    return p.parse_args()

def _ensure_sample_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = np.arange(1, len(df) + 1).astype(str)
    return df

def _prep_input(df: pd.DataFrame, lat_col: str, lon_col: str, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        raise ValueError("Input CSV has no rows.")
    for c in [lat_col, lon_col, date_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # --------- Skip NA/blank dates (and common 'NA' strings) ---------
    orig_n = len(df)
    na_tokens = {"", "na", "nan", "n/a", "null", "none"}
    date_raw_stripped = df[date_col].astype(str).str.strip()
    valid_mask = df[date_col].notna() & (~date_raw_stripped.str.lower().isin(na_tokens))
    df = df.loc[valid_mask].copy()
    dropped_na = orig_n - len(df)
    if dropped_na > 0:
        print(f"[Info] Skipped {dropped_na} rows with NA/blank '{date_col}'")

    if df.empty:
        raise ValueError("All rows were dropped due to missing dates. Nothing to process.")

    # --------- Robust ISO 8601 parsing with timezone handling ---------
    dt = pd.to_datetime(date_raw_stripped.loc[valid_mask], errors="coerce", utc=True)

    # Drop unparsable dates (be forgiving)
    bad_mask = dt.isna()
    if bad_mask.any():
        bad_count = int(bad_mask.sum())
        print(f"[Info] Skipped {bad_count} rows with unparsable '{date_col}' values")
        good_idx = dt.index[~bad_mask]
        df = df.loc[good_idx].copy()
        dt = dt.loc[good_idx]

    if df.empty:
        raise ValueError("All remaining rows had unparsable dates. Nothing to process.")

    # UTC day, then naive (no tz)
    df[date_col] = dt.dt.floor("D").dt.tz_localize(None)

    # --------- Coordinates ---------
    lon = df[lon_col].astype(float).values
    lon360 = np.where(lon < 0, lon + 360.0, lon)  # ERDDAP uses 0..360°E
    df["_lon360"] = lon360
    df["_lat"] = df[lat_col].astype(float).values
    return df

import importlib

def _dask_chunks():
    """Return a chunks dict if dask is available, else None (avoid dask requirement)."""
    has_dask = importlib.util.find_spec("dask") is not None
    return {"time": 1} if has_dask else None

def _open_dataset(which: str, timeout: int, insecure: bool = False) -> xr.Dataset:
    """
    Open OISST via OPeNDAP.
    - secure (default): netCDF4 engine + .dods endpoint
    - insecure: Pydap + requests (verify=False) and dataset BASE url (no .dods)
    """
    FINAL_BASE  = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon"
    PRELIM_BASE = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_prelim_by_time_zlev_lat_lon"

    base = FINAL_BASE if which == "final" else PRELIM_BASE
    dods = base + ".dods"
    chunks = _dask_chunks()  # None if dask not installed

    if insecure:
        # ---- Pydap path (no .dods here) ----
        try:
            from xarray.backends import PydapDataStore
        except Exception as e:
            raise RuntimeError("Pydap is required for --insecure mode. Install with: pip install pydap") from e
        import requests

        # Build a requests.Session with verify disabled; use identity encoding to avoid some server quirks
        s = requests.Session()
        s.verify = False                 # INSECURE: skip SSL verification
        s.headers.update({"Accept-Encoding": "identity"})
        s.trust_env = True               # try honoring corporate proxies first

        # 1) Try base dataset URL (preferred for Pydap)
        try:
            store = PydapDataStore.open(base, session=s)
            ds = xr.open_dataset(store, chunks=chunks)  # chunks=None if no dask
        except Exception as e1:
            # 2) Fallback: try again without proxies (some networks require this)
            try:
                s.trust_env = False
                store = PydapDataStore.open(base, session=s)
                ds = xr.open_dataset(store, chunks=chunks)
            except Exception as e2:
                # (Do NOT try ".dods" with Pydap—ERDDAP will treat it as part of datasetID and 404.)
                raise RuntimeError(
                    "Pydap insecure open failed.\n"
                    f"First error (trust_env=True): {e1}\n"
                    f"Second error (trust_env=False): {e2}"
                )
    else:
        # ---- Secure path: netCDF4 + .dods (correct for OPeNDAP) ----
        ds = xr.open_dataset(dods, engine="netcdf4", chunks=chunks)

    # ---- Normalize coordinate names (unchanged from your version) ----
    rename = {}
    for cand in ["lat", "latitude"]:
        if cand in ds.coords:
            rename[cand] = "latitude"
            break
    for cand in ["lon", "longitude"]:
        if cand in ds.coords:
            rename[cand] = "longitude"
            break
    for cand in ["zlev", "depth"]:
        if cand in ds.coords:
            rename[cand] = "depth"
            break
    if rename:
        ds = ds.rename(rename)

    for req in ["time", "latitude", "longitude", "depth"]:
        if req not in ds.coords:
            raise RuntimeError(f"Expected coordinate '{req}' not found in dataset. Available: {list(ds.coords)}")
    return ds


def _select_time(ds: xr.Dataset, when: pd.Timestamp) -> xr.Dataset:
    # OISST is daily; align to the nearest daily label
    return ds.sel(time=np.datetime64(when), method="nearest")

def _extract_for_rows(ds_t: xr.Dataset, rows: pd.DataFrame, interp: str, vars_to_get: list) -> pd.DataFrame:
    # Surface layer
    if "depth" in ds_t.coords:
        if ds_t.sizes.get("depth", 0) > 1:
            ds_t = ds_t.isel(depth=0)
        else:
            ds_t = ds_t.sel(depth=ds_t["depth"].values[0])

    lats = xr.DataArray(rows["_lat"].values, dims=("points",))
    lons = xr.DataArray(rows["_lon360"].values, dims=("points",))

    if interp == "linear":
        sampled = ds_t[vars_to_get].interp(latitude=lats, longitude=lons, method="linear")
    else:
        sampled = ds_t[vars_to_get].sel(latitude=lats, longitude=lons, method="nearest")

    df_vars = sampled.to_dataframe().reset_index().rename(columns={"points": "_row"})
    df_vars["_row"] = np.arange(len(rows))
    merged = pd.concat([rows.reset_index(drop=True), df_vars.drop(columns=["_row"])], axis=1)
    return merged

def main():
    args = parse_args()

    # Read
    df = pd.read_csv(args.input, sep = '\t')

    # Prep (drop NA/blank/unparsable dates, convert coords)
    df = _prep_input(df, args.lat_col, args.lon_col, args.date_col)

    # Assign sample_id only to rows we will actually process
    df = _ensure_sample_id(df, args.id_col)

    print(f"[Info] Opening OISST dataset: {args.dataset} (insecure={args.insecure})")
    ds = _open_dataset(args.dataset, args.timeout, insecure=args.insecure)

    # Validate variables
    missing = [v for v in args.vars if v not in ds.data_vars]
    if missing:
        raise RuntimeError(f"Requested variables not in dataset: {missing}. Available: {list(ds.data_vars)}")

    results = []
    unique_dates = df[args.date_col].drop_duplicates().sort_values().tolist()

    for t0 in tqdm(unique_dates, desc="Sampling times"):
        sub = df[df[args.date_col] == t0]
        ds_t = _select_time(ds, t0)

        for start in range(0, len(sub), args.chunksize):
            chunk = sub.iloc[start:start + args.chunksize]
            res_chunk = _extract_for_rows(ds_t, chunk, args.interp, args.vars)
            results.append(res_chunk)

    out = pd.concat(results, ignore_index=True)

    # Order columns nicely
    cols_front = [c for c in [args.id_col, args.lat_col, args.lon_col, args.date_col] if c in out.columns]
    cols_hidden = ["_lat", "_lon360"]
    cols_rest = [c for c in out.columns if c not in set(cols_front + cols_hidden + args.vars)]
    out = out[cols_front + cols_rest + args.vars]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[Saved] {args.output} ({len(out)} rows)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

