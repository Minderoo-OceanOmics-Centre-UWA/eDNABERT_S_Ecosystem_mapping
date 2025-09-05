#!/usr/bin/env python3

import os
import math
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def process_excel_to_dataframes(files_by_assay: Dict[str, List[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process Excel files by assay to create ASV sequences and reads DataFrames.
    Args:
        files_by_assay: Dictionary mapping assay names ('12S', '16S') to lists of file paths
    Returns: (asv_seqs_df, reads_long_df)
    """
    all_asv_seqs = []
    all_reads_long = []
    
    for assay, excel_paths in files_by_assay.items():
        if not excel_paths:
            continue
            
        print(f"[Processing] {len(excel_paths)} {assay} files")
        
        for excel_path in excel_paths:
            print(f"[Processing] {assay} file: {excel_path}")
            
            # Process ASV sequences from taxaRaw sheet
            asvs = pd.read_excel(excel_path, sheet_name='taxaRaw', skiprows=2)
            asv_seqs = asvs.loc[:, ['seq_id', 'dna_sequence']]
            asv_seqs = asv_seqs.drop_duplicates()
            asv_seqs = asv_seqs.rename(columns={'seq_id': 'asv_id', 'dna_sequence': 'sequence'})
            asv_seqs['assay'] = assay
            asv_seqs['source_file'] = excel_path.name  # Track source file
            asv_seqs = asv_seqs[['asv_id', 'assay', 'sequence', 'source_file']]
            all_asv_seqs.append(asv_seqs)
            
            # Process reads from otuRaw sheet
            reads = pd.read_excel(excel_path, sheet_name='otuRaw')
            reads = reads.drop('Unnamed: 0', axis=1)
            reads = reads.rename(columns={'ASV': 'asv_id'})
            reads_long = reads.drop('ASV_sequence', axis=1).melt(id_vars=['asv_id'], var_name='site_id', value_name='reads')
            reads_long['assay'] = assay
            reads_long['source_file'] = excel_path.name  # Track source file
            reads_long = reads_long[['site_id', 'assay', 'asv_id', 'reads', 'source_file']]
            all_reads_long.append(reads_long)
            
            print(f"[Loaded] {len(asv_seqs)} {assay} ASVs, {len(reads_long)} site-ASV records from {excel_path.name}")
    
    if not all_asv_seqs:
        raise ValueError("No Excel files were processed. Please provide --12s-files and/or --16s-files")
    
    # Concatenate all dataframes
    combined_asv_seqs = pd.concat(all_asv_seqs, ignore_index=True)
    combined_reads_long = pd.concat(all_reads_long, ignore_index=True)
    
    # Remove duplicates (same ASV across files should be identical)
    combined_asv_seqs = combined_asv_seqs.drop_duplicates(subset=['asv_id', 'assay'])
    
    # Drop source_file column for final output (keep internal structure consistent)
    combined_asv_seqs = combined_asv_seqs[['asv_id', 'assay', 'sequence']]
    combined_reads_long = combined_reads_long[['site_id', 'assay', 'asv_id', 'reads']]
    
    # Count by assay
    assay_counts = combined_asv_seqs['assay'].value_counts()
    print(f"[Combined] Total: {len(combined_asv_seqs)} unique ASVs ({', '.join([f'{count} {assay}' for assay, count in assay_counts.items()])})")
    print(f"[Combined] Total: {len(combined_reads_long)} site-ASV records")
    
    return combined_asv_seqs, combined_reads_long


def process_excel_to_dataframes_legacy(excel_paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy function for backwards compatibility with old argument format."""
    # Convert to new format (assume 12S for backwards compatibility)
    if isinstance(excel_paths, (str, Path)):
        excel_paths = [Path(excel_paths)]
    else:
        excel_paths = [Path(p) for p in excel_paths]
    files_by_assay = {'12S': excel_paths}
    return process_excel_to_dataframes(files_by_assay)


def write_parquet(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False)
    print(f"[Saved] {path}")


def clean_seq(seq: str) -> str:
    seq = (seq or "").upper()
    allowed = set("ACGTN")
    return "".join(ch if ch in allowed else "N" for ch in seq)



def load_model_and_tokenizer(assay: str, base_config: str, model_name: str, cache_dir: str = None):
    config = AutoConfig.from_pretrained(base_config, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config, cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"[{assay}] Loaded model on device: {device}")
    return tokenizer, model, device


@torch.inference_mode()
def embed_sequences(df: pd.DataFrame,
                    tokenizer,
                    model,
                    device,
                    pooling: str = "mean",
                    batch_size: int = 128,
                    use_amp: bool = True,
                    max_length: int = 512) -> pd.DataFrame:
    """
    df: columns ["asv_id", "sequence"]
    Returns DataFrame: ["asv_id", "dim_0", ..., "dim_{D-1}"]
    """
    asv_ids = df["asv_id"].tolist()
    seqs = [clean_seq(s) for s in df["sequence"].tolist()]

    all_vecs = []
    steps = math.ceil(len(seqs) / batch_size)
    for i in tqdm(range(0, len(seqs), batch_size), total=steps, desc="Embedding"):
        batch = seqs[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)

        # Handle both tuple output and ModelOutput with last_hidden_state
        # TODO: is this correct??
        if isinstance(out, tuple):
            hidden = out[0]  # First element is typically the hidden states
        else:
            hidden = out.last_hidden_state  # (B, L, D)
        if pooling == "cls":
            vec = hidden[:, 0, :]  # (B, D)
        elif pooling == "mean":
            attn = enc.get("attention_mask", None)
            if attn is None:
                vec = hidden.mean(dim=1)
            else:
                mask = attn.unsqueeze(-1)  # (B, L, 1)
                vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

        all_vecs.append(vec.detach().float().cpu().numpy())

    if not all_vecs:
        raise ValueError("No sequences were embedded—check your input.")
    X = np.vstack(all_vecs)
    dim_cols = [f"dim_{i}" for i in range(X.shape[1])]
    emb_df = pd.DataFrame(X, columns=dim_cols, copy=False)
    emb_df.insert(0, "asv_id", asv_ids)
    return emb_df



def weights_from_counts(counts, mode="hellinger", tau=3.0, eps=1e-12):
    # TODO: I think we need only hellinger...
    c = np.asarray(counts, dtype=float)
    if c.sum() <= eps:
        return np.ones_like(c) / max(len(c), 1)
    if mode == "relative":
        w = c / c.sum()
    elif mode == "log":
        w = np.log1p(c); w /= (w.sum() + eps)
    elif mode == "hellinger":
        a = c / (c.sum() + eps)
        w = np.sqrt(a); w /= (w.sum() + eps)
    elif mode.startswith("softmax"):
        # e.g., softmax_tau3
        if "_tau" in mode:
            try:
                tau = float(mode.split("_tau")[-1])
            except Exception:
                tau = 3.0
        z = np.log1p(c) / float(tau)
        z -= z.max()
        expz = np.exp(z)
        w = expz / (expz.sum() + eps)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return w


def site_embed_weighted_mean(embeds, weights):
    wsum = weights.sum()
    return (embeds * weights[:, None]).sum(axis=0) / (wsum + 1e-12)


def site_embed_l2_weighted_mean(embeds, weights, eps=1e-12, renorm=True):
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    v = embeds / norms
    site = (v * weights[:, None]).sum(axis=0)
    if renorm:
        site_norm = max(np.linalg.norm(site), eps)
        site = site / site_norm
    return site


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def site_embed_gem(embeds, weights, p=2.0):
    vpos = softplus(embeds)
    pooled = (weights[:, None] * (vpos ** p)).sum(axis=0) / (weights.sum() + 1e-12)
    return np.power(pooled, 1.0 / p)


def compute_site_embeddings_from_dfs(reads_long: pd.DataFrame,
                                     asv_emb_df: pd.DataFrame,
                                     per_assay: bool = True,
                                     weight_mode: str = "hellinger",
                                     pooling: str = "l2_weighted_mean") -> pd.DataFrame:
    """
    reads_long: columns [site_id, assay, asv_id, reads]
    asv_emb_df: columns [asv_id, assay, dim_*]
    returns DataFrame with per-(site,assay) vectors split across dim_* columns
    """
    assert {"site_id", "assay", "asv_id", "reads"} <= set(reads_long.columns)
    dim_cols = [c for c in asv_emb_df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("asv_emb_df contains no embedding columns (dim_*)")

    df = reads_long.merge(asv_emb_df[["asv_id", "assay"] + dim_cols],
                          on=["asv_id", "assay"], how="inner")
    df = df[df["reads"] > 0].copy()
    if df.empty:
        raise ValueError("No overlapping ASVs with reads > 0 and embeddings for this assay.")

    group_cols = ["site_id", "assay"] if per_assay else ["site_id"]
    records = []
    unique_groups = df[group_cols].drop_duplicates().shape[0]
    for keys, g in tqdm(df.groupby(group_cols, sort=False), total=unique_groups, desc="Pooling sites"):
        embeds = g[dim_cols].to_numpy(dtype=np.float32)
        counts = g["reads"].to_numpy(dtype=np.float64)
        w = weights_from_counts(counts, mode=weight_mode)

        if pooling == "weighted_mean":
            vec = site_embed_weighted_mean(embeds, w)
        elif pooling == "l2_weighted_mean":
            vec = site_embed_l2_weighted_mean(embeds, w)
        elif pooling.startswith("gem"):
            p = 2.0
            if "_p" in pooling:
                try:
                    p = float(pooling.split("_p")[-1])
                except Exception:
                    p = 2.0
            vec = site_embed_gem(embeds, w, p=p)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        rec = {"site_id": keys[0]}
        if per_assay:
            rec["assay"] = keys[1]
        for i, val in enumerate(vec):
            rec[f"dim_{i}"] = float(val)
        records.append(rec)

    return pd.DataFrame(records)



def run_tsne(site_df: pd.DataFrame,
             label_cols: List[str],
             metric="cosine",
             perplexity=30,
             random_state=42) -> pd.DataFrame:
    dim_cols = [c for c in site_df.columns if c.startswith("dim_")]
    X = site_df[dim_cols].to_numpy(dtype=np.float32)
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 sites for t-SNE.")
    p = min(perplexity, max(5, (X.shape[0]-1)//3))
    tsne = TSNE(n_components=2, metric=metric, perplexity=p, random_state=random_state)
    Y = tsne.fit_transform(X)
    out = site_df[label_cols].copy()
    out["tsne_x"] = Y[:, 0]
    out["tsne_y"] = Y[:, 1]
    return out


def run_umap(site_df: pd.DataFrame,
             label_cols: List[str],
             metric="cosine",
             n_neighbors=15,
             random_state=42) -> pd.DataFrame:
    if not HAS_UMAP:
        raise RuntimeError("umap-learn is not installed. pip install umap-learn")
    dim_cols = [c for c in site_df.columns if c.startswith("dim_")]
    X = site_df[dim_cols].to_numpy(dtype=np.float32)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, metric=metric, random_state=random_state)
    Y = reducer.fit_transform(X)
    out = site_df[label_cols].copy()
    out["umap_x"] = Y[:, 0]
    out["umap_y"] = Y[:, 1]
    return out



def main():
    parser = argparse.ArgumentParser(description="eDNA DNABERT-S embedding pipeline (Excel -> ASVs -> sites -> t-SNE/UMAP)")
    parser.add_argument("--12s-files", nargs='*', type=Path, default=[],
                        help="Path(s) to Excel file(s) containing 12S data")
    parser.add_argument("--16s-files", nargs='*', type=Path, default=[],
                        help="Path(s) to Excel file(s) containing 16S data")
    parser.add_argument("--excel-files", nargs='+', type=Path,
                        help="Deprecated: Use --12s-files and --16s-files instead")
    parser.add_argument("--excel-file", type=Path,
                        help="Deprecated: Use --12s-files and --16s-files instead")
    parser.add_argument("--asv-seqs", type=Path,
                        help="Optional: Path to asv_sequences.[csv|parquet] (columns: asv_id, assay, sequence)")
    parser.add_argument("--reads", type=Path,
                        help="Optional: Path to reads_long.[csv|parquet] (columns: site_id, assay, asv_id, reads)")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--cache-dir", default=None, type=str, help="HuggingFace cache dir (optional)")

    parser.add_argument("--model-12s", default="OceanOmics/eDNABERT-S_12S")
    parser.add_argument("--model-16s", default="OceanOmics/eDNABERT-S_16S")
    parser.add_argument("--base-config", default="zhihan1996/DNABERT-S")

    parser.add_argument("--pooling-token", default="mean", choices=["mean", "cls"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--weight-mode", default="hellinger",
                        choices=["hellinger", "log", "relative", "softmax_tau3"])
    parser.add_argument("--site-pooling", default="l2_weighted_mean",
                        choices=["l2_weighted_mean", "weighted_mean", "gem_p2", "gem_p3"])

    parser.add_argument("--run-tsne", action="store_true")
    parser.add_argument("--run-umap", action="store_true")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fuse", default="concat", choices=["none", "concat"],
                        help="How to fuse 12S+16S site vectors (concat or none)")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Check for new assay-specific arguments
    if hasattr(args, '12s_files') and args.__dict__['12s_files'] or hasattr(args, '16s_files') and args.__dict__['16s_files']:
        # New format: separate files by assay
        files_by_assay = {}
        if hasattr(args, '12s_files') and args.__dict__['12s_files']:
            files_by_assay['12S'] = args.__dict__['12s_files']
        if hasattr(args, '16s_files') and args.__dict__['16s_files']:
            files_by_assay['16S'] = args.__dict__['16s_files']
        
        print(f"[Loading] Processing Excel files by assay")
        asv_seqs, reads_long = process_excel_to_dataframes(files_by_assay)
        print(f"[Loaded] ASV sequences: {len(asv_seqs)} rows, Reads: {len(reads_long)} rows")
        
    elif args.excel_files:
        # Legacy format: assume 12S
        print(f"[Loading] Processing {len(args.excel_files)} Excel files (legacy format, assuming 12S)")
        asv_seqs, reads_long = process_excel_to_dataframes_legacy(args.excel_files)
        print(f"[Loaded] ASV sequences: {len(asv_seqs)} rows, Reads: {len(reads_long)} rows")
        
    elif args.excel_file:
        # Legacy format: assume 12S
        print(f"[Loading] Processing single Excel file: {args.excel_file} (legacy format, assuming 12S)")
        asv_seqs, reads_long = process_excel_to_dataframes_legacy(args.excel_file)
        print(f"[Loaded] ASV sequences: {len(asv_seqs)} rows, Reads: {len(reads_long)} rows")
        
    elif args.asv_seqs and args.reads:
        print(f"[Loading] Using parquet/csv files")
        asv_seqs = read_table(args.asv_seqs)
        reads_long = read_table(args.reads)
        
    else:
        raise ValueError("Please provide either --12s-files/--16s-files, or --asv-seqs and --reads, or legacy --excel-files/--excel-file")

    for cols, name in [({"asv_id", "assay", "sequence"}, "asv_sequences"),
                       ({"site_id", "assay", "asv_id", "reads"}, "reads_long")]:
        have = set(asv_seqs.columns) if name == "asv_sequences" else set(reads_long.columns)
        missing = cols - have
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    assays_available = []
    assay_models: Dict[str, str] = {}
    if (asv_seqs["assay"] == "12S").any():
        assays_available.append("12S")
        assay_models["12S"] = args.model_12s
    if (asv_seqs["assay"] == "16S").any():
        assays_available.append("16S")
        assay_models["16S"] = args.model_16s
    if not assays_available:
        raise ValueError("No 12S or 16S rows found in asv_sequences.")

    print(f"[Assays] Detected in input: {assays_available}")

    # embed ASVs per assay
    asv_embeddings_paths = {}
    for assay in assays_available:
        print(f"\n=== Embedding {assay} ASVs ===")
        tok, mdl, device = load_model_and_tokenizer(assay, args.base_config, assay_models[assay], args.cache_dir)
        sub = asv_seqs[asv_seqs["assay"] == assay][["asv_id", "sequence"]].drop_duplicates()
        emb_df = embed_sequences(sub, tok, mdl, device,
                                 pooling=args.pooling_token,
                                 batch_size=args.batch_size,
                                 use_amp=args.use_amp,
                                 max_length=args.max_length)
        emb_df.insert(1, "assay", assay)
        outp = args.outdir / f"asv_embeddings_{assay}.parquet"
        write_parquet(emb_df, outp)
        asv_embeddings_paths[assay] = outp

    # pool to site embeddings per assay
    site_embeddings: Dict[str, pd.DataFrame] = {}
    for assay in assays_available:
        print(f"\n=== Site pooling for {assay} ===")
        asv_emb = read_table(asv_embeddings_paths[assay])  # asv_id, assay, dim_*
        sub_reads = reads_long[reads_long["assay"] == assay]
        site_df = compute_site_embeddings_from_dfs(
            reads_long=sub_reads,
            asv_emb_df=asv_emb,
            per_assay=True,
            weight_mode=args.weight_mode,
            pooling=args.site_pooling
        )
        outp = args.outdir / f"site_embeddings_{assay}.parquet"
        write_parquet(site_df, outp)
        site_embeddings[assay] = site_df

    # optional fusion/12S+16S concatenation
    fused_df = pd.DataFrame()
    if args.fuse == "concat" and {"12S", "16S"} <= set(site_embeddings.keys()):
        print("\n=== Fusing 12S+16S by concatenation ===")
        s12 = site_embeddings["12S"][["site_id"] + [c for c in site_embeddings["12S"].columns if c.startswith("dim_")]].set_index("site_id")
        s16 = site_embeddings["16S"][["site_id"] + [c for c in site_embeddings["16S"].columns if c.startswith("dim_")]].set_index("site_id")
        common = s12.index.intersection(s16.index)

        fused_records = []
        for sid in common:
            v12 = s12.loc[sid].to_numpy(dtype=np.float32)
            v16 = s16.loc[sid].to_numpy(dtype=np.float32)
            vec = np.concatenate([v12, v16])
            rec = {"site_id": sid}
            for i, val in enumerate(vec):
                rec[f"dim_{i}"] = float(val)
            fused_records.append(rec)
        fused_df = pd.DataFrame(fused_records)
        outp = args.outdir / "site_embeddings_fused.parquet"
        write_parquet(fused_df, outp)
    else:
        print("\n[Info] Fusion skipped (either --fuse none or not both assays present).")

    # optional t-SNE / UMAP on site vectors
    def _run_ordinations(tag: str, df: pd.DataFrame):
        if df.empty:
            print(f"[{tag}] No data—skipping ordinations")
            return
        label_cols = ["site_id"]
        if "assay" in df.columns:
            label_cols.append("assay")

        if args.run_tsne:
            print(f"[{tag}] Running t-SNE (metric={args.metric}, perplexity={args.perplexity})")
            tsne_df = run_tsne(df, label_cols=label_cols, metric=args.metric, perplexity=args.perplexity, random_state=args.seed)
            path = args.outdir / f"tsne_{tag}.csv"
            tsne_df.to_csv(path, index=False)
            print(f"[Saved] {path}")
        if args.run_umap:
            if not HAS_UMAP:
                print("[Warn] umap-learn is not installed; skipping UMAP.")
            else:
                print(f"[{tag}] Running UMAP (metric={args.metric}, n_neighbors={args.n_neighbors})")
                umap_df = run_umap(df, label_cols=label_cols, metric=args.metric, n_neighbors=args.n_neighbors, random_state=args.seed)
                path = args.outdir / f"umap_{tag}.csv"
                umap_df.to_csv(path, index=False)
                print(f"[Saved] {path}")

    for assay in assays_available:
        _run_ordinations(assay, site_embeddings[assay])

    if not fused_df.empty:
        fused_df2 = fused_df.copy()
        fused_df2["assay"] = "fused"
        _run_ordinations("fused", fused_df2)

    print("\n[Done] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
