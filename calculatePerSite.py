#!/usr/bin/env python3

import os
import math
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import openpyxl

# get rid of boring user warning
# UserWarning: Data Validation extension is not supported and will be removed
openpyxl.reader.excel.warnings.simplefilter(action="ignore")

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


def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """
    Parse a FASTA file and return a dictionary of {seq_id: sequence}.
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # Start new sequence
                current_id = line[1:]  # Remove '>' character
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)

    # Don't forget the last sequence
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences


def process_fasta_tsv_to_dataframes(
    fasta_12s: Path = None,
    fasta_16s: Path = None,
    counts_12s: Path = None,
    counts_16s: Path = None,
    min_length: int = None,
    max_length: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process FASTA and TSV files to create ASV sequences and reads DataFrames.
    Args:
        fasta_12s: Path to 12S FASTA file
        fasta_16s: Path to 16S FASTA file
        counts_12s: Path to 12S counts TSV file
        counts_16s: Path to 16S counts TSV file
        min_length: Minimum ASV sequence length (optional)
        max_length: Maximum ASV sequence length (optional)
    Returns: (asv_seqs_df, reads_long_df)
    """
    all_asv_seqs = []
    all_reads_long = []

    # Process each assay
    for assay, fasta_path, counts_path in [
        ("12S", fasta_12s, counts_12s),
        ("16S", fasta_16s, counts_16s),
    ]:
        if fasta_path is None or counts_path is None:
            continue

        print(f"[Processing] {assay} FASTA: {fasta_path}")
        print(f"[Processing] {assay} counts: {counts_path}")

        # Parse FASTA file
        sequences = parse_fasta(fasta_path)
        print(f"[Loaded] {len(sequences)} sequences from {fasta_path}")

        # Create ASV sequences DataFrame
        asv_seqs = pd.DataFrame(
            [{"asv_id": seq_id, "sequence": seq} for seq_id, seq in sequences.items()]
        )

        # Apply length filtering if specified
        initial_count = len(asv_seqs)
        if min_length is not None:
            asv_seqs = asv_seqs[asv_seqs["sequence"].str.len() >= min_length]
        if max_length is not None:
            asv_seqs = asv_seqs[asv_seqs["sequence"].str.len() <= max_length]

        filtered_count = len(asv_seqs)
        if initial_count != filtered_count:
            print(
                f"[Filtered] {assay}: {initial_count} -> {filtered_count} ASVs (length filter: {min_length or 'no min'} - {max_length or 'no max'})"
            )

        asv_seqs["assay"] = assay
        asv_seqs = asv_seqs[["asv_id", "assay", "sequence"]]

        # Load counts TSV file
        # Expected format: first column is ASV IDs, remaining columns are site counts
        counts_df = pd.read_csv(counts_path, sep="\t", index_col=0)
        print(
            f"[Loaded] Counts table: {counts_df.shape[0]} ASVs x {counts_df.shape[1]} sites"
        )

        # Convert to long format
        counts_df.index.name = "asv_id"
        reads_long = counts_df.reset_index().melt(
            id_vars=["asv_id"], var_name="site_id", value_name="reads"
        )

        # Filter reads to only include ASVs that passed length filtering
        valid_asvs = set(asv_seqs["asv_id"])
        reads_long = reads_long[reads_long["asv_id"].isin(valid_asvs)]

        reads_long["assay"] = assay
        reads_long = reads_long[["site_id", "assay", "asv_id", "reads"]]

        all_asv_seqs.append(asv_seqs)
        all_reads_long.append(reads_long)

        print(
            f"[Loaded] {len(asv_seqs)} {assay} ASVs, {len(reads_long)} site-ASV records"
        )

    if not all_asv_seqs:
        raise ValueError(
            "No FASTA/TSV files were processed. Please provide FASTA and TSV file pairs."
        )

    combined_asv_seqs = pd.concat(all_asv_seqs, ignore_index=True)
    combined_reads_long = pd.concat(all_reads_long, ignore_index=True)

    # Count by assay
    assay_counts = combined_asv_seqs["assay"].value_counts()
    print(
        f"[Combined] Total: {len(combined_asv_seqs)} unique ASVs ({', '.join([f'{count} {assay}' for assay, count in assay_counts.items()])})"
    )
    print(f"[Combined] Total: {len(combined_reads_long)} site-ASV records")

    return combined_asv_seqs, combined_reads_long


def process_excel_to_dataframes(
    files_by_assay: Dict[str, List[Path]],
    min_length: int = None,
    max_length: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process Excel files by assay to create ASV sequences and reads DataFrames.
    Args:
        files_by_assay: Dictionary mapping assay names ('12S', '16S') to lists of file paths
        min_length: Minimum ASV sequence length (optional)
        max_length: Maximum ASV sequence length (optional)
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
            asvs = pd.read_excel(excel_path, sheet_name="taxaRaw", skiprows=2)
            asv_seqs = asvs.loc[:, ["seq_id", "dna_sequence"]]
            asv_seqs = asv_seqs.drop_duplicates()
            asv_seqs = asv_seqs.rename(
                columns={"seq_id": "asv_id", "dna_sequence": "sequence"}
            )

            # Apply length filtering if specified
            initial_count = len(asv_seqs)
            if min_length is not None:
                asv_seqs = asv_seqs[asv_seqs["sequence"].str.len() >= min_length]
            if max_length is not None:
                asv_seqs = asv_seqs[asv_seqs["sequence"].str.len() <= max_length]

            filtered_count = len(asv_seqs)
            if initial_count != filtered_count:
                print(
                    f"[Filtered] {assay}: {initial_count} -> {filtered_count} ASVs (length filter: {min_length or 'no min'} - {max_length or 'no max'})"
                )

            asv_seqs["assay"] = assay
            asv_seqs["source_file"] = excel_path.name  # Track source file
            asv_seqs = asv_seqs[["asv_id", "assay", "sequence", "source_file"]]

            # Process reads from otuRaw sheet
            reads = pd.read_excel(excel_path, sheet_name="otuRaw")
            reads = reads.drop("Unnamed: 0", axis=1)
            reads = reads.rename(columns={"ASV": "asv_id"})
            reads_long = reads.drop("ASV_sequence", axis=1).melt(
                id_vars=["asv_id"], var_name="site_id", value_name="reads"
            )

            # Filter reads to only include ASVs that passed length filtering
            valid_asvs = set(asv_seqs["asv_id"])
            reads_long = reads_long[reads_long["asv_id"].isin(valid_asvs)]

            reads_long["assay"] = assay
            reads_long["source_file"] = excel_path.name  # Track source file
            reads_long = reads_long[
                ["site_id", "assay", "asv_id", "reads", "source_file"]
            ]

            all_asv_seqs.append(asv_seqs)
            all_reads_long.append(reads_long)

            print(
                f"[Loaded] {len(asv_seqs)} {assay} ASVs, {len(reads_long)} site-ASV records from {excel_path.name}"
            )

    if not all_asv_seqs:
        raise ValueError(
            "No Excel files were processed. Please provide --12s-files and/or --16s-files"
        )

    combined_asv_seqs = pd.concat(all_asv_seqs, ignore_index=True)
    combined_reads_long = pd.concat(all_reads_long, ignore_index=True)

    # Drop source_file column for final output (keep internal structure consistent)
    combined_asv_seqs = combined_asv_seqs[["asv_id", "assay", "sequence"]]
    combined_reads_long = combined_reads_long[["site_id", "assay", "asv_id", "reads"]]

    # Count by assay
    assay_counts = combined_asv_seqs["assay"].value_counts()
    print(
        f"[Combined] Total: {len(combined_asv_seqs)} unique ASVs ({', '.join([f'{count} {assay}' for assay, count in assay_counts.items()])})"
    )
    print(f"[Combined] Total: {len(combined_reads_long)} site-ASV records")

    return combined_asv_seqs, combined_reads_long


def write_parquet(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False)
    print(f"[Saved] {path}")


def check_intermediate_files(
    outdir: Path, assays_available: List[str]
) -> Dict[str, bool]:
    """
    Check which intermediate files already exist.
    Returns: Dictionary indicating which steps can be resumed
    """
    status = {
        "can_resume_asv_embeddings": {},
        "can_resume_site_embeddings": {},
        "can_resume_fused": False,
    }

    # Check ASV embedding files
    for assay in assays_available:
        asv_emb_path = outdir / f"asv_embeddings_{assay}.parquet"
        site_emb_path = outdir / f"site_embeddings_{assay}.parquet"

        status["can_resume_asv_embeddings"][assay] = asv_emb_path.exists()
        status["can_resume_site_embeddings"][assay] = site_emb_path.exists()

    # Check fused file
    fused_path = outdir / "site_embeddings_fused.parquet"
    status["can_resume_fused"] = fused_path.exists()

    return status


def clean_seq(seq: str) -> str:
    seq = (seq or "").upper()
    allowed = set("ACGTN")
    return "".join(ch if ch in allowed else "N" for ch in seq)


def load_model_and_tokenizer(
    assay: str,
    base_config: str,
    model_name: str,
    revision: str,
    config_revision: str,
    cache_dir: str = None,
):
    """
    Load config (base), tokenizer and model for a given assay and revision.
    The revision is applied to both tokenizer and model for reproducibility.
    """
    config = AutoConfig.from_pretrained(
        base_config, trust_remote_code=True, cache_dir=cache_dir, revision=config_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir, revision=revision
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        cache_dir=cache_dir,
        revision=revision,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(
        f"[{assay}] Loaded model '{model_name}' (revision={revision}) on device: {device}"
    )
    return tokenizer, model, device


@torch.inference_mode()
def embed_sequences(
    df: pd.DataFrame,
    tokenizer,
    model,
    device,
    pooling: str = "mean",
    batch_size: int = 128,
    use_amp: bool = True,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    df: columns ["asv_id", "sequence"]
    Returns DataFrame: ["asv_id", "dim_0", ..., "dim_{D-1}"]
    """
    asv_ids = df["asv_id"].tolist()
    seqs = [clean_seq(s) for s in df["sequence"].tolist()]

    all_vecs = []
    steps = math.ceil(len(seqs) / batch_size)
    for i in tqdm(range(0, len(seqs), batch_size), total=steps, desc="Embedding"):
        batch = seqs[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)

        # Handle both tuple output and ModelOutput with last_hidden_state
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
        w = np.log1p(c)
        w /= w.sum() + eps
    elif mode == "hellinger":
        a = c / (c.sum() + eps)
        w = np.sqrt(a)
        w /= w.sum() + eps
    elif mode == "clr":
        c_adj = c + eps
        log_c = np.log(c_adj)
        clr_vals = log_c - np.mean(log_c)
        # Convert back to weights (could also use softmax, should be same??)
        w = np.exp(clr_vals)
        w /= w.sum() + eps
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


def site_embed_simple_mean(embeds):
    """Simple arithmetic mean of ASV embeddings with no normalisation."""
    return embeds.mean(axis=0)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def site_embed_gem(embeds, weights, p=2.0):
    vpos = softplus(embeds)
    pooled = (weights[:, None] * (vpos**p)).sum(axis=0) / (weights.sum() + 1e-12)
    return np.power(pooled, 1.0 / p)


def compute_site_embeddings_from_dfs(
    reads_long: pd.DataFrame,
    asv_emb_df: pd.DataFrame,
    per_assay: bool = True,
    weight_mode: str = "hellinger",
    pooling: str = "l2_weighted_mean",
) -> pd.DataFrame:
    """
    reads_long: columns [site_id, assay, asv_id, reads]
    asv_emb_df: columns [asv_id, assay, dim_*]
    returns DataFrame with per-(site,assay) vectors split across dim_* columns
    """
    assert {"site_id", "assay", "asv_id", "reads"} <= set(reads_long.columns)
    dim_cols = [c for c in asv_emb_df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("asv_emb_df contains no embedding columns (dim_*)")

    df = reads_long.merge(
        asv_emb_df[["asv_id", "assay"] + dim_cols], on=["asv_id", "assay"], how="inner"
    )
    df = df[df["reads"] > 0].copy()
    if df.empty:
        raise ValueError(
            "No overlapping ASVs with reads > 0 and embeddings for this assay."
        )

    group_cols = ["site_id", "assay"] if per_assay else ["site_id"]
    records = []
    unique_groups = df[group_cols].drop_duplicates().shape[0]
    for keys, g in tqdm(
        df.groupby(group_cols, sort=False), total=unique_groups, desc="Pooling sites"
    ):
        embeds = g[dim_cols].to_numpy(dtype=np.float32)
        counts = g["reads"].to_numpy(dtype=np.float64)

        if pooling == "simple_mean":
            # No normalisation - just simple arithmetic mean
            vec = site_embed_simple_mean(embeds)
        else:
            # All other pooling methods use weights
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


def run_tsne(
    site_df: pd.DataFrame,
    label_cols: List[str],
    metric="cosine",
    perplexity=5,
    random_state=42,
) -> pd.DataFrame:
    dim_cols = [c for c in site_df.columns if c.startswith("dim_")]
    X = site_df[dim_cols].to_numpy(dtype=np.float32)
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 sites for t-SNE.")
    p = min(perplexity, max(5, (X.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, metric=metric, perplexity=p, random_state=random_state)
    Y = tsne.fit_transform(X)
    out = site_df[label_cols].copy()
    out["tsne_x"] = Y[:, 0]
    out["tsne_y"] = Y[:, 1]
    return out


def run_umap(
    site_df: pd.DataFrame,
    label_cols: List[str],
    metric="cosine",
    n_neighbors=15,
    random_state=42,
) -> pd.DataFrame:
    if not HAS_UMAP:
        raise RuntimeError("umap-learn is not installed. pip install umap-learn")
    dim_cols = [c for c in site_df.columns if c.startswith("dim_")]
    X = site_df[dim_cols].to_numpy(dtype=np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
    )
    Y = reducer.fit_transform(X)
    out = site_df[label_cols].copy()
    out["umap_x"] = Y[:, 0]
    out["umap_y"] = Y[:, 1]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="eDNA DNABERT-S embedding pipeline (Excel/FASTA+TSV -> ASVs -> sites -> t-SNE/UMAP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Excel input options (original)
    parser.add_argument(
        "--12s-files",
        nargs="*",
        type=Path,
        default=[],
        help="Path(s) to Excel file(s) containing 12S data",
    )
    parser.add_argument(
        "--16s-files",
        nargs="*",
        type=Path,
        default=[],
        help="Path(s) to Excel file(s) containing 16S data",
    )

    # FASTA + TSV input options (new)
    parser.add_argument(
        "--12s-fasta",
        type=Path,
        help="Path to FASTA file containing 12S ASV sequences",
    )
    parser.add_argument(
        "--12s-counts",
        type=Path,
        help="Path to TSV file containing 12S ASV counts (ASVs as rows, sites as columns)",
    )
    parser.add_argument(
        "--16s-fasta",
        type=Path,
        help="Path to FASTA file containing 16S ASV sequences",
    )
    parser.add_argument(
        "--16s-counts",
        type=Path,
        help="Path to TSV file containing 16S ASV counts (ASVs as rows, sites as columns)",
    )

    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--cache-dir", default=None, type=str, help="HuggingFace cache dir (optional)"
    )

    parser.add_argument(
        "--min-asv-length",
        type=int,
        default=None,
        help="Minimum ASV sequence length (optional)",
    )
    parser.add_argument(
        "--max-asv-length",
        type=int,
        default=None,
        help="Maximum ASV sequence length (optional)",
    )

    # Resume functionality
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation of all steps, ignoring existing intermediate files",
    )

    parser.add_argument("--model-12s", default="OceanOmics/eDNABERT-S_12S")
    parser.add_argument("--model-16s", default="OceanOmics/eDNABERT-S_16S")
    parser.add_argument("--base-config", default="zhihan1996/DNABERT-S")

    parser.add_argument(
        "--revision-12s",
        default="72923454c20c3b8c28ecd8d601e4140d92667e46",
        help="Git commit hash or tag for the 12S model",
    )
    parser.add_argument(
        "--revision-16s",
        default="d762ca73d44292a0b1074a7d760475f80154580c",
        help="Git commit hash or tag for the 16S model",
    )
    parser.add_argument(
        "--config-revision",
        default="00e47f96cdea35e4b6f5df89e5419cbe47d490c6",
        help="Git commit hash for the base config",
    )

    parser.add_argument("--pooling-token", default="mean", choices=["mean", "cls"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--use-amp", action="store_true", help="Enable mixed precision on CUDA"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Longest tokenized length for the tokenizer",
    )

    parser.add_argument(
        "--weight-mode",
        default="hellinger",
        choices=["hellinger", "log", "relative", "softmax_tau3", "clr"],
    )
    parser.add_argument(
        "--site-pooling",
        default="l2_weighted_mean",
        choices=[
            "l2_weighted_mean",
            "weighted_mean",
            "gem_p2",
            "gem_p3",
            "simple_mean",
        ],
        help="Method for pooling ASV embeddings to site embeddings. 'simple_mean' performs no normalisation.",
    )

    parser.add_argument("--run-tsne", action="store_true")
    parser.add_argument("--run-umap", action="store_true")
    parser.add_argument(
        "--perplexity", type=int, default=5, help="Perplexity setting for tSNE"
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15, help="Number of neighbours for UMAP"
    )
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--fuse",
        default="concat",
        choices=["none", "concat"],
        help="How to fuse 12S+16S site vectors (concat or none)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Validate input: user must provide either Excel files OR FASTA+TSV files, but not both
    has_excel = bool(args.__dict__["12s_files"] or args.__dict__["16s_files"])
    has_fasta_tsv = bool(
        args.__dict__["12s_fasta"]
        or args.__dict__["16s_fasta"]
        or args.__dict__["12s_counts"]
        or args.__dict__["16s_counts"]
    )

    if has_excel and has_fasta_tsv:
        raise ValueError(
            "Please provide either Excel files (--12s-files/--16s-files) OR "
            "FASTA+TSV files (--12s-fasta/--12s-counts/--16s-fasta/--16s-counts), but not both."
        )

    if not has_excel and not has_fasta_tsv:
        raise ValueError(
            "Please provide input files: either Excel files (--12s-files/--16s-files) OR "
            "FASTA+TSV files (--12s-fasta/--12s-counts/--16s-fasta/--16s-counts)."
        )

    # Validate FASTA+TSV pairs if using that input method
    if has_fasta_tsv:
        if args.__dict__["12s_fasta"] and not args.__dict__["12s_counts"]:
            raise ValueError(
                "If providing --12s-fasta, you must also provide --12s-counts"
            )
        if args.__dict__["12s_counts"] and not args.__dict__["12s_fasta"]:
            raise ValueError(
                "If providing --12s-counts, you must also provide --12s-fasta"
            )
        if args.__dict__["16s_fasta"] and not args.__dict__["16s_counts"]:
            raise ValueError(
                "If providing --16s-fasta, you must also provide --16s-counts"
            )
        if args.__dict__["16s_counts"] and not args.__dict__["16s_fasta"]:
            raise ValueError(
                "If providing --16s-counts, you must also provide --16s-fasta"
            )

    # Process input files based on input method
    if has_excel:
        # Original Excel processing
        files_by_assay = {}
        if args.__dict__["12s_files"]:
            files_by_assay["12S"] = args.__dict__["12s_files"]
        if args.__dict__["16s_files"]:
            files_by_assay["16S"] = args.__dict__["16s_files"]

        print(f"[Loading] Processing Excel files by assay")
        asv_seqs, reads_long = process_excel_to_dataframes(
            files_by_assay,
            min_length=args.min_asv_length,
            max_length=args.max_asv_length,
        )
    else:
        # New FASTA+TSV processing
        print(f"[Loading] Processing FASTA and TSV files")
        asv_seqs, reads_long = process_fasta_tsv_to_dataframes(
            fasta_12s=args.__dict__["12s_fasta"],
            fasta_16s=args.__dict__["16s_fasta"],
            counts_12s=args.__dict__["12s_counts"],
            counts_16s=args.__dict__["16s_counts"],
            min_length=args.min_asv_length,
            max_length=args.max_asv_length,
        )

    print(
        f"[Loaded] ASV sequences: {len(asv_seqs)} rows, Reads: {len(reads_long)} rows"
    )

    for cols, name in [
        ({"asv_id", "assay", "sequence"}, "asv_sequences"),
        ({"site_id", "assay", "asv_id", "reads"}, "reads_long"),
    ]:
        have = (
            set(asv_seqs.columns)
            if name == "asv_sequences"
            else set(reads_long.columns)
        )
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

    # Check resume status
    force_recalc = args.force
    resume_status = (
        check_intermediate_files(args.outdir, assays_available)
        if not force_recalc
        else {}
    )

    if not force_recalc:
        resumable_asvs = [
            assay
            for assay in assays_available
            if resume_status.get("can_resume_asv_embeddings", {}).get(assay, False)
        ]
        resumable_sites = [
            assay
            for assay in assays_available
            if resume_status.get("can_resume_site_embeddings", {}).get(assay, False)
        ]

        if resumable_asvs:
            print(f"[Resume] Found existing ASV embeddings for: {resumable_asvs}")
        if resumable_sites:
            print(f"[Resume] Found existing site embeddings for: {resumable_sites}")

    # embed ASVs per assay
    asv_embeddings_paths = {}
    for assay in assays_available:
        outp = args.outdir / f"asv_embeddings_{assay}.parquet"
        asv_embeddings_paths[assay] = outp

        # Check if we can resume from existing ASV embeddings
        if not force_recalc and resume_status.get("can_resume_asv_embeddings", {}).get(
            assay, False
        ):
            print(f"\n=== Resuming {assay} ASV embeddings from {outp} ===")
            continue

        print(f"\n=== Embedding {assay} ASVs ===")
        revision = args.revision_12s if assay == "12S" else args.revision_16s
        config_revision = arg.config_revision
        tok, mdl, device = load_model_and_tokenizer(
            assay=assay,
            base_config=args.base_config,
            model_name=assay_models[assay],
            revision=revision,
            config_revision=config_revision,
            cache_dir=args.cache_dir,
        )
        sub = asv_seqs[asv_seqs["assay"] == assay][
            ["asv_id", "sequence"]
        ].drop_duplicates()
        emb_df = embed_sequences(
            sub,
            tok,
            mdl,
            device,
            pooling=args.pooling_token,
            batch_size=args.batch_size,
            use_amp=args.use_amp,
            max_length=args.max_length,
        )
        emb_df.insert(1, "assay", assay)
        write_parquet(emb_df, outp)

    # pool to site embeddings per assay
    site_embeddings: Dict[str, pd.DataFrame] = {}
    for assay in assays_available:
        outp = args.outdir / f"site_embeddings_{assay}.parquet"

        # Check if we can resume from existing site embeddings
        if not force_recalc and resume_status.get("can_resume_site_embeddings", {}).get(
            assay, False
        ):
            print(f"\n=== Resuming {assay} site embeddings from {outp} ===")
            site_df = pd.read_parquet(outp)
            site_embeddings[assay] = site_df
            continue
        print(f"\n=== Site pooling for {assay} ===")
        asv_emb = pd.read_parquet(asv_embeddings_paths[assay])  # asv_id, assay, dim_*
        sub_reads = reads_long[reads_long["assay"] == assay]
        site_df = compute_site_embeddings_from_dfs(
            reads_long=sub_reads,
            asv_emb_df=asv_emb,
            per_assay=True,
            weight_mode=args.weight_mode,
            pooling=args.site_pooling,
        )
        write_parquet(site_df, outp)
        site_embeddings[assay] = site_df

    # optional fusion/12S+16S concatenation
    fused_df = pd.DataFrame()
    if args.fuse == "concat" and {"12S", "16S"} <= set(site_embeddings.keys()):
        outp = args.outdir / "site_embeddings_fused.parquet"

        # Check if we can resume from existing fused embeddings
        if not force_recalc and resume_status.get("can_resume_fused", False):
            print(f"\n=== Resuming fused embeddings from {outp} ===")
            fused_df = pd.read_parquet(outp)
        else:
            print("\n=== Fusing 12S+16S by concatenation ===")
            s12 = site_embeddings["12S"][
                ["site_id"]
                + [c for c in site_embeddings["12S"].columns if c.startswith("dim_")]
            ].set_index("site_id")
            s16 = site_embeddings["16S"][
                ["site_id"]
                + [c for c in site_embeddings["16S"].columns if c.startswith("dim_")]
            ].set_index("site_id")
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
            write_parquet(fused_df, outp)
    else:
        print(
            "\n[Info] Fusion skipped (either --fuse none or not both assays present)."
        )

    # optional t-SNE / UMAP on site vectors
    def _run_ordinations(tag: str, df: pd.DataFrame):
        if df.empty:
            print(f"[{tag}] No data—skipping ordinations")
            return
        label_cols = ["site_id"]
        if "assay" in df.columns:
            label_cols.append("assay")

        if args.run_tsne:
            print(
                f"[{tag}] Running t-SNE (metric={args.metric}, perplexity={args.perplexity})"
            )
            tsne_df = run_tsne(
                df,
                label_cols=label_cols,
                metric=args.metric,
                perplexity=args.perplexity,
                random_state=args.seed,
            )
            path = args.outdir / f"tsne_{tag}.csv"
            tsne_df.to_csv(path, index=False)
            print(f"[Saved] {path}")
        if args.run_umap:
            if not HAS_UMAP:
                print("[Warn] umap-learn is not installed; skipping UMAP.")
            else:
                print(
                    f"[{tag}] Running UMAP (metric={args.metric}, n_neighbors={args.n_neighbors})"
                )
                umap_df = run_umap(
                    df,
                    label_cols=label_cols,
                    metric=args.metric,
                    n_neighbors=args.n_neighbors,
                    random_state=args.seed,
                )
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
