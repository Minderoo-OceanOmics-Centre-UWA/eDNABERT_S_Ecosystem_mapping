#!/usr/bin/env python3
import os
import sys
import math
import argparse
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import openpyxl

from util.diagnostics import run_diagnostics

# get rid of boring user warning
# UserWarning: Data Validation extension is not supported and will be removed
openpyxl.reader.excel.warnings.simplefilter(action="ignore")
# get rid of boring tokenizers warning
# The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from sklearn.manifold import TSNE

try:
    import umap

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Optional psutil for resource monitoring
try:
    import psutil

    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


class _StreamToLogger:
    """File-like stream object that redirects writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level

    def write(self, buf):
        # tqdm writes carriage returns/partial lines; splitlines keeps useful parts
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def setup_logging(
    outdir: Path, capture_stdio: bool = True, level: str = "INFO"
) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()  # root
    logger.handlers.clear()

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_log = outdir / f"pipeline_{ts}.log"
    latest_log = outdir / "latest.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File: timestamped. can be old log
    fh_ts = logging.FileHandler(ts_log, mode="w")
    fh_ts.setLevel(log_level)
    fh_ts.setFormatter(fmt)
    logger.addHandler(fh_ts)

    # File: latest (always overwritten)
    fh_latest = logging.FileHandler(latest_log, mode="w")
    fh_latest.setLevel(log_level)
    fh_latest.setFormatter(fmt)
    logger.addHandler(fh_latest)

    # Console: IMPORTANT — write to the original stdout to avoid recursion
    ch = logging.StreamHandler(stream=sys.__stdout__)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logging initialised")
    logger.info(f"Log file (timestamped): {ts_log}")
    logger.info(f"Log file (latest):     {latest_log}")

    if capture_stdio:
        # Redirect stdout & stderr so everything lands in the logs (incl. tqdm/HF)
        # tqdm writes to stderr by default — map to INFO to avoid wonkylooking ERRORs
        # that aren't errors
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.INFO)
        logger.info("STDOUT/STDERR are captured into log files.")
    else:
        logger.info(
            "STDOUT/STDERR capture disabled (progress bars cleaner in console)."
        )

    return logger


class ResourceMonitor:
    """
    Background sampler for CPU & memory (process + system) and GPU memory via torch.cuda.
    """

    def __init__(
        self,
        interval: float = 1.0,
        include_gpu: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.interval = max(0.2, float(interval))  # don't go too fast
        self.include_gpu = include_gpu
        self.logger = logger or logging.getLogger(__name__)
        self._thread = None
        self._stop_evt = threading.Event()
        self._started = False
        self._start_time = None
        self._end_time = None

        self._samples = 0
        self._rss_sum = 0.0
        self._cpu_sum = 0.0
        self.max_rss = 0
        self.max_vms = 0
        self.max_cpu_percent = 0.0
        self.system_mem_max_percent = 0.0

        self.gpu_info = []  # populated at start from torch
        self.gpu_peak_alloc = {}  # bytes (from sampling)
        self.gpu_peak_reserved = {}  # bytes (from sampling)

        self._proc = psutil.Process(os.getpid()) if HAS_PSUTIL else None

    def _bytes_to_mib(self, b: Union[int, float]) -> float:
        return float(b) / (1024.0 * 1024.0)

    def _init_gpu(self):
        self.gpu_info = []
        if self.include_gpu and torch.cuda.is_available():
            try:
                # Reset PyTorch peak stats so "max_memory_allocated" is scoped to this run
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)
            except Exception:
                pass

            for d in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(d)
                    total_mem = getattr(props, "total_memory", 0)
                    name = getattr(props, "name", f"cuda:{d}")
                except Exception:
                    total_mem = 0
                    name = f"cuda:{d}"
                self.gpu_info.append(
                    {"index": d, "name": name, "total_bytes": total_mem}
                )
                self.gpu_peak_alloc[d] = 0
                self.gpu_peak_reserved[d] = 0

    def _gpu_snapshot(self):
        """Return current per-device (allocated, reserved) in bytes, and update internal peaks."""
        if not (self.include_gpu and torch.cuda.is_available()):
            return
        for d in range(torch.cuda.device_count()):
            try:
                alloc = int(torch.cuda.memory_allocated(d))
                reserv = int(torch.cuda.memory_reserved(d))
            except Exception:
                alloc, reserv = 0, 0
            self.gpu_peak_alloc[d] = max(self.gpu_peak_alloc.get(d, 0), alloc)
            self.gpu_peak_reserved[d] = max(self.gpu_peak_reserved.get(d, 0), reserv)

    def _sample_once(self):
        # CPU percent: first call establishes baseline; subsequent calls produce % since last call
        if HAS_PSUTIL and self._proc is not None:
            try:
                cpu = self._proc.cpu_percent(interval=None)  # non-blocking
                mem = self._proc.memory_info()
                sys_mem = psutil.virtual_memory()

                self.max_rss = max(self.max_rss, mem.rss)
                self.max_vms = max(self.max_vms, mem.vms)
                self.max_cpu_percent = max(self.max_cpu_percent, float(cpu))
                self.system_mem_max_percent = max(
                    self.system_mem_max_percent, float(sys_mem.percent)
                )

                # running averages
                self._samples += 1
                self._rss_sum += float(mem.rss)
                self._cpu_sum += float(cpu)
            except Exception:
                # ignore transient psutil errors
                pass

        # GPU snapshot (also updates internal peaks)
        self._gpu_snapshot()

    def _run(self):
        # Prime the process CPU percent to get a realistic first number
        if HAS_PSUTIL and self._proc is not None:
            try:
                self._proc.cpu_percent(interval=None)
            except Exception:
                pass

        # Continuous sampling
        while not self._stop_evt.is_set():
            self._sample_once()
            self._stop_evt.wait(self.interval)

    def start(self):
        if self._started:
            return
        self._start_time = time.time()
        self._init_gpu()
        self._thread = threading.Thread(
            target=self._run, name="ResourceMonitor", daemon=True
        )
        self._thread.start()
        self._started = True
        # initial sample (so we have at least one)
        time.sleep(0.05)
        self._sample_once()

    def stop(self):
        if not self._started:
            return
        self._stop_evt.set()
        try:
            self._thread.join(timeout=5)
        except Exception:
            pass
        self._end_time = time.time()
        # One last sample at end
        self._sample_once()

    def summary(self) -> dict:
        duration = (
            (self._end_time - self._start_time)
            if (self._end_time and self._start_time)
            else 0.0
        )
        avg_rss = (self._rss_sum / self._samples) if self._samples else 0.0
        avg_cpu = (self._cpu_sum / self._samples) if self._samples else 0.0

        # PyTorch's own max_memory_allocated may be more accurate than our sampling
        torch_peaks = {}
        if self.include_gpu and torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                try:
                    torch_peaks[d] = int(torch.cuda.max_memory_allocated(d))
                except Exception:
                    torch_peaks[d] = None

        return {
            "duration_sec": duration,
            "samples": self._samples,
            "proc_rss_avg_mib": self._bytes_to_mib(avg_rss),
            "proc_rss_peak_mib": self._bytes_to_mib(self.max_rss),
            "proc_vms_peak_mib": self._bytes_to_mib(self.max_vms),
            "proc_cpu_avg_percent": avg_cpu,
            "proc_cpu_peak_percent": self.max_cpu_percent,
            "system_mem_peak_percent": self.system_mem_max_percent,
            "gpu": {
                "devices": self.gpu_info,
                "peak_alloc_mib": {
                    d: self._bytes_to_mib(self.gpu_peak_alloc.get(d, 0))
                    for d in self.gpu_peak_alloc
                },
                "peak_reserved_mib": {
                    d: self._bytes_to_mib(self.gpu_peak_reserved.get(d, 0))
                    for d in self.gpu_peak_reserved
                },
                "torch_peak_alloc_mib": {
                    d: (self._bytes_to_mib(b) if b is not None else None)
                    for d, b in (torch_peaks or {}).items()
                },
            },
        }

    def log_summary(self, logger: Optional[logging.Logger] = None):
        lg = logger or self.logger
        s = self.summary()
        lg.info("=== Resource usage summary ===")
        lg.info("Duration: %.2f sec | Samples: %d", s["duration_sec"], s["samples"])
        lg.info(
            "Process RSS avg/peak: %.1f / %.1f MiB | VMS peak: %.1f MiB",
            s["proc_rss_avg_mib"],
            s["proc_rss_peak_mib"],
            s["proc_vms_peak_mib"],
        )
        lg.info(
            "Process CPU avg/peak: %.1f%% / %.1f%%",
            s["proc_cpu_avg_percent"],
            s["proc_cpu_peak_percent"],
        )
        lg.info("System memory peak: %.1f%%", s["system_mem_peak_percent"])

        if s["gpu"]["devices"]:
            lg.info("GPU devices:")
            for dev in s["gpu"]["devices"]:
                idx, name, total_b = dev["index"], dev["name"], dev["total_bytes"]
                lg.info(
                    "  GPU %d: %s (total %.1f MiB)",
                    idx,
                    name,
                    self._bytes_to_mib(total_b),
                )
                peak_alloc = s["gpu"]["peak_alloc_mib"].get(idx, 0.0)
                peak_reserved = s["gpu"]["peak_reserved_mib"].get(idx, 0.0)
                torch_peak = s["gpu"]["torch_peak_alloc_mib"].get(idx, None)
                lg.info(
                    "    peak allocated: %.1f MiB | peak reserved: %.1f MiB | torch peak: %s",
                    peak_alloc,
                    peak_reserved,
                    f"{torch_peak:.1f} MiB" if torch_peak is not None else "n/a",
                )
        else:
            lg.info("No CUDA devices detected.")

        if not HAS_PSUTIL:
            lg.warning(
                "psutil not installed: CPU/RSS metrics may be limited. `pip install psutil` for full stats."
            )


# logger needs to be global
logger = logging.getLogger(__name__)


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
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    for assay, fasta_path, counts_path in [
        ("12S", fasta_12s, counts_12s),
        ("16S", fasta_16s, counts_16s),
    ]:
        if fasta_path is None or counts_path is None:
            continue

        logger.info(f"[Processing] {assay} FASTA: {fasta_path}")
        logger.info(f"[Processing] {assay} counts: {counts_path}")

        sequences = parse_fasta(fasta_path)
        logger.info(f"[Loaded] {len(sequences)} sequences from {fasta_path}")

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
            logger.info(
                f"[Filtered] {assay}: {initial_count} -> {filtered_count} ASVs "
                f"(length filter: {min_length or 'no min'} - {max_length or 'no max'})"
            )

        asv_seqs["assay"] = assay
        asv_seqs = asv_seqs[["asv_id", "assay", "sequence"]]

        # Expected format: first column is ASV IDs, remaining columns are site counts
        counts_df = pd.read_csv(counts_path, sep="\t", index_col=0)
        logger.info(
            f"[Loaded] Counts table: {counts_df.shape[0]} ASVs x {counts_df.shape[1]} sites"
        )

        counts_df.index.name = "asv_id"
        reads_long = counts_df.reset_index().melt(
            id_vars=["asv_id"], var_name="site_id", value_name="reads"
        )

        valid_asvs = set(asv_seqs["asv_id"])
        reads_long = reads_long[reads_long["asv_id"].isin(valid_asvs)]

        reads_long["assay"] = assay
        reads_long = reads_long[["site_id", "assay", "asv_id", "reads"]]

        all_asv_seqs.append(asv_seqs)
        all_reads_long.append(reads_long)

        logger.info(
            f"[Loaded] {len(asv_seqs)} {assay} ASVs, {len(reads_long)} site-ASV records"
        )

    if not all_asv_seqs:
        raise ValueError(
            "No FASTA/TSV files were processed. Please provide FASTA and TSV file pairs."
        )

    combined_asv_seqs = pd.concat(all_asv_seqs, ignore_index=True)
    combined_reads_long = pd.concat(all_reads_long, ignore_index=True)

    assay_counts = combined_asv_seqs["assay"].value_counts()
    logger.info(
        "[Combined] Total: %d unique ASVs (%s)",
        len(combined_asv_seqs),
        ", ".join([f"{count} {assay}" for assay, count in assay_counts.items()]),
    )
    logger.info(f"[Combined] Total: {len(combined_reads_long)} site-ASV records")

    return combined_asv_seqs, combined_reads_long


def process_excel_to_dataframes(
    files_by_assay: Dict[str, List[Path]],
    min_length: int = None,
    max_length: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        logger.info(f"[Processing] {len(excel_paths)} {assay} files")

        for excel_path in excel_paths:
            logger.info(f"[Processing] {assay} file: {excel_path}")

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
                logger.info(
                    f"[Filtered] {assay}: {initial_count} -> {filtered_count} ASVs "
                    f"(length filter: {min_length or 'no min'} - {max_length or 'no max'})"
                )

            asv_seqs["assay"] = assay
            asv_seqs["source_file"] = excel_path.name  # Track source file
            asv_seqs = asv_seqs[["asv_id", "assay", "sequence", "source_file"]]

            # Process reads from otuRaw sheet
            reads = pd.read_excel(excel_path, sheet_name="otuRaw")
            # some of our FAIRe sheets have this column, not all
            reads = reads.drop("Unnamed: 0", axis=1, errors="ignore")
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

            logger.info(
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
    logger.info(
        "[Combined] Total: %d unique ASVs (%s)",
        len(combined_asv_seqs),
        ", ".join([f"{count} {assay}" for assay, count in assay_counts.items()]),
    )
    logger.info(f"[Combined] Total: {len(combined_reads_long)} site-ASV records")

    return combined_asv_seqs, combined_reads_long


def write_parquet(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False)
    logger.info(f"[Saved] {path}")


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
        base_config,
        trust_remote_code=True,
        cache_dir=cache_dir,
        revision=config_revision,
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
    logger.info(
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


def weights_from_counts(counts, mode="clr", tau=3.0, eps=1e-12):
    # TODO: I think we need only CLR?
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
    weight_mode: str = "clr",
    pooling: str = "l2_weighted_mean",
    chunk_size: int = 50000,  # TODO: fiddle with this param
) -> pd.DataFrame:
    """
    Memory-efficient version that processes site embeddings in chunks.
    Optimized for systems with large amounts of RAM (500GB+).

    Args:
        reads_long: columns [site_id, assay, asv_id, reads]
        asv_emb_df: columns [asv_id, assay, dim_*]
        per_assay: whether to compute per-(site,assay) or just per-site
        weight_mode: weighting method for pooling
        pooling: pooling method
        chunk_size: number of site-assay groups to process at once

    Returns:
        DataFrame with per-(site,assay) vectors split across dim_* columns
    """
    import gc

    assert {"site_id", "assay", "asv_id", "reads"} <= set(reads_long.columns)
    dim_cols = [c for c in asv_emb_df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("asv_emb_df contains no embedding columns (dim_*)")

    # Filter out zero reads first to reduce data size
    # most sightings are 0.
    logger.info("[Memory-Efficient] Filtering zero reads...")
    reads_filtered = reads_long[reads_long["reads"] > 0].copy()
    if reads_filtered.empty:
        raise ValueError("No ASVs with reads > 0")

    logger.info(
        f"[Memory-Efficient] Filtered from {len(reads_long):,} to {len(reads_filtered):,} records"
    )

    logger.info("[Memory-Efficient] Creating ASV embedding lookup...")
    asv_lookup = {}
    embedding_dim = len(dim_cols)

    for _, row in tqdm(
        asv_emb_df.iterrows(), total=len(asv_emb_df), desc="Building lookup"
    ):
        key = (row["asv_id"], row["assay"])
        # Use float32 to save memory
        asv_lookup[key] = row[dim_cols].to_numpy(dtype=np.float32)

    logger.info(
        f"[Memory-Efficient] Created lookup for {len(asv_lookup):,} ASV embeddings (dim={embedding_dim})"
    )

    # Get unique site-assay combinations
    group_cols = ["site_id", "assay"] if per_assay else ["site_id"]
    unique_groups = reads_filtered[group_cols].drop_duplicates().reset_index(drop=True)
    total_groups = len(unique_groups)

    logger.info(
        f"[Memory-Efficient] Processing {total_groups:,} unique groups in chunks of {chunk_size:,}"
    )

    all_records = []
    processed_groups = 0

    # Process in chunks
    for chunk_start in tqdm(
        range(0, total_groups, chunk_size), desc="Processing chunks"
    ):
        chunk_end = min(chunk_start + chunk_size, total_groups)
        chunk_groups = unique_groups.iloc[chunk_start:chunk_end]

        chunk_records = []

        for _, group_row in chunk_groups.iterrows():
            if per_assay:
                site_id, assay = group_row["site_id"], group_row["assay"]
                group_filter = (reads_filtered["site_id"] == site_id) & (
                    reads_filtered["assay"] == assay
                )
            else:
                site_id = group_row["site_id"]
                group_filter = reads_filtered["site_id"] == site_id

            group_data = reads_filtered[group_filter]
            if group_data.empty:
                continue

            # Collect embeddings and counts for this group
            embeds_list = []
            counts_list = []

            for _, row in group_data.iterrows():
                key = (row["asv_id"], row["assay"])
                if key in asv_lookup:
                    embeds_list.append(asv_lookup[key])
                    counts_list.append(row["reads"])

            if not embeds_list:
                continue  # No valid embeddings for this group

            # Convert to numpy arrays
            embeds = np.stack(embeds_list, axis=0)
            counts = np.array(counts_list, dtype=np.float64)

            # Compute pooled embedding using existing functions
            if pooling == "simple_mean":
                vec = site_embed_simple_mean(embeds)
            else:
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

            # Build record
            rec = {"site_id": site_id}
            if per_assay:
                rec["assay"] = assay
            for i, val in enumerate(vec):
                rec[f"dim_{i}"] = float(val)
            chunk_records.append(rec)

        all_records.extend(chunk_records)
        processed_groups += len(chunk_groups)

        # Print progress and memory info
        if chunk_start % (chunk_size * 5) == 0:  # Every 5 chunks
            logger.info(
                f"[Memory-Efficient] Processed {processed_groups:,}/{total_groups:,} groups "
                f"({processed_groups / total_groups * 100:.1f}%)"
            )

        # Force garbage collection every 10 chunks to keep memory usage down
        if chunk_start % (chunk_size * 10) == 0:
            gc.collect()

    logger.info(
        f"[Memory-Efficient] Completed processing {len(all_records):,} site embeddings"
    )
    return pd.DataFrame(all_records)


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


def estimate_memory_usage(reads_long, asv_emb_df):
    """
    Estimate peak memory usage to help choose appropriate chunk size.
    """
    n_reads = len(reads_long[reads_long["reads"] > 0])
    n_asvs = len(asv_emb_df)
    embedding_dim = len([c for c in asv_emb_df.columns if c.startswith("dim_")])

    # Estimate memory for ASV lookup (embeddings)
    asv_lookup_gb = (n_asvs * embedding_dim * 4) / (1024**3)  # float32 = 4 bytes

    # Estimate memory for intermediate processing
    # This is rough - depends on data sparsity
    max_asvs_per_site = 1000  # Conservative estimate
    temp_memory_gb = (max_asvs_per_site * embedding_dim * 4) / (1024**3)

    total_gb = asv_lookup_gb + temp_memory_gb

    logger.info(f"[Memory Estimate] ASV lookup: {asv_lookup_gb:.2f} GB")
    logger.info(f"[Memory Estimate] Temp processing: {temp_memory_gb:.2f} GB")
    logger.info(f"[Memory Estimate] Total estimated: {total_gb:.2f} GB")

    return total_gb


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
    parser.add_argument(
        "--min-reads",
        type=int,
        default=100,
        help="Minimum read count per ASV per site to include in site embedding calculations"
    )

    # Resume functionality
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation of all steps, ignoring existing intermediate files",
    )
    parser.add_argument(
        "--diagnose-quality",
        action="store_true",
        help="Run embedding quality diagnostics to identify problematic samples"
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
        default="clr",
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

    # Logging controls
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--no-capture-stdio",
        action="store_true",
        help="Do not capture STDOUT/STDERR into log (keeps tqdm bars cleaner in console).",
    )

    # Resource monitor controls
    parser.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Track CPU/memory/GPU usage during the run and log a summary at the end.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds for resource monitoring.",
    )

    args = parser.parse_args()

    # Ensure outdir exists before logging init
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger_root = setup_logging(
        args.outdir,
        capture_stdio=(not args.no_capture_stdio),
        level=args.log_level,
    )

    # Start resource monitor (optional)
    monitor = None
    if args.monitor_resources:
        monitor = ResourceMonitor(
            interval=args.monitor_interval, include_gpu=True, logger=logger_root
        )
        monitor.start()
        logger_root.info(
            "Resource monitoring enabled (interval=%.2fs).", args.monitor_interval
        )

    try:
        # Repro
        seed_everything(args.seed)

        # Log environment/config summary
        logger_root.info("=== Pipeline Started ===")
        logger_root.info(f"Output directory: {args.outdir.resolve()}")
        logger_root.info(f"Torch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger_root.info(f"CUDA device count: {torch.cuda.device_count()}")
            try:
                logger_root.info(f"Current device: {torch.cuda.current_device()}")
            except Exception:
                pass
        # Validate input: user must provide either Excel files OR FASTA+TSV files, but not both
        has_excel = bool(args.__dict__["12s_files"] or args.__dict__["16s_files"])
        has_fasta_tsv = bool(
            args.__dict__["12s_fasta"]
            or args.__dict__["16s_fasta"]
            or args.__dict__["12s_counts"]
            or args.__dict__["16s_counts"]
        )

        # Log input names/paths
        logger_root.info("--- Input specification ---")
        if args.__dict__["12s_files"]:
            logger_root.info("12S Excel files:")
            for p in args.__dict__["12s_files"]:
                logger_root.info(f"  - {Path(p).resolve()}")
        if args.__dict__["16s_files"]:
            logger_root.info("16S Excel files:")
            for p in args.__dict__["16s_files"]:
                logger_root.info(f"  - {Path(p).resolve()}")

        for key in ["12s_fasta", "12s_counts", "16s_fasta", "16s_counts"]:
            if args.__dict__.get(key):
                logger_root.info(f"{key}: {Path(args.__dict__[key]).resolve()}")

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

            logger_root.info(f"[Loading] Processing Excel files by assay")
            asv_seqs, reads_long = process_excel_to_dataframes(
                files_by_assay,
                min_length=args.min_asv_length,
                max_length=args.max_asv_length,
            )
        else:
            # New FASTA+TSV processing
            logger_root.info(f"[Loading] Processing FASTA and TSV files")
            asv_seqs, reads_long = process_fasta_tsv_to_dataframes(
                fasta_12s=args.__dict__["12s_fasta"],
                fasta_16s=args.__dict__["16s_fasta"],
                counts_12s=args.__dict__["12s_counts"],
                counts_16s=args.__dict__["16s_counts"],
                min_length=args.min_asv_length,
                max_length=args.max_asv_length,
            )

        logger_root.info(
            f"[Loaded] ASV sequences: {len(asv_seqs)} rows, Reads: {len(reads_long)} rows"
        )

        if args.min_reads > 0:
            before = len(reads_long)
            reads_long = reads_long[reads_long["reads"] >= args.min_reads]
            logger_root.info(f"[Filter] Applied min-reads={args.min_reads}: {before:,} → {len(reads_long):,} records")

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

        logger_root.info(f"[Assays] Detected in input: {assays_available}")

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
                logger_root.info(
                    f"[Resume] Found existing ASV embeddings for: {resumable_asvs}"
                )
            if resumable_sites:
                logger_root.info(
                    f"[Resume] Found existing site embeddings for: {resumable_sites}"
                )

        # embed ASVs per assay
        asv_embeddings_paths = {}
        for assay in assays_available:
            outp = args.outdir / f"asv_embeddings_{assay}.parquet"
            asv_embeddings_paths[assay] = outp

            # Check if we can resume from existing ASV embeddings
            if not force_recalc and resume_status.get(
                "can_resume_asv_embeddings", {}
            ).get(assay, False):
                logger_root.info(
                    f"\n=== Resuming {assay} ASV embeddings from {outp} ==="
                )
                continue

            logger_root.info(f"\n=== Embedding {assay} ASVs ===")
            revision = args.revision_12s if assay == "12S" else args.revision_16s
            config_revision = args.config_revision
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
            if not force_recalc and resume_status.get(
                "can_resume_site_embeddings", {}
            ).get(assay, False):
                logger_root.info(
                    f"\n=== Resuming {assay} site embeddings from {outp} ==="
                )
                site_df = pd.read_parquet(outp)
                site_embeddings[assay] = site_df
                continue
            logger_root.info(f"\n=== Site pooling for {assay} ===")
            asv_emb = pd.read_parquet(
                asv_embeddings_paths[assay]
            )  # asv_id, assay, dim_*
            sub_reads = reads_long[reads_long["assay"] == assay]
            # memory estimation
            estimate_memory_usage(sub_reads, asv_emb)

            site_df = compute_site_embeddings_from_dfs(
                reads_long=sub_reads,
                asv_emb_df=asv_emb,
                per_assay=True,
                weight_mode=args.weight_mode,
                pooling=args.site_pooling,
            )
            write_parquet(site_df, outp)
            site_embeddings[assay] = site_df

        if args.diagnose_quality: 
            logger_root.info("=== Running Quality Diagnostics ===")
            for assay in assays_available:
                asv_emb = pd.read_parquet(asv_embeddings_paths[assay])
                sub_reads = reads_long[reads_long["assay"] == assay]
                sub_seqs = asv_seqs[asv_seqs["assay"] == assay]

                diagnostics = run_diagnostics(
                    assay, sub_seqs, asv_emb, sub_reads,
                    site_embeddings[assay], args.outdir
                )
            logger_root.info(f"Diagnostics completed for {assay}")


        # optional fusion/12S+16S concatenation
        fused_df = pd.DataFrame()
        if args.fuse == "concat" and {"12S", "16S"} <= set(site_embeddings.keys()):
            outp = args.outdir / "site_embeddings_fused.parquet"

            # Check if we can resume from existing fused embeddings
            if not force_recalc and resume_status.get("can_resume_fused", False):
                logger_root.info(f"=== Resuming fused embeddings from {outp} ===")
                fused_df = pd.read_parquet(outp)
            else:
                logger_root.info("=== Fusing 12S+16S by concatenation ===")
                s12 = site_embeddings["12S"][
                    ["site_id"]
                    + [
                        c
                        for c in site_embeddings["12S"].columns
                        if c.startswith("dim_")
                    ]
                ].set_index("site_id")
                s16 = site_embeddings["16S"][
                    ["site_id"]
                    + [
                        c
                        for c in site_embeddings["16S"].columns
                        if c.startswith("dim_")
                    ]
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
            logger_root.info(
                "\n[Info] Fusion skipped (either --fuse none or not both assays present)."
            )

        # optional t-SNE / UMAP on site vectors
        def _run_ordinations(tag: str, df: pd.DataFrame):
            if df.empty:
                logger_root.info(f"[{tag}] No data—skipping ordinations")
                return
            label_cols = ["site_id"]
            if "assay" in df.columns:
                label_cols.append("assay")

            if args.run_tsne:
                logger_root.info(
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
                logger_root.info(f"[Saved] {path}")
            if args.run_umap:
                if not HAS_UMAP:
                    logger_root.warning(
                        "[Warn] umap-learn is not installed; skipping UMAP."
                    )
                else:
                    logger_root.info(
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
                    logger_root.info(f"[Saved] {path}")

        for assay in assays_available:
            _run_ordinations(assay, site_embeddings[assay])

        if not fused_df.empty:
            fused_df2 = fused_df.copy()
            fused_df2["assay"] = "fused"
            _run_ordinations("fused", fused_df2)

        logger_root.info("[Done] Pipeline completed successfully.")

    finally:
        # Always stop monitor and log summary, even on exceptions
        if monitor:
            monitor.stop()
            monitor.log_summary(logger_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure any uncaught exceptions hit the logs
        logging.getLogger().exception("Fatal error: %s", e)
        raise
