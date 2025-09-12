from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import calculatePerSite as mod

def test_clean_seq_basic():
    assert mod.clean_seq("acgtn") == "ACGTN"
    assert mod.clean_seq("acgtxyz") == "ACGTNNN"
    assert mod.clean_seq("") == ""
    assert mod.clean_seq(None) == ""


def test_parse_fasta_and_fasta_tsv_processing(tmp_path: Path):
    # Create tiny FASTA for 12S with 2 sequences
    fasta12 = tmp_path / "12s.fasta"
    fasta12.write_text(">asv1\nACGTACGT\n>asv2\nACGTN\n")

    # Create counts TSV for 12S (ASVs as rows, sites as columns)
    counts12 = tmp_path / "12s_counts.tsv"
    df_counts = pd.DataFrame({"SiteA": [10, 0], "SiteB": [0, 5]}, index=["asv1", "asv2"])
    df_counts.to_csv(counts12, sep="\t")

    # Process with min length filter to drop asv2 (length 5)
    asv_df, reads_df = mod.process_fasta_tsv_to_dataframes(
        fasta_12s=fasta12,
        counts_12s=counts12,
        fasta_16s=None,
        counts_16s=None,
        min_length=6,
        max_length=None,
    )
    # asv2 should be filtered out
    assert set(asv_df["asv_id"]) == {"asv1"}
    # reads_long should only include asv1 rows
    assert set(reads_df["asv_id"]) == {"asv1"}
    assert set(reads_df["site_id"]) == {"SiteA", "SiteB"}


def test_process_excel_to_dataframes(tmp_path: Path):
    # Build an Excel file for one assay with required sheets/columns
    xlsx = tmp_path / "assay_12s.xlsx"

    # Write taxaRaw with two dummy rows BEFORE the header (to match skiprows=2)
    # Row1: dummy | Row2: dummy | Row3: header | Rows 4..: data
    from openpyxl import Workbook
    wb = Workbook()

    ws = wb.active
    ws.title = "taxaRaw"
    ws.append(["dummy", "dummy"])
    ws.append(["dummy", "dummy"])
    ws.append(["seq_id", "dna_sequence"])   # header row that pandas will use after skiprows=2
    ws.append(["asv1", "ACGTAC"])
    ws.append(["asv2", "ACGT"])
    ws.append(["asv2", "ACGT"])             # duplicate to test drop_duplicates

    # otuRaw sheet — IMPORTANT: include the "Unnamed: 0" column
    ws2 = wb.create_sheet("otuRaw")
    ws2.append(["Unnamed: 0", "ASV", "ASV_sequence", "Site1", "Site2"])
    ws2.append([0, "asv1", "ACGTAC", 7, 0])
    ws2.append([1, "asv2", "ACGT", 0, 4])

    wb.save(xlsx)

    asv_df, reads_df = mod.process_excel_to_dataframes(
        files_by_assay={"12S": [xlsx]}, min_length=4, max_length=None
    )

    # Both ASVs present (after filtering min_length=4)
    assert set(asv_df["asv_id"]) == {"asv1", "asv2"}
    # reads_long filtered to ASVs present and melted correctly
    assert set(reads_df["asv_id"]) == {"asv1", "asv2"}
    assert set(reads_df["site_id"]) == {"Site1", "Site2"}
    # Ensure assay column is set
    assert set(asv_df["assay"]) == {"12S"}
    assert set(reads_df["assay"]) == {"12S"}


def test_weights_from_counts_modes_sum_to_one():
    counts = np.array([10, 0, 5, 5])
    for mode in ["relative", "log", "hellinger", "clr", "softmax_tau3"]:
        w = mod.weights_from_counts(counts, mode=mode)
        assert np.all(w >= 0)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-6, atol=1e-6)


def test_site_embedding_pooling_simple_and_weighted():
    # Two ASVs, 3 dims
    embeds = np.array(
        [
            [1.0, 2.0, 3.0],  # asv1
            [3.0, 2.0, 1.0],  # asv2
        ],
        dtype=np.float32,
    )

    counts = np.array([9, 1], dtype=float)
    w_hell = mod.weights_from_counts(counts, mode="hellinger")  # prefers first

    # simple mean
    simple = mod.site_embed_simple_mean(embeds)
    np.testing.assert_allclose(simple, np.array([2.0, 2.0, 2.0], dtype=np.float32))

    # weighted mean with hellinger weights:
    # dim0 (1 vs 3): weighting toward 1 pulls BELOW 2
    # dim2 (3 vs 1): weighting toward 3 pushes ABOVE 2
    wmean = mod.site_embed_weighted_mean(embeds, w_hell)
    assert wmean[0] < 2.0 and wmean[2] > 2.0  # corrected directional check

    # l2 weighted mean returns finite result
    l2w = mod.site_embed_l2_weighted_mean(embeds, w_hell)
    assert np.isfinite(l2w).all()


def test_compute_site_embeddings_from_dfs_simple_mean():
    # Build a tiny asv_emb_df with 2 ASVs (2 dims) for 1 assay
    asv_emb_df = pd.DataFrame(
        {
            "asv_id": ["a1", "a2"],
            "assay": ["12S", "12S"],
            "dim_0": [1.0, 3.0],
            "dim_1": [2.0, 4.0],
        }
    )

    # reads_long for two sites; ensure some zeros get filtered
    reads_long = pd.DataFrame(
        {
            "site_id": ["S1", "S1", "S2", "S2"],
            "assay": ["12S", "12S", "12S", "12S"],
            "asv_id": ["a1", "a2", "a1", "a2"],
            "reads": [5, 5, 10, 0],  # S2 should only use a1
        }
    )

    site_df = mod.compute_site_embeddings_from_dfs(
        reads_long=reads_long,
        asv_emb_df=asv_emb_df,
        per_assay=True,
        weight_mode="hellinger",
        pooling="simple_mean",  # test simple path
        chunk_size=1,  # tiny to stress chunking pathway
    )

    # Expect two rows (S1 & S2) with dim columns
    assert set(site_df["site_id"]) == {"S1", "S2"}
    dim_cols = [c for c in site_df.columns if c.startswith("dim_")]
    assert len(dim_cols) == 2

    # For S1 simple mean of [[1,2],[3,4]] -> [2,3]
    s1 = (
        site_df.set_index("site_id")
        .loc["S1", dim_cols]
        .astype(float)
        .to_numpy(dtype=float)
    )
    np.testing.assert_allclose(s1, np.array([2.0, 3.0]), rtol=1e-6, atol=1e-6)

    # For S2 only a1 present -> [1,2]
    s2 = (
        site_df.set_index("site_id")
        .loc["S2", dim_cols]
        .astype(float)
        .to_numpy(dtype=float)
    )
    np.testing.assert_allclose(s2, np.array([1.0, 2.0]), rtol=1e-6, atol=1e-6)


# --- Dummy tokenizer/model to unit test embed_sequences without HuggingFace ---

class DummyTokenizer:
    def __call__(self, batch, return_tensors="pt", padding=True, truncation=True, max_length=512):
        B = len(batch)
        L = 5
        input_ids = torch.ones((B, L), dtype=torch.long)
        attention_mask = torch.ones((B, L), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyOut:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyModel(torch.nn.Module):
    def forward(self, **kwargs):
        # Produce per-sample constant-hidden states of shape (B, L, D)
        input_ids = kwargs["input_ids"]
        B, L = input_ids.shape
        D = 8
        # sample i has all values == i (easy to check pooling)
        hidden = torch.stack([torch.full((L, D), float(i)) for i in range(B)], dim=0)
        return _DummyOut(hidden)


def test_embed_sequences_mean_and_cls():
    df = pd.DataFrame({"asv_id": ["asvA", "asvB"], "sequence": ["ACGT", "ACGTN"]})
    tok = DummyTokenizer()
    mdl = DummyModel()
    device = torch.device("cpu")

    # Use batch_size=2 so both rows are processed together (i=0,1)
    emb_mean = mod.embed_sequences(
        df, tok, mdl, device, pooling="mean", batch_size=2, use_amp=False, max_length=32
    )
    dim_cols = [c for c in emb_mean.columns if c.startswith("dim_")]
    v0 = emb_mean.iloc[0][dim_cols].astype(float).to_numpy(dtype=float)
    v1 = emb_mean.iloc[1][dim_cols].astype(float).to_numpy(dtype=float)
    np.testing.assert_allclose(v0, np.zeros(8))
    np.testing.assert_allclose(v1, np.ones(8))

    emb_cls = mod.embed_sequences(
        df, tok, mdl, device, pooling="cls", batch_size=2, use_amp=False, max_length=32
    )
    v0 = emb_cls.iloc[0][dim_cols].astype(float).to_numpy(dtype=float)
    v1 = emb_cls.iloc[1][dim_cols].astype(float).to_numpy(dtype=float)
    np.testing.assert_allclose(v0, np.zeros(8))
    np.testing.assert_allclose(v1, np.ones(8))


def test_run_tsne_shape():
    # Build a minimal site_df with 5 rows and 3 dims
    site_df = pd.DataFrame(
        {
            "site_id": [f"S{i}" for i in range(5)],
            "assay": ["12S"] * 5,
            "dim_0": np.linspace(0, 1, 5),
            "dim_1": np.linspace(1, 2, 5),
            "dim_2": np.linspace(-1, 0, 5),
        }
    )
    # perplexity must be strictly less than n_samples (5)
    out = mod.run_tsne(site_df, label_cols=["site_id", "assay"], metric="cosine", perplexity=4, random_state=0)
    assert set(["site_id", "assay", "tsne_x", "tsne_y"]).issubset(out.columns)
    assert len(out) == 5


@pytest.mark.skipif(not getattr(mod, "HAS_UMAP", False), reason="umap-learn not installed")
def test_run_umap_shape():
    site_df = pd.DataFrame(
        {
            "site_id": [f"S{i}" for i in range(5)],
            "assay": ["12S"] * 5,
            "dim_0": np.linspace(0, 1, 5),
            "dim_1": np.linspace(1, 2, 5),
        }
    )
    out = mod.run_umap(site_df, label_cols=["site_id", "assay"], metric="cosine", n_neighbors=3, random_state=0)
    assert set(["site_id", "assay", "umap_x", "umap_y"]).issubset(out.columns)
    assert len(out) == 5


def test_check_intermediate_files(tmp_path: Path):
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    # Touch files to simulate existing intermediate artifacts
    (outdir / "asv_embeddings_12S.parquet").touch()
    (outdir / "site_embeddings_12S.parquet").touch()
    (outdir / "site_embeddings_fused.parquet").touch()

    status = mod.check_intermediate_files(outdir, ["12S"])
    assert status["can_resume_asv_embeddings"]["12S"] is True
    assert status["can_resume_site_embeddings"]["12S"] is True
    assert status["can_resume_fused"] is True


def test_estimate_memory_usage_logs_and_returns_number():
    # build toy inputs
    asv_emb_df = pd.DataFrame(
        {"asv_id": ["a"], "assay": ["12S"], "dim_0": [0.1], "dim_1": [0.2]}
    )
    reads_long = pd.DataFrame(
        {"site_id": ["S1"], "assay": ["12S"], "asv_id": ["a"], "reads": [1]}
    )
    total_gb = mod.estimate_memory_usage(reads_long, asv_emb_df)
    assert isinstance(total_gb, float)
    assert total_gb >= 0.0

