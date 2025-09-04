# Calculate per-site embeddings using 12S/16S finetuned DNABERT-S


This script takes a FAIRe-formatted Excel sheet with ASV results of 12S (MiFishU, Miya et al) or 16S sequencing (Berry et al) ASVs, downloads the corresponding finetuned OceanOmics DNABERT-S models, and averages all embeddings for each site by weighing the embeddings using per-site read counts of all ASVs.

It then stores the per-site embeddings in a paraquet file, and optionally runs tSNE or UMAP on those embeddings to get per-site representations.

This is what the tSNE clustering of those per-site embeddings looks like:

<img width="1024" height="653" alt="image" src="https://github.com/user-attachments/assets/dcbb4415-f25d-46f7-a10b-d9c547ec437e" />

Each dot is one site, colored by their latitude/longitude.

# Usage

    python calculate.py --cache-dir './cache' --asv-seqs asv_seqs.12S.parquet 
       --reads asv_reads.12S.parquet --run-tsne   --outdir './results' --run-umap


<pre>
usage: calculatePerSite.py [-h] --excel-file EXCEL_FILE [--asv-seqs ASV_SEQS] [--reads READS] --outdir OUTDIR [--cache-dir CACHE_DIR]
                           [--model-12s MODEL_12S] [--model-16s MODEL_16S] [--base-config BASE_CONFIG] [--pooling-token {mean,cls}]
                           [--batch-size BATCH_SIZE] [--use-amp] [--max-length MAX_LENGTH] [--weight-mode {hellinger,log,relative,softmax_tau3}]
                           [--site-pooling {l2_weighted_mean,weighted_mean,gem_p2,gem_p3}] [--run-tsne] [--run-umap] [--perplexity PERPLEXITY]
                           [--n-neighbors N_NEIGHBORS] [--metric {cosine,euclidean}] [--seed SEED] [--fuse {none,concat}]

eDNA DNABERT-S embedding pipeline (Excel -> ASVs -> sites -> t-SNE/UMAP)

optional arguments:
  -h, --help            show this help message and exit
  --excel-file EXCEL_FILE
                        Path to FAIRe-formatted Excel file with taxaRaw and otuRaw sheets
  --asv-seqs ASV_SEQS   Optional: Path to asv_sequences.[csv|parquet] (columns: asv_id, assay, sequence)
  --reads READS         Optional: Path to reads_long.[csv|parquet] (columns: site_id, assay, asv_id, reads)
  --outdir OUTDIR       Output directory
  --cache-dir CACHE_DIR
                        HuggingFace cache dir (optional)
  --model-12s MODEL_12S
  --model-16s MODEL_16S
  --base-config BASE_CONFIG
  --pooling-token {mean,cls}
  --batch-size BATCH_SIZE
  --use-amp             Enable mixed precision on CUDA
  --max-length MAX_LENGTH
  --weight-mode {hellinger,log,relative,softmax_tau3}
  --site-pooling {l2_weighted_mean,weighted_mean,gem_p2,gem_p3}
  --run-tsne
  --run-umap
  --perplexity PERPLEXITY
  --n-neighbors N_NEIGHBORS
  --metric {cosine,euclidean}
  --seed SEED
  --fuse {none,concat}  How to fuse 12S+16S site vectors (concat or none)
</pre>

# Installation

There's a conda environment with the DNABERT-S needed dependencies in DNABERT_S.yml

    conda env create -f DNABERT_S.yml

# Making your own paraquet file

The format of the input paraquet files is fairly easy (see the help), there's a script in `util/makeParquetFiles.py` with some example code.

# Example data

This repository comes with a FAIRe-formatted excel sheet from an OceanOmics transect from Perth to the Cocos Keeling Islands, only 12S (MiFish-U, Miya et al.)

OceanOmics - Computational Biology\Bioinformatics\Results\V10_CKIP1\results_MiFishU_curateddb\07-faire
