import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import argparse
# Assuming your data is in a DataFrame called 'df'

parser = argparse.ArgumentParser(
    description="Calculates embeddings for a single site from a parquet file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Excel input options (original)
parser.add_argument(
    "--site_parquet",
    type=Path,
    default=[],
    help="Path to Parquet file containing per-ASV embeddings for a site",
)

parser.add_argument(
    "--meta_csv",
    type=Path,
    default=[],
    help="Path to big CSV file with all metadata and taxonomic labels - will extract family",
)
parser.add_argument(
    "--output",
    type=Path,
    default=[],
    help="Path for the resulting tSNE dimensions CSV file",
)


args = parser.parse_args()

df = pd.read_parquet(args.site_parquet)

feature_cols = [col for col in df.columns if col.startswith('dim_')]
X = df[feature_cols].values
metadata = df[['asv_id', 'assay']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=5,
    learning_rate='auto',     
    n_iter=1000,
    random_state=42,
    verbose=1
)

X_tsne = tsne.fit_transform(X_scaled)

tsne_results = pd.DataFrame({
    'tsne_1': X_tsne[:, 0],
    'tsne_2': X_tsne[:, 1],
    'asv_id': metadata['asv_id'],
    'assay': metadata['assay']
})

#tsne_results.to_csv(args.output)

# now we add the taxonomic labels
taxa = pd.read_csv(args.meta_csv, sep='\t')
print(taxa)
print(tsne_results)
pd.merge(tsne_results, taxa, how = 'left', on = ['asv_id', 'assay']).to_csv(args.output)
