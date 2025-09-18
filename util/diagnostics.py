#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_embedding_quality(asv_seqs, asv_embeddings, reads_long, site_embeddings):
    """
    Comprehensive diagnostic pipeline to identify problematic samples
    """
    diagnostics = {}
    
    print("=== Analyzing Sequence Quality ===")
    seq_issues = analyse_sequence_quality(asv_seqs)
    diagnostics['problematic_sequences'] = seq_issues
    print(f"Found {len(seq_issues)} potentially problematic sequences")
    
    print("=== Detecting Embedding Outliers ===")
    embedding_outliers = detect_embedding_outliers(asv_embeddings)
    diagnostics['embedding_outliers'] = embedding_outliers
    print(f"Found {len(embedding_outliers)} embedding outliers")
    
    print("=== Analyzing Site Characteristics ===")
    site_diagnostics = analyse_site_characteristics(reads_long, site_embeddings)
    diagnostics['site_characteristics'] = site_diagnostics
    
    print("=== Cross-validating Issues ===")
    problematic_asvs = set(seq_issues['asv_id']).union(set(embedding_outliers['asv_id']))
    problematic_sites = identify_problematic_sites(reads_long, problematic_asvs, site_diagnostics)
    diagnostics['problematic_sites'] = problematic_sites
    
    return diagnostics

def analyse_sequence_quality(asv_seqs):
    """Identify sequences with quality issues"""
    df = asv_seqs.copy()
    
    df['length'] = df['sequence'].str.len()
    df['n_count'] = df['sequence'].str.count('N')
    df['n_fraction'] = df['n_count'] / df['length']
    df['gc_content'] = (df['sequence'].str.count('G') + df['sequence'].str.count('C')) / df['length']
    
    df['max_homopolymer'] = df['sequence'].apply(find_max_homopolymer)
    
    issues = (
        (df['n_fraction'] > 0.1) |
        (df['length'] < 50) |   
        (df['length'] > 300) |   
        (df['gc_content'] < 0.2) | 
        (df['gc_content'] > 0.8) |
        (df['max_homopolymer'] > 20)
    )
    
    problematic = df[issues].copy()
    
    problematic['issues'] = ''
    problematic.loc[problematic['n_fraction'] > 0.1, 'issues'] += 'high_N_content;'
    problematic.loc[problematic['length'] < 50, 'issues'] += 'too_short;'
    problematic.loc[problematic['length'] > 500, 'issues'] += 'too_long;'
    problematic.loc[problematic['gc_content'] < 0.2, 'issues'] += 'extreme_AT;'
    problematic.loc[problematic['gc_content'] > 0.8, 'issues'] += 'extreme_GC;'
    problematic.loc[problematic['max_homopolymer'] > 20, 'issues'] += 'long_homopolymer;'
    
    return problematic

def find_max_homopolymer(sequence):
    """Find the longest homopolymer run in a sequence"""
    if not sequence:
        return 0
    
    max_run = 1
    current_run = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    return max_run

def detect_embedding_outliers(asv_emb_df, contamination=0.05):
    """Detect outlier embeddings using multiple methods"""
    dim_cols = [c for c in asv_emb_df.columns if c.startswith('dim_')]
    X = asv_emb_df[dim_cols].values
    
    outliers_mask = np.zeros(len(asv_emb_df), dtype=bool)
    
    # Method 1: Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_outliers = iso_forest.fit_predict(X) == -1
    
    # Method 2: Statistical outliers (Z-score > 3 in any dimension)
    z_scores = np.abs(stats.zscore(X, axis=0))
    stat_outliers = (z_scores > 4).any(axis=1)
    
    # Method 3: Mahalanobis distance outliers
    try:
        cov_inv = np.linalg.pinv(np.cov(X.T))
        mean = np.mean(X, axis=0)
        mahal_dist = np.array([
            np.sqrt((x - mean).T @ cov_inv @ (x - mean)) 
            for x in X
        ])
        mahal_threshold = np.percentile(mahal_dist, 95)
        mahal_outliers = mahal_dist > mahal_threshold
    except:
        mahal_outliers = np.zeros(len(X), dtype=bool)
    
    norms = np.linalg.norm(X, axis=1)
    norm_outliers = (norms < np.percentile(norms, 2)) | (norms > np.percentile(norms, 98))
    
    outliers_mask = iso_outliers | stat_outliers | mahal_outliers | norm_outliers
    
    result = asv_emb_df[outliers_mask].copy()
    result['outlier_methods'] = ''
    result.loc[result.index[iso_outliers[outliers_mask]], 'outlier_methods'] += 'isolation_forest;'
    result.loc[result.index[stat_outliers[outliers_mask]], 'outlier_methods'] += 'statistical;'
    result.loc[result.index[mahal_outliers[outliers_mask]], 'outlier_methods'] += 'mahalanobis;'
    result.loc[result.index[norm_outliers[outliers_mask]], 'outlier_methods'] += 'norm_based;'
    
    return result

def analyse_site_characteristics(reads_long, site_embeddings):
    """Analyse characteristics of each site"""
    site_stats = []
    
    for site_id in reads_long['site_id'].unique():
        site_data = reads_long[reads_long['site_id'] == site_id]
        
        total_reads = site_data['reads'].sum()
        n_asvs = len(site_data[site_data['reads'] > 0])
        
        if total_reads > 0:
            rel_abundance = site_data['reads'] / total_reads
            shannon_diversity = -np.sum(rel_abundance * np.log(rel_abundance + 1e-10))
            simpson_diversity = 1 - np.sum(rel_abundance ** 2)
            evenness = shannon_diversity / np.log(n_asvs) if n_asvs > 1 else 0
            dominant_fraction = rel_abundance.max()
        else:
            shannon_diversity = simpson_diversity = evenness = dominant_fraction = 0
        
        site_stats.append({
            'site_id': site_id,
            'total_reads': total_reads,
            'n_asvs': n_asvs,
            'shannon_diversity': shannon_diversity,
            'simpson_diversity': simpson_diversity,
            'evenness': evenness,
            'dominant_fraction': dominant_fraction
        })
    
    site_stats_df = pd.DataFrame(site_stats)
    
    unusual_sites = (
        (site_stats_df['total_reads'] < 100) | 
        (site_stats_df['n_asvs'] < 3) |       
        (site_stats_df['dominant_fraction'] > 0.9) | 
        (site_stats_df['evenness'] < 0.1)
    )
    
    return site_stats_df, site_stats_df[unusual_sites]

def identify_problematic_sites(reads_long, problematic_asvs, site_diagnostics):
    """Identify sites heavily influenced by problematic ASVs"""
    site_stats, unusual_sites = site_diagnostics
    
    problematic_sites = []
    
    for site_id in reads_long['site_id'].unique():
        site_data = reads_long[reads_long['site_id'] == site_id]
        
        problematic_reads = site_data[site_data['asv_id'].isin(problematic_asvs)]['reads'].sum()
        total_reads = site_data['reads'].sum()
        
        if total_reads > 0:
            problematic_fraction = problematic_reads / total_reads
            
            if problematic_fraction > 0.3:  #  TODO: good cutoff?
                prob_asv_details = site_data[site_data['asv_id'].isin(problematic_asvs)].copy()
                prob_asv_details['rel_abundance'] = prob_asv_details['reads'] / total_reads
                
                top_prob_asvs = prob_asv_details.nlargest(3, 'reads')['asv_id'].tolist()
                
                reason_parts = [
                    f"problematic_reads_{problematic_fraction:.1%}",
                    f"n_prob_asvs_{len(prob_asv_details)}",
                    f"top_prob_asvs_{'-'.join(top_prob_asvs[:2])}"  # Show top 2 ASV IDs
                ]
                
                problematic_sites.append({
                    'site_id': site_id,
                    'problematic_fraction': problematic_fraction,
                    'total_reads': total_reads,
                    'n_problematic_asvs': len(prob_asv_details),
                    'top_problematic_asvs': ';'.join(top_prob_asvs),
                    'reason': '|'.join(reason_parts)
                })
    
    for _, site in unusual_sites.iterrows():
        site_id = site['site_id']
        
        issues = []
        # TODO: check if these numbers make sense
        if site['total_reads'] < 100:
            issues.append(f"low_reads_{site['total_reads']}")
        if site['n_asvs'] < 3:
            issues.append(f"low_diversity_{site['n_asvs']}_asvs")
        if site['dominant_fraction'] > 0.9:
            issues.append(f"single_dominant_{site['dominant_fraction']:.2f}")
        if site['evenness'] < 0.1:
            issues.append(f"low_evenness_{site['evenness']:.2f}")
        
        problematic_sites.append({
            'site_id': site_id,
            'problematic_fraction': 0,
            'total_reads': int(site['total_reads']),
            'n_problematic_asvs': 0,
            'top_problematic_asvs': '',
            'reason': '|'.join(issues) if issues else 'unusual_diversity_metrics'
        })
    
    return pd.DataFrame(problematic_sites)

def visualize_diagnostics(diagnostics, output_dir):
    """Create diagnostic visualizations"""
    
    asv_embeddings = diagnostics.get('embedding_outliers', pd.DataFrame())
    if not asv_embeddings.empty:
        dim_cols = [c for c in asv_embeddings.columns if c.startswith('dim_')]
        if len(dim_cols) > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(asv_embeddings[dim_cols])
            
            plt.figure(figsize=(10, 8))
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('Embedding Outliers in PCA Space')
            plt.savefig(f'{output_dir}/embedding_outliers_pca.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    site_stats, _ = diagnostics.get('site_characteristics', (pd.DataFrame(), pd.DataFrame()))
    if not site_stats.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        site_stats['shannon_diversity'].hist(ax=axes[0,0], bins=30)
        axes[0,0].set_title('Shannon Diversity Distribution')
        
        site_stats['dominant_fraction'].hist(ax=axes[0,1], bins=30)
        axes[0,1].set_title('Dominant ASV Fraction')
        
        site_stats['total_reads'].hist(ax=axes[1,0], bins=30, log=True)
        axes[1,0].set_title('Total Reads (log scale)')
        
        site_stats['n_asvs'].hist(ax=axes[1,1], bins=30)
        axes[1,1].set_title('Number of ASVs per Site')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/site_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_diagnostics(assay, asv_seqs, asv_embeddings, reads_long, site_embeddings, output_dir):
    """Run the complete diagnostic pipeline"""
    
    print(f"Running embedding quality diagnostics for {assay}...")
    diagnostics = diagnose_embedding_quality(asv_seqs, asv_embeddings, reads_long, site_embeddings)
    
    visualize_diagnostics(diagnostics, output_dir)
    
    for key, df in diagnostics.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(f'{output_dir}/{assay}_diagnostics_{key}.csv', index=False)
            print(f"Saved {key} diagnostics: {len(df)} entries")
    
    return diagnostics
