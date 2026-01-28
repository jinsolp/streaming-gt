"""
Configuration Classes for Scalable GT Generation and ANN Benchmarking

Contains:
- ClusterConfig: Configuration for cluster-based data generation
- BenchmarkConfig: Configuration for streaming ANN benchmarks
"""

import numpy as np
from typing import Optional, Literal, Union, Any
from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    """
    Configuration for cluster-based data generation using pre-extracted cluster statistics.
    
    All cluster information (centers, variances, densities) should be loaded from
    real dataset statistics extracted via extract_values.py.
    
    Parameters:
        cluster_centers: (nclusters, ncols) cluster centroids (required)
        cluster_variances: (nclusters, ncols) per-dimension variance per cluster (required)
        cluster_densities: (nclusters,) relative density per cluster (required)
    
        # Low-rank covariance parameters (for better capturing correlations)
        pca_components_list: Optional list of (k, ncols) arrays - principal directions per cluster
        pca_explained_var_list: Optional list of (k,) arrays - variance along each PC per cluster
        pca_noise_var: Optional (nclusters,) residual noise variance per cluster
    """
    nclusters: int
    ncols: int  # dimensions
    seed: int
    cluster_centers: np.ndarray  # (nclusters, ncols) - required
    cluster_variances: np.ndarray  # (nclusters, ncols) per-dim variance - required
    cluster_densities: np.ndarray  # (nclusters,) - required

    # Low-rank covariance (captures correlations between dimensions)
    pca_components_list: Optional[list] = None  # list of (k, ncols) or None per cluster
    pca_explained_var_list: Optional[list] = None  # list of (k,) or None per cluster
    pca_noise_var: Optional[np.ndarray] = None  # (nclusters,) residual noise variance
    
    def __post_init__(self):
        # Normalize densities to sum to 1
        self.cluster_densities = self.cluster_densities / self.cluster_densities.sum()
    

@dataclass
class BenchmarkConfig:
    """Configuration for streaming ANN benchmark."""
    # Dataset parameters
    total_rows: int
    seed: int = 42
    batch_size: int = 100_000
    
    # Query/GT parameters
    nqueries: int = 1000
    k: int = 10
    nprobes: int = 10  # Number of nearby clusters to probe for GT computation
    gt_backend: Literal["cuvs", "sklearn"] = "cuvs"
    
    # Pre-extracted cluster statistics (from extract_values.py) - required
    cluster_stats_path: str = None
    
    # ANN index (CagraIndex or IvfPqIndex from ann_indices.py)
    # Pass a pre-configured index instance
    ann_index: Any = None  # type: Union[CagraIndex, IvfPqIndex]


def save_cluster_stats(filepath: str, stats: dict):
    """
    Save cluster statistics to a .npz file.
    
    Args:
        filepath: Output path for .npz file
        stats: Dict from extract_cluster_stats containing:
            - centroids, densities (required)
            - variances_per_dim (per-dimension variance)
    """
    np.savez(
        filepath,
        centroids=stats['centroids'],
        densities=stats['densities'],
        variances_per_dim=stats['variances_per_dim'],
    )
    print(f"Saved cluster statistics to {filepath}")


def load_cluster_stats(filepath: str) -> dict:
    """
    Load cluster statistics from a .npz file.
    
    Returns dict with:
        - centroids, densities: Always present
        - variances_per_dim: Per-dimension variance (n_clusters, n_dim)
    """
    data = np.load(filepath)
    
    result = {
        'centroids': data['centroids'],
        'densities': data['densities'],
        'variances_per_dim': data['variances_per_dim'],
    }
    
    return result


def get_cluster_config(config: BenchmarkConfig) -> ClusterConfig:
    """
    Create ClusterConfig from BenchmarkConfig by loading pre-extracted cluster stats.
    
    cluster_stats_path must be provided - this loads centroids, variances_per_dim, densities
    from a .npz file created by extract_values.py.
    """
    if config.cluster_stats_path is None:
        raise ValueError("cluster_stats_path is required. Use extract_values.py to create cluster stats from your dataset.")
    
    print(f"Loading cluster stats from {config.cluster_stats_path}...")
    stats = load_cluster_stats(config.cluster_stats_path)
    
    centroids = stats['centroids']
    densities = stats['densities']
    variances_per_dim = stats['variances_per_dim']
    
    n_clusters = len(centroids)
    n_cols = centroids.shape[1]
    
    print(f"  Loaded {n_clusters} clusters, {n_cols} dimensions")
    
    return ClusterConfig(
        nclusters=n_clusters,
        ncols=n_cols,
        seed=config.seed,
        cluster_centers=centroids,
        cluster_variances=variances_per_dim,
        cluster_densities=densities,
    )
