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
        cluster_variances: (nclusters,) or (nclusters, ncols) variance per cluster (required)
        cluster_densities: (nclusters,) relative density per cluster (required)
        cluster_mins: Optional (nclusters, ncols) - per-dimension min bounds
        cluster_maxs: Optional (nclusters, ncols) - per-dimension max bounds
        cluster_means: Optional (nclusters, ncols) - actual means of clusters (for diagnostics)
        
        # Low-rank covariance parameters (for better capturing correlations)
        pca_components_list: Optional list of (k, ncols) arrays - principal directions per cluster
        pca_explained_var_list: Optional list of (k,) arrays - variance along each PC per cluster
        pca_noise_var: Optional (nclusters,) residual noise variance per cluster
    """
    nclusters: int
    ncols: int  # dimensions
    seed: int
    cluster_centers: np.ndarray  # (nclusters, ncols) - required
    cluster_variances: np.ndarray  # (nclusters,) or (nclusters, ncols) - required
    cluster_densities: np.ndarray  # (nclusters,) - required
    # Optional: extracted from real data for more realistic generation
    cluster_means: Optional[np.ndarray] = None  # (nclusters, ncols), actual cluster means
    cluster_mins: Optional[np.ndarray] = None  # (nclusters, ncols), per-dim min bounds
    cluster_maxs: Optional[np.ndarray] = None  # (nclusters, ncols), per-dim max bounds
    # Low-rank covariance (captures correlations between dimensions)
    pca_components_list: Optional[list] = None  # list of (k, ncols) or None per cluster
    pca_explained_var_list: Optional[list] = None  # list of (k,) or None per cluster
    pca_noise_var: Optional[np.ndarray] = None  # (nclusters,) residual noise variance
    # Structure scaling (to make mock data harder/more realistic)
    pca_scale: float = 1.0  # Multiplier for PCA explained variance (<1 = less structure = harder)
    outlier_fraction: float = 0.0  # Fraction of points to replace with outliers (makes data harder)
    student_df: Optional[float] = None  # Degrees of freedom for Student's t (None=Gaussian, lower=heavier tails, e.g. 5)
    
    def __post_init__(self):
        # Normalize densities to sum to 1
        self.cluster_densities = self.cluster_densities / self.cluster_densities.sum()
    
    @property
    def is_lowrank(self) -> bool:
        """Check if low-rank covariance is available."""
        return self.pca_components_list is not None
    

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


def load_cluster_stats(filepath: str) -> dict:
    """
    Load cluster statistics from a .npz file.
    
    Returns dict with:
        - centroids, variances, densities: Always present
        - variances_per_dim, mins_per_dim, maxs_per_dim, means_per_dim: 
          Present if saved with new format, None otherwise
        - pca_components_list, pca_explained_var_list, pca_noise_var: 
          Present if low-rank extraction was used
        - is_lowrank: True if PCA components are available
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Check if low-rank (PCA) data is available
    has_pca = 'pca_components_list' in data
    
    result = {
        'centroids': data['centroids'],
        'variances': data['variances'],
        'densities': data['densities'],
        # New fields - None if not present (backward compat)
        'variances_per_dim': data['variances_per_dim'] if 'variances_per_dim' in data else None,
        'mins_per_dim': data['mins_per_dim'] if 'mins_per_dim' in data else None,
        'maxs_per_dim': data['maxs_per_dim'] if 'maxs_per_dim' in data else None,
        'means_per_dim': data['means_per_dim'] if 'means_per_dim' in data else None,
        # Low-rank (PCA) fields
        'is_lowrank': has_pca,
        'pca_components_list': list(data['pca_components_list']) if has_pca else None,
        'pca_explained_var_list': list(data['pca_explained_var_list']) if has_pca else None,
        'pca_noise_var': data['pca_noise_var'] if has_pca else None,
    }
    
    return result


def get_cluster_config(config: BenchmarkConfig) -> ClusterConfig:
    """
    Create ClusterConfig from BenchmarkConfig by loading pre-extracted cluster stats.
    
    cluster_stats_path must be provided - this loads centroids, variances, densities
    from a .npz file created by extract_values.py.
    """
    if config.cluster_stats_path is None:
        raise ValueError("cluster_stats_path is required. Use extract_values.py to create cluster stats from your dataset.")
    
    print(f"Loading cluster stats from {config.cluster_stats_path}...")
    stats = load_cluster_stats(config.cluster_stats_path)
    
    centroids = stats['centroids']
    densities = stats['densities']
    # Use per-dimension variances if available, else scalar variances
    variances = stats['variances_per_dim'] if stats['variances_per_dim'] is not None else stats['variances']
    mins = stats['mins_per_dim']
    maxs = stats['maxs_per_dim']
    means = stats['means_per_dim']
    
    n_clusters = len(centroids)
    n_cols = centroids.shape[1]
    
    print(f"  Loaded {n_clusters} clusters, {n_cols} dimensions")
    
    return ClusterConfig(
        nclusters=n_clusters,
        ncols=n_cols,
        seed=config.seed,
        cluster_centers=centroids,
        cluster_variances=variances,
        cluster_densities=densities,
        cluster_mins=mins,
        cluster_maxs=maxs,
        cluster_means=means,
    )
