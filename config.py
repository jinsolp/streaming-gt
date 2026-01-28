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
    
    @property
    def is_lowrank(self) -> bool:
        """Check if low-rank covariance (PCA) is available."""
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


def save_cluster_stats(filepath: str, stats: dict):
    """
    Save cluster statistics to a .npz file.
    
    Handles both basic stats and lowrank (PCA) stats automatically.
    
    Args:
        filepath: Output path for .npz file
        stats: Dict from extract_cluster_stats or extract_lowrank_cluster_stats containing:
            - centroids, densities, variances_per_dim (required)
            - pca_components_list, pca_explained_var_list, pca_noise_var, pca_n_components (optional, lowrank)
    """
    # Check if this is lowrank stats
    is_lowrank = 'pca_components_list' in stats and stats['pca_components_list'] is not None
    
    if is_lowrank:
        # Save lowrank stats with special handling for variable-length arrays
        n_clusters = len(stats['centroids'])
        
        # Convert lists to object arrays for npz storage
        # Handle None entries by storing empty arrays
        pca_components_list = stats['pca_components_list']
        pca_explained_var_list = stats['pca_explained_var_list']
        
        # Store as object arrays (allows variable-length arrays per cluster)
        pca_components_arr = np.empty(n_clusters, dtype=object)
        pca_explained_var_arr = np.empty(n_clusters, dtype=object)
        
        for i in range(n_clusters):
            if pca_components_list[i] is not None:
                pca_components_arr[i] = pca_components_list[i]
                pca_explained_var_arr[i] = pca_explained_var_list[i]
            else:
                # Store empty arrays for None (fallback clusters)
                pca_components_arr[i] = np.array([], dtype=np.float32)
                pca_explained_var_arr[i] = np.array([], dtype=np.float32)
        
        np.savez(
            filepath,
            centroids=stats['centroids'],
            densities=stats['densities'],
            variances_per_dim=stats['variances_per_dim'],
            # Lowrank specific
            is_lowrank=np.array([True]),
            pca_components_arr=pca_components_arr,
            pca_explained_var_arr=pca_explained_var_arr,
            pca_noise_var=stats['pca_noise_var'],
            pca_n_components=np.array([stats['pca_n_components']]),
        )
        print(f"Saved lowrank cluster statistics to {filepath}")
    else:
        # Save basic stats
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
    
    Handles both basic stats and lowrank (PCA) stats automatically.
    
    Returns dict with:
        - centroids, densities, variances_per_dim: Always present
        - pca_components_list, pca_explained_var_list, pca_noise_var, pca_n_components: Present if lowrank
    """
    data = np.load(filepath, allow_pickle=True)
    
    result = {
        'centroids': data['centroids'],
        'densities': data['densities'],
        'variances_per_dim': data['variances_per_dim'],
    }
    
    # Check if this is lowrank stats
    if 'is_lowrank' in data and data['is_lowrank'][0]:
        n_clusters = len(result['centroids'])
        
        # Convert object arrays back to lists, handling empty arrays as None
        pca_components_arr = data['pca_components_arr']
        pca_explained_var_arr = data['pca_explained_var_arr']
        
        pca_components_list = []
        pca_explained_var_list = []
        
        for i in range(n_clusters):
            if len(pca_components_arr[i]) > 0:
                pca_components_list.append(pca_components_arr[i])
                pca_explained_var_list.append(pca_explained_var_arr[i])
            else:
                # Empty array means this cluster uses diagonal fallback
                pca_components_list.append(None)
                pca_explained_var_list.append(None)
        
        result['pca_components_list'] = pca_components_list
        result['pca_explained_var_list'] = pca_explained_var_list
        result['pca_noise_var'] = data['pca_noise_var']
        result['pca_n_components'] = int(data['pca_n_components'][0])
    
    return result


def get_cluster_config(config: BenchmarkConfig) -> ClusterConfig:
    """
    Create ClusterConfig from BenchmarkConfig by loading pre-extracted cluster stats.
    
    cluster_stats_path must be provided - this loads centroids, variances_per_dim, densities
    from a .npz file created by extract_values.py. If lowrank stats are present, also loads
    PCA components for better covariance modeling.
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
    
    # Check for lowrank stats
    is_lowrank = 'pca_components_list' in stats
    
    if is_lowrank:
        print(f"  Loaded {n_clusters} clusters, {n_cols} dimensions (lowrank, k={stats['pca_n_components']})")
        return ClusterConfig(
            nclusters=n_clusters,
            ncols=n_cols,
            seed=config.seed,
            cluster_centers=centroids,
            cluster_variances=variances_per_dim,
            cluster_densities=densities,
            pca_components_list=stats['pca_components_list'],
            pca_explained_var_list=stats['pca_explained_var_list'],
            pca_noise_var=stats['pca_noise_var'],
        )
    else:
        print(f"  Loaded {n_clusters} clusters, {n_cols} dimensions")
        return ClusterConfig(
            nclusters=n_clusters,
            ncols=n_cols,
            seed=config.seed,
            cluster_centers=centroids,
            cluster_variances=variances_per_dim,
            cluster_densities=densities,
        )
