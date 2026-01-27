"""
Utility Functions for Scalable GT Generation

Contains:
- Brute Force KNN functions (GPU and CPU)
- Recall computation functions
- Visualization functions
- Pretty printing functions
"""

import numpy as np
import cupy as cp
from cuvs.neighbors import brute_force
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from config import ClusterConfig, BenchmarkConfig


# =============================================================================
# Brute Force KNN Functions
# =============================================================================

def cuvs_brute_force_knn(
    database: np.ndarray, 
    queries: np.ndarray, 
    k: int, 
    metric: str = "sqeuclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    db_array = cp.array(database)
    index = brute_force.build(db_array, metric=metric)

    query_array = cp.array(queries)
    distances, indices = brute_force.search(index, query_array, k)

    return indices.copy_to_host(), distances.copy_to_host()


def cuvs_brute_force_knn_gpu(
    database: cp.ndarray, 
    queries: cp.ndarray, 
    k: int, 
    metric: str = "sqeuclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbors using brute force on GPU.
    Accepts CuPy arrays directly to avoid CPU->GPU transfer overhead.
    
    Args:
        database: CuPy array (n, d) of database vectors (already on GPU)
        queries: CuPy array (m, d) of query vectors (already on GPU)
        k: number of nearest neighbors
        metric: distance metric
    
    Returns:
        indices: NumPy array (m, k) of neighbor indices
        distances: NumPy array (m, k) of neighbor distances
    """
    index = brute_force.build(database, metric=metric)
    distances, indices = brute_force.search(index, queries, k)
    return indices.copy_to_host(), distances.copy_to_host()


def sklearn_brute_force_knn(
    database: np.ndarray, 
    queries: np.ndarray, 
    k: int, 
    metric: str = "sqeuclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    # Map metric names to sklearn equivalents
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "euclidean": "euclidean",
        "inner_product": "dot",
        "cosine": "cosine",
    }.get(metric, metric)
    
    nn_skl = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=skl_metric)
    nn_skl.fit(database)
    distances, indices = nn_skl.kneighbors(queries, return_distance=True)
    return indices, distances


def brute_force_knn(
    database: np.ndarray, 
    queries: np.ndarray, 
    k: int, 
    metric: str = "sqeuclidean",
    backend: Literal["cuvs", "sklearn"] = "cuvs"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbors using brute force.
    
    Args:
        database: (n, d) array of database vectors
        queries: (m, d) array of query vectors
        k: number of nearest neighbors
        metric: distance metric (default: sqeuclidean for squared L2)
        backend: "cuvs" for GPU or "sklearn" for CPU
    
    Returns:
        indices: (m, k) array of neighbor indices
        distances: (m, k) array of neighbor distances
    """
    if backend == "cuvs":
        return cuvs_brute_force_knn(database.astype(np.float32), queries.astype(np.float32), k, metric)
    elif backend == "sklearn":
        return sklearn_brute_force_knn(database, queries, k, metric)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cuvs' or 'sklearn'.")


# =============================================================================
# Recall Computation
# =============================================================================

def compute_recall(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute recall@k."""
    nqueries, k = ground_truth.shape
    hits = 0
    for i in range(nqueries):
        hits += len(set(predicted[i]) & set(ground_truth[i]))
    return hits / (nqueries * k)


def compute_recall_with_ties(
    predicted_indices: np.ndarray,
    predicted_distances: np.ndarray,
    gt_indices: np.ndarray,
    gt_distances: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> Tuple[float, List[int]]:
    """
    Compute recall@k accounting for distance ties.
    
    If two results have the same distance at position k, different indices
    are considered equivalent.
    
    Returns:
        recall: Recall score accounting for ties
        mismatched_queries: List of query indices that don't match even with ties
    """
    nqueries, k = gt_indices.shape
    matches = 0
    mismatched = []
    
    for i in range(nqueries):
        # Check each position
        query_matches = 0
        for j in range(k):
            pred_idx = predicted_indices[i, j]
            pred_dist = predicted_distances[i, j]
            
            # Check if this predicted index is in GT
            if pred_idx in gt_indices[i]:
                query_matches += 1
            else:
                # Check if there's a GT index with the same distance (tie)
                gt_dists = gt_distances[i]
                if np.any(np.isclose(gt_dists, pred_dist, rtol=rtol, atol=atol)):
                    query_matches += 1  # Tied distance - consider it a match
        
        matches += query_matches
        if query_matches < k:
            mismatched.append(i)
    
    return matches / (nqueries * k), mismatched


# =============================================================================
# Pretty Printing
# =============================================================================

def print_config(config: Union["BenchmarkConfig", "ClusterConfig"], title: str = None) -> None:
    """
    Pretty-print a config object in a box format.
    
    Args:
        config: Either a BenchmarkConfig or ClusterConfig
        title: Optional title for the box (auto-detected if None)
    """
    from config import BenchmarkConfig, ClusterConfig
    
    if isinstance(config, BenchmarkConfig):
        _print_benchmark_config(config, title)
    elif isinstance(config, ClusterConfig):
        _print_cluster_config(config, title)
    else:
        raise TypeError(f"Unknown config type: {type(config)}")


def _print_box(title: str, lines: list, width: int = 60) -> None:
    """Print content in a Unicode box."""
    print(f"┌{'─' * (width - 2)}┐")
    print(f"│ {title.center(width - 4)} │")
    print(f"├{'─' * (width - 2)}┤")
    for line in lines:
        # Pad line to fit box width
        padded = f" {line}".ljust(width - 2)
        print(f"│{padded}│")
    print(f"└{'─' * (width - 2)}┘")


def _print_benchmark_config(config: "BenchmarkConfig", title: str = None) -> None:
    """Pretty-print BenchmarkConfig."""
    title = title or "Benchmark Configuration"
    
    lines = [
        "── Dataset ──",
        f"  Total rows:        {config.total_rows:,}",
        f"  Seed:              {config.seed}",
        f"  Batch size:        {config.batch_size:,}",
        f"  Cluster stats:     {config.cluster_stats_path}",
        "",
        "── Query/GT ──",
        f"  Queries:           {config.nqueries}",
        f"  k:                 {config.k}",
        f"  nprobes:           {config.nprobes}",
        f"  GT backend:        {config.gt_backend}",
    ]
    
    # Show ANN index info if available
    if config.ann_index is not None:
        lines.extend([
            "",
            "── ANN Index ──",
            f"  Type:              {type(config.ann_index).__name__}",
        ])
    
    _print_box(title, lines)


def _print_cluster_config(config: "ClusterConfig", title: str = None) -> None:
    """Pretty-print ClusterConfig."""
    title = title or "Cluster Configuration"
    
    lines = [
        "── Loaded Stats ──",
        f"  Clusters:          {config.nclusters}",
        f"  Dimensions:        {config.ncols}",
        f"  Seed:              {config.seed}",
    ]
    
    # Show variance info
    if config.cluster_variances.ndim == 1:
        lines.append(f"  Variance type:     scalar per cluster")
    else:
        lines.append(f"  Variance type:     per-dimension ({config.cluster_variances.shape})")
    
    # Show optional fields if present
    if config.cluster_mins is not None:
        lines.append(f"  Has min bounds:    yes")
    if config.cluster_maxs is not None:
        lines.append(f"  Has max bounds:    yes")
    
    _print_box(title, lines)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_clusters_2d(
    config: "ClusterConfig",
    total_points: int,
    output_path: str = "clusters_2d.png",
    points_per_cluster_to_plot: int = -1,
    figsize: Tuple[int, int] = (12, 10),
    show_radii: bool = False,
    title: str = None
) -> None:
    """
    Plot 2D cluster configuration and save as PNG.
    
    Args:
        config: ClusterConfig with ncols=2
        total_points: Total number of points (used for calculating cluster sizes)
        output_path: Path to save the PNG file
        points_per_cluster_to_plot: Max points to plot per cluster (for visualization)
        figsize: Figure size (width, height)
        show_radii: Whether to draw circles showing 2σ radius for each cluster
        title: Plot title (auto-generated if None)
    
    Raises:
        ValueError: If config.ncols != 2
    """
    import matplotlib.pyplot as plt
    
    if config.ncols != 2:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colormap
    cmap = plt.cm.get_cmap('tab20', config.nclusters)
    colors = [cmap(i) for i in range(config.nclusters)]
    
    # Calculate points per cluster
    from config import ClusterConfig  # Import here to avoid circular import
    points_per_cluster = (config.cluster_densities * total_points).astype(np.int64)
    points_per_cluster[0] += total_points - points_per_cluster.sum()  # Adjust rounding
    
    # Plot each cluster
    for cluster_id in range(config.nclusters):
        center = config.cluster_centers[cluster_id]
        variance = config.cluster_variances[cluster_id]
        std = np.sqrt(variance)
        n_points = points_per_cluster[cluster_id] if points_per_cluster_to_plot == -1 else min(points_per_cluster[cluster_id], points_per_cluster_to_plot)
        
        # Generate sample points for this cluster
        rng = np.random.default_rng(config.seed * 1000000 + cluster_id)
        points = rng.normal(loc=center, scale=std, size=(n_points, 2))
        
        # Plot points
        ax.scatter(
            points[:, 0], points[:, 1], 
            c=[colors[cluster_id]], 
            alpha=0.5, 
            s=10,
            label=f'Cluster {cluster_id} (n={points_per_cluster[cluster_id]:,})'
        )
        
        # Plot center
        ax.scatter(
            center[0], center[1], 
            c=[colors[cluster_id]], 
            marker='X', 
            s=100, 
            edgecolors='black',
            linewidths=1.5
        )
        
        # Draw 2σ radius circle
        if show_radii:
            radius_2sigma = 2 * std
            circle = plt.Circle(
                (center[0], center[1]), 
                radius_2sigma,
                fill=False, 
                color=colors[cluster_id],
                linestyle='--',
                linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(circle)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    # Title
    if title is None:
        title = (
            f"Cluster Configuration: {config.nclusters} clusters, {total_points:,} points\n"
            f"Hypercube Scale: {config.hypercube_scale}, Variance Scale: {config.variance_scale}"
        )
    ax.set_title(title)
    
    # Legend (only show if not too many clusters)
    if config.nclusters <= 10:
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster plot to: {output_path}")

