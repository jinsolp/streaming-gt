"""
Compare Pareto curves between original and mock-generated datasets.

This validates that the synthetic data mimics the real dataset's ANN search characteristics.
If the Pareto curves are similar, the synthetic data is a good approximation.
"""

import numpy as np
import time
import re
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import argparse
import sys
import os
from tqdm import tqdm
import cupy as cp

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extract_values import load_memmap_bin
from config import load_cluster_stats, ClusterConfig
from utils import brute_force_knn, compute_recall_with_ties
from scalable_gt import gen_cluster, get_num_points_per_cluster, generate_mock_data, generate_queries_and_gt_batched
from ann_indices import CagraIndex, IvfPqIndex


def analyze_centroid_quality(
    centroids: np.ndarray,
    densities: np.ndarray,
    variances_per_dim: np.ndarray,
    means: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze centroid quality to detect poor KMeans convergence.
    
    Checks for:
    - Redundant centroids (too close together) using L2 and cosine distance
    - Centroid drift (if means provided)
    - Cluster size imbalance
    
    Args:
        centroids: (n_clusters, n_dim) cluster centers
        densities: (n_clusters,) relative density per cluster
        variances_per_dim: (n_clusters, n_dim) per-dimension variance per cluster
        means: Optional (n_clusters, n_dim) actual cluster means
    
    Returns:
        Dict with quality metrics
    """
    from scipy.spatial.distance import pdist, squareform, cdist
    
    n_clusters, n_dim = centroids.shape
    
    print("\n" + "=" * 60)
    print("Centroid Quality Analysis")
    print("=" * 60)
    
    # === L2 Distance Analysis ===
    l2_dists = squareform(pdist(centroids, metric='euclidean'))
    np.fill_diagonal(l2_dists, np.inf)
    
    l2_min = l2_dists.min()
    l2_median = np.median(l2_dists[l2_dists < np.inf])
    l2_mean = l2_dists[l2_dists < np.inf].mean()
    l2_proximity_ratio = l2_min / l2_median
    
    # Find closest pair
    min_idx = np.unravel_index(np.argmin(l2_dists), l2_dists.shape)
    
    print(f"\n[L2 Distance]")
    print(f"  Min distance: {l2_min:.6f}")
    print(f"  Median distance: {l2_median:.6f}")
    print(f"  Mean distance: {l2_mean:.6f}")
    print(f"  Proximity ratio (min/median): {l2_proximity_ratio:.4f}")
    print(f"  Closest pair: cluster {min_idx[0]} <-> cluster {min_idx[1]}")
    
    if l2_proximity_ratio < 0.1:
        print("  ⚠️ WARNING: Some centroids are very close (L2) - possible redundancy!")
    elif l2_proximity_ratio < 0.2:
        print("  ⚡ NOTICE: Some centroids are moderately close (L2)")
    else:
        print("  ✅ Centroids are well-separated (L2)")
    
    # === Cosine Distance Analysis ===
    # Normalize centroids for cosine similarity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized = centroids / (norms + 1e-10)
    
    # Cosine distance = 1 - cosine similarity
    cosine_sim = normalized @ normalized.T
    cosine_dist = 1 - cosine_sim
    np.fill_diagonal(cosine_dist, np.inf)
    
    cos_min = cosine_dist.min()
    cos_median = np.median(cosine_dist[cosine_dist < np.inf])
    cos_mean = cosine_dist[cosine_dist < np.inf].mean()
    cos_proximity_ratio = cos_min / cos_median if cos_median > 0 else 0
    
    # Find closest pair by cosine
    cos_min_idx = np.unravel_index(np.argmin(cosine_dist), cosine_dist.shape)
    
    print(f"\n[Cosine Distance]")
    print(f"  Min distance: {cos_min:.6f}")
    print(f"  Median distance: {cos_median:.6f}")
    print(f"  Mean distance: {cos_mean:.6f}")
    print(f"  Proximity ratio (min/median): {cos_proximity_ratio:.4f}")
    print(f"  Closest pair: cluster {cos_min_idx[0]} <-> cluster {cos_min_idx[1]}")
    
    if cos_proximity_ratio < 0.1:
        print("  ⚠️ WARNING: Some centroids are very similar (cosine) - possible redundancy!")
    elif cos_proximity_ratio < 0.2:
        print("  ⚡ NOTICE: Some centroids are moderately similar (cosine)")
    else:
        print("  ✅ Centroids are well-separated (cosine)")
    
    # === Centroid Drift Analysis (if means provided) ===
    if means is not None:
        drift_l2 = np.linalg.norm(centroids - means, axis=1)
        drift_cos = 1 - np.sum(normalized * (means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-10)), axis=1)
        
        print(f"\n[Centroid Drift (centroid vs actual mean)]")
        print(f"  L2 drift - mean: {drift_l2.mean():.6f}, max: {drift_l2.max():.6f}")
        print(f"  Cosine drift - mean: {drift_cos.mean():.6f}, max: {drift_cos.max():.6f}")
        
        # Flag clusters with high drift
        high_drift_l2 = np.where(drift_l2 > l2_median * 0.5)[0]
        if len(high_drift_l2) > 0:
            print(f"  ⚠️ {len(high_drift_l2)} clusters have high L2 drift: {high_drift_l2[:10].tolist()}...")
    else:
        drift_l2 = None
        drift_cos = None
    
    # === Cluster Size Analysis ===
    size_cv = densities.std() / densities.mean()  # Coefficient of variation
    tiny_threshold = 1.0 / n_clusters * 0.1  # < 10% of expected size
    tiny_clusters = np.where(densities < tiny_threshold)[0]
    large_threshold = 1.0 / n_clusters * 5.0  # > 5x expected size
    large_clusters = np.where(densities > large_threshold)[0]
    
    print(f"\n[Cluster Size Distribution]")
    print(f"  Size CV (lower=more balanced): {size_cv:.4f}")
    print(f"  Min density: {densities.min():.6f} ({densities.min() * 100:.4f}%)")
    print(f"  Max density: {densities.max():.6f} ({densities.max() * 100:.4f}%)")
    print(f"  Mean density: {densities.mean():.6f} ({densities.mean() * 100:.4f}%)")
    print(f"  Tiny clusters (<10% expected): {len(tiny_clusters)}")
    print(f"  Large clusters (>5x expected): {len(large_clusters)}")
    
    if len(tiny_clusters) > n_clusters * 0.1:
        print(f"  ⚠️ WARNING: Many tiny clusters - possible poor convergence")
        print(f"    Tiny cluster IDs: {tiny_clusters[:10].tolist()}...")
    
    # === Variance Analysis ===
    # Per-dimension variance: take mean across dimensions
    var_per_cluster = variances_per_dim.mean(axis=1)
    var_min, var_max, var_mean = var_per_cluster.min(), var_per_cluster.max(), var_per_cluster.mean()
    
    print(f"\n[Variance Distribution]")
    print(f"  Min: {var_min:.6f}, Max: {var_max:.6f}, Mean: {var_mean:.6f}")
    print(f"  Ratio (max/min): {var_max / var_min:.2f}x")
    
    print("=" * 60)
    
    return {
        "l2_min": l2_min,
        "l2_median": l2_median,
        "l2_proximity_ratio": l2_proximity_ratio,
        "l2_closest_pair": min_idx,
        "cos_min": cos_min,
        "cos_median": cos_median,
        "cos_proximity_ratio": cos_proximity_ratio,
        "cos_closest_pair": cos_min_idx,
        "size_cv": size_cv,
        "n_tiny_clusters": len(tiny_clusters),
        "n_large_clusters": len(large_clusters),
        "drift_l2": drift_l2,
        "drift_cos": drift_cos
    }


def analyze_data_to_centroid_distances(
    original_data: np.ndarray,
    mock_data: np.ndarray,
    centroids: np.ndarray,
) -> Dict:
    """
    Compare distance-to-closest-centroid distributions between original and mock data.
    
    This validates that mock data has similar "tightness" around centroids as real data.
    
    Args:
        original_data: Original dataset
        mock_data: Mock generated dataset
        centroids: Cluster centroids
    
    Returns:
        Dict with distance statistics for both datasets
    """
    print("\n" + "=" * 60)
    print("Data-to-Centroid Distance Analysis")
    print("=" * 60)
    
    print(f"Analyzing {len(original_data):,} original and {len(mock_data):,} mock points...")
    
    # Use GPU brute force to find distance to closest centroid (k=1)
    # This is much faster than cdist for large datasets
    print("  Computing distances to closest centroid (GPU brute force)...")
    
    # Original data -> closest centroid
    _, orig_min_dists_sq = brute_force_knn(centroids, original_data, k=1, backend="cuvs")
    orig_min_dists = np.sqrt(orig_min_dists_sq.flatten())
    
    # Mock data -> closest centroid  
    _, mock_min_dists_sq = brute_force_knn(centroids, mock_data, k=1, backend="cuvs")
    mock_min_dists = np.sqrt(mock_min_dists_sq.flatten())
    
    # Inertia = sum of squared distances to closest centroid
    # Normalize by dataset size for fair comparison
    orig_inertia = (orig_min_dists ** 2).sum() / len(original_data)
    mock_inertia = (mock_min_dists ** 2).sum() / len(mock_data)
    
    print(f"\n[Inertia (avg squared distance to closest centroid)]")
    print(f"  Original: {orig_inertia:.6f}")
    print(f"  Mock:     {mock_inertia:.6f}")
    inertia_diff_pct = (mock_inertia - orig_inertia) / orig_inertia * 100
    print(f"  Diff:     {inertia_diff_pct:+.1f}%")
    
    if abs(inertia_diff_pct) < 10:
        print("  ✅ Inertia is similar - mock data has similar cluster tightness!")
    elif mock_inertia > orig_inertia * 1.2:
        print("  ⚠️ Mock inertia is higher - points are more spread out than original")
    elif mock_inertia < orig_inertia * 0.8:
        print("  ⚠️ Mock inertia is lower - points are tighter than original")
    
    # Statistics
    orig_stats = {
        "mean": orig_min_dists.mean(),
        "std": orig_min_dists.std(),
        "median": np.median(orig_min_dists),
        "p5": np.percentile(orig_min_dists, 5),
        "p95": np.percentile(orig_min_dists, 95),
        "min": orig_min_dists.min(),
        "max": orig_min_dists.max()
    }
    
    mock_stats = {
        "mean": mock_min_dists.mean(),
        "std": mock_min_dists.std(),
        "median": np.median(mock_min_dists),
        "p5": np.percentile(mock_min_dists, 5),
        "p95": np.percentile(mock_min_dists, 95),
        "min": mock_min_dists.min(),
        "max": mock_min_dists.max()
    }
    
    print(f"\n[Distance to Closest Centroid (L2)]")
    print(f"{'Metric':<12} {'Original':>12} {'Mock':>12} {'Diff %':>12}")
    print("-" * 50)
    for key in ["mean", "std", "median", "p5", "p95", "min", "max"]:
        orig_val = orig_stats[key]
        mock_val = mock_stats[key]
        diff_pct = (mock_val - orig_val) / orig_val * 100 if orig_val != 0 else 0
        print(f"{key:<12} {orig_val:>12.4f} {mock_val:>12.4f} {diff_pct:>+11.1f}%")
    
    # Compute cosine distances to closest centroid (GPU brute force with inner product)
    # print("  Computing cosine distances to closest centroid (GPU brute force)...")
    
    # # Normalize vectors for cosine similarity
    # orig_norms = np.linalg.norm(original_data, axis=1, keepdims=True)
    # mock_norms = np.linalg.norm(mock_data, axis=1, keepdims=True)
    # centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # orig_normalized = (original_data / (orig_norms + 1e-10)).astype(np.float32)
    # mock_normalized = (mock_data / (mock_norms + 1e-10)).astype(np.float32)
    # centroids_normalized = (centroids / (centroid_norms + 1e-10)).astype(np.float32)
    
    # # inner_product of normalized vectors = cosine similarity
    # # brute_force_knn returns max inner product (closest = highest similarity)
    # _, orig_cos_sim_vals = brute_force_knn(centroids_normalized, orig_normalized, k=1, metric="inner_product", backend="cuvs")
    # _, mock_cos_sim_vals = brute_force_knn(centroids_normalized, mock_normalized, k=1, metric="inner_product", backend="cuvs")
    
    # # Cosine distance = 1 - cosine similarity
    # orig_cos_dists = 1 - orig_cos_sim_vals.flatten()
    # mock_cos_dists = 1 - mock_cos_sim_vals.flatten()
    
    # orig_cos_stats = {
    #     "mean": orig_cos_dists.mean(),
    #     "std": orig_cos_dists.std(),
    #     "median": np.median(orig_cos_dists),
    #     "p5": np.percentile(orig_cos_dists, 5),
    #     "p95": np.percentile(orig_cos_dists, 95),
    # }
    
    # mock_cos_stats = {
    #     "mean": mock_cos_dists.mean(),
    #     "std": mock_cos_dists.std(),
    #     "median": np.median(mock_cos_dists),
    #     "p5": np.percentile(mock_cos_dists, 5),
    #     "p95": np.percentile(mock_cos_dists, 95),
    # }
    
    # print(f"\n[Distance to Closest Centroid (Cosine)]")
    # print(f"{'Metric':<12} {'Original':>12} {'Mock':>12} {'Diff %':>12}")
    # print("-" * 50)
    # for key in ["mean", "std", "median", "p5", "p95"]:
    #     orig_val = orig_cos_stats[key]
    #     mock_val = mock_cos_stats[key]
    #     diff_pct = (mock_val - orig_val) / orig_val * 100 if orig_val != 0 else 0
    #     print(f"{key:<12} {orig_val:>12.6f} {mock_val:>12.6f} {diff_pct:>+11.1f}%")
    
    # # Overall assessment
    # mean_diff_l2 = abs(mock_stats["mean"] - orig_stats["mean"]) / orig_stats["mean"]
    # std_diff_l2 = abs(mock_stats["std"] - orig_stats["std"]) / orig_stats["std"]
    
    # print(f"\n[Assessment]")
    # if mean_diff_l2 < 0.1 and std_diff_l2 < 0.2:
    #     print("  ✅ Mock data distribution closely matches original!")
    # elif mean_diff_l2 < 0.2 and std_diff_l2 < 0.3:
    #     print("  ⚡ Mock data distribution is reasonably similar to original")
    # else:
    #     print("  ⚠️ Mock data distribution differs from original - consider adjusting parameters")
    #     if mock_stats["mean"] > orig_stats["mean"] * 1.2:
    #         print("     → Mock points are too spread out (variance too high or clipping not working)")
    #     elif mock_stats["mean"] < orig_stats["mean"] * 0.8:
    #         print("     → Mock points are too clustered (variance too low)")
    
    # print("=" * 60)
    
    # return {
    #     "original_l2": orig_stats,
    #     "mock_l2": mock_stats,
    #     "original_cosine": orig_cos_stats,
    #     "mock_cosine": mock_cos_stats,
    #     "orig_min_dists": orig_min_dists,
    #     "mock_min_dists": mock_min_dists,
    #     "orig_inertia": orig_inertia,
    #     "mock_inertia": mock_inertia,
    #     "inertia_diff_pct": inertia_diff_pct
    # }

    return None


def generate_queries_from_gmm(gmm, n_queries: int) -> np.ndarray:
    """
    Generate queries by sampling from the GMM.
    
    Args:
        gmm: Fitted GaussianMixture model
        n_queries: Number of queries to generate
    
    Returns:
        queries: (n_queries, n_features) array
    """
    print(f"Generating {n_queries} queries using gmm.sample()...")
    queries, _ = gmm.sample(n_samples=n_queries)
    return queries.astype(np.float32)


def build_cagra_index(data: np.ndarray, graph_degree: int = 128) -> CagraIndex:
    """Build CAGRA index using CagraIndex wrapper."""
    from cuvs.neighbors import cagra
    
    build_params = cagra.IndexParams(
        intermediate_graph_degree=graph_degree * 2,
        graph_degree=graph_degree,
        build_algo="nn_descent"
    )
    
    print(f"Building CAGRA index (graph_degree={graph_degree})...")
    start = time.perf_counter()
    index = CagraIndex(build_params=build_params)
    index.build(data)
    build_time = time.perf_counter() - start
    print(f"  Build time: {build_time:.2f}s")
    
    return index


def build_ivf_pq_index(data: np.ndarray, n_lists: int = 1024, pq_dim: int = 0) -> IvfPqIndex:
    """Build IVF-PQ index using IvfPqIndex wrapper."""
    from cuvs.neighbors import ivf_pq
    
    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        pq_dim=pq_dim if pq_dim > 0 else data.shape[1] // 4,
        pq_bits=8
    )
    
    print(f"Building IVF-PQ index (n_lists={n_lists})...")
    start = time.perf_counter()
    index = IvfPqIndex(build_params=build_params)
    index.build(data)
    build_time = time.perf_counter() - start
    print(f"  Build time: {build_time:.2f}s")
    
    return index


def sweep_cagra(
    index: CagraIndex,
    queries: np.ndarray,
    gt_indices: np.ndarray,
    gt_distances: np.ndarray,
    k: int,
    itopk_sizes: List[int] = [32, 64, 128, 256, 512]
) -> List[Dict]:
    """
    Sweep CAGRA search parameters and collect recall/QPS.
    """
    results = []
    
    for itopk in itopk_sizes:
        # warmup
        _ = index.search(queries, k, itopk=itopk)
        
        # Timed search
        n_runs = 3
        times = []
        for _ in tqdm(range(n_runs), desc="Sweeping CAGRA"):
            start = time.perf_counter()
            pred_indices, pred_distances = index.search(queries, k, itopk=itopk)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        qps = queries.shape[0] / avg_time
       
        recall, _ = compute_recall_with_ties(pred_indices, pred_distances, gt_indices, gt_distances)
        
        results.append({
            "param": f"itopk={itopk}",
            "recall": recall,
            "qps": qps,
            "latency_ms": avg_time * 1000 / len(queries)
        })
        print(f"  itopk={itopk}: recall={recall:.4f}, QPS={qps:.0f}")
    
    return results


def sweep_ivf_pq(
    index: IvfPqIndex,
    queries: np.ndarray,
    gt_indices: np.ndarray,
    gt_distances: np.ndarray,
    k: int,
    n_probes_list: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]
) -> List[Dict]:
    """
    Sweep IVF-PQ search parameters and collect recall/QPS.
    """
    results = []
    
    for n_probes in n_probes_list:
        # Warmup
        _ = index.search(queries[:10], k, n_probes=n_probes)
        
        # Timed search
        n_runs = 3
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            pred_indices, pred_distances = index.search(queries, k, n_probes=n_probes)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        qps = len(queries) / avg_time
        
        # Compute recall with tie-awareness
        recall, _ = compute_recall_with_ties(pred_indices, pred_distances, gt_indices, gt_distances)
        
        results.append({
            "param": f"nprobes={n_probes}",
            "recall": recall,
            "qps": qps,
            "latency_ms": avg_time * 1000 / len(queries)
        })
        print(f"  nprobes={n_probes}: recall={recall:.4f}, QPS={qps:.0f}")
    
    return results


def plot_pareto_curves(
    original_results: List[Dict],
    mock_results: List[Dict],
    title: str = "Pareto Curve Comparison",
    output_path: Optional[str] = None
):
    """
    Plot Pareto curves for original and mock datasets.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    orig_recalls = [r["recall"] for r in original_results]
    orig_qps = [r["qps"] for r in original_results]
    mock_recalls = [r["recall"] for r in mock_results]
    mock_qps = [r["qps"] for r in mock_results]
    
    # Plot
    ax.plot(orig_recalls, orig_qps, 'o-', label='Original Dataset', markersize=8, linewidth=2)
    ax.plot(mock_recalls, mock_qps, 's--', label='Mock Dataset', markersize=8, linewidth=2)
    
    # Labels
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Queries per Second (QPS)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    # ax.set_xlim(0, 1.05)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def compare_datasets(
    original_data: np.ndarray,
    mock_data: np.ndarray,
    n_clusters: int,
    dataset_name: str = "unknown",
    subsample_size: int = 0,
    cluster_extract_size: int = 0,
    n_queries: int = 1000,
    k: int = 10,
    algorithm: str = "ivf_pq",
    output_path: Optional[str] = None,
    backend: str = "cuvs",
    gmm_method: Optional[str] = None,
    gmm_queries: Optional[np.ndarray] = None
):
    """
    Full comparison pipeline between original and mock datasets.
    
    Uses brute force GT for both datasets for fair comparison.
    
    Args:
        original_data: Original dataset
        mock_data: Mock generated dataset
        n_clusters: Number of clusters used in mock data generation
        dataset_name: Name of the original dataset
        subsample_size: Subsample size used for comparison
        cluster_extract_size: Sample size used for cluster extraction (KMeans)
        n_queries: Number of queries
        k: Number of nearest neighbors
        algorithm: "cagra" or "ivf_pq"
        output_path: Path to save plot
        backend: "cuvs" or "sklearn" for brute force
        gmm_method: GMM method used ("gmm", "gmm_gpu_init", or None)
        gmm_queries: Optional pre-generated queries from GMM (used for both datasets)
    """
    print("=" * 60)
    print("Pareto Curve Comparison: Original vs Mock Dataset")
    print("=" * 60)
    print(f"Original data shape: {original_data.shape}")
    print(f"Mock data shape: {mock_data.shape}")
    print(f"Algorithm: {algorithm}")
    print(f"Queries: {n_queries}, k: {k}")
    if gmm_queries is not None:
        print(f"Using GMM-generated queries: {gmm_queries.shape}")
    
    # Generate queries and GT for original (always brute force since no cluster structure)
    print("\n[Original Dataset]")
    orig_queries, orig_gt_idx, orig_gt_dist = generate_queries_and_gt_batched(
        original_data, n_queries, k, seed=12345, backend=backend
    )
    
    # Generate queries and GT for mock (use brute force for fair comparison)
    print("\n[Mock Dataset]")
    mock_queries, mock_gt_idx, mock_gt_dist = generate_queries_and_gt_batched(
        mock_data, n_queries, k, seed=12345, backend=backend, queries=gmm_queries
    )
    
    # Build and sweep one dataset at a time to reduce memory usage
    if algorithm == "cagra":
        sweep_func = sweep_cagra
        sweep_params = {"itopk_sizes": [32, 64, 128, 256]}
        param_str = f"itopk={sweep_params['itopk_sizes']}"
        build_func = build_cagra_index
        build_kwargs = {}
    else:  # ivf_pq
        sweep_func = sweep_ivf_pq
        sweep_params = {"n_probes_list": [1, 2, 4, 8, 16, 32, 64, 128, 256]}
        param_str = f"nprobes={sweep_params['n_probes_list']}"
        build_func = build_ivf_pq_index
        n_lists = min(1024, len(original_data) // 100)
        build_kwargs = {"n_lists": n_lists}
    
    # Build, sweep, and delete original index
    print("\n[Building Original Index]")
    orig_index = build_func(original_data, **build_kwargs)
    print("\n[Sweeping Original Dataset]")
    orig_results = sweep_func(orig_index, orig_queries, orig_gt_idx, orig_gt_dist, k, **sweep_params)
    del orig_index  # Free memory before building next index
    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
    
    # Build, sweep, and delete mock index
    print("\n[Building Mock Index]")
    mock_index = build_func(mock_data, **build_kwargs)
    print("\n[Sweeping Mock Dataset]")
    mock_results = sweep_func(mock_index, mock_queries, mock_gt_idx, mock_gt_dist, k, **sweep_params)
    del mock_index  # Free memory
    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
    
    # Plot comparison
    print("\n[Plotting Pareto Curves]")
    n_points = subsample_size if subsample_size > 0 else len(original_data)
    subsample_str = f"{n_points / 1_000_000:.1f}M"
    cluster_extract_str = f"{cluster_extract_size / 1_000_000:.1f}M" if cluster_extract_size > 0 else "?"
    gmm_str = f" [{gmm_method}]" if gmm_method else ""
    
    plot_pareto_curves(
        orig_results, mock_results,
        title=f"Pareto Comparison ({algorithm.upper()}): {dataset_name} (N={subsample_str})\nMock data {n_clusters} clusters from {cluster_extract_str} samples{gmm_str}\n{algorithm.upper()} config:{param_str}",
        output_path=output_path
    )
    
    # Compute similarity metric
    orig_recalls = np.array([r["recall"] for r in orig_results])
    mock_recalls = np.array([r["recall"] for r in mock_results])
    recall_diff = np.abs(orig_recalls - mock_recalls).mean()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Average recall difference: {recall_diff:.4f}")
    if recall_diff < 0.05:
        print("✅ Pareto curves are very similar - mock data is a good approximation!")
    elif recall_diff < 0.10:
        print("⚠️ Pareto curves are somewhat similar - may need parameter tuning")
    else:
        print("❌ Pareto curves differ significantly - mock data may not match well")
    
    return orig_results, mock_results


if __name__ == "__main__":
    import pickle
    import joblib
    
    parser = argparse.ArgumentParser(description="Compare Pareto curves between original and mock datasets")
    parser.add_argument("--original", type=str, default="wiki",
                        help="Original dataset: 'wiki', 'food', or path")
    parser.add_argument("--cluster_stats", type=str, default=None,
                        help="Path to cluster_stats.npz from extract_values.py")
    parser.add_argument("--gmm_model", type=str, default=None,
                        help="Path to saved GMM model (.joblib) - uses gmm.sample() to generate mock data")
    parser.add_argument("--subsample", type=int, default=-1,
                        help="Subsample size for original data (-1 = use full dataset)")
    parser.add_argument("--n_queries", type=int, default=1000,
                        help="Number of queries")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of nearest neighbors")
    parser.add_argument("--algorithm", type=str, default="cagra",
                        choices=["cagra", "ivf_pq"],
                        help="ANN algorithm to benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Load original dataset
    print("Loading original dataset...")
    if args.original == "wiki":
        data_path = "/datasets/jinsolp/data/wiki_all/base.88M.fbin"
        original_data, _, n_dim = load_memmap_bin(data_path, np.float32, extra=args.subsample)
    elif args.original == "food":
        data_path = "/datasets/jinsolp/data/amazon_reviews/Grocery_and_Gourmet_Food.pkl"
        with open(data_path, "rb") as f:
            original_data = pickle.load(f)
        if not isinstance(original_data, np.ndarray):
            original_data = np.array(original_data, dtype=np.float32)
        n_dim = original_data.shape[1]
        if args.subsample > 0 and args.subsample < len(original_data):
            rng = np.random.default_rng(100)
            indices = rng.choice(len(original_data), args.subsample, replace=False)
            original_data = original_data[indices]
    elif args.original.endswith(".npy"):
        data_path = args.original
        print(f"Loading numpy dataset from {data_path}...")
        original_data = np.load(data_path)
        n_rows, n_dim = original_data.shape
    else:
        original_data, _, n_dim = load_memmap_bin(args.original, np.float32, extra=args.subsample)
    
    print(f"Original data shape: {original_data.shape}")
    
    # Validate that at least one of cluster_stats or gmm_model is provided
    if args.cluster_stats is None and args.gmm_model is None:
        raise ValueError("Must provide either --cluster_stats or --gmm_model")
    
    # ===================== GMM SAMPLE MODE =====================
    if args.gmm_model is not None:
        print(f"\nLoading GMM model from {args.gmm_model}...")
        gmm = joblib.load(args.gmm_model)
        
        n_clusters = gmm.n_components
        n_dim = gmm.means_.shape[1]
        print(f"  GMM components: {n_clusters}")
        print(f"  Dimensions: {n_dim}")
        print(f"  Covariance type: {gmm.covariance_type}")
        
        # Generate mock data using GMM sample
        print(f"\nGenerating {len(original_data):,} samples using gmm.sample()...")
        start = time.time()
        mock_data, mock_labels = gmm.sample(n_samples=len(original_data))
        mock_data = mock_data.astype(np.float32)
        sample_time = time.time() - start
        print(f"  Sample time: {sample_time:.2f}s")
        print(f"  Mock data shape: {mock_data.shape}")
        
        # Generate queries using GMM sample
        gmm_queries = generate_queries_from_gmm(gmm, args.n_queries)
        print(f"  GMM queries shape: {gmm_queries.shape}")
        
        # For analysis
        centroids = gmm.means_.astype(np.float32)
        original_n_clusters = n_clusters
        cluster_extract_size = 0
        
        # Parse extraction info from filename if available
        match = re.search(r'sample(\d+)', args.gmm_model)
        if match:
            cluster_extract_size = int(match.group(1))
        
        # Check if GMM method from filename
        if "_gmm_gpu_init" in args.gmm_model.lower():
            gmm_method = "gmm_gpu_init"
        elif "_gmm" in args.gmm_model.lower():
            gmm_method = "gmm_sample"
        else:
            gmm_method = "gmm_sample"
        
    else:
        # No GMM queries in cluster_stats mode
        gmm_queries = None
        
        # Parse cluster extraction sample size from filename
        cluster_extract_size = 0
        match = re.search(r'sample(\d+)', args.cluster_stats)
        if match:
            cluster_extract_size = int(match.group(1))
        
        print(f"Loading cluster stats from {args.cluster_stats}...")
        stats = load_cluster_stats(args.cluster_stats)
        centroids = stats['centroids']
        densities = stats['densities']
        variances_per_dim = stats['variances_per_dim']
        
        # Save original cluster count for title/filename
        original_n_clusters = len(centroids)
        
        # Identify problematic clusters (near-zero variance)
        var_per_cluster = variances_per_dim.mean(axis=1)
        var_threshold = np.median(var_per_cluster) * 0.01
        low_var_mask = var_per_cluster < var_threshold
        n_low_var = low_var_mask.sum()
        
        if n_low_var > 0:
            print(f"ℹ️  Found {n_low_var} clusters with near-zero variance (< {var_threshold:.2e})")
            print(f"    These will generate 1 point each (the centroid)")
        
        if cluster_extract_size > 0:
            print(f"Cluster extraction sample size: {cluster_extract_size / 1_000_000:.1f}M")
        
        cluster_config = ClusterConfig(
            nclusters=len(centroids),
            ncols=n_dim,
            seed=42,
            cluster_centers=centroids,
            cluster_variances=variances_per_dim,
            cluster_densities=densities,
        )
        
        # Generate mock data
        print("Generating mock data...")
        mock_data = generate_mock_data(cluster_config, len(original_data))
        print(f"Mock data shape: {mock_data.shape}")
        
        # Check if GMM was used (based on filename)
        if "_gmm_gpu_init" in args.cluster_stats.lower():
            gmm_method = "gmm_gpu_init"
        elif "_gmm" in args.cluster_stats.lower():
            gmm_method = "gmm"
        else:
            gmm_method = None
    
    # # Analyze distance-to-centroid distributions (validates mock data quality)
    # dist_analysis = analyze_data_to_centroid_distances(original_data, mock_data, centroids)
    
    # Compare (both use brute force GT for fair comparison)
    actual_subsample_size = len(original_data) if args.subsample < 0 else args.subsample
    
    # Generate output path if not provided
    if args.output is None:
        n_clusters = original_n_clusters
        subsample_str = f"{actual_subsample_size / 1_000_000:.1f}M"
        extract_str = f"{cluster_extract_size / 1_000_000:.1f}M" if cluster_extract_size > 0 else "full"
        gmm_str = f"_{gmm_method}" if gmm_method else ""
        output_path = f"pareto_curves/pareto_{args.original}_N{subsample_str}_{n_clusters}clusters{gmm_str}_extract{extract_str}_{args.algorithm}.png"
    else:
        output_path = args.output
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    compare_datasets(
        original_data,
        mock_data,
        n_clusters=original_n_clusters,
        dataset_name=args.original,
        subsample_size=actual_subsample_size,
        cluster_extract_size=cluster_extract_size,
        n_queries=args.n_queries,
        k=args.k,
        algorithm=args.algorithm,
        output_path=output_path,
        gmm_method=gmm_method,
        gmm_queries=gmm_queries
    )

