"""
Scalable Ground Truth Generation for 100B-scale ANN Benchmarking

Key idea: Generate clustered data deterministically so we can:
1. Stream/batch data generation without storing everything
2. Compute ground truth by only generating nearby clusters (like IVF nprobes)
"""

import numpy as np
from typing import Tuple, List, Literal
from tqdm import tqdm
from cuvs.neighbors import cagra

from config import ClusterConfig, BenchmarkConfig, get_cluster_config
from ann_indices import CagraIndex, IvfPqIndex
from utils import brute_force_knn, compute_recall, compute_recall_with_ties, plot_clusters_2d, print_config


# =============================================================================
# Data Generation Functions
# =============================================================================

def get_cluster_seed(base_seed: int, cluster_id: int) -> int:
    """
    Get deterministic seed for a specific cluster.
    Muliply the base seed by 1000000 for different sequences for different base_seeds.
    """
    return base_seed * 1000000 + cluster_id


def gen_cluster(cluster_id: int, n_points: int, config: ClusterConfig) -> np.ndarray:
    """
    Generate points for a single cluster deterministically using normal distribution.
    Same cluster_id + config always produces identical points.
    
    Supports:
    - Scalar variance per cluster: config.cluster_variances shape (nclusters,)
    - Per-dimension variance: config.cluster_variances shape (nclusters, ncols)
    """
    seed = get_cluster_seed(config.seed, cluster_id)
    rng = np.random.default_rng(seed)
    
    center = config.cluster_centers[cluster_id]
    variance = config.cluster_variances[cluster_id]
    
    # variance can be scalar (single value) or per-dimension array (ncols,)
    scale = np.sqrt(variance)
    
    # Generate points using normal distribution
    points = rng.normal(loc=center, scale=scale, size=(n_points, config.ncols))
    
    return points.astype(np.float32)


def get_num_points_per_cluster(total_points: int, config: ClusterConfig, min_points_per_cluster: int = 1) -> np.ndarray:
    """
    Calculate how many points each cluster should have based on densities.
    Ensures each cluster gets at least min_points_per_cluster points.
    """
    nclusters = config.nclusters
    
    # Check if we have enough total points
    min_required = nclusters * min_points_per_cluster
    if total_points < min_required:
        raise ValueError(
            f"total_points={total_points} is less than nclusters * min_points_per_cluster = {min_required}. "
            f"Increase total_points or decrease nclusters."
        )
    
    # First, give each cluster the minimum
    points_per_cluster = np.full(nclusters, min_points_per_cluster, dtype=np.int64)
    
    # Distribute remaining points proportionally based on densities
    remaining = total_points - min_required
    if remaining > 0:
        extra_points = (config.cluster_densities * remaining).astype(np.int64)
        points_per_cluster += extra_points
        
        # Add remainder to first cluster
        diff = total_points - points_per_cluster.sum()
        points_per_cluster[0] += diff
    
    return points_per_cluster


def gen_batch(
    batch_num: int,
    batch_size: int,
    total_rows: int,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of batch_size vectors deterministically.
    
    Returns:
        vectors: (batch_size, ncols) array of vectors
        cluster_ids: (batch_size,) array of cluster assignments
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)
    
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    actual_batch_size = end_idx - start_idx
    if actual_batch_size <= 0:
        return np.array([]), np.array([])
    
    vectors = []
    cluster_ids = []
    
    # start_idx and end_idx are the global indices of the batch
    # generate vectors from the clusters that overlap with this batch
    for cluster_id in range(config.nclusters):
        cluster_start = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
        cluster_end = cumsum[cluster_id]
        
        # Early exit: if this cluster starts after our batch ends, we're done
        if cluster_start >= end_idx:
            break
        
        # Skip: if this cluster ends before our batch starts
        if cluster_end <= start_idx:
            continue
        
        # This cluster overlaps with batch range
        overlap_start = max(start_idx, cluster_start)
        overlap_end = min(end_idx, cluster_end)
        
        # Generate full cluster, then slice the portion we need
        n_cluster_points = points_per_cluster[cluster_id]
        cluster_points = gen_cluster(cluster_id, n_cluster_points, config)
        
        # Calculate local indices within this cluster
        local_start = overlap_start - cluster_start
        local_end = overlap_end - cluster_start

        vectors.append(cluster_points[local_start:local_end])
        cluster_ids.extend([cluster_id] * (local_end - local_start))
    
    return np.vstack(vectors), np.array(cluster_ids)


def find_nearby_clusters(
    query_points: np.ndarray,
    config: ClusterConfig,
    nprobes: int,
    backend: Literal["cuvs", "sklearn"] = "cuvs",
    metric: str = "sqeuclidean"
) -> List[np.ndarray]:
    """
    Find the nprobes nearest cluster centers for each query point.
    """
    nearby_clusters, _ = brute_force_knn(
        config.cluster_centers,
        query_points,
        nprobes,
        metric=metric,
        backend=backend
    )

    return nearby_clusters


# =============================================================================
# Ground Truth Generation
# =============================================================================

def gen_gt(
    nqueries: int,
    total_rows: int,
    config: ClusterConfig,
    k: int = 10,
    nprobes: int = 10,
    query_seed_offset: int = 999999,
    backend: Literal["cuvs", "sklearn"] = "cuvs",
    metric: str = "sqeuclidean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate queries and compute ground truth
    
    Instead of brute-forcing against all total_rows points,
    we only generate points from the nprobes nearest clusters.
    
    Args:
        nqueries: Number of query points to generate
        total_rows: Total dataset size (for calculating points per cluster)
        config: Cluster configuration
        k: Number of nearest neighbors for ground truth
        nprobes: Number of nearby clusters to search (like IVF)
        query_seed_offset: Offset for query generation seed
        backend: "cuvs" for GPU brute force or "sklearn" for CPU brute force
    
    Returns:
        queries: (nqueries, ncols) query vectors
        gt_indices: (nqueries, k) ground truth neighbor indices  
        gt_distances: (nqueries, k) ground truth distances
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)
    
    # Generate query points deterministically
    # Queries come from random clusters (weighted by density)
    query_rng = np.random.default_rng(config.seed + query_seed_offset)
    
    # Sample which clusters queries come from
    query_cluster_ids = query_rng.choice(
        config.nclusters, 
        size=nqueries, 
        p=config.cluster_densities
    )
    
    # Generate query points from their respective clusters
    queries = []
    for i, cluster_id in enumerate(query_cluster_ids):
        # Use a unique seed for each query point
        q_seed = config.seed + query_seed_offset + i + 1
        q_rng = np.random.default_rng(q_seed)
        center = config.cluster_centers[cluster_id]
        variance = config.cluster_variances[cluster_id]
        q = q_rng.normal(loc=center, scale=np.sqrt(variance), size=(config.ncols,))
        queries.append(q)
    
    queries = np.array(queries, dtype=np.float32)

    # Find nearby clusters for each query
    nearby_clusters = find_nearby_clusters(queries, config, nprobes, backend, metric)

    # Compute ground truth by only searching nearby clusters
    gt_indices = np.full((nqueries, k), -1, dtype=np.int64)
    gt_distances = np.full((nqueries, k), np.inf, dtype=np.float32)
    
    for q_idx in tqdm(range(nqueries), desc="Computing ground truth for each query"):
        query = queries[q_idx]
        
        # Collect points from nearby clusters only
        candidate_points = []
        candidate_global_indices = []
        
        for cluster_id in nearby_clusters[q_idx]:
            cluster_id = int(cluster_id)
            n_points = points_per_cluster[cluster_id]
            
            # Generate this cluster's points
            cluster_points = gen_cluster(cluster_id, n_points, config)
            
            # Calculate global indices for this cluster's points
            global_start = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
            global_indices = np.arange(global_start, global_start + n_points)
            
            candidate_points.append(cluster_points)
            candidate_global_indices.append(global_indices)
        
        # Stack all candidates
        all_candidates = np.vstack(candidate_points)
        all_indices = np.concatenate(candidate_global_indices)
        # Brute force on this smaller subset
        local_indices, local_dists = brute_force_knn(
            all_candidates, 
            query.reshape(1, -1), 
            k,
            backend=backend
        )

        gt_indices[q_idx] = all_indices[local_indices[0]]
        gt_distances[q_idx] = local_dists[0]

    return queries, gt_indices, gt_distances


# =============================================================================
# Verification Functions
# =============================================================================

def verify_clustering_gt_accuracy(
    config: BenchmarkConfig,
    verbose: bool = True
) -> dict:
    """
    Verify that the clustering-based GT approach produces accurate results against full brute force.
    
    This function:
    1. Generates queries using the clustering approach
    2. Computes GT by probing the nprobes nearest clusters for each query
    3. Computes TRUE GT using brute force on ALL data (Stacked from all clusters)
    4. Compares to measure GT accuracy
    
    Args:
        config: BenchmarkConfig with cluster_stats_path and benchmark parameters
        verbose: Print progress
    
    Returns:
        Dictionary with verification results
    """
    if verbose:
        print("=" * 60)
        print("Verifying Clustering-based Ground Truth Accuracy")
        print("=" * 60)
    
    # Create cluster config from loaded stats
    cluster_config = get_cluster_config(config)
    
    plot_clusters_2d(cluster_config, total_points=config.total_rows)
    
    if verbose:
        print_config(config)
        print_config(cluster_config)
    
    # Step 1: Generate ALL data at once
    if verbose:
        print(f"\n[Step 1] Generating all {config.total_rows:,} vectors for {cluster_config.nclusters} clusters...")
    
    points_per_cluster = get_num_points_per_cluster(config.total_rows, cluster_config)
    
    # Ensure we have enough points across nprobes clusters for k neighbors
    min_points = points_per_cluster.min()
    min_total_from_nprobes = config.nprobes * min_points
    if min_total_from_nprobes < config.k:
        raise ValueError(
            f"With nprobes={config.nprobes} and min cluster size={min_points}, "
            f"we may only have {min_total_from_nprobes} candidates which is less than k={config.k}. "
            f"Increase total_rows, nprobes, or decrease nclusters/k."
        )
    
    all_vectors = []
    for cluster_id in tqdm(range(cluster_config.nclusters)):
        cluster_points = gen_cluster(cluster_id, points_per_cluster[cluster_id], cluster_config)
        all_vectors.append(cluster_points)
    all_data = np.vstack(all_vectors).astype(np.float32)
    
    if verbose:
        print(f"  Generated {all_data.shape[0]:,} vectors")
    
    # Step 2: Generate queries and cluster-based GT
    if verbose:
        print(f"\n[Step 2] Generating queries and cluster-based GT (nprobes={config.nprobes})...")

    queries, cluster_gt_indices, cluster_gt_distances = gen_gt(
        nqueries=config.nqueries,
        total_rows=config.total_rows,
        config=cluster_config,
        k=config.k,
        nprobes=config.nprobes,
        backend=config.gt_backend
    )

    if verbose:
        print(f"  Generated {config.nqueries} queries with cluster-based GT")
    
    # Step 3: Compute TRUE brute force GT on ALL data
    if verbose:
        print(f"\n[Step 3] Computing TRUE brute force GT on all {config.total_rows:,} vectors...")
    
    true_gt_indices, true_gt_distances = brute_force_knn(
        all_data,
        queries,
        config.k,
        backend=config.gt_backend
    )

    if verbose:
        print(f"  Computed true GT for {config.nqueries} queries")
    
    # Step 4: Compare cluster-based GT vs true GT
    if verbose:
        print(f"\n[Step 4] Comparing cluster-based GT vs true brute force GT...")
    
    # Compute recall with tie-awareness
    gt_accuracy, mismatched_queries = compute_recall_with_ties(
        cluster_gt_indices, cluster_gt_distances,
        true_gt_indices, true_gt_distances
    )
    
    # Also compute simple recall for comparison
    simple_recall = compute_recall(cluster_gt_indices, true_gt_indices)
    
    # Detailed analysis: which queries have incorrect GT (accounting for ties)?
    incorrect_queries = []
    for i in mismatched_queries:
        cluster_set = set(cluster_gt_indices[i])
        true_set = set(true_gt_indices[i])
        missing = true_set - cluster_set
        incorrect_queries.append({
            'query_idx': i,
            'missing_neighbors': list(missing),
            'overlap': len(cluster_set & true_set) / config.k
        })
    
    results = {
        'gt_accuracy': gt_accuracy,
        'total_queries': config.nqueries,
        'incorrect_queries': len(incorrect_queries),
        'nprobes_used': config.nprobes,
    }
    
    if verbose:
        print(f"\n" + "=" * 60)
        print("Verification Results")
        print("=" * 60)
        print(f"  GT Accuracy (with distance ties):    {gt_accuracy:.4f} ({gt_accuracy*100:.2f}%)")
        print(f"  GT Accuracy (naive):       {simple_recall:.4f} ({simple_recall*100:.2f}%)")
        
        if gt_accuracy < 1.0:
            print(f"\n  🚨 Some true neighbors are in clusters outside the {config.nprobes} probed.")
            print(f"  Queries with imperfect GT: {len(incorrect_queries)}/{config.nqueries}")
        else:
            print(f"\n  ✨ Perfect GT accuracy! Clustering approach is valid for this config.")
    
    return results


# =============================================================================
# Streaming Benchmark
# =============================================================================

def run_streaming_benchmark(config: BenchmarkConfig, verbose: bool = True) -> dict:
    """
    Streaming ANN benchmark without storing the full dataset.
    
    This function:
    1. Generates cluster config deterministically from seed
    2. Streams data batches into ANN index (vectors discarded after indexing)
    3. Generates queries and ground truth on-the-fly (only nearby clusters generated)
    4. Searches ANN and computes recall
    
    Args:
        config: Benchmark configuration
        verbose: Print progress information
    
    Returns:
        Dictionary with benchmark results (recall, timing, etc.)
    """
    import time
    results = {}
    
    if verbose:
        print("=" * 60)
        print("Streaming ANN Benchmark")
        print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Load cluster config from pre-extracted stats
    # -------------------------------------------------------------------------
    cluster_config = get_cluster_config(config)
    
    
    if verbose:
        print_config(config)
        print_config(cluster_config)
    
    # -------------------------------------------------------------------------
    # Step 2: Stream data into ANN index (NO STORAGE)
    # -------------------------------------------------------------------------
    if verbose:
        index_name = type(config.ann_index).__name__
        print(f"\n[Step 1] Streaming data into {index_name}...")
    
    # Use the ANN index from config
    if config.ann_index is None:
        raise ValueError("ann_index must be provided in BenchmarkConfig")
    ann_index = config.ann_index
    
    num_batches = (config.total_rows + config.batch_size - 1) // config.batch_size
    total_indexed = 0
    
    build_start = time.perf_counter()

    for batch_num in tqdm(range(num_batches), desc="Building index in data batches"):
        # Generate batch
        vectors, _ = gen_batch(batch_num, config.batch_size, config.total_rows, cluster_config)
        total_indexed += len(vectors)
        # Build or extend index
        if batch_num == 0:
            ann_index.build(vectors)
        else:
            ann_index.extend(vectors)

    # ===for debugging===
    all_vectors = []
    points_per_cluster = get_num_points_per_cluster(config.total_rows, cluster_config)
    for cluster_id in tqdm(range(cluster_config.nclusters)):
        cluster_points = gen_cluster(cluster_id, points_per_cluster[cluster_id], cluster_config)
        all_vectors.append(cluster_points)
    all_data = np.vstack(all_vectors).astype(np.float32)
    # ===for debugging===

    build_time = time.perf_counter() - build_start
    results['build_time_sec'] = build_time
    results['total_indexed'] = total_indexed
    
    if verbose:
        print(f"  Total indexed: {total_indexed:,} vectors")
        print(f"  Build time: {build_time:.2f}s")

    # -------------------------------------------------------------------------
    # Step 3: Generate queries and GT on-the-fly (only nearby clusters)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 2] Generating queries and ground truth on-the-fly using {config.gt_backend} backend, probing {config.nprobes}/{cluster_config.nclusters} clusters...")
    
    gt_start = time.perf_counter()
    queries, gt_indices, gt_distances = gen_gt(
        nqueries=config.nqueries,
        total_rows=config.total_rows,
        config=cluster_config,
        k=config.k,
        nprobes=config.nprobes,
        backend=config.gt_backend
    )
    gt_time = time.perf_counter() - gt_start
    results['gt_time_sec'] = gt_time

    print(f"gt_indices[0]: {gt_indices[0]}")
    print(f"gt_distances[0]: {gt_distances[0]}")

    # Estimate how many points we actually searched for GT
    points_per_cluster = get_num_points_per_cluster(config.total_rows, cluster_config)
    avg_points_searched = config.nprobes * points_per_cluster.mean()
    
    if verbose:
        print(f"  Queries: {config.nqueries}, k: {config.k}")
        print(f"  Avg points searched per query: {avg_points_searched:,.0f} / {config.total_rows:,}")
        print(f"  GT generation time: {gt_time:.2f}s")
    
    # -------------------------------------------------------------------------
    # Step 4: Search ANN index and compute recall
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 3] Searching {index_name}...")
    
    search_start = time.perf_counter()
    predicted_indices, predicted_distances = ann_index.search(queries, config.k)
    search_time = time.perf_counter() - search_start
    results['search_time_sec'] = search_time

    print(f"predicted_indices[0]: {predicted_indices[0]}")
    print(f"predicted_distances[0]: {predicted_distances[0]}")
    
    recall, _ = compute_recall_with_ties(
        predicted_indices, predicted_distances,
        gt_indices, gt_distances
    )
    results['recall'] = recall
    
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"  🙌 Recall@{config.k} (with distance ties): {recall:.4f}")
        print(f"  Index build time: {build_time:.2f}s")
        print(f"  GT generation time: {gt_time:.2f}s")
        print(f"  Search time: {search_time:.4f}s")
        
        # Memory analysis
        print(f"\n[Memory Analysis]")
        print(f"  If storing all {config.total_rows:,} vectors at {cluster_config.ncols} dims:")
        data_size_gb = (config.total_rows * cluster_config.ncols * 4) / (1024**3)
        print(f"    Storage needed: {data_size_gb:.2f} GB")
        print(f"  With streaming approach:")
        batch_size_gb = (config.batch_size * cluster_config.ncols * 4) / (1024**3)
        print(f"    Max batch memory: {batch_size_gb:.4f} GB")
    
    return results


# =============================================================================
# Demo / Test Functions
# =============================================================================

def demo_benchmark(config: BenchmarkConfig):
    """
    Demonstrate the streaming benchmark with example configuration.
    """
    
    results = run_streaming_benchmark(config, verbose=True)
    return results


def test_gt_accuracy(config: BenchmarkConfig):
    """Test that clustering-based GT is accurate compared to full brute force."""
    results = verify_clustering_gt_accuracy(
        config=config,
        verbose=True
    )
    
    print("\nGT accuracy tests complete!\n\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scalable Ground Truth Generation and ANN Benchmarking")
    parser.add_argument("--cluster_stats", type=str, required=True,
                        help="Path to cluster_stats.npz from extract_values.py (required)")
    parser.add_argument("--total_rows", type=int, default=1_000_000,
                        help="Total number of vectors to generate (default: 1M)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--batch_size", type=int, default=100_000,
                        help="Batch size for streaming (default: 100K)")
    parser.add_argument("--nqueries", type=int, default=100,
                        help="Number of queries (default: 100)")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of nearest neighbors (default: 16)")
    parser.add_argument("--nprobes", type=int, default=1,
                        help="Number of clusters to probe for GT (default: 1)")
    parser.add_argument("--gt_backend", type=str, default="cuvs", choices=["cuvs", "sklearn"],
                        help="Backend for brute force GT computation (default: cuvs)")
    parser.add_argument("--mode", type=str, default="test_gt", choices=["test_gt", "benchmark"],
                        help="Mode: test_gt (verify GT accuracy) or benchmark (streaming ANN benchmark)")
    
    args = parser.parse_args()
    
    # Create the ANN index
    build_params = cagra.IndexParams(
        metric="sqeuclidean",
        intermediate_graph_degree=256,
        graph_degree=128,
        build_algo="nn_descent",
    )
    ann_index = CagraIndex(build_params=build_params)
    
    config = BenchmarkConfig(
        total_rows=args.total_rows,
        seed=args.seed,
        batch_size=args.batch_size,
        nqueries=args.nqueries,
        k=args.k,
        nprobes=args.nprobes,
        gt_backend=args.gt_backend,
        cluster_stats_path=args.cluster_stats,
        ann_index=ann_index,
    )

    if args.mode == "test_gt":
        # Test for checking if clustering-based GT is accurate compared to full brute force.
        test_gt_accuracy(config)
    else:
        # Streaming benchmark to compare the performance of different ANN algorithms.
        demo_benchmark(config)
