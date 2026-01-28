"""
Scalable Ground Truth Generation for 100B-scale ANN Benchmarking

Key idea: Generate clustered data deterministically so we can:
1. Stream/batch data generation without storing everything
2. Compute ground truth by only generating nearby clusters (like IVF nprobes)
"""

import numpy as np
import cupy as cp
from typing import Tuple, List, Literal, Optional, Union
from tqdm import tqdm
from cuvs.neighbors import cagra

from config import ClusterConfig, BenchmarkConfig, get_cluster_config
from ann_indices import CagraIndex, IvfPqIndex
from utils import brute_force_knn, cuvs_brute_force_knn_gpu, compute_recall, compute_recall_with_ties, plot_clusters_2d, print_config


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
    Generate points for a single cluster deterministically using normal distribution (CPU).
    Same cluster_id + config always produces identical points.
    
    Uses per-dimension variance: config.cluster_variances shape (nclusters, ncols)
    """
    seed = get_cluster_seed(config.seed, cluster_id)
    rng = np.random.default_rng(seed)
    
    center = config.cluster_centers[cluster_id]
    variance = config.cluster_variances[cluster_id]  # (ncols,) per-dimension
    scale = np.sqrt(variance)
    
    # Generate points using normal distribution
    points = rng.normal(loc=center, scale=scale, size=(n_points, config.ncols))
    
    return points.astype(np.float32)


def gen_cluster_gpu(cluster_id: int, n_points: int, config: ClusterConfig, return_cupy: bool = True) -> Union[cp.ndarray, np.ndarray]:
    """
    Generate points for a single cluster deterministically using GPU (CuPy).
    Same cluster_id + config always produces identical points (within GPU implementation).
    
    Note: GPU random numbers will differ from CPU version but are deterministic on GPU.
    
    Uses per-dimension variance: config.cluster_variances shape (nclusters, ncols)
    
    Args:
        cluster_id: ID of the cluster to generate
        n_points: Number of points to generate
        config: Cluster configuration
        return_cupy: If True, return CuPy array (stays on GPU). If False, return NumPy array.
    
    Returns:
        Generated points as CuPy array (on GPU) or NumPy array (on CPU)
    """
    seed = get_cluster_seed(config.seed, cluster_id)
    
    # Use CuPy's RandomState for deterministic seeding (compatible API)
    rng = cp.random.RandomState(seed)
    
    # Transfer center and variance to GPU (these are small, can be cached)
    center = cp.asarray(config.cluster_centers[cluster_id])
    variance = cp.asarray(config.cluster_variances[cluster_id])  # (ncols,) per-dimension
    scale = cp.sqrt(variance)
    
    # Generate standard normal samples directly into output array
    # Then transform IN-PLACE to minimize GPU memory usage: X = center + scale * Z
    # This uses ~30GB instead of ~90GB for large clusters
    points = rng.standard_normal(size=(n_points, config.ncols), dtype=cp.float32)
    points *= scale   # In-place multiplication (no new array)
    points += center  # In-place addition (no new array)
    
    if return_cupy:
        return points
    return cp.asnumpy(points)


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


def generate_mock_data(
    cluster_config: ClusterConfig,
    total_points: int,
    use_gpu_gen: bool = True,
) -> np.ndarray:
    """
    Generate full mock dataset using cluster configuration.
    
    For clusters with near-zero variance, generates just 1 point (the centroid).
    The remaining points are redistributed to normal clusters proportionally.
    
    Args:
        cluster_config: ClusterConfig with centroids, variances, densities
        total_points: Total number of points to generate
        use_gpu_gen: If True, use GPU for data generation (faster). Default False.
    
    Returns:
        Generated data array (total_points, ncols)
    """
    # Get points per cluster using existing logic
    points_per_cluster = get_num_points_per_cluster(total_points, cluster_config)

    # Pre-allocate result array
    ncols = cluster_config.cluster_centers.shape[1]
    result = np.empty((total_points, ncols), dtype=np.float32)
    write_idx = 0

    for cluster_id in tqdm(range(cluster_config.nclusters), desc="Generating mock data"):
        n_points = points_per_cluster[cluster_id]
        if n_points <= 0:
            continue
        if use_gpu_gen:
            cluster_points = gen_cluster_gpu(cluster_id, n_points, cluster_config, return_cupy=False)
        else:
            cluster_points = gen_cluster(cluster_id, n_points, cluster_config)
        result[write_idx:write_idx + n_points] = cluster_points
        write_idx += n_points

    return result


def gen_build_sample(
    sample_size: int,
    total_rows: int,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a build sample that includes vectors from ALL clusters proportionally.
    
    This ensures the initial index build has representative data from every cluster,
    which is important for algorithms that use build data to set up internal structures.
    
    Args:
        sample_size: Total number of vectors to sample for build
        total_rows: Total dataset size
        config: Cluster configuration
    
    Returns:
        vectors: (sample_size, ncols) array of vectors
        cluster_ids: (sample_size,) array of cluster assignments
        n_sampled_per_cluster: (nclusters,) how many vectors sampled from each cluster
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    
    # Calculate how many to sample from each cluster
    # Sample proportionally based on cluster sizes, with minimum 1 per cluster if possible
    base_per_cluster = sample_size // config.nclusters
    n_sampled_per_cluster = np.minimum(
        np.full(config.nclusters, base_per_cluster, dtype=np.int64),
        points_per_cluster
    )
    
    # Distribute remaining samples to clusters that have capacity
    remaining = sample_size - n_sampled_per_cluster.sum()
    for cluster_id in range(config.nclusters):
        if remaining <= 0:
            break
        capacity = points_per_cluster[cluster_id] - n_sampled_per_cluster[cluster_id]
        add = min(remaining, capacity)
        n_sampled_per_cluster[cluster_id] += add
        remaining -= add
    
    # Generate sampled vectors from each cluster (take first N from each)
    vectors = []
    cluster_ids = []
    
    for cluster_id in range(config.nclusters):
        n_sample = n_sampled_per_cluster[cluster_id]
        if n_sample > 0:
            # Generate full cluster and take first n_sample vectors
            n_cluster_points = points_per_cluster[cluster_id]
            cluster_points = gen_cluster(cluster_id, n_cluster_points, config)
            vectors.append(cluster_points[:n_sample])
            cluster_ids.append(np.full(n_sample, cluster_id, dtype=np.int32))
    
    vectors = np.vstack(vectors).astype(np.float32)
    cluster_ids = np.concatenate(cluster_ids)
    
    return vectors, cluster_ids, n_sampled_per_cluster


def gen_extend_batch(
    batch_num: int,
    batch_size: int,
    total_rows: int,
    config: ClusterConfig,
    n_sampled_per_cluster: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of vectors for extend, skipping vectors already used in build.
    
    Args:
        batch_num: Batch number (0-indexed, for the extend phase)
        batch_size: Number of vectors per batch
        total_rows: Total dataset size
        config: Cluster configuration
        n_sampled_per_cluster: How many vectors were sampled from each cluster for build
    
    Returns:
        vectors: (batch_size, ncols) array of vectors
        cluster_ids: (batch_size,) array of cluster assignments
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    remaining_per_cluster = points_per_cluster - n_sampled_per_cluster
    
    # Calculate cumulative sum for remaining vectors
    cumsum_remaining = np.cumsum(remaining_per_cluster)
    total_remaining = cumsum_remaining[-1]
    
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, total_remaining)
    actual_batch_size = end_idx - start_idx
    if actual_batch_size <= 0:
        return np.array([]).reshape(0, config.ncols), np.array([])
    
    vectors = []
    cluster_ids = []
    
    for cluster_id in range(config.nclusters):
        cluster_start = 0 if cluster_id == 0 else cumsum_remaining[cluster_id - 1]
        cluster_end = cumsum_remaining[cluster_id]
        
        if cluster_start >= end_idx:
            break
        if cluster_end <= start_idx:
            continue
        
        # This cluster overlaps with batch range
        overlap_start = max(start_idx, cluster_start)
        overlap_end = min(end_idx, cluster_end)
        
        # Generate full cluster, then slice the portion we need
        # Skip the first n_sampled vectors (already used in build)
        n_cluster_points = points_per_cluster[cluster_id]
        cluster_points = gen_cluster(cluster_id, n_cluster_points, config)
        
        # Local indices within the REMAINING portion of this cluster
        local_start = overlap_start - cluster_start
        local_end = overlap_end - cluster_start
        
        # Offset by n_sampled to skip build vectors
        offset = n_sampled_per_cluster[cluster_id]
        vectors.append(cluster_points[offset + local_start:offset + local_end])
        cluster_ids.append(np.full(local_end - local_start, cluster_id, dtype=np.int32))
    
    if len(vectors) == 0:
        return np.array([]).reshape(0, config.ncols), np.array([])
    
    vectors = np.vstack(vectors).astype(np.float32)
    cluster_ids = np.concatenate(cluster_ids)
    
    return vectors, cluster_ids


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

def compute_true_gt_streaming(
    queries: np.ndarray,
    cluster_config: ClusterConfig,
    total_rows: int,
    k: int,
    backend: str = "cuvs",
    use_gpu_gen: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute true brute-force GT by streaming through ALL clusters.
    
    Instead of loading all data into memory, generates each cluster's data
    on-the-fly, computes brute force, and merges results incrementally.
    
    Args:
        queries: (nqueries, ncols) query vectors
        cluster_config: Cluster configuration
        total_rows: Total number of vectors
        k: Number of nearest neighbors
        backend: "cuvs" or "sklearn" for brute force
        use_gpu_gen: If True, use GPU for data generation (faster). Default True.
    
    Returns:
        gt_indices: (nqueries, k) ground truth neighbor indices
        gt_distances: (nqueries, k) ground truth distances
        timing: dict with 'gen_time' and 'bf_time' breakdowns
    """
    import time
    
    nqueries = len(queries)
    points_per_cluster = get_num_points_per_cluster(total_rows, cluster_config)
    cumsum = np.cumsum(points_per_cluster)
    
    # Initialize running top-k
    gt_indices = np.full((nqueries, k), -1, dtype=np.int64)
    gt_distances = np.full((nqueries, k), np.inf, dtype=np.float32)
    
    # Track timing
    total_gen_time = 0.0
    total_bf_time = 0.0
    total_merge_time = 0.0
    
    # For GPU path: keep queries on GPU to avoid repeated transfers
    if use_gpu_gen and backend == "cuvs":
        queries_gpu = cp.asarray(queries)
    
    print(f"Computing true GT by streaming through {cluster_config.nclusters} clusters...")
    
    # Process each cluster one at a time
    for cluster_id in tqdm(range(cluster_config.nclusters), desc="Streaming clusters for true GT"):
        n_points = points_per_cluster[cluster_id]
        
        # Calculate global index offset for this cluster
        global_start = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
        
        if use_gpu_gen and backend == "cuvs":
            # GPU path: generate on GPU and keep data there
            gen_start = time.perf_counter()
            cluster_points_gpu = gen_cluster_gpu(cluster_id, n_points, cluster_config, return_cupy=True)
            total_gen_time += time.perf_counter() - gen_start
            
            # Brute force directly on GPU (no CPU->GPU transfer)
            bf_start = time.perf_counter()
            k_cluster = min(k, n_points)
            local_indices, local_dists = cuvs_brute_force_knn_gpu(
                cluster_points_gpu,
                queries_gpu,
                k_cluster
            )
            total_bf_time += time.perf_counter() - bf_start
        else:
            # CPU path: original behavior
            gen_start = time.perf_counter()
            cluster_points = gen_cluster(cluster_id, n_points, cluster_config)
            total_gen_time += time.perf_counter() - gen_start
            
            bf_start = time.perf_counter()
            k_cluster = min(k, n_points)
            local_indices, local_dists = brute_force_knn(
                cluster_points,
                queries,
                k_cluster,
                backend=backend
            )
            total_bf_time += time.perf_counter() - bf_start
        
        # Convert local indices to global indices
        global_indices = local_indices + global_start
        
        # Time: Merge results for each query
        merge_start = time.perf_counter()
        for q_idx in range(nqueries):
            merged_indices = np.concatenate([gt_indices[q_idx], global_indices[q_idx]])
            merged_dists = np.concatenate([gt_distances[q_idx], local_dists[q_idx]])
            sort_order = np.argsort(merged_dists)[:k]
            gt_indices[q_idx] = merged_indices[sort_order]
            gt_distances[q_idx] = merged_dists[sort_order]
        total_merge_time += time.perf_counter() - merge_start
    
    timing = {'gen_time': total_gen_time, 'bf_time': total_bf_time, 'merge_time': total_merge_time}
    return gt_indices, gt_distances, timing


def generate_queries_and_gt_batched(
    data: np.ndarray,
    n_queries: int,
    k: int,
    batch_size: int = 1_000_000,
    seed: int = 12345,
    backend: str = "cuvs",
    queries: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate queries and compute ground truth via batched brute force.
    
    For large datasets (e.g., 50M vectors), processes data in batches and 
    merges results incrementally to avoid OOM.
    
    Args:
        data: Dataset to query (can be very large)
        n_queries: Number of queries
        k: Number of nearest neighbors
        batch_size: Number of vectors to process per batch
        seed: Random seed
        backend: "cuvs" or "sklearn" for brute force
        queries: Optional pre-generated queries (if None, samples from data)
    
    Returns:
        queries: (n_queries, ncols)
        gt_indices: (n_queries, k)
        gt_distances: (n_queries, k)
    """
    n_rows = len(data)
    
    if queries is None:
        rng = np.random.default_rng(seed)
        
        # Sample queries from the dataset (with small noise to avoid exact matches)
        query_indices = rng.choice(n_rows, size=n_queries, replace=False)
        queries = data[query_indices].copy()
        # Add small noise
        noise_scale = np.std(data[:min(10000000, n_rows)]) * 0.1  # Use subset for std to avoid loading all data
        queries += rng.normal(0, noise_scale, queries.shape).astype(np.float32)
    
    # Initialize running top-k
    gt_indices = np.full((n_queries, k), -1, dtype=np.int64)
    gt_distances = np.full((n_queries, k), np.inf, dtype=np.float32)
    
    num_batches = (n_rows + batch_size - 1) // batch_size
    print(f"Computing batched GT for {n_queries} queries over {n_rows:,} vectors ({num_batches} batches)...")
    
    for batch_num in tqdm(range(num_batches), desc="GT batches"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        batch_data = data[start_idx:end_idx]
        
        # Compute brute force KNN for this batch
        batch_indices, batch_distances = brute_force_knn(batch_data, queries, k, backend=backend)
        
        # Convert local batch indices to global indices
        batch_indices = batch_indices + start_idx
        
        # Merge with running top-k
        for q_idx in range(n_queries):
            # Concatenate current top-k with batch results
            merged_indices = np.concatenate([gt_indices[q_idx], batch_indices[q_idx]])
            merged_distances = np.concatenate([gt_distances[q_idx], batch_distances[q_idx]])
            
            # Sort by distance and keep top k
            sort_order = np.argsort(merged_distances)[:k]
            gt_indices[q_idx] = merged_indices[sort_order]
            gt_distances[q_idx] = merged_distances[sort_order]
    
    return queries, gt_indices, gt_distances


def gen_queries(
    nqueries: int,
    total_rows: int,
    config: ClusterConfig,
    query_seed_offset: int = 999999,
    use_gpu_gen: bool = True,
) -> np.ndarray:
    """
    Generate query vectors by sampling from the synthetic dataset.
    
    Args:
        nqueries: Number of query points to generate
        total_rows: Total dataset size (for calculating points per cluster)
        config: Cluster configuration
        query_seed_offset: Offset for query generation seed
        use_gpu_gen: If True, use GPU for data generation (faster). Default True.
    
    Returns:
        queries: (nqueries, ncols) query vectors
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)
    
    # Generate query points by sampling from actual data points
    query_rng = np.random.default_rng(config.seed + query_seed_offset)
    
    # Sample global indices uniformly from the dataset
    query_global_indices = query_rng.choice(total_rows, size=nqueries, replace=False)
    
    # Convert global indices to (cluster_id, local_index) pairs
    query_cluster_ids = np.searchsorted(cumsum, query_global_indices, side='right')
    
    # Group queries by cluster to generate efficiently
    cluster_to_query_info = {}
    for q_idx, (global_idx, cluster_id) in tqdm(enumerate(zip(query_global_indices, query_cluster_ids)), 
                                                  total=nqueries, desc="Grouping queries by cluster"):
        local_idx = global_idx if cluster_id == 0 else global_idx - cumsum[cluster_id - 1]
        if cluster_id not in cluster_to_query_info:
            cluster_to_query_info[cluster_id] = []
        cluster_to_query_info[cluster_id].append((q_idx, local_idx))
    
    # Generate query points from actual data
    queries = np.zeros((nqueries, config.ncols), dtype=np.float32)
    for cluster_id, query_info_list in tqdm(cluster_to_query_info.items(), desc="Generating query points"):
        n_points = points_per_cluster[cluster_id]
        
        if use_gpu_gen:
            # GPU path: generate on GPU, then copy back
            cluster_points_gpu = gen_cluster_gpu(cluster_id, n_points, config, return_cupy=True)
            cluster_points = cp.asnumpy(cluster_points_gpu)
        else:
            cluster_points = gen_cluster(cluster_id, n_points, config)
        
        for q_idx, local_idx in query_info_list:
            queries[q_idx] = cluster_points[local_idx]
    
    # Add small noise to avoid exact matches
    noise_scale = np.std(queries) * 0.1
    queries += query_rng.normal(0, noise_scale, queries.shape).astype(np.float32)
    
    return queries


def compute_cluster_gt(
    queries: np.ndarray,
    total_rows: int,
    config: ClusterConfig,
    k: int = 10,
    nprobes: int = 10,
    backend: Literal["cuvs", "sklearn"] = "cuvs",
    metric: str = "sqeuclidean",
    use_gpu_gen: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute cluster-based ground truth for given queries.
    
    Instead of brute-forcing against all total_rows points,
    we only generate points from the nprobes nearest clusters.
    
    Args:
        queries: (nqueries, ncols) query vectors
        total_rows: Total dataset size (for calculating points per cluster)
        config: Cluster configuration
        k: Number of nearest neighbors for ground truth
        nprobes: Number of nearby clusters to search (like IVF)
        backend: "cuvs" for GPU brute force or "sklearn" for CPU brute force
        metric: Distance metric
        use_gpu_gen: If True, use GPU for data generation (faster). Default True.
    
    Returns:
        gt_indices: (nqueries, k) ground truth neighbor indices  
        gt_distances: (nqueries, k) ground truth distances
        timing: dict with 'gen_time' and 'bf_time' breakdowns
    """
    import time
    
    nqueries = len(queries)
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)

    # Time: Find nearby clusters for each query
    find_clusters_start = time.perf_counter()
    nearby_clusters = find_nearby_clusters(queries, config, nprobes, backend, metric)
    total_find_clusters_time = time.perf_counter() - find_clusters_start

    # Initialize GT arrays
    gt_indices = np.full((nqueries, k), -1, dtype=np.int64)
    gt_distances = np.full((nqueries, k), np.inf, dtype=np.float32)
    
    # Track timing
    total_gen_time = 0.0
    total_bf_time = 0.0
    total_merge_time = 0.0
    
    # Time: Build reverse mapping
    mapping_start = time.perf_counter()
    cluster_to_queries = {}
    for q_idx in tqdm(range(nqueries), desc="Building reverse mapping"):
        for cluster_id in nearby_clusters[q_idx]:
            cluster_id = int(cluster_id)
            if cluster_id not in cluster_to_queries:
                cluster_to_queries[cluster_id] = []
            cluster_to_queries[cluster_id].append(q_idx)
    total_mapping_time = time.perf_counter() - mapping_start
    
    # Process each cluster once
    for cluster_id in tqdm(cluster_to_queries.keys(), desc="Processing clusters for GT"):
        query_indices = cluster_to_queries[cluster_id]
        n_points = points_per_cluster[cluster_id]
        
        global_start = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
        batch_queries = queries[query_indices]
        
        if use_gpu_gen and backend == "cuvs":
            # GPU path: generate on GPU and keep data there
            gen_start = time.perf_counter()
            cluster_points_gpu = gen_cluster_gpu(cluster_id, n_points, config, return_cupy=True)
            total_gen_time += time.perf_counter() - gen_start
            
            # Brute force directly on GPU (no CPU->GPU transfer)
            bf_start = time.perf_counter()
            k_cluster = min(k, n_points)
            batch_queries_gpu = cp.asarray(batch_queries)
            local_indices, local_dists = cuvs_brute_force_knn_gpu(
                cluster_points_gpu, batch_queries_gpu, k_cluster, metric=metric
            )
            total_bf_time += time.perf_counter() - bf_start
        else:
            # CPU path: original behavior
            gen_start = time.perf_counter()
            cluster_points = gen_cluster(cluster_id, n_points, config)
            total_gen_time += time.perf_counter() - gen_start
            
            bf_start = time.perf_counter()
            k_cluster = min(k, n_points)
            local_indices, local_dists = brute_force_knn(
                cluster_points, batch_queries, k_cluster, backend=backend
            )
            total_bf_time += time.perf_counter() - bf_start
        
        global_indices = local_indices + global_start
        
        # Time: Merge results
        merge_start = time.perf_counter()
        for i, q_idx in enumerate(query_indices):
            merged_indices = np.concatenate([gt_indices[q_idx], global_indices[i]])
            merged_dists = np.concatenate([gt_distances[q_idx], local_dists[i]])
            sort_order = np.argsort(merged_dists)[:k]
            gt_indices[q_idx] = merged_indices[sort_order]
            gt_distances[q_idx] = merged_dists[sort_order]
        total_merge_time += time.perf_counter() - merge_start

    timing = {
        'find_clusters_time': total_find_clusters_time,
        'mapping_time': total_mapping_time,
        'gen_time': total_gen_time,
        'bf_time': total_bf_time,
        'merge_time': total_merge_time
    }
    return gt_indices, gt_distances, timing


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
    Generate queries and compute ground truth (convenience wrapper).
    
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
    queries = gen_queries(nqueries, total_rows, config, query_seed_offset)
    gt_indices, gt_distances, _ = compute_cluster_gt(queries, total_rows, config, k, nprobes, backend, metric)
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
    
    # Step 1: Validate configuration (no longer generating all data at once to save memory)
    if verbose:
        print(f"\n[Step 1] Validating configuration for {config.total_rows:,} vectors, {cluster_config.nclusters} clusters...")
    
    import time
    step1_start = time.perf_counter()
    
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
    
    step1_time = time.perf_counter() - step1_start
    if verbose:
        print(f"  Cluster sizes: min={min_points}, max={points_per_cluster.max()}, mean={points_per_cluster.mean():.0f}")
        print(f"  ⏱️  Step 1 time: {step1_time:.2f}s")
    
    # Step 2: Generate queries
    if verbose:
        print(f"\n[Step 2] Generating {config.nqueries} queries...")

    step2_start = time.perf_counter()
    queries = gen_queries(
        nqueries=config.nqueries,
        total_rows=config.total_rows,
        config=cluster_config,
    )
    step2_time = time.perf_counter() - step2_start

    if verbose:
        print(f"  Generated {config.nqueries} queries")
        print(f"  ⏱️  Step 2 time: {step2_time:.2f}s")
    
    # Step 3: Compute cluster-based GT
    if verbose:
        print(f"\n[Step 3] Computing cluster-based GT (nprobes={config.nprobes})...")

    step3_start = time.perf_counter()
    cluster_gt_indices, cluster_gt_distances, step3_timing = compute_cluster_gt(
        queries=queries,
        total_rows=config.total_rows,
        config=cluster_config,
        k=config.k,
        nprobes=config.nprobes,
        backend=config.gt_backend
    )
    step3_time = time.perf_counter() - step3_start

    if verbose:
        print(f"  Computed cluster-based GT for {config.nqueries} queries")
        print(f"  ⏱️  Step 3 time: {step3_time:.2f}s")
        print(f"      └─ Find nearby clusters: {step3_timing['find_clusters_time']:.2f}s")
        print(f"      └─ Build mapping:        {step3_timing['mapping_time']:.2f}s")
        print(f"      └─ Data generation:      {step3_timing['gen_time']:.2f}s")
        print(f"      └─ Brute force KNN:      {step3_timing['bf_time']:.2f}s")
        print(f"      └─ Merge results:        {step3_timing['merge_time']:.2f}s")
    
    # Step 4: Compute TRUE brute force GT by streaming through ALL clusters (no OOM!)
    if verbose:
        print(f"\n[Step 4] Computing TRUE brute force GT on all {config.total_rows:,} vectors (streaming)...")
    
    step4_start = time.perf_counter()
    true_gt_indices, true_gt_distances, step4_timing = compute_true_gt_streaming(
        queries=queries,
        cluster_config=cluster_config,
        total_rows=config.total_rows,
        k=config.k,
        backend=config.gt_backend
    )
    step4_time = time.perf_counter() - step4_start

    if verbose:
        print(f"  Computed true GT for {config.nqueries} queries")
        print(f"  ⏱️  Step 4 time: {step4_time:.2f}s")
        print(f"      └─ Data generation:      {step4_timing['gen_time']:.2f}s")
        print(f"      └─ Brute force KNN:      {step4_timing['bf_time']:.2f}s")
        print(f"      └─ Merge results:        {step4_timing['merge_time']:.2f}s")
    
    # Step 5: Compare cluster-based GT vs true GT
    if verbose:
        print(f"\n[Step 5] Comparing cluster-based GT vs true brute force GT...")
    
    step5_start = time.perf_counter()
    
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
    
    step5_time = time.perf_counter() - step5_start
    total_time = step1_time + step2_time + step3_time + step4_time + step5_time
    
    results = {
        'gt_accuracy': gt_accuracy,
        'total_queries': config.nqueries,
        'incorrect_queries': len(incorrect_queries),
        'nprobes_used': config.nprobes,
        'step1_time': step1_time,
        'step2_time': step2_time,
        'step3_time': step3_time,
        'step3_find_clusters_time': step3_timing['find_clusters_time'],
        'step3_mapping_time': step3_timing['mapping_time'],
        'step3_gen_time': step3_timing['gen_time'],
        'step3_bf_time': step3_timing['bf_time'],
        'step3_merge_time': step3_timing['merge_time'],
        'step4_time': step4_time,
        'step4_gen_time': step4_timing['gen_time'],
        'step4_bf_time': step4_timing['bf_time'],
        'step4_merge_time': step4_timing['merge_time'],
        'step5_time': step5_time,
        'total_time': total_time,
    }
    
    if verbose:
        print(f"  ⏱️  Step 5 time: {step5_time:.2f}s")
        
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
        
        print(f"\n" + "-" * 60)
        print("Timing Summary")
        print("-" * 60)
        print(f"  Step 1 (Validate config):    {step1_time:6.2f}s")
        print(f"  Step 2 (Generate queries):   {step2_time:6.2f}s")
        print(f"  Step 3 (Cluster-based GT):   {step3_time:6.2f}s")
        print(f"      └─ Find nearby clusters: {step3_timing['find_clusters_time']:6.2f}s")
        print(f"      └─ Build mapping:        {step3_timing['mapping_time']:6.2f}s")
        print(f"      └─ Data generation:      {step3_timing['gen_time']:6.2f}s")
        print(f"      └─ Brute force KNN:      {step3_timing['bf_time']:6.2f}s")
        print(f"      └─ Merge results:        {step3_timing['merge_time']:6.2f}s")
        print(f"  Step 4 (True GT streaming):  {step4_time:6.2f}s")
        print(f"      └─ Data generation:      {step4_timing['gen_time']:6.2f}s")
        print(f"      └─ Brute force KNN:      {step4_timing['bf_time']:6.2f}s")
        print(f"      └─ Merge results:        {step4_timing['merge_time']:6.2f}s")
        print(f"  Step 5 (Compare):            {step5_time:6.2f}s")
        print(f"  Total:                       {total_time:6.2f}s")
    
    return results


# =============================================================================
# Streaming Benchmark
# =============================================================================

def run_streaming_benchmark(config: BenchmarkConfig, itopk_list: list = None, verbose: bool = True) -> dict:
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
    
    total_indexed = 0
    build_start = time.perf_counter()

    # Step 2a: Build with sample from ALL clusters (ensures representative initial index)
    if verbose:
        print(f"  Building initial index with sample from all clusters...")
    build_vectors, _, n_sampled_per_cluster = gen_build_sample(
        sample_size=config.batch_size,
        total_rows=config.total_rows,
        config=cluster_config
    )
    ann_index.build(build_vectors)
    total_indexed += len(build_vectors)
    if verbose:
        print(f"  Built with {len(build_vectors):,} vectors from {(n_sampled_per_cluster > 0).sum()} clusters")
    
    # Step 2b: Extend with remaining vectors (skipping those used in build)
    total_remaining = config.total_rows - n_sampled_per_cluster.sum()
    num_extend_batches = (total_remaining + config.batch_size - 1) // config.batch_size
    
    for batch_num in tqdm(range(num_extend_batches), desc="Extending index with remaining batches"):
        vectors, _ = gen_extend_batch(
            batch_num, config.batch_size, config.total_rows, 
            cluster_config, n_sampled_per_cluster
        )
        if len(vectors) > 0:
            ann_index.extend(vectors)
            total_indexed += len(vectors)

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

    # Estimate how many points we actually searched for GT
    points_per_cluster = get_num_points_per_cluster(config.total_rows, cluster_config)
    avg_points_searched = config.nprobes * points_per_cluster.mean()
    
    if verbose:
        print(f"  Queries: {config.nqueries}, k: {config.k}")
        print(f"  Avg points searched per query: {avg_points_searched:,.0f} / {config.total_rows:,}")
        print(f"  GT generation time: {gt_time:.2f}s")
    
    # -------------------------------------------------------------------------
    # Step 4: Search ANN index and compute recall (sweep itopk if provided)
    # -------------------------------------------------------------------------
    if itopk_list is None:
        itopk_list = [None]  # Use default search params
    
    if verbose:
        print(f"\n[Step 3] Searching {index_name} with itopk sweep: {itopk_list}...")
    
    # Store results for each itopk value
    pareto_results = []
    
    for itopk in itopk_list:
        # Warmup search (not timed)
        _ = ann_index.search(queries, config.k, itopk=itopk)
        
        # Timed search
        search_start = time.perf_counter()
        predicted_indices, predicted_distances = ann_index.search(queries, config.k, itopk=itopk)
        search_time = time.perf_counter() - search_start
        
        recall, _ = compute_recall_with_ties(
            predicted_indices, predicted_distances,
            gt_indices, gt_distances
        )
        
        qps = config.nqueries / search_time
        
        pareto_results.append({
            'itopk': itopk,
            'recall': recall,
            'search_time_sec': search_time,
            'qps': qps,
        })
        
        if verbose:
            itopk_str = itopk if itopk is not None else "default"
            print(f"  itopk={itopk_str}: recall={recall:.4f}, search_time={search_time:.4f}s, QPS={qps:,.0f}")
    
    results['pareto_results'] = pareto_results
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Benchmark Results Summary")
        print("=" * 60)
        print(f"  Index build time: {build_time:.2f}s")
        print(f"  GT generation time: {gt_time:.2f}s")
        print(f"\n  Pareto curve data (itopk sweep):")
        print(f"  {'itopk':>8} {'recall':>10} {'QPS':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*12}")
        for pr in pareto_results:
            itopk_str = str(pr['itopk']) if pr['itopk'] is not None else "default"
            print(f"  {itopk_str:>8} {pr['recall']:>10.4f} {pr['qps']:>12,.0f}")
        
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

def demo_benchmark(config: BenchmarkConfig, itopk_list: list = None, output_path: str = None):
    """
    Demonstrate the streaming benchmark with example configuration.
    
    Args:
        config: Benchmark configuration
        itopk_list: List of itopk values to sweep (e.g., [32, 64, 128, 256])
        output_path: Optional path to save pareto results as .npz file
    """
    results = run_streaming_benchmark(config, itopk_list=itopk_list, verbose=True)
    
    # Save pareto results if output path provided
    if output_path is not None and 'pareto_results' in results:
        pareto = results['pareto_results']
        itopks = np.array([p['itopk'] if p['itopk'] is not None else 0 for p in pareto])
        recalls = np.array([p['recall'] for p in pareto])
        qps_arr = np.array([p['qps'] for p in pareto])
        search_times = np.array([p['search_time_sec'] for p in pareto])
        
        np.savez(
            output_path,
            itopk=itopks,
            recall=recalls,
            qps=qps_arr,
            search_time_sec=search_times,
            k=config.k,
            nqueries=config.nqueries,
            total_rows=config.total_rows,
        )
        print(f"\nPareto results saved to: {output_path}")
    
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
    parser.add_argument("--nqueries", type=int, default=1000,
                        help="Number of queries (default: 100)")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of nearest neighbors (default: 16)")
    parser.add_argument("--nprobes", type=int, default=1,
                        help="Number of clusters to probe for GT (default: 1)")
    parser.add_argument("--gt_backend", type=str, default="cuvs", choices=["cuvs", "sklearn"],
                        help="Backend for brute force GT computation (default: cuvs)")
    parser.add_argument("--mode", type=str, default="test_gt", choices=["test_gt", "benchmark"],
                        help="Mode: test_gt (verify GT accuracy) or benchmark (streaming ANN benchmark)")
    parser.add_argument("--itopk", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="List of itopk values to sweep for Pareto curve (default: 32 64 128 256)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save pareto results as .npz file (optional)")
    
    args = parser.parse_args()
    
    # Create the ANN index
    build_params = cagra.IndexParams(
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
        demo_benchmark(config, itopk_list=args.itopk, output_path=args.output)
    
    # Explicit GPU cleanup to avoid CUDA errors on exit
    import gc
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass
