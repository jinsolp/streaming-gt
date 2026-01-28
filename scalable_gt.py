"""
Scalable Ground Truth Generation for 100B-scale ANN Benchmarking

Key idea: Generate clustered data deterministically so we can:
1. Stream/batch data generation without storing everything
2. Compute ground truth by only generating nearby clusters (like IVF nprobes)
"""

import numpy as np
import cupy as cp
from typing import Tuple, List, Literal
from tqdm import tqdm
from cuvs.neighbors import cagra
import time

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


def _sample_standard(rng, shape, dtype, student_df=None):
    """
    Sample from standard normal or Student's t distribution.
    
    When student_df is set, uses Student's t with that many degrees of freedom,
    scaled to have unit variance (like standard normal).
    
    Lower df = heavier tails = more extreme points:
    - df=None or inf: Standard normal
    - df=5: Moderately heavy tails
    - df=3: Very heavy tails
    """
    if student_df is None or student_df > 100:
        return rng.standard_normal(shape, dtype=dtype)
    else:
        # Student's t = Z / sqrt(V/df) where Z ~ N(0,1), V ~ chi-squared(df)
        # CuPy Generator doesn't have standard_t, so we construct it
        z = rng.standard_normal(shape, dtype=dtype)
        # Use one chi-squared per row (shared across dimensions for correlation)
        n_samples = shape[0] if len(shape) > 1 else shape[0]
        chi2 = rng.chisquare(student_df, (n_samples, 1)).astype(dtype)
        t_samples = z / cp.sqrt(chi2 / student_df)
        
        # Student's t has variance = df/(df-2) for df > 2
        # Scale to unit variance by multiplying by sqrt((df-2)/df)
        if student_df > 2:
            scale_to_unit_var = cp.sqrt((student_df - 2) / student_df)
            t_samples = t_samples * scale_to_unit_var
        return t_samples


def gen_cluster(cluster_id: int, n_points: int, config: ClusterConfig) -> np.ndarray:
    """
    Generate points for a single cluster deterministically using GPU (CuPy).
    Same cluster_id + config always produces identical points.
    
    Supports:
    - Scalar variance per cluster: config.cluster_variances shape (nclusters,)
    - Per-dimension variance: config.cluster_variances shape (nclusters, ncols)
    - Low-rank covariance: if config.pca_components_list is provided, uses PCA-based sampling
    - Heavy-tailed distribution: if config.student_df is set, uses Student's t instead of Gaussian
    
    For low-variance clusters (n_points=1), returns the centroid directly.
    
    Args:
        cluster_id: Which cluster to generate
        n_points: Number of points to generate
        config: Cluster configuration
    
    Returns:
        numpy array of shape (n_points, ncols)
    """
    center = config.cluster_centers[cluster_id]
    
    # For low-variance clusters with 1 point, just return the centroid
    if n_points == 1:
        return center.reshape(1, -1).astype(np.float32)
    
    seed = get_cluster_seed(config.seed, cluster_id)
    rng = cp.random.default_rng(seed)
    
    # Check if low-rank covariance is available for this cluster
    if config.is_lowrank and config.pca_components_list[cluster_id] is not None:
        # Low-rank sampling: sample in PCA space, project back
        pca_components = cp.asarray(config.pca_components_list[cluster_id])  # (k, ncols)
        explained_var = cp.asarray(config.pca_explained_var_list[cluster_id])  # (k,)
        noise_var = float(config.pca_noise_var[cluster_id])
        center_gpu = cp.asarray(center)
        
        k = len(explained_var)
        
        # Sample in PCA space: z ~ N(0, explained_var * pca_scale) or Student's t
        # pca_scale < 1 reduces structure, making data harder to search
        raw_scaled_var = explained_var * config.pca_scale
        scaled_var = cp.maximum(raw_scaled_var, 0.0)
        z = _sample_standard(rng, (n_points, k), cp.float32, config.student_df) * cp.sqrt(scaled_var)
        
        # Project back to original space: z @ components
        projected = z @ pca_components  # (n_points, k) @ (k, ncols) = (n_points, ncols)
        
        # Add isotropic noise for unexplained variance (also heavy-tailed if specified)
        noise_std = float(cp.sqrt(max(noise_var, 0.0)))
        noise = _sample_standard(rng, (n_points, config.ncols), cp.float32, config.student_df) * noise_std
        
        # Combine projected + noise (centered points before adding center)
        centered_points = projected + noise
        
        # First: Scale to match original cluster variance (affects directions via stretching)
        actual_var_before = float(cp.var(centered_points))
        
        if config.cluster_variances.ndim == 2:
            # Per-dimension scaling for more accurate variance matching
            target_var_per_dim = cp.asarray(config.cluster_variances[cluster_id])  # (ncols,)
            actual_var_per_dim = cp.var(centered_points, axis=0)  # (ncols,)
            # Scale each dimension separately (avoid div by zero)
            scale_factors = cp.sqrt(target_var_per_dim / cp.maximum(actual_var_per_dim, 1e-10))
            centered_points = centered_points * scale_factors  # broadcasts (n_points, ncols) * (ncols,)
        else:
            # Scalar variance: single scale factor for all dimensions
            target_var = float(config.cluster_variances[cluster_id])
            if actual_var_before > 0 and target_var > 0:
                scale_factor = float(cp.sqrt(target_var / actual_var_before))
                centered_points = centered_points * scale_factor
        
        # Final points: center + scaled centered_points
        points = center_gpu + centered_points
    else:
        # Diagonal covariance fallback
        center_gpu = cp.asarray(center)
        
        # First: Generate points with correct per-dimension variance scaling
        # This affects the "directions" (relative contribution of each dimension)
        variance = config.cluster_variances[cluster_id]
        scale = cp.sqrt(cp.asarray(variance))
        centered_points = _sample_standard(rng, (n_points, config.ncols), cp.float32, config.student_df) * scale
        
        points = center_gpu + centered_points
    
    # Return as numpy array
    return cp.asnumpy(points).astype(np.float32)


def get_low_variance_mask(config: ClusterConfig) -> np.ndarray:
    """
    Identify clusters with near-zero variance.
    These clusters should only generate 1 point (the centroid).
    
    Returns:
        Boolean mask of shape (nclusters,) - True for low-variance clusters
    """
    variances = config.cluster_variances
    if variances.ndim == 2:
        # Per-dimension variance: use mean across dimensions
        var_per_cluster = variances.mean(axis=1)
    else:
        var_per_cluster = variances
    
    var_threshold = np.median(var_per_cluster) * 0.01
    return var_per_cluster < var_threshold


def get_num_points_per_cluster(total_points: int, config: ClusterConfig) -> np.ndarray:
    """
    Calculate how many points each cluster should have based on densities.
    
    Low-variance clusters (variance < median * 0.01) get exactly 1 point (centroid).
    Remaining points are distributed to normal clusters proportionally.
    """
    nclusters = config.nclusters
    
    # Identify low-variance clusters
    low_var_mask = get_low_variance_mask(config)
    n_low_var = low_var_mask.sum()
    n_normal = nclusters - n_low_var
    
    # Initialize: low-var clusters get 1 point each
    points_per_cluster = np.ones(nclusters, dtype=np.int64)
    
    if n_normal == 0:
        # All clusters are low-variance, just return 1 per cluster
        # (this shouldn't happen in practice)
        return points_per_cluster
    
    # Remaining points go to normal clusters
    points_for_normal = total_points - n_low_var
    
    # Get densities for normal clusters only and renormalize
    normal_cluster_ids = np.where(~low_var_mask)[0]
    normal_densities = config.cluster_densities[~low_var_mask]
    normal_densities = normal_densities / normal_densities.sum()
    
    # Allocate points proportionally (matches generate_mock_data logic)
    base_points = points_for_normal // n_normal
    extra_points = (normal_densities * (points_for_normal - base_points * n_normal)).astype(np.int64)
    normal_points_per_cluster = np.full(n_normal, base_points, dtype=np.int64) + extra_points
    
    # Fix any rounding errors
    diff = points_for_normal - normal_points_per_cluster.sum()
    normal_points_per_cluster[0] += diff
    
    # Assign to the result array
    for i, cid in enumerate(normal_cluster_ids):
        points_per_cluster[cid] = normal_points_per_cluster[i]
    
    return points_per_cluster


def generate_mock_data(
    cluster_config: ClusterConfig,
    total_points: int,
) -> np.ndarray:
    """
    Generate full mock dataset using cluster configuration (GPU-accelerated).
    
    For clusters with near-zero variance, generates just 1 point (the centroid).
    The remaining points are redistributed to normal clusters proportionally.
    
    Optionally injects outliers (points scattered outside cluster structure) to
    make the data harder for ANN algorithms.
    
    Args:
        cluster_config: ClusterConfig with centroids, variances, densities
        total_points: Total number of points to generate
    
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
        cluster_points = gen_cluster(cluster_id, n_points, cluster_config)
        result[write_idx:write_idx + n_points] = cluster_points
        write_idx += n_points
    
    # Inject outliers if specified
    if cluster_config.outlier_fraction > 0:
        n_outliers = int(total_points * cluster_config.outlier_fraction)
        if n_outliers > 0:
            rng = np.random.default_rng(cluster_config.seed + 999)  # Different seed for outliers
            
            # Compute data bounds from cluster centers + variance spread
            centers = cluster_config.cluster_centers
            # Use wider bounds: min/max of centers ± 3*std (covers ~99.7% of normal data)
            if cluster_config.cluster_variances.ndim == 2:
                stds = np.sqrt(cluster_config.cluster_variances.mean(axis=0))  # (ncols,)
            else:
                stds = np.sqrt(cluster_config.cluster_variances.mean()) * np.ones(ncols)
            
            data_min = centers.min(axis=0) - 3 * stds
            data_max = centers.max(axis=0) + 3 * stds
            
            # Generate uniformly distributed outliers
            outliers = rng.uniform(data_min, data_max, (n_outliers, ncols)).astype(np.float32)
            
            # Randomly select indices to replace
            outlier_indices = rng.choice(total_points, n_outliers, replace=False)
            result[outlier_indices] = outliers
            
            print(f"  Injected {n_outliers} outliers ({cluster_config.outlier_fraction*100:.1f}% of data)")
    
    return result


def gen_gt_brute_force(
    data: np.ndarray,
    n_queries: int,
    k: int,
    seed: int = 12345,
    backend: str = "cuvs",
    queries: np.ndarray = None,
    batch_size: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random queries from the data and compute ground truth via brute force.
    
    For large datasets, use batch_size > 0 to process data in batches and 
    merge results incrementally to avoid OOM.
    
    Args:
        data: Dataset to query
        n_queries: Number of queries
        k: Number of nearest neighbors
        seed: Random seed
        backend: "cuvs" or "sklearn" for brute force
        queries: Optional pre-generated queries (if None, samples from data)
        batch_size: If > 0, process data in batches of this size
    
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
        # Add small noise (use subset for std to avoid loading all data for large datasets)
        noise_scale = np.std(data[:min(100000, n_rows)]) * 0.1
        queries += rng.normal(0, noise_scale, queries.shape).astype(np.float32)
    
    # Non-batched: single brute force call
    if batch_size <= 0 or batch_size >= n_rows:
        print(f"Computing ground truth for {len(queries)} queries (k={k})...")
        gt_indices, gt_distances = brute_force_knn(data, queries, k, backend=backend)
        return queries, gt_indices, gt_distances
    
    # Batched: process in chunks and merge
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


def gen_build_sample(
    sample_size: int,
    total_rows: int,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        global_indices: (sample_size,) array of global indices for each vector
        cluster_ids: (sample_size,) array of cluster assignments
        n_sampled_per_cluster: (nclusters,) how many vectors sampled from each cluster
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)
    
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
    global_indices = []
    cluster_ids = []
    
    for cluster_id in tqdm(range(config.nclusters), desc="Getting samples from clusters"):
        n_sample = n_sampled_per_cluster[cluster_id]
        if n_sample > 0:
            # Generate full cluster and take first n_sample vectors
            n_cluster_points = points_per_cluster[cluster_id]
            cluster_points = gen_cluster(cluster_id, n_cluster_points, config)
            vectors.append(cluster_points[:n_sample])
            cluster_ids.append(np.full(n_sample, cluster_id, dtype=np.int32))
            
            # Compute global indices for these vectors
            global_start = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
            global_indices.append(np.arange(global_start, global_start + n_sample, dtype=np.int64))
    
    vectors = np.vstack(vectors).astype(np.float32)
    global_indices = np.concatenate(global_indices)
    cluster_ids = np.concatenate(cluster_ids)
    
    return vectors, global_indices, cluster_ids, n_sampled_per_cluster


def gen_extend_batch(
    batch_num: int,
    batch_size: int,
    total_rows: int,
    config: ClusterConfig,
    n_already_sampled_per_cluster: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of vectors for extend, DISTRIBUTED across all clusters.
    
    Unlike sequential batching, this samples proportionally from all clusters
    to ensure each extend batch has vectors spread across the data distribution.
    This helps CAGRA maintain better graph connectivity.
    
    Args:
        batch_num: Batch number (0-indexed, for the extend phase)
        batch_size: Number of vectors per batch
        total_rows: Total dataset size
        config: Cluster configuration
        n_already_sampled_per_cluster: How many vectors have ALREADY been sampled 
                                        from each cluster (cumulative from build + previous extends)
    
    Returns:
        vectors: (batch_size, ncols) array of vectors
        global_indices: (batch_size,) array of global indices for each vector
        cluster_ids: (batch_size,) array of cluster assignments
    """
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)
    remaining_per_cluster = points_per_cluster - n_already_sampled_per_cluster
    
    total_remaining = remaining_per_cluster.sum()
    if total_remaining <= 0:
        return np.array([]).reshape(0, config.ncols), np.array([], dtype=np.int64), np.array([])
    
    # Calculate how many to sample from each cluster for THIS batch
    # Sample proportionally based on remaining capacity
    actual_batch_size = min(batch_size, total_remaining)
    
    # Distribute batch_size proportionally across clusters based on remaining vectors
    remaining_fractions = remaining_per_cluster / total_remaining
    n_to_sample = (remaining_fractions * actual_batch_size).astype(np.int64)
    
    # Ensure we don't exceed remaining capacity
    n_to_sample = np.minimum(n_to_sample, remaining_per_cluster)
    
    # Distribute any leftover due to rounding
    leftover = actual_batch_size - n_to_sample.sum()
    for cluster_id in range(config.nclusters):
        if leftover <= 0:
            break
        capacity = remaining_per_cluster[cluster_id] - n_to_sample[cluster_id]
        add = min(leftover, capacity)
        n_to_sample[cluster_id] += add
        leftover -= add
    
    # Generate sampled vectors from each cluster
    vectors = []
    global_indices = []
    cluster_ids = []
    
    for cluster_id in tqdm(range(config.nclusters), desc="Generating vectors from clusters"):
        n_sample = n_to_sample[cluster_id]
        if n_sample <= 0:
            continue
        
        # Generate full cluster
        n_cluster_points = points_per_cluster[cluster_id]
        cluster_points = gen_cluster(cluster_id, n_cluster_points, config)
        
        # Take vectors starting from where we left off (after already sampled)
        offset = n_already_sampled_per_cluster[cluster_id]
        vectors.append(cluster_points[offset:offset + n_sample])
        cluster_ids.append(np.full(n_sample, cluster_id, dtype=np.int32))
        
        # Compute global indices for these vectors
        global_base = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
        global_indices.append(np.arange(
            global_base + offset,
            global_base + offset + n_sample,
            dtype=np.int64
        ))
    
    if len(vectors) == 0:
        return np.array([]).reshape(0, config.ncols), np.array([], dtype=np.int64), np.array([])
    
    vectors = np.vstack(vectors).astype(np.float32)
    global_indices = np.concatenate(global_indices)
    cluster_ids = np.concatenate(cluster_ids)
    
    # Return the n_to_sample so caller can update n_already_sampled_per_cluster
    return vectors, global_indices, cluster_ids, n_to_sample


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

def gen_gt_with_nprobes(
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
    
    # Generate query points by sampling from actual data points (like compare_pareto.py)
    query_rng = np.random.default_rng(config.seed + query_seed_offset)
    
    # TODO: maybe instead of adding noise just generate from normal distribution as well
    # Sample global indices uniformly from the dataset
    query_global_indices = query_rng.choice(total_rows, size=nqueries, replace=False)
    
    # Convert global indices to (cluster_id, local_index) pairs
    # For each global index, find which cluster it belongs to
    query_cluster_ids = np.searchsorted(cumsum, query_global_indices, side='right')
    
    # Group queries by cluster to generate efficiently
    cluster_to_query_info = {}  # cluster_id -> list of (query_idx, local_index)
    for q_idx, (global_idx, cluster_id) in enumerate(zip(query_global_indices, query_cluster_ids)):
        local_idx = global_idx if cluster_id == 0 else global_idx - cumsum[cluster_id - 1]
        if cluster_id not in cluster_to_query_info:
            cluster_to_query_info[cluster_id] = []
        cluster_to_query_info[cluster_id].append((q_idx, local_idx))
    
    # Generate query points from actual data
    queries = np.zeros((nqueries, config.ncols), dtype=np.float32)
    for cluster_id, query_info_list in cluster_to_query_info.items():
        # Generate the full cluster (we need specific indices from it)
        n_points = points_per_cluster[cluster_id]
        cluster_points = gen_cluster(cluster_id, n_points, config)
        
        # Extract the specific points for queries
        for q_idx, local_idx in query_info_list:
            queries[q_idx] = cluster_points[local_idx]
    
    # Add small noise to avoid exact matches (like compare_pareto.py)
    noise_scale = np.std(queries) * 0.1
    queries += query_rng.normal(0, noise_scale, queries.shape).astype(np.float32)

    # Find nearby clusters for each query
    nearby_clusters = find_nearby_clusters(queries, config, nprobes, backend, metric)

    # Compute ground truth by iterating over clusters (more efficient)
    # For each cluster, batch process all queries that have it as a nearby cluster
    gt_indices = np.full((nqueries, k), -1, dtype=np.int64)
    gt_distances = np.full((nqueries, k), np.inf, dtype=np.float32)
    
    # Build reverse mapping: cluster_id -> list of query indices that need this cluster
    cluster_to_queries = {}
    for q_idx in tqdm(range(nqueries), desc="Building reverse mapping"):
        for cluster_id in nearby_clusters[q_idx]:
            cluster_id = int(cluster_id)
            if cluster_id not in cluster_to_queries:
                cluster_to_queries[cluster_id] = []
            cluster_to_queries[cluster_id].append(q_idx)
    
    # Process each cluster once
    for cluster_id in tqdm(cluster_to_queries.keys(), desc="Processing clusters for GT"):
        query_indices = cluster_to_queries[cluster_id]
        n_points = points_per_cluster[cluster_id]
        
        # Generate this cluster's points (only once!)
        cluster_points = gen_cluster(cluster_id, n_points, config)
        
        # Calculate global index offset for this cluster
        global_start = 0 if cluster_id == 0 else cumsum[cluster_id - 1]
        
        # Gather queries that need this cluster
        batch_queries = queries[query_indices]  # (num_queries_for_cluster, ncols)
        
        # Brute force KNN: batch_queries vs cluster_points
        k_cluster = min(k, n_points)
        local_indices, local_dists = brute_force_knn(
            cluster_points,
            batch_queries,
            k_cluster,
            backend=backend
        )
        
        # Convert local indices to global indices
        global_indices = local_indices + global_start  # (num_queries, k_cluster)
        
        # Merge results for each query in this batch
        for i, q_idx in enumerate(query_indices):
            # Combine current top-k with new cluster results
            merged_indices = np.concatenate([gt_indices[q_idx], global_indices[i]])
            merged_dists = np.concatenate([gt_distances[q_idx], local_dists[i]])
            
            # Sort by distance and keep top k
            sort_order = np.argsort(merged_dists)[:k]
            gt_indices[q_idx] = merged_indices[sort_order]
            gt_distances[q_idx] = merged_dists[sort_order]

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
    
    all_data = generate_mock_data(cluster_config, config.total_rows)
    
    if verbose:
        print(f"  Generated {all_data.shape[0]:,} vectors")
    
    # Step 2: Generate queries and cluster-based GT
    if verbose:
        print(f"\n[Step 2] Generating queries and cluster-based GT (nprobes={config.nprobes})...")

    queries, cluster_gt_indices, cluster_gt_distances = gen_gt_with_nprobes(
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
    
    _, true_gt_indices, true_gt_distances = gen_gt_brute_force(
        all_data,
        n_queries=config.nqueries,
        k=config.k,
        backend=config.gt_backend,
        queries=queries
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

def run_streaming_benchmark(
    config: BenchmarkConfig, 
    itopk_list: list = None, 
    nprobes_list: list = None,
    verbose: bool = True
) -> dict:
    """
    Streaming ANN benchmark without storing the full dataset.
    
    This function:
    1. Generates cluster config deterministically from seed
    2. Streams data batches into ANN index (vectors discarded after indexing)
    3. Generates queries and ground truth on-the-fly (only nearby clusters generated)
    4. Searches ANN and computes recall
    
    Args:
        config: Benchmark configuration
        itopk_list: List of itopk values to sweep for CAGRA
        nprobes_list: List of n_probes values to sweep for IVF-PQ
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
    if config.ann_index is None:
        raise ValueError("ann_index must be provided in BenchmarkConfig")
    ann_index = config.ann_index
    index_name = type(ann_index).__name__
    is_ivfpq = isinstance(ann_index, IvfPqIndex)
    
    if verbose:
        print(f"\n[Step 1] Streaming data into {index_name}...")
    
    total_indexed = 0
    build_start = time.perf_counter()

    # Use representative sampling with explicit indices for both CAGRA and IVF-PQ
    if verbose:
        print(f"  Building with representative sample from all clusters...")
    
    # TODO: consolidate since we're taking the distributed sampling approach anyway
    # Sample representative build data from all clusters (includes global indices)
    build_vectors, build_indices, _, n_sampled_per_cluster = gen_build_sample(
        sample_size=config.batch_size,
        total_rows=config.total_rows,
        config=cluster_config
    )
    
    
    ann_index.build(build_vectors, indices=build_indices)
    total_indexed += len(build_vectors)
    if verbose:
        print(f"  Built with {len(build_vectors):,} vectors from {(n_sampled_per_cluster > 0).sum()} clusters")
    
    # Extend with remaining vectors (distributed across clusters)
    # Track cumulative samples per cluster
    n_cumulative_sampled = n_sampled_per_cluster.copy()
    total_remaining = config.total_rows - n_cumulative_sampled.sum()
    num_extend_batches = (total_remaining + config.batch_size - 1) // config.batch_size
    
    for batch_num in tqdm(range(num_extend_batches), desc=f"Extending {index_name} in batches"):
        vectors, batch_indices, _, n_batch_sampled = gen_extend_batch(
            batch_num, config.batch_size, config.total_rows, 
            cluster_config, n_cumulative_sampled
        )
        if len(vectors) > 0:
            ann_index.extend(vectors, indices=batch_indices)
            total_indexed += len(vectors)
            n_cumulative_sampled += n_batch_sampled

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
    queries, gt_indices, gt_distances = gen_gt_with_nprobes(
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
    # Step 4: Search ANN index and compute recall (sweep search params)
    # -------------------------------------------------------------------------
    pareto_results = []
    
    if is_ivfpq:
        # IVF-PQ: Sweep n_probes
        if nprobes_list is None:
            nprobes_list = [16, 32, 64, 128, 256]
        param_name = "n_probes"
        
        if verbose:
            print(f"\n[Step 3] Searching {index_name} with n_probes sweep: {nprobes_list}...")
        
        for n_probes in nprobes_list:
            # Warmup search (not timed)
            _ = ann_index.search(queries, config.k, nprobes=n_probes)
            
            # Timed search
            search_start = time.perf_counter()
            predicted_indices, predicted_distances = ann_index.search(queries, config.k, nprobes=n_probes)
            search_time = time.perf_counter() - search_start
            
            recall, _ = compute_recall_with_ties(
                predicted_indices, predicted_distances,
                gt_indices, gt_distances
            )
            
            qps = config.nqueries / search_time
            
            pareto_results.append({
                'param': n_probes,
                'recall': recall,
                'search_time_sec': search_time,
                'qps': qps,
            })
            
            if verbose:
                print(f"  n_probes={n_probes}: recall={recall:.4f}, QPS={qps:,.0f}")
    else:
        # CAGRA: Sweep itopk
        if itopk_list is None:
            itopk_list = [32, 64, 128, 256]
        param_name = "itopk"
        
        if verbose:
            print(f"\n[Step 3] Searching {index_name} with itopk sweep: {itopk_list}...")
        
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
                'param': itopk,
                'recall': recall,
                'search_time_sec': search_time,
                'qps': qps,
            })
            
            if verbose:
                itopk_str = itopk if itopk is not None else "default"
                print(f"  itopk={itopk_str}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    results['pareto_results'] = pareto_results
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print(f"Benchmark Results Summary ({index_name})")
        print("=" * 60)
        print(f"  Index build time: {build_time:.2f}s")
        print(f"  GT generation time: {gt_time:.2f}s")
        print(f"\n  Pareto curve data ({param_name} sweep):")
        print(f"  {param_name:>8} {'recall':>10} {'QPS':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*12}")
        for pr in pareto_results:
            param_str = str(pr['param']) if pr['param'] is not None else "default"
            print(f"  {param_str:>8} {pr['recall']:>10.4f} {pr['qps']:>12,.0f}")
        
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

def demo_benchmark(
    config: BenchmarkConfig, 
    itopk_list: list = None, 
    nprobes_list: list = None,
    output_path: str = None
):
    """
    Demonstrate the streaming benchmark with example configuration.
    
    Args:
        config: Benchmark configuration
        itopk_list: List of itopk values to sweep for CAGRA (e.g., [32, 64, 128, 256])
        nprobes_list: List of n_probes values to sweep for IVF-PQ (e.g., [16, 32, 64, 128])
        output_path: Optional path to save pareto results as .npz file
    """
    results = run_streaming_benchmark(
        config, 
        itopk_list=itopk_list, 
        nprobes_list=nprobes_list,
        verbose=True
    )
    
    # Save pareto results if output path provided
    if output_path is not None and 'pareto_results' in results:
        pareto = results['pareto_results']
        params = np.array([p['param'] if p['param'] is not None else 0 for p in pareto])
        recalls = np.array([p['recall'] for p in pareto])
        qps_arr = np.array([p['qps'] for p in pareto])
        search_times = np.array([p['search_time_sec'] for p in pareto])
        
        np.savez(
            output_path,
            param=params,
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
    from cuvs.neighbors import ivf_pq
    
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
    parser.add_argument("--algorithm", type=str, default="cagra", choices=["cagra", "ivfpq"],
                        help="ANN algorithm: cagra or ivfpq (default: cagra)")
    parser.add_argument("--itopk", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="List of itopk values to sweep for CAGRA (default: 32 64 128 256)")
    parser.add_argument("--search_nprobes", type=int, nargs="+", default=[16, 32, 64, 128, 256],
                        help="List of n_probes values to sweep for IVF-PQ (default: 16 32 64 128 256)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save pareto results as .npz file (optional)")
    
    args = parser.parse_args()
    
    # Load cluster stats to get ncols
    from config import load_cluster_stats
    stats = load_cluster_stats(args.cluster_stats)
    ncols = stats['centroids'].shape[1]
    
    # Create the ANN index based on algorithm choice
    if args.algorithm == "cagra":
        build_params = cagra.IndexParams(
            intermediate_graph_degree=256,
            graph_degree=128,
            build_algo="nn_descent",
        )
        ann_index = CagraIndex(build_params=build_params)
        itopk_list = args.itopk
        nprobes_list = None
    else:  # ivfpq
        n_lists = min(1024, args.total_rows // 100)
        pq_dim = min(ncols, 128)
        build_params = ivf_pq.IndexParams(
            n_lists=n_lists,
            pq_dim=pq_dim,
            pq_bits=8,
        )
        ann_index = IvfPqIndex(build_params=build_params)
        itopk_list = None
        nprobes_list = args.search_nprobes
        print(f"IVF-PQ params: n_lists={n_lists}, pq_dim={pq_dim}")
    
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
        demo_benchmark(
            config, 
            itopk_list=itopk_list, 
            nprobes_list=nprobes_list,
            output_path=args.output
        )
    
    # Explicit GPU cleanup to avoid CUDA errors on exit
    import gc
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass
