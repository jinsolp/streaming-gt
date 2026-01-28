"""
Extract cluster characteristics from a real dataset.

This script:
1. Loads the wiki dataset (or other datasets)
2. Runs KMeans clustering on a subsample
3. Extracts cluster centroids, variances, and densities

These values can then be used to configure synthetic data generation
that mimics the real dataset's characteristics.
"""

import numpy as np
import struct
import pickle
import os
import joblib
from typing import Tuple, Optional
import argparse
import time
from datetime import datetime, timezone, timedelta
from sklearn import config_context
from tqdm import tqdm

def load_memmap_bin(input_bin_file: str, dtype=np.float32, extra: int = -1):
    """
    Load binary file with header format: [n_rows (uint32), n_dim (uint32), data...]
    
    Args:
        input_bin_file: Path to binary file
        dtype: Data type (default float32)
        extra: If > 0, only load first `extra` rows (for subsampling)
    
    Returns:
        data: numpy array of shape (n_rows, n_dim)
        n_rows: total rows in file
        n_dim: dimensionality
    """
    with open(input_bin_file, 'rb') as f:
        header = f.read(8)
        n_rows, n_dim = struct.unpack('II', header)

    if extra > -1:
        data = np.fromfile(input_bin_file, dtype=dtype, offset=8, count=extra * n_dim)
        data = data.reshape((extra, n_dim))
    else:
        data = np.fromfile(input_bin_file, dtype=dtype, offset=8)
        data = data.reshape((n_rows, n_dim))

    return data, n_rows, n_dim


def extract_cluster_stats(
    data: np.ndarray,
    n_clusters: int,
    subsample_size: Optional[int] = None,
    seed: int = 42,
    use_gpu: bool = True,
    verbose: bool = True,
    max_iter: int = 500,
    method: str = "kmeans",
    save_gmm_path: Optional[str] = None
) -> dict:
    """
    Run clustering and extract cluster statistics.
    
    Args:
        data: Input data array (n_samples, n_features)
        n_clusters: Number of clusters
        subsample_size: If provided, subsample data to this size before clustering
        seed: Random seed for reproducibility
        use_gpu: Use cuML GPU KMeans (faster) or sklearn CPU KMeans (ignored for GMM)
        verbose: Print progress
        max_iter: Maximum iterations for clustering algorithm
        method: Clustering method - "kmeans" or "gmm" (Gaussian Mixture Model)
        save_gmm_path: If provided, save the fitted GMM model to this path (only for gmm methods)
    
    Returns:
        dict with:
            centroids: (n_clusters, n_features) cluster centers
            densities: (n_clusters,) relative density (fraction of points) per cluster
            variances_per_dim: (n_clusters, n_features) per-dimension variance
    """
    rng = np.random.default_rng(seed)
    n_dim = data.shape[1]
    
    # Subsample if requested
    if subsample_size is not None and subsample_size < len(data):
        if verbose:
            print(f"Subsampling from {len(data):,} to {subsample_size:,} points...")
        indices = rng.choice(len(data), size=subsample_size, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data
    
    if verbose:
        pst = timezone(timedelta(hours=-8))
        start_time_pst = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")
        method_names = {"gmm": "GMM", "gmm_gpu_init": "GMM (GPU KMeans init)", "kmeans": "KMeans"}
        method_name = method_names.get(method, method)
        print(f"[{start_time_pst}] Running {method_name} with {n_clusters} clusters on {len(data_sample):,} points...")
    
    if method == "gmm":
        # Use sklearn's Gaussian Mixture Model (CPU k-means++ init)
        from sklearn.mixture import GaussianMixture
        start = time.perf_counter()
        gmm = GaussianMixture(
            n_components=n_clusters, 
            random_state=seed, 
            max_iter=max_iter,
            covariance_type='diag',  # diagonal covariance for efficiency
            reg_covar=1e-4,  # regularization to prevent singular covariance
            verbose=1 if verbose else 0,
            init_params='k-means++'
        )
        gmm.fit(data_sample)
        end = time.perf_counter()
        if verbose:
            print(f"GMM time: {end - start:.2f} seconds")
        
        labels = gmm.predict(data_sample)
        centroids = gmm.means_
        # GMM provides weights directly (already normalized)
        gmm_weights = gmm.weights_
        # GMM provides per-dimension variances directly (for diag covariance)
        gmm_variances_per_dim = gmm.covariances_  # (n_clusters, n_dim) for 'diag'
        
        # Save GMM model if path provided
        if save_gmm_path:
            os.makedirs(os.path.dirname(save_gmm_path) if os.path.dirname(save_gmm_path) else "saved_gmm", exist_ok=True)
            joblib.dump(gmm, save_gmm_path)
            if verbose:
                print(f"Saved GMM model to {save_gmm_path}")
    
    elif method == "gmm_gpu_init":
        # Use cuML GPU KMeans for fast initialization, then sklearn GMM for refinement
        from sklearn.mixture import GaussianMixture
        
        # Step 1: Fast GPU KMeans initialization
        if verbose:
            print("  Step 1: Running cuML GPU KMeans for initialization...")
        try:
            from cuml.cluster import KMeans as cuMLKMeans
            start = time.perf_counter()
            kmeans = cuMLKMeans(n_clusters=n_clusters, random_state=seed, max_iter=100, verbose=0)
            kmeans.fit(data_sample)
            kmeans_time = time.perf_counter() - start
            if verbose:
                print(f"    KMeans init time: {kmeans_time:.2f}s")
            km_labels = kmeans.labels_.get() if hasattr(kmeans.labels_, 'get') else np.array(kmeans.labels_)
            km_centroids = kmeans.cluster_centers_.get() if hasattr(kmeans.cluster_centers_, 'get') else np.array(kmeans.cluster_centers_)
        except ImportError:
            raise ImportError("cuML not available - gmm_gpu_init requires cuML for GPU KMeans")
        
        # Step 2: Compute initial weights from cluster sizes
        counts = np.bincount(km_labels, minlength=n_clusters).astype(np.float64)
        weights_init = counts / counts.sum()
        
        # Step 3: Compute initial precisions (1/variance) for 'diag' covariance
        if verbose:
            print("  Step 2: Computing initial variances from KMeans clusters...")
        variances_init = np.zeros((n_clusters, n_dim), dtype=np.float64)
        for i in range(n_clusters):
            mask = km_labels == i
            if mask.sum() > 1:
                variances_init[i] = np.var(data_sample[mask], axis=0) + 1e-6
            else:
                variances_init[i] = 1.0  # fallback for singleton clusters
        precisions_init = 1.0 / variances_init
        
        # Step 4: Initialize GMM with KMeans results and refine
        if verbose:
            print("  Step 3: Running GMM EM refinement...")
        
        start = time.perf_counter()
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=seed,
            max_iter=max_iter,
            covariance_type='diag',
            reg_covar=1e-4,
            weights_init=weights_init,
            means_init=km_centroids.astype(np.float64),
            precisions_init=precisions_init,
            verbose=1 if verbose else 0,
        )
        gmm.fit(data_sample.astype(np.float64))  # float64 for numerical stability
        gmm_time = time.perf_counter() - start

        if verbose:
            print(f"    GMM refinement time: {gmm_time:.2f}s")
            print(f"    Total time: {kmeans_time + gmm_time:.2f}s")
        
        labels = gmm.predict(data_sample)
        centroids = gmm.means_.astype(np.float32)
        gmm_weights = gmm.weights_
        gmm_variances_per_dim = gmm.covariances_.astype(np.float32)
        
        # Save GMM model if path provided
        if save_gmm_path:
            os.makedirs(os.path.dirname(save_gmm_path) if os.path.dirname(save_gmm_path) else "saved_gmm", exist_ok=True)
            joblib.dump(gmm, save_gmm_path)
            if verbose:
                print(f"Saved GMM model to {save_gmm_path}")
        
    else:
        # Use KMeans
        gmm_weights = None
        gmm_variances_per_dim = None
        
        if use_gpu:
            try:
                from cuml.cluster import KMeans
                start = time.perf_counter()
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed, max_iter=max_iter, verbose=verbose)
                kmeans.fit(data_sample)
                end = time.perf_counter()
                print(f"KMeans time: {end - start:.2f} seconds")
                labels = kmeans.labels_.get() if hasattr(kmeans.labels_, 'get') else np.array(kmeans.labels_)
                centroids = kmeans.cluster_centers_.get() if hasattr(kmeans.cluster_centers_, 'get') else np.array(kmeans.cluster_centers_)
            except ImportError:
                print("cuML not available, falling back to sklearn...")
                use_gpu = False
        
        if not use_gpu:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, max_iter=max_iter, n_init=10)
            kmeans.fit(data_sample)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
    
    if verbose:
        print("Computing cluster statistics...")
    
    # Initialize arrays for statistics
    densities = np.zeros(n_clusters)
    variances_per_dim = np.zeros((n_clusters, n_dim))
    means_per_dim = np.zeros((n_clusters, n_dim))  # For quality analysis only
    
    # For GMM, we can use the model's parameters directly for variances/densities
    if method == "gmm" and gmm_weights is not None:
        densities = gmm_weights.copy()
        variances_per_dim = gmm_variances_per_dim.copy()
    
    for i in range(n_clusters):
        mask = labels == i
        cluster_points = data_sample[mask]
        n_points = len(cluster_points)
        
        # For KMeans, compute density from labels
        if method != "gmm":
            densities[i] = n_points / len(data_sample)
        
        if n_points > 0:
            # For KMeans, compute variances from data
            if method != "gmm":
                variances_per_dim[i] = np.var(cluster_points, axis=0)
            
            means_per_dim[i] = np.mean(cluster_points, axis=0)
        else:
            # Empty cluster - use centroid as mean, zero variance
            means_per_dim[i] = centroids[i]
    
    # Quality analysis: detect potential convergence issues
    if verbose:
        _print_quality_analysis(centroids, means_per_dim, variances_per_dim, densities, data_sample, n_clusters)
    
    return {
        'centroids': centroids,
        'densities': densities,
        'variances_per_dim': variances_per_dim,
    }


def _print_quality_analysis(centroids, means_per_dim, variances_per_dim, densities, data_sample, n_clusters):
    """Print quality analysis to detect poor KMeans convergence."""
    # Compute mean variance per cluster for summary stats
    variances = variances_per_dim.mean(axis=1)
    print(f"\nCluster Statistics Summary:")
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Variance (mean per cluster) range: [{variances.min():.4f}, {variances.max():.4f}]")
    print(f"  Variance (mean per cluster) mean: {variances.mean():.4f}")
    print(f"  Density range: [{densities.min():.4f}, {densities.max():.4f}]")
    print(f"  Points per cluster: {int(densities.min() * len(data_sample))} - {int(densities.max() * len(data_sample))}")
    
    # === Quality Analysis ===
    print(f"\n=== Centroid Quality Analysis ===")
    
    # 1. Centroid Proximity: Are any centroids too close together?
    from scipy.spatial.distance import pdist, squareform
    centroid_dists = squareform(pdist(centroids))
    np.fill_diagonal(centroid_dists, np.inf)
    
    min_centroid_dist = centroid_dists.min()
    median_centroid_dist = np.median(centroid_dists[centroid_dists < np.inf])
    proximity_ratio = min_centroid_dist / median_centroid_dist
    
    print(f"  Min centroid pair distance: {min_centroid_dist:.4f}")
    print(f"  Median centroid pair distance: {median_centroid_dist:.4f}")
    print(f"  Proximity ratio (higher=better): {proximity_ratio:.4f}")
    if proximity_ratio < 0.1:
        print("    ⚠️ WARNING: Some centroids are very close together!")
    
    # 2. Centroid vs actual mean drift
    centroid_drift = np.linalg.norm(centroids - means_per_dim, axis=1)
    mean_drift = centroid_drift.mean()
    max_drift = centroid_drift.max()
    print(f"  Centroid drift (mean): {mean_drift:.4f}")
    print(f"  Centroid drift (max): {max_drift:.4f}")
    if max_drift > median_centroid_dist * 0.5:
        print("    ⚠️ WARNING: Some centroids drifted far from actual cluster mean!")
    
    # 3. Cluster size imbalance
    size_cv = densities.std() / densities.mean()  # Coefficient of variation
    tiny_clusters = (densities < 1.0 / n_clusters * 0.1).sum()
    print(f"  Cluster size CV (lower=more balanced): {size_cv:.4f}")
    print(f"  Tiny clusters (<10% expected size): {tiny_clusters}")
    if tiny_clusters > n_clusters * 0.1:
        print("    ⚠️ WARNING: Many clusters are undersized!")


def load_cluster_stats_legacy(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cluster statistics from a .npz file (legacy format).
    Returns (centroids, variances_per_dim, densities) tuple for backward compatibility.
    """
    data = np.load(filepath)
    return data['centroids'], data['variances_per_dim'], data['densities']


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import save_cluster_stats
    
    parser = argparse.ArgumentParser(description="Extract cluster statistics from a dataset")
    parser.add_argument("--dataset", type=str, default="wiki", 
                        help="Dataset to load: 'wiki', 'food', 'small' (synthetic test data), or path to .fbin/.pkl file")
    parser.add_argument("--n_clusters", type=int, default=100,
                        help="Number of clusters for KMeans")
    parser.add_argument("--subsample", type=int, default=-1,
                        help="Subsample size for clustering (-1 = use full dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path (default: cluster_stats_{dataset}_{n_clusters}.npz)")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU sklearn instead of GPU cuML")
    parser.add_argument("--max_iter", type=int, default=300,
                        help="Max iterations for clustering (default: 300)")
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "gmm", "gmm_gpu_init"],
                        help="Clustering method: 'kmeans', 'gmm' (CPU), or 'gmm_gpu_init' (GPU KMeans init + GMM refinement)")
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "small":
        # Small synthetic dataset for testing
        from sklearn.datasets import make_blobs
        n_samples = 100000
        n_dim = 384  # Small dimensionality for fast testing
        n_true_clusters = 50
        print(f"Generating small test dataset: {n_samples} points, {n_dim}D, {n_true_clusters} true clusters...")
        data, _ = make_blobs(n_samples=n_samples, n_features=n_dim, centers=n_true_clusters, 
                             cluster_std=1.0, random_state=args.seed)
        data = data.astype(np.float32)
        n_rows = n_samples
    elif args.dataset == "wiki":
        data_path = "/datasets/jinsolp/data/wiki_all/base.88M.fbin"
        print(f"Loading wiki dataset from {data_path}...")
        data, n_rows, n_dim = load_memmap_bin(data_path, np.float32, extra=args.subsample)
    elif args.dataset == "food":
        data_path = "/datasets/jinsolp/data/amazon_reviews/Grocery_and_Gourmet_Food.pkl"
        print(f"Loading food dataset from {data_path}...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        n_rows, n_dim = data.shape
    elif args.dataset.endswith(".pkl"):
        data_path = args.dataset
        print(f"Loading pickle dataset from {data_path}...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        n_rows, n_dim = data.shape
    elif args.dataset.endswith(".npy"):
        data_path = args.dataset
        print(f"Loading numpy dataset from {data_path}...")
        data = np.load(data_path)
        n_rows, n_dim = data.shape
    else:
        data_path = args.dataset
        print(f"Loading binary dataset from {data_path}...")
        data, n_rows, n_dim = load_memmap_bin(data_path, np.float32, extra=args.subsample)
    
    print(f"Loaded data shape: {data.shape}")
    n_dim = data.shape[1]
    
    # Extract cluster statistics
    # Use None for subsample_size if -1 (full dataset)
    subsample_size = None if args.subsample < 0 else args.subsample
    
    import os
    
    # Generate GMM save path if using GMM method
    gmm_save_path = None
    if args.method in ["gmm", "gmm_gpu_init"]:
        dataset_name = args.dataset.split("/")[-1].replace(".fbin", "") if "/" in args.dataset else args.dataset
        sample_size = len(data) if args.subsample < 0 else args.subsample
        method_str = "_gmm_gpu_init" if args.method == "gmm_gpu_init" else "_gmm"
        gmm_save_path = f"saved_gmm/gmm_{dataset_name}_sample{sample_size}_n{args.n_clusters}{method_str}_{args.max_iter}.joblib"
    
    stats = extract_cluster_stats(
        data=data,
        n_clusters=args.n_clusters,
        subsample_size=subsample_size,
        seed=args.seed,
        use_gpu=not args.cpu,
        verbose=True,
        max_iter=args.max_iter,
        method=args.method,
        save_gmm_path=gmm_save_path
    )

    # Save results
    if args.output is None:
        dataset_name = args.dataset.split("/")[-1].replace(".fbin", "") if "/" in args.dataset else args.dataset
        sample_size = len(data) if args.subsample < 0 else args.subsample
        method_str = "_gmm_gpu_init" if args.method == "gmm_gpu_init" else ("_gmm" if args.method == "gmm" else "")
        args.output = f"cluster_stats/{dataset_name}_sample{sample_size}_n{args.n_clusters}{method_str}_{args.max_iter}.npz"
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_cluster_stats(args.output, stats)
    
    # Print summary for use in ClusterConfig
    print("\n" + "=" * 60)
    print("To use these stats in ClusterConfig:")
    print("=" * 60)
    print(f"""
from config import load_cluster_stats
stats = load_cluster_stats("{args.output}")

cluster_config = ClusterConfig(
    nclusters={args.n_clusters},
    ncols={n_dim},
    seed=42,
    cluster_centers=stats['centroids'],
    cluster_variances=stats['variances_per_dim'],  # (n_clusters, n_dim) per-dim variance
    cluster_densities=stats['densities']
)
""")
