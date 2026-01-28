"""
Compare 2 approaches for ANN benchmarking (full single builds only):
1. Full mock data: Mock data generated from cluster stats, single build, brute-force GT
2. Original single: Original data, single build, brute-force GT

Usage:
    python investigate/compare_full_builds.py \
        --original investigate/subsampled_data/clothes_pareto_1M_seed123.fbin \
        --cluster_stats investigate/cluster_stats/cluster_stats_clothes_extract_5M_seed42_sample5000000_n1000.npz \
        --algorithm ivfpq
"""

import numpy as np
import time
import argparse
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extract_values import load_memmap_bin, load_cluster_stats
from config import ClusterConfig
from utils import compute_recall_with_ties, brute_force_knn
from cuvs.neighbors import cagra, ivf_pq
from ann_indices import CagraIndex, IvfPqIndex

# Reuse from scalable_gt.py
from scalable_gt import generate_mock_data, gen_gt_brute_force, gen_cluster, get_num_points_per_cluster

# Reuse from stream_vs_single.py
from stream_vs_single import build_single_cagra, build_single_ivfpq


def verify_variance_distribution(
    original_data: np.ndarray,
    cluster_config: ClusterConfig,
    total_rows: int,
    n_clusters_to_check: int = 10
):
    """
    Verify that mock data generation produces correct variance distribution.
    Compares global stats and spot-checks individual clusters.
    """
    print("\n" + "=" * 60)
    print("VARIANCE VERIFICATION")
    print("=" * 60)
    
    # Generate mock data for comparison
    print("\n[Generating mock data for verification...]")
    mock_data = generate_mock_data(cluster_config, total_rows)
    
    # Global statistics comparison
    print("\n[Global Statistics Comparison]")
    print(f"  {'Metric':<25} {'Original':>15} {'Mock':>15} {'Ratio':>10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")
    
    orig_var = original_data.var()
    mock_var = mock_data.var()
    print(f"  {'Overall variance':<25} {orig_var:>15.6f} {mock_var:>15.6f} {mock_var/orig_var:>10.3f}")
    
    # Per-dimension statistics
    orig_dim_std = original_data.std(axis=0)
    mock_dim_std = mock_data.std(axis=0)
    print(f"\n[Per-Dimension Std Comparison]")
    print(f"  {'Metric':<25} {'Original':>15} {'Mock':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Min std across dims':<25} {orig_dim_std.min():>15.6f} {mock_dim_std.min():>15.6f}")
    print(f"  {'Max std across dims':<25} {orig_dim_std.max():>15.6f} {mock_dim_std.max():>15.6f}")
    print(f"  {'Mean std across dims':<25} {orig_dim_std.mean():>15.6f} {mock_dim_std.mean():>15.6f}")
    print(f"  {'Std of stds':<25} {orig_dim_std.std():>15.6f} {mock_dim_std.std():>15.6f}")
    
    # Cluster-level variance check (spot check)
    print(f"\n[Per-Cluster Variance Check (spot check {n_clusters_to_check} clusters)]")
    points_per_cluster = get_num_points_per_cluster(total_rows, cluster_config)
    
    # Select clusters to check: some small, some large, some random
    n_clusters = cluster_config.nclusters
    cluster_ids_to_check = []
    if n_clusters >= n_clusters_to_check:
        # Mix of indices: first, last, and evenly spaced
        cluster_ids_to_check = [0, n_clusters-1]
        step = n_clusters // (n_clusters_to_check - 2)
        cluster_ids_to_check.extend([i * step for i in range(1, n_clusters_to_check - 1)])
        cluster_ids_to_check = sorted(set(cluster_ids_to_check))[:n_clusters_to_check]
    else:
        cluster_ids_to_check = list(range(n_clusters))
    
    print(f"  (Ratio = mean of per-dimension ratios, more robust than ratio of means)")
    print(f"  {'Cluster':<10} {'N_points':>10} {'Expected Var':>15} {'Actual Var':>15} {'Ratio':>10} {'Status':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*15} {'-'*10} {'-'*10}")
    
    variance_ratios = []
    for cid in cluster_ids_to_check:
        n_points = points_per_cluster[cid]
        if n_points < 2:
            continue
        
        cluster_points = gen_cluster(cid, n_points, cluster_config)
        
        # Compute per-dimension variance
        actual_var_per_dim = cluster_points.var(axis=0)  # (ncols,)
        
        # Get expected variance per dimension
        if cluster_config.cluster_variances.ndim == 2:
            expected_var_per_dim = cluster_config.cluster_variances[cid]  # (ncols,)
        else:
            # Scalar variance - broadcast to all dimensions
            expected_var_per_dim = np.full(cluster_config.ncols, cluster_config.cluster_variances[cid])
        
        # Compute ratio per dimension, then average (more robust than ratio of averages)
        # Avoid division by zero
        ratios_per_dim = actual_var_per_dim / np.maximum(expected_var_per_dim, 1e-10)
        ratio = ratios_per_dim.mean()
        
        # For display, also compute mean variances
        expected_var = expected_var_per_dim.mean()
        actual_var = actual_var_per_dim.mean()
        
        # Debug: check first few clusters
        if cid < 3:
            print(f"[VERIFY DEBUG] Cluster {cid}: mean_ratio={ratio:.3f}, ratio_std={ratios_per_dim.std():.3f}")
        variance_ratios.append(ratio)
        
        # Status: OK if ratio is between 0.5 and 2.0
        if 0.5 <= ratio <= 2.0:
            status = "OK"
        elif ratio < 0.5:
            status = "TOO TIGHT"
        else:
            status = "TOO SPREAD"
        
        print(f"  {cid:<10} {n_points:>10} {expected_var:>15.6f} {actual_var:>15.6f} {ratio:>10.3f} {status:>10}")
    
    # Inertia check (average squared distance to closest centroid)
    print(f"\n[Inertia Check (avg squared distance to closest centroid)]")
    centroids = cluster_config.cluster_centers
    
    # Original data -> closest centroid
    _, orig_min_dists_sq = brute_force_knn(centroids, original_data, k=1, backend="cuvs")
    orig_inertia = orig_min_dists_sq.mean()
    
    # Mock data -> closest centroid
    _, mock_min_dists_sq = brute_force_knn(centroids, mock_data, k=1, backend="cuvs")
    mock_inertia = mock_min_dists_sq.mean()
    
    print(f"  Original inertia: {orig_inertia:.6f}")
    print(f"  Mock inertia:     {mock_inertia:.6f}")
    inertia_ratio = mock_inertia / orig_inertia if orig_inertia > 0 else float('inf')
    inertia_diff_pct = (mock_inertia - orig_inertia) / orig_inertia * 100 if orig_inertia > 0 else 0
    print(f"  Ratio:            {inertia_ratio:.3f}")
    print(f"  Diff:             {inertia_diff_pct:+.1f}%")
    
    if abs(inertia_diff_pct) < 10:
        print("  Status: GOOD - inertia is similar, mock data has similar cluster tightness")
    elif inertia_diff_pct > 20:
        print("  Status: WARNING - mock inertia higher, points more spread out than original")
    elif inertia_diff_pct < -20:
        print("  Status: WARNING - mock inertia lower, points tighter than original (easier for ANN)")
    else:
        print("  Status: OK - minor difference in inertia")
    
    # Summary
    print(f"\n[Variance Verification Summary]")
    if variance_ratios:
        mean_ratio = np.mean(variance_ratios)
        print(f"  Mean variance ratio: {mean_ratio:.3f}")
        if 0.8 <= mean_ratio <= 1.2:
            print(f"  Status: GOOD - variance distribution matches expected")
        elif mean_ratio < 0.8:
            print(f"  Status: WARNING - mock data may be TOO TIGHT (easier for ANN)")
        else:
            print(f"  Status: WARNING - mock data may be TOO SPREAD")
    
    print("=" * 60 + "\n")
    
    return mock_data  # Return for reuse if needed


def run_full_mock_data_approach(
    cluster_config: ClusterConfig,
    total_rows: int,
    n_queries: int,
    k: int,
    build_params,
    param_list: list,
    is_ivfpq: bool = True,
    refine_k: int = 0,
    gt_batch_size: int = 0,
    post_noise: float = 0.0,
):
    """
    Approach 1: full mock data approach
    - Generate full mock dataset from cluster stats
    - Build index in one shot
    - Use brute-force GT
    """
    print("\n" + "=" * 60)
    print("Approach 1: Full mock data (single build + brute-force GT)")
    print("=" * 60)
    
    # Generate full mock dataset
    print(f"\n[Generating full mock dataset]")
    gen_start = time.perf_counter()
    mock_data = generate_mock_data(cluster_config, total_rows)
    gen_time = time.perf_counter() - gen_start
    print(f"  Mock data shape: {mock_data.shape}")
    print(f"  Generation time: {gen_time:.2f}s")
    
    # Apply post-generation noise if specified
    if post_noise > 0:
        print(f"\n[Applying post-generation noise]")
        # Scale noise relative to the data's standard deviation
        data_std = mock_data.std()
        noise_std = data_std * post_noise
        print(f"  Data std: {data_std:.6f}, noise std: {noise_std:.6f} (post_noise={post_noise})")
        rng = np.random.default_rng(seed=12345)
        noise = rng.normal(0, noise_std, mock_data.shape).astype(np.float32)
        mock_data = mock_data + noise
        print(f"  Added Gaussian noise to mock data")
    
    # Generate queries and brute-force GT
    print(f"\n[Generating queries and brute-force GT]")
    queries, gt_indices, gt_distances = gen_gt_brute_force(
        mock_data, n_queries, k, seed=12345, backend="cuvs", batch_size=gt_batch_size
    )
    print(f"  mock_data shape: {mock_data.shape}, queries shape: {queries.shape}")

    # Build index
    print(f"\n[Building index (single call)]")
    build_start = time.perf_counter()
    if is_ivfpq:
        ann_index = IvfPqIndex(build_params=build_params, store_vectors=(refine_k > 0))
    else:
        ann_index = CagraIndex(build_params=build_params)
    ann_index.build(mock_data)
    build_time = time.perf_counter() - build_start
    print(f"  Build time: {build_time:.2f}s")

    # Sweep search params
    results = []
    search_kwarg = 'nprobes' if is_ivfpq else 'itopk'
    n_runs = 5
    use_refine = is_ivfpq and refine_k > 0
    print(f"\n[Sweeping search params]{' (with refine_k=' + str(refine_k) + ')' if use_refine else ''}")
    for param in param_list:
        # Warmup
        if use_refine:
            _ = ann_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
        else:
            _ = ann_index.search(queries, k, **{search_kwarg: param})
        
        # Timed search (average over n_runs)
        times = []
        for _ in range(n_runs):
            search_start = time.perf_counter()
            if use_refine:
                pred_indices, pred_distances = ann_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
            else:
                pred_indices, pred_distances = ann_index.search(queries, k, **{search_kwarg: param})
            times.append(time.perf_counter() - search_start)
        search_time = np.mean(times)
        
        recall, _ = compute_recall_with_ties(pred_indices, pred_distances, gt_indices, gt_distances)
        qps = n_queries / search_time
        results.append({'param': param, 'recall': recall, 'qps': qps})
        times_str = ", ".join([f"{t*1000:.2f}ms" for t in times])
        print(f"  param={param}: recall={recall:.4f}, QPS={qps:,.0f}, times=[{times_str}]")
    
    return results, build_time


def run_full_original_data_approach(
    original_data: np.ndarray,
    n_queries: int,
    k: int,
    build_params,
    param_list: list,
    is_ivfpq: bool = True,
    refine_k: int = 0,
    gt_batch_size: int = 0
):
    """
    Approach 2: Full original data approach
    - Use original dataset
    - Build index in one shot
    - Use brute-force GT
    """
    print("\n" + "=" * 60)
    print("Approach 2: Full original data (single build + brute-force GT)")
    print("=" * 60)
    
    # Generate queries and brute-force GT
    print(f"\n[Generating queries and brute-force GT on original data]")
    queries, gt_indices, gt_distances = gen_gt_brute_force(
        original_data, n_queries, k, seed=12345, backend="cuvs", batch_size=gt_batch_size
    )
    
    # Single build on original
    print(f"\n[Building index (single call)]")
    build_start = time.perf_counter()
    if is_ivfpq:
        ann_index = build_single_ivfpq(original_data, build_params, store_vectors=(refine_k > 0))
    else:
        ann_index = build_single_cagra(original_data, build_params)
    build_time = time.perf_counter() - build_start
    print(f"  Build time: {build_time:.2f}s")
    
    # Sweep search params
    results = []
    search_kwarg = 'nprobes' if is_ivfpq else 'itopk'
    n_runs = 5
    use_refine = is_ivfpq and refine_k > 0
    print(f"\n[Sweeping search params]{' (with refine_k=' + str(refine_k) + ')' if use_refine else ''}")
    for param in param_list:
        # Warmup
        if use_refine:
            _ = ann_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
        else:
            _ = ann_index.search(queries, k, **{search_kwarg: param})
        
        # Timed search (average over n_runs)
        times = []
        for _ in range(n_runs):
            search_start = time.perf_counter()
            if use_refine:
                pred_indices, pred_distances = ann_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
            else:
                pred_indices, pred_distances = ann_index.search(queries, k, **{search_kwarg: param})
            times.append(time.perf_counter() - search_start)
        search_time = np.mean(times)
        
        recall, _ = compute_recall_with_ties(pred_indices, pred_distances, gt_indices, gt_distances)
        qps = n_queries / search_time
        results.append({'param': param, 'recall': recall, 'qps': qps})
        times_str = ", ".join([f"{t*1000:.2f}ms" for t in times])
        print(f"  param={param}: recall={recall:.4f}, QPS={qps:,.0f}, times=[{times_str}]")
    
    return results, build_time


def print_summary(
    mock_results, orig_results,
    mock_time, orig_time,
    param_name: str
):
    """Print comparison summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Full Single Builds Comparison")
    print("=" * 80)
    
    print(f"\nBuild Times:")
    print(f"  1. Full mock (single build):    {mock_time:.2f}s")
    print(f"  2. Original data (single build): {orig_time:.2f}s")
    
    print(f"\nRecall Comparison:")
    print(f"  {param_name:>8}  {'full_mock':>12}  {'original':>12}  {'diff':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
    
    for i in range(len(mock_results)):
        param = mock_results[i]['param']
        r1 = mock_results[i]['recall']
        r2 = orig_results[i]['recall']
        diff = r1 - r2
        print(f"  {param:>8}  {r1:>12.4f}  {r2:>12.4f}  {diff:>+10.4f}")
    
    print(f"\nQPS Comparison:")
    print(f"  {param_name:>8}  {'full_mock':>12}  {'original':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")
    
    for i in range(len(mock_results)):
        param = mock_results[i]['param']
        q1 = mock_results[i]['qps']
        q2 = orig_results[i]['qps']
        print(f"  {param:>8}  {q1:>12,.0f}  {q2:>12,.0f}")


def plot_pareto_curves(
    mock_results, orig_results,
    dataset_name: str,
    cluster_stats_name: str,
    total_rows: int,
    algorithm: str,
    n_lists: int = 0,
    pq_dim: int = 0,
    refine_k: int = 0,
    variance_scale: float = 1.0,
    output_dir: str = "pareto_curves"
):
    """Plot Pareto curves for both approaches."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data
    def get_recall_qps(results):
        recalls = [r['recall'] for r in results]
        qps = [r['qps'] for r in results]
        return recalls, qps
    
    mock_recalls, mock_qps = get_recall_qps(mock_results)
    orig_recalls, orig_qps = get_recall_qps(orig_results)
    
    # Plot each approach
    ax.plot(mock_recalls, mock_qps, 's-', label='1. Full mock (single build + brute-force GT)', 
            markersize=10, linewidth=2, color='tab:orange')
    ax.plot(orig_recalls, orig_qps, 'd-', label='2. Original (single build + brute-force GT)', 
            markersize=10, linewidth=2, color='tab:red')
    
    # Labels and title
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Queries per Second (QPS)', fontsize=12)
    
    # Format data size
    if total_rows >= 1_000_000:
        size_str = f"{total_rows / 1_000_000:.1f}M"
    else:
        size_str = f"{total_rows / 1_000:.0f}K"
    
    # Build title based on algorithm
    var_str = f", var_scale={variance_scale}x" if variance_scale != 1.0 else ""
    if algorithm.lower() == "ivfpq":
        algo_params = f"n_lists={n_lists}, pq_dim={pq_dim}"
        if refine_k > 0:
            algo_params += f", refine_k={refine_k}"
        title = (f"Pareto Comparison: Mock vs Original (Single Builds)\n"
                 f"Dataset: {dataset_name} | Clusters: {cluster_stats_name}\n"
                 f"Size: {size_str}{var_str}\n"
                 f"IVF-PQ: {algo_params}")
    else:
        title = (f"Pareto Comparison: Mock vs Original (Single Builds)\n"
                 f"Dataset: {dataset_name} | Clusters: {cluster_stats_name}\n"
                 f"Size: {size_str}{var_str}\n"
                 f"Algorithm: {algorithm.upper()}")
    ax.set_title(title, fontsize=11)
    
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save - include params in filename
    os.makedirs(output_dir, exist_ok=True)
    if algorithm.lower() == "ivfpq":
        filename = f"full_builds_{dataset_name}_N{size_str}_nlist{n_lists}_pqdim{pq_dim}"
        if refine_k > 0:
            filename += f"_refine{refine_k}"
        if variance_scale != 1.0:
            filename += f"_varscale{variance_scale}"
        filename += ".png"
    else:
        filename = f"full_builds_{dataset_name}_N{size_str}"
        if variance_scale != 1.0:
            filename += f"_varscale{variance_scale}"
        filename += f"_{algorithm}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    
    plt.show()
    
    return output_path


if __name__ == "__main__":
    import pickle
    
    parser = argparse.ArgumentParser(description="Compare full single builds: mock vs original")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original dataset (.fbin, .npy, or 'wiki', 'clothes')")
    parser.add_argument("--cluster_stats", type=str, required=True,
                        help="Path to cluster_stats.npz")
    parser.add_argument("--algorithm", type=str, default="ivfpq", choices=["cagra", "ivfpq"])
    parser.add_argument("--total_rows", type=int, default=-1,
                        help="Total rows for mock data (-1 = match original)")
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--refine_k", type=int, default=0,
                        help="For IVF-PQ: retrieve refine_k candidates and re-rank (0=disable)")
    parser.add_argument("--n_lists", type=int, default=0,
                        help="For IVF-PQ: number of IVF lists (0=auto: min(1024, total_rows//100))")
    parser.add_argument("--pq_dim", type=int, default=0,
                        help="For IVF-PQ: PQ dimension (0=auto: min(n_cols, 128))")
    parser.add_argument("--variance_scale", type=float, default=1.0,
                        help="Scale factor for cluster variances (>1 = more overlap between clusters)")
    parser.add_argument("--pca_scale", type=float, default=1.0,
                        help="Scale factor for PCA explained variance (<1 = less structure = harder)")
    parser.add_argument("--outlier_fraction", type=float, default=0.0,
                        help="Fraction of points to replace with outliers (e.g., 0.05 = 5%% outliers)")
    parser.add_argument("--student_df", type=float, default=None,
                        help="Degrees of freedom for Student's t distribution (None=Gaussian, 5=moderate, 3=heavy tails)")
    parser.add_argument("--gt_batch_size", type=int, default=0,
                        help="Batch size for GT computation on large datasets (0=no batching, all in memory)")
    parser.add_argument("--post_noise", type=float, default=0.0,
                        help="Post-generation noise as fraction of data std (e.g., 0.1 = 10%% of data std)")
    # CAGRA with IVF-PQ build params
    parser.add_argument("--cagra_ivf_n_lists", type=int, default=0,
                        help="For CAGRA with build_algo='ivfpq': number of IVF lists (0=use cuvs default)")
    parser.add_argument("--cagra_ivf_pq_dim", type=int, default=0,
                        help="For CAGRA with build_algo='ivfpq': PQ dimension (0=use cuvs default)")
    parser.add_argument("--cagra_ivf_nprobes", type=int, default=0,
                        help="For CAGRA with build_algo='ivfpq': number of probes for search (0=use cuvs default)")
    
    args = parser.parse_args()
    
    # Load original dataset
    print("Loading original dataset...")
    if args.original == "wiki":
        data_path = "/datasets/jinsolp/data/wiki_all/base.88M.fbin"
        original_data, _, n_dim = load_memmap_bin(data_path, np.float32, extra=args.total_rows)
    elif args.original == "clothes":
        data_path = "/datasets/jinsolp/data/amazon_reviews/Clothing_Shoes_and_Jewelry.pkl"
        with open(data_path, "rb") as f:
            original_data = pickle.load(f)
        _, n_dim = original_data.shape
    elif args.original.endswith(".npy"):
        original_data = np.load(args.original)
        if args.total_rows > 0 and args.total_rows < len(original_data):
            original_data = original_data[:args.total_rows]
    elif args.original.endswith(".pkl"):
        with open(args.original, "rb") as f:
            original_data = pickle.load(f)
        if not isinstance(original_data, np.ndarray):
            original_data = np.array(original_data, dtype=np.float32)
    else:
        original_data, _, n_dim = load_memmap_bin(args.original, np.float32, extra=args.total_rows)
    
    n_rows, n_cols = original_data.shape
    print(f"Original data shape: {original_data.shape}")
    
    total_rows = args.total_rows if args.total_rows > 0 else n_rows
    
    # Load cluster stats and create config
    print(f"Loading cluster stats from {args.cluster_stats}...")
    stats = load_cluster_stats(args.cluster_stats)
    
    # Apply variance scaling (>1 creates more cluster overlap)
    base_variances = stats['variances_per_dim'] if stats['variances_per_dim'] is not None else stats['variances']
    scaled_variances = base_variances * args.variance_scale
    if args.variance_scale != 1.0:
        print(f"  Scaling variances by {args.variance_scale}x")
    
    # Check if low-rank stats are available
    is_lowrank = stats.get('is_lowrank', False)
    
    if is_lowrank:
        print(f"  Low-rank covariance detected (PCA components per cluster)")
        if args.pca_scale != 1.0:
            print(f"  PCA scale: {args.pca_scale}x (less structure)")
        if args.outlier_fraction > 0:
            print(f"  Outlier fraction: {args.outlier_fraction*100:.1f}% (makes data harder)")
        if args.student_df is not None:
            print(f"  Student's t df: {args.student_df} (heavy-tailed distribution)")
        cluster_config = ClusterConfig(
            nclusters=len(stats['centroids']),
            ncols=n_cols,
            seed=42,
            cluster_centers=stats['centroids'],
            cluster_variances=scaled_variances,
            cluster_densities=stats['densities'],
            cluster_mins=stats.get('mins_per_dim'),
            cluster_maxs=stats.get('maxs_per_dim'),
            # Low-rank specific
            pca_components_list=stats['pca_components_list'],
            pca_explained_var_list=stats['pca_explained_var_list'],
            pca_noise_var=stats['pca_noise_var'],
            pca_scale=args.pca_scale,
            outlier_fraction=args.outlier_fraction,
            student_df=args.student_df,
        )
    else:
        if args.outlier_fraction > 0:
            print(f"  Outlier fraction: {args.outlier_fraction*100:.1f}% (makes data harder)")
        if args.student_df is not None:
            print(f"  Student's t df: {args.student_df} (heavy-tailed distribution)")
        cluster_config = ClusterConfig(
            nclusters=len(stats['centroids']),
            ncols=n_cols,
            seed=42,
            cluster_centers=stats['centroids'],
            cluster_variances=scaled_variances,
            cluster_densities=stats['densities'],
            cluster_mins=stats.get('mins_per_dim'),
            cluster_maxs=stats.get('maxs_per_dim'),
            outlier_fraction=args.outlier_fraction,
            student_df=args.student_df,
        )
    print(f"  Loaded {cluster_config.nclusters} clusters, {cluster_config.ncols} dimensions")
    
    # Setup algorithm-specific params
    is_ivfpq = args.algorithm == "ivfpq"
    
    if is_ivfpq:
        param_list = [16, 32, 64, 128, 256]
        param_name = "n_probes"
        n_lists = args.n_lists if args.n_lists > 0 else min(1024, total_rows // 100)
        pq_dim = args.pq_dim if args.pq_dim > 0 else min(n_cols, 128)
        build_params = ivf_pq.IndexParams(
            n_lists=n_lists,
            pq_dim=pq_dim,
            pq_bits=8,
        )
        print(f"IVF-PQ params: n_lists={n_lists}, pq_dim={pq_dim}")
    else:
        param_list = [32, 64, 128, 256]
        param_name = "itopk"
        
        # Build IVF-PQ params for CAGRA if any are specified
        ivf_pq_build_params = None
        ivf_pq_search_params = None
        
        if args.cagra_ivf_n_lists > 0 or args.cagra_ivf_pq_dim > 0:
            ivf_pq_kwargs = {}
            if args.cagra_ivf_n_lists > 0:
                ivf_pq_kwargs['n_lists'] = args.cagra_ivf_n_lists
            if args.cagra_ivf_pq_dim > 0:
                ivf_pq_kwargs['pq_dim'] = args.cagra_ivf_pq_dim
            ivf_pq_build_params = ivf_pq.IndexParams(**ivf_pq_kwargs)
            print(f"CAGRA IVF-PQ build params: {ivf_pq_kwargs}")
        
        if args.cagra_ivf_nprobes > 0:
            ivf_pq_search_params = ivf_pq.SearchParams(n_probes=args.cagra_ivf_nprobes)
            print(f"CAGRA IVF-PQ search params: n_probes={args.cagra_ivf_nprobes}")
        
        build_params = cagra.IndexParams(
            intermediate_graph_degree=128,
            graph_degree=64,
            build_algo="ivf_pq",
            refinement_rate=2.0,
            ivf_pq_build_params=ivf_pq_build_params,
            ivf_pq_search_params=ivf_pq_search_params
        )

        # ace_params = cagra.AceParams(
        #     # npartitions=4,
        # )

        # build_params = cagra.IndexParams(
        #     intermediate_graph_degree=256,
        #     graph_degree=128,
        #     build_algo="ace",
        #     ace_params=ace_params,
        # )

        # build_params = cagra.IndexParams(
        #     intermediate_graph_degree=128,
        #     graph_degree=64,
        #     build_algo="nn_descent",
        #     nn_descent_n_clusters=1,
        #     nn_descent_overlap_factor=2
        # )
    

    # Verify variance distribution before running benchmarks
    # NOTE: Disabled to save GPU memory - verification generates ~47GB extra data
    # verify_variance_distribution(original_data, cluster_config, total_rows, n_clusters_to_check=20)
    
    # Run both approaches
    if args.refine_k > 0 and is_ivfpq:
        print(f"IVF-PQ refinement enabled: refine_k={args.refine_k}")
    
    mock_results, mock_time = run_full_mock_data_approach(
        cluster_config, total_rows,
        args.n_queries, args.k,
        build_params, param_list, is_ivfpq, refine_k=args.refine_k,
        gt_batch_size=args.gt_batch_size,
        post_noise=args.post_noise,
    )
    
    orig_results, orig_time = run_full_original_data_approach(
        original_data,
        args.n_queries, args.k,
        build_params, param_list, is_ivfpq, refine_k=args.refine_k,
        gt_batch_size=args.gt_batch_size
    )
    
    # Print summary
    print_summary(
        mock_results, orig_results,
        mock_time, orig_time,
        param_name
    )
    
    # Plot Pareto curves
    dataset_name = os.path.basename(args.original).replace('.fbin', '').replace('.npy', '')
    cluster_stats_name = os.path.basename(args.cluster_stats).replace('.npz', '')
    plot_pareto_curves(
        mock_results, orig_results,
        dataset_name=dataset_name,
        cluster_stats_name=cluster_stats_name,
        total_rows=total_rows,
        algorithm=args.algorithm,
        n_lists=n_lists if is_ivfpq else 0,
        pq_dim=pq_dim if is_ivfpq else 0,
        refine_k=args.refine_k,
        variance_scale=args.variance_scale,
        output_dir="pareto_curves"
    )
