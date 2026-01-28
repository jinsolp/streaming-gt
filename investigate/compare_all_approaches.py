"""
Compare 4 approaches for ANN benchmarking:
1. scalable_gt: Mock data generated on-the-fly, streaming build, cluster-based GT
2. compare_pareto mock: Full mock dataset, single build, brute-force GT
3. Original streaming: Original data, streaming build, brute-force GT
4. Original single: Original data, single build, brute-force GT

Usage:
    python investigate/compare_all_approaches.py \
        --original investigate/subsampled_data/clothes_pareto_1M_seed123.fbin \
        --cluster_stats investigate/cluster_stats/cluster_stats_clothes_extract_5M_seed42_sample5000000_n1000.npz \
        --algorithm ivfpq
"""

import numpy as np
import time
import argparse
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extract_values import load_memmap_bin, load_cluster_stats
from config import ClusterConfig, get_cluster_config, BenchmarkConfig
from utils import brute_force_knn, compute_recall_with_ties
from cuvs.neighbors import cagra, ivf_pq
from ann_indices import CagraIndex, IvfPqIndex

# Reuse from compare_pareto.py (none needed anymore)

# Reuse from stream_vs_single.py
from stream_vs_single import (
    build_streaming_cagra, build_single_cagra, sweep_cagra,
    build_streaming_ivfpq, build_single_ivfpq, sweep_ivfpq
)

# Reuse from scalable_gt.py
from scalable_gt import (
    gen_gt_with_nprobes, gen_build_sample, gen_extend_batch,
    get_num_points_per_cluster, generate_mock_data, gen_gt_brute_force
)


def run_scalable_gt_approach(
    cluster_config: ClusterConfig,
    total_rows: int,
    batch_size: int,
    n_queries: int,
    k: int,
    nprobes_gt: int,
    build_params,
    param_list: list,
    is_ivfpq: bool = True,
    refine_k: int = 0
):
    """
    Approach 1: scalable_gt style
    - Generate mock data on-the-fly during build
    - Use cluster-based GT (only search nearby clusters)
    """
    print("\n" + "=" * 60)
    print("Approach 1: scalable_gt (streaming mock + cluster GT)")
    print("=" * 60)
    
    # Build index with streaming (same as scalable_gt.py)
    print(f"\n[Building index with streaming]")
    build_start = time.perf_counter()
    
    if is_ivfpq:
        ann_index = IvfPqIndex(build_params=build_params, store_vectors=(refine_k > 0))
    else:
        ann_index = CagraIndex(build_params=build_params)
    
    # Sample representative build data from all clusters
    build_vectors, build_indices, _, n_sampled_per_cluster = gen_build_sample(
        sample_size=batch_size,
        total_rows=total_rows,
        config=cluster_config
    )
    
    ann_index.build(build_vectors, indices=build_indices)
    print(f"  Built with {len(build_vectors):,} vectors from {(n_sampled_per_cluster > 0).sum()} clusters")
    
    # Extend with remaining vectors (distributed across clusters for better graph connectivity)
    n_cumulative_sampled = n_sampled_per_cluster.copy()
    total_remaining = total_rows - n_cumulative_sampled.sum()
    num_extend_batches = (total_remaining + batch_size - 1) // batch_size
    
    for batch_num in tqdm(range(num_extend_batches), desc="Extending"):
        vectors, batch_indices, _, n_batch_sampled = gen_extend_batch(
            batch_num, batch_size, total_rows, 
            cluster_config, n_cumulative_sampled
        )
        if len(vectors) > 0:
            ann_index.extend(vectors, indices=batch_indices)
            n_cumulative_sampled += n_batch_sampled
    
    build_time = time.perf_counter() - build_start
    print(f"  Build time: {build_time:.2f}s")
    
    # Generate queries and cluster-based GT
    print(f"\n[Generating queries and cluster-based GT (nprobes={nprobes_gt})]")
    gt_start = time.perf_counter()
    queries, gt_indices, gt_distances = gen_gt_with_nprobes(
        nqueries=n_queries,
        total_rows=total_rows,
        config=cluster_config,
        k=k,
        nprobes=nprobes_gt,
        backend="cuvs"
    )
    gt_time = time.perf_counter() - gt_start
    print(f"  GT generation time: {gt_time:.2f}s")
    
    # Sweep search params
    results = []
    search_kwarg = 'nprobes' if is_ivfpq else 'itopk'
    n_runs = 3
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
        print(f"  param={param}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    return results, build_time


def run_full_mock_data_approch(
    cluster_config: ClusterConfig,
    total_rows: int,
    n_queries: int,
    k: int,
    build_params,
    param_list: list,
    is_ivfpq: bool = True,
    refine_k: int = 0,
    gt_batch_size: int = 0
):
    """
    Approach 2: full mock data approach
    - Generate full mock dataset
    - Build index in one shot
    - Use brute-force GT
    """
    print("\n" + "=" * 60)
    print("Approach 2: full mock data (full mock + brute-force GT)")
    print("=" * 60)
    
    # Generate full mock dataset
    print(f"\n[Generating full mock dataset]")
    gen_start = time.perf_counter()
    mock_data = generate_mock_data(cluster_config, total_rows)
    gen_time = time.perf_counter() - gen_start
    print(f"  Mock data shape: {mock_data.shape}")
    print(f"  Generation time: {gen_time:.2f}s")
    
    # Generate queries and brute-force GT
    print(f"\n[Generating queries and brute-force GT]")
    queries, gt_indices, gt_distances = gen_gt_brute_force(
        mock_data, n_queries, k, seed=12345, backend="cuvs", batch_size=gt_batch_size
    )
    
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
    n_runs = 3
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
        print(f"  param={param}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    return results, build_time, mock_data


def run_original_data_approaches(
    original_data: np.ndarray,
    batch_size: int,
    n_queries: int,
    k: int,
    build_params,
    param_list: list,
    is_ivfpq: bool = True,
    refine_k: int = 0,
    gt_batch_size: int = 0,
    skip_single_build: bool = False
):
    """
    Approach 3 & 4: stream_vs_single style on original data
    - Use original dataset
    - Compare streaming vs single build
    - Use brute-force GT
    """
    print("\n" + "=" * 60)
    print(f"Approach 3{'' if skip_single_build else ' & 4'}: Original data (streaming{'' if skip_single_build else ' vs single'})")
    print("=" * 60)
    
    # Generate queries and brute-force GT (shared for both approaches)
    print(f"\n[Generating queries and brute-force GT on original data]")
    queries, gt_indices, gt_distances = gen_gt_brute_force(
        original_data, n_queries, k, seed=12345, backend="cuvs", batch_size=gt_batch_size
    )
    
    # Approach 3: Streaming build on original
    print(f"\n[Approach 3: Streaming build on original data]")
    stream_start = time.perf_counter()
    if is_ivfpq:
        stream_index = build_streaming_ivfpq(original_data, batch_size, build_params, store_vectors=(refine_k > 0))
    else:
        stream_index = build_streaming_cagra(original_data, batch_size, build_params)
    stream_time = time.perf_counter() - stream_start
    print(f"  Build time: {stream_time:.2f}s")
    
    # Approach 4: Single build on original (skip if data too large)
    if skip_single_build:
        print(f"\n[Approach 4: SKIPPED (data too large for single build)]")
        single_index = None
        single_time = 0.0
    else:
        print(f"\n[Approach 4: Single build on original data]")
        single_start = time.perf_counter()
        if is_ivfpq:
            single_index = build_single_ivfpq(original_data, build_params, store_vectors=(refine_k > 0))
        else:
            single_index = build_single_cagra(original_data, build_params)
        single_time = time.perf_counter() - single_start
        print(f"  Build time: {single_time:.2f}s")
    
    # Sweep for streaming
    search_kwarg = 'nprobes' if is_ivfpq else 'itopk'
    n_runs = 3
    use_refine = is_ivfpq and refine_k > 0
    stream_results = []
    print(f"\n[Sweeping - Streaming Index]{' (with refine_k=' + str(refine_k) + ')' if use_refine else ''}")
    for param in param_list:
        # Warmup
        if use_refine:
            _ = stream_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
        else:
            _ = stream_index.search(queries, k, **{search_kwarg: param})
        
        # Timed search (average over n_runs)
        times = []
        for _ in range(n_runs):
            search_start = time.perf_counter()
            if use_refine:
                pred_indices, pred_distances = stream_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
            else:
                pred_indices, pred_distances = stream_index.search(queries, k, **{search_kwarg: param})
            times.append(time.perf_counter() - search_start)
        search_time = np.mean(times)
        
        recall, _ = compute_recall_with_ties(pred_indices, pred_distances, gt_indices, gt_distances)
        qps = n_queries / search_time
        stream_results.append({'param': param, 'recall': recall, 'qps': qps})
        print(f"  param={param}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    # Sweep for single (skip if data too large)
    single_results = []
    if skip_single_build:
        # Create dummy results
        single_results = [{'param': r['param'], 'recall': np.nan, 'qps': np.nan} for r in stream_results]
        print(f"\n[Sweeping - Single Build Index]: SKIPPED")
    else:
        print(f"\n[Sweeping - Single Build Index]{' (with refine_k=' + str(refine_k) + ')' if use_refine else ''}")
        for param in param_list:
            # Warmup
            if use_refine:
                _ = single_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
            else:
                _ = single_index.search(queries, k, **{search_kwarg: param})
            
            # Timed search (average over n_runs)
            times = []
            for _ in range(n_runs):
                search_start = time.perf_counter()
                if use_refine:
                    pred_indices, pred_distances = single_index.search_with_refine(queries, k, refine_k=refine_k, nprobes=param)
                else:
                    pred_indices, pred_distances = single_index.search(queries, k, **{search_kwarg: param})
                times.append(time.perf_counter() - search_start)
            search_time = np.mean(times)
            
            recall, _ = compute_recall_with_ties(pred_indices, pred_distances, gt_indices, gt_distances)
            qps = n_queries / search_time
            single_results.append({'param': param, 'recall': recall, 'qps': qps})
            print(f"  param={param}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    return stream_results, single_results, stream_time, single_time


def print_summary(
    scalable_results, pareto_results, orig_stream_results, orig_single_results,
    scalable_time, pareto_time, orig_stream_time, orig_single_time,
    param_name: str
):
    """Print comparison summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY: All 4 Approaches")
    print("=" * 100)
    
    print(f"\nBuild Times:")
    print(f"  1. scalable_gt (streaming mock):     {scalable_time:.2f}s")
    print(f"  2. compare_pareto (full mock):       {pareto_time:.2f}s")
    print(f"  3. Original data (streaming):        {orig_stream_time:.2f}s")
    print(f"  4. Original data (single):           {orig_single_time:.2f}s")
    
    print(f"\nRecall Comparison:")
    print(f"  {param_name:>8}  {'scalable_gt':>12}  {'full_mock':>12}  {'orig_stream':>12}  {'full_single':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    
    for i in range(len(scalable_results)):
        param = scalable_results[i]['param']
        r1 = scalable_results[i]['recall']
        r2 = pareto_results[i]['recall']
        r3 = orig_stream_results[i]['recall']
        r4 = orig_single_results[i]['recall']
        print(f"  {param:>8}  {r1:>12.4f}  {r2:>12.4f}  {r3:>12.4f}  {r4:>12.4f}")
    
    print(f"\nQPS Comparison:")
    print(f"  {param_name:>8}  {'scalable_gt':>12}  {'full_mock':>12}  {'orig_stream':>12}  {'full_single':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    
    for i in range(len(scalable_results)):
        param = scalable_results[i]['param']
        q1 = scalable_results[i]['qps']
        q2 = pareto_results[i]['qps']
        q3 = orig_stream_results[i]['qps']
        q4 = orig_single_results[i]['qps']
        print(f"  {param:>8}  {q1:>12,.0f}  {q2:>12,.0f}  {q3:>12,.0f}  {q4:>12,.0f}")


def plot_pareto_curves(
    scalable_results, pareto_results, orig_stream_results, orig_single_results,
    dataset_name: str,
    cluster_stats_name: str,
    total_rows: int,
    batch_size: int,
    nprobes_gt: int,
    algorithm: str,
    n_lists: int = 0,
    pq_dim: int = 0,
    refine_k: int = 0,
    variance_scale: float = 1.0,
    output_dir: str = "pareto_curves"
):
    """Plot Pareto curves for all 4 approaches."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    def get_recall_qps(results):
        recalls = [r['recall'] for r in results]
        qps = [r['qps'] for r in results]
        return recalls, qps
    
    scalable_recalls, scalable_qps = get_recall_qps(scalable_results)
    pareto_recalls, pareto_qps = get_recall_qps(pareto_results)
    orig_stream_recalls, orig_stream_qps = get_recall_qps(orig_stream_results)
    orig_single_recalls, orig_single_qps = get_recall_qps(orig_single_results)
    
    # Plot each approach
    ax.plot(scalable_recalls, scalable_qps, 'o-', label='1. scalable_gt (streaming mock + cluster GT)', 
            markersize=8, linewidth=2)
    ax.plot(pareto_recalls, pareto_qps, 's-', label='2. Full mock (single build + brute-force GT)', 
            markersize=8, linewidth=2)
    ax.plot(orig_stream_recalls, orig_stream_qps, '^--', label='3. Original (streaming + brute-force GT)', 
            markersize=8, linewidth=2)
    if not all(np.isnan(r) for r in orig_single_recalls):
        ax.plot(orig_single_recalls, orig_single_qps, 'd--', label='4. Original (single + brute-force GT)', 
                markersize=8, linewidth=2)
    
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
        title = (f"Pareto Comparison: All 4 Approaches\n"
                 f"Dataset: {dataset_name} | Clusters: {cluster_stats_name}\n"
                 f"Size: {size_str} | Batch: {batch_size:,} | GT nprobes: {nprobes_gt}{var_str}\n"
                 f"IVF-PQ: {algo_params}")
    else:
        title = (f"Pareto Comparison: All 4 Approaches\n"
                 f"Dataset: {dataset_name} | Clusters: {cluster_stats_name}\n"
                 f"Size: {size_str} | Batch: {batch_size:,} | GT nprobes: {nprobes_gt}{var_str}\n"
                 f"Algorithm: {algorithm.upper()}")
    ax.set_title(title, fontsize=11)
    
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save - include params in filename
    os.makedirs(output_dir, exist_ok=True)
    if algorithm.lower() == "ivfpq":
        filename = f"all_approaches_{dataset_name}_N{size_str}_nlist{n_lists}_pqdim{pq_dim}"
        if refine_k > 0:
            filename += f"_refine{refine_k}"
        if variance_scale != 1.0:
            filename += f"_varscale{variance_scale}"
        filename += ".png"
    else:
        filename = f"all_approaches_{dataset_name}_N{size_str}_batch{batch_size}"
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
    
    parser = argparse.ArgumentParser(description="Compare all 4 approaches")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original dataset (.fbin, .npy, or 'wiki')")
    parser.add_argument("--cluster_stats", type=str, required=True,
                        help="Path to cluster_stats.npz")
    parser.add_argument("--algorithm", type=str, default="ivfpq", choices=["cagra", "ivfpq"])
    parser.add_argument("--total_rows", type=int, default=-1,
                        help="Total rows for mock data (-1 = match original)")
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--nprobes_gt", type=int, default=200,
                        help="Number of probes for cluster-based GT in scalable_gt approach")
    parser.add_argument("--refine_k", type=int, default=0,
                        help="For IVF-PQ: retrieve refine_k candidates and re-rank (0=disable)")
    parser.add_argument("--n_lists", type=int, default=0,
                        help="For IVF-PQ: number of IVF lists (0=auto: min(1024, total_rows//100))")
    parser.add_argument("--pq_dim", type=int, default=0,
                        help="For IVF-PQ: PQ dimension (0=auto: min(n_cols, 128))")
    parser.add_argument("--variance_scale", type=float, default=1.0,
                        help="Scale factor for cluster variances (>1 = more overlap between clusters)")
    parser.add_argument("--gt_batch_size", type=int, default=0,
                        help="Batch size for GT computation on large datasets (0=no batching, all in memory)")
    parser.add_argument("--max_single_build_rows", type=int, default=15_000_000,
                        help="Max rows for single build (larger datasets skip approaches 2 & 4). Default: 15M")
    
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
    
    original_data = original_data.astype(np.float32)
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
    
    cluster_config = ClusterConfig(
        nclusters=len(stats['centroids']),
        ncols=n_cols,
        seed=42,
        cluster_centers=stats['centroids'],
        cluster_variances=scaled_variances,
        cluster_densities=stats['densities'],
        cluster_mins=stats['mins_per_dim'],
        cluster_maxs=stats['maxs_per_dim'],
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
        build_params = cagra.IndexParams(
            intermediate_graph_degree=256,
            graph_degree=128,
            build_algo="nn_descent",
        )
    
    # Run all 4 approaches
    if args.refine_k > 0 and is_ivfpq:
        print(f"IVF-PQ refinement enabled: refine_k={args.refine_k}")
    
    # Auto-detect if data is too large for single build
    skip_single_build = total_rows > args.max_single_build_rows
    if skip_single_build:
        print(f"\n*** Data size ({total_rows:,}) > max_single_build_rows ({args.max_single_build_rows:,})")
        print(f"*** Skipping single-build approaches (2 & 4) to avoid OOM ***\n")
    
    scalable_results, scalable_time = run_scalable_gt_approach(
        cluster_config, total_rows, args.batch_size,
        args.n_queries, args.k, args.nprobes_gt,
        build_params, param_list, is_ivfpq, refine_k=args.refine_k
    )
    
    if skip_single_build:
        print("\n" + "=" * 60)
        print("Approach 2: SKIPPED (data too large for single build)")
        print("=" * 60)
        pareto_results = [{'param': r['param'], 'recall': np.nan, 'qps': np.nan} for r in scalable_results]
        pareto_time = 0.0
        mock_data = None
    else:
        pareto_results, pareto_time, mock_data = run_full_mock_data_approch(
            cluster_config, total_rows,
            args.n_queries, args.k,
            build_params, param_list, is_ivfpq, refine_k=args.refine_k,
            gt_batch_size=args.gt_batch_size
        )
    
    orig_stream_results, orig_single_results, orig_stream_time, orig_single_time = run_original_data_approaches(
        original_data, args.batch_size,
        args.n_queries, args.k,
        build_params, param_list, is_ivfpq, refine_k=args.refine_k,
        gt_batch_size=args.gt_batch_size,
        skip_single_build=skip_single_build
    )
    
    # Print summary
    print_summary(
        scalable_results, pareto_results, orig_stream_results, orig_single_results,
        scalable_time, pareto_time, orig_stream_time, orig_single_time,
        param_name
    )
    
    # Plot Pareto curves
    dataset_name = os.path.basename(args.original).replace('.fbin', '').replace('.npy', '')
    cluster_stats_name = os.path.basename(args.cluster_stats).replace('.npz', '')
    plot_pareto_curves(
        scalable_results, pareto_results, orig_stream_results, orig_single_results,
        dataset_name=dataset_name,
        cluster_stats_name=cluster_stats_name,
        total_rows=total_rows,
        batch_size=args.batch_size,
        nprobes_gt=args.nprobes_gt,
        algorithm=args.algorithm,
        n_lists=n_lists if is_ivfpq else 0,
        pq_dim=pq_dim if is_ivfpq else 0,
        refine_k=args.refine_k,
        variance_scale=args.variance_scale,
        output_dir="pareto_curves"
    )