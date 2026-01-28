"""
MRE: Compare streaming build/extend vs single build for ANN index.
Supports both CAGRA and IVF-PQ algorithms.
This is not for generating the mock dataset, but to check if the build/extend is functioning correctly
on the same dataset and query.
"""

import numpy as np
import time
import argparse
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extract_values import load_memmap_bin
from utils import brute_force_knn, compute_recall_with_ties
from cuvs.neighbors import cagra, ivf_pq
from ann_indices import CagraIndex, IvfPqIndex


def generate_queries_and_gt(data, n_queries, k, seed=12345):
    """Generate queries from data + noise, compute brute force GT."""
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(len(data), size=n_queries, replace=False)
    queries = data[query_indices].copy()
    # queries = data[:n_queries].copy()

    noise_scale = np.std(data) * 0.1
    queries += rng.normal(0, noise_scale, queries.shape).astype(np.float32)
    
    print(f"Computing GT for {n_queries} queries (k={k})...")
    gt_indices, gt_distances = brute_force_knn(data, queries, k, backend="cuvs")
    return queries, gt_indices, gt_distances


# ============ CAGRA Functions ============

def build_streaming_cagra(data, batch_size, build_params, seed=42):
    """Build CAGRA index with streaming: sequential batches."""
    n_rows = len(data)
    num_batches = (n_rows + batch_size - 1) // batch_size
    
    index = CagraIndex(build_params=build_params)

    for batch_num in tqdm(range(num_batches), desc="Streaming build (CAGRA)"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        batch_vectors = data[start_idx:end_idx]
        
        if len(batch_vectors) == 0:
            continue
        
        if batch_num == 0:
            index.build(batch_vectors)
        else:
            index.extend(batch_vectors)
    
    return index


def build_single_cagra(data, build_params):
    """Build CAGRA index with all data at once."""
    print("Single build (CAGRA)...")
    index = CagraIndex(build_params=build_params)
    index.build(data)
    return index


def sweep_cagra(index, queries, gt_indices, gt_distances, k, itopk_list, label):
    """Sweep CAGRA itopk values and print recall."""
    print(f"\n[{label}]")
    results = []
    
    for itopk in itopk_list:
        # Warmup
        _ = index.search(queries, k, itopk=itopk)
        
        # Timed search
        search_start = time.perf_counter()
        pred_indices, pred_distances = index.search(queries, k, itopk=itopk)
        search_time = time.perf_counter() - search_start
        
        recall, _ = compute_recall_with_ties(
            pred_indices, pred_distances,
            gt_indices, gt_distances
        )
        
        qps = len(queries) / search_time
        results.append({'param': itopk, 'recall': recall, 'qps': qps})
        print(f"  itopk={itopk}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    return results


# ============ IVF-PQ Functions ============

def build_streaming_ivfpq(data, batch_size, build_params, seed=42):
    """Build IVF-PQ index with streaming: representative first batch, rest extend."""
    n_rows = len(data)
    rng = np.random.default_rng(seed)
    
    # Sample representative first batch from entire dataset
    # This ensures IVF centroids are trained on representative data
    build_sample_size = min(batch_size, n_rows)
    build_indices = rng.choice(n_rows, size=build_sample_size, replace=False)
    build_vectors = data[build_indices]
    
    # Get remaining indices for extend batches
    all_indices = np.arange(n_rows)
    extend_mask = np.ones(n_rows, dtype=bool)
    extend_mask[build_indices] = False
    extend_indices = all_indices[extend_mask]
    
    index = IvfPqIndex(build_params=build_params)
    
    # Build with representative sample (trains IVF centroids on this data)
    # Pass original indices so vectors are indexed correctly
    print(f"  Building with {len(build_vectors):,} randomly sampled vectors...")
    index.build(build_vectors, indices=build_indices)
    
    # Extend with remaining data in batches (with correct original indices)
    num_extend_batches = (len(extend_indices) + batch_size - 1) // batch_size
    
    for batch_num in tqdm(range(num_extend_batches), desc="Extending (IVF-PQ)"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(extend_indices))
        batch_idx = extend_indices[start_idx:end_idx]
        batch_vectors = data[batch_idx]
        
        if len(batch_vectors) > 0:
            index.extend(batch_vectors, indices=batch_idx)
    
    return index


def build_single_ivfpq(data, build_params, store_vectors=False):
    """Build IVF-PQ index with all data at once."""
    print("Single build (IVF-PQ)...")
    index = IvfPqIndex(build_params=build_params, store_vectors=store_vectors)
    index.build(data)
    return index


def sweep_ivfpq(index, queries, gt_indices, gt_distances, k, nprobes_list, label):
    """Sweep IVF-PQ n_probes values and print recall."""
    print(f"\n[{label}]")
    results = []
    
    for n_probes in nprobes_list:
        # Warmup
        _ = index.search(queries, k, nprobes=n_probes)
        
        # Timed search
        search_start = time.perf_counter()
        pred_indices, pred_distances = index.search(queries, k, nprobes=n_probes)
        search_time = time.perf_counter() - search_start
        
        recall, _ = compute_recall_with_ties(
            pred_indices, pred_distances,
            gt_indices, gt_distances
        )
        
        qps = len(queries) / search_time
        results.append({'param': n_probes, 'recall': recall, 'qps': qps})
        print(f"  n_probes={n_probes}: recall={recall:.4f}, QPS={qps:,.0f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare streaming vs single build")
    parser.add_argument("--dataset", type=str, default="wiki")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--algorithm", type=str, default="cagra", choices=["cagra", "ivfpq"],
                        help="ANN algorithm to use")
    parser.add_argument("--pq_dim", type=int, default=0,
                        help="PQ dimension for IVF-PQ (0=auto, higher=better recall)")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "wiki":
        data_path = "/datasets/jinsolp/data/wiki_all/base.88M.fbin"
        data, _, _ = load_memmap_bin(data_path, np.float32, extra=args.subsample)
    elif args.dataset.endswith(".npy"):
        data = np.load(args.dataset)
        if args.subsample > 0 and args.subsample < len(data):
            data = data[:args.subsample]
    else:
        data, _, _ = load_memmap_bin(args.dataset, np.float32, extra=args.subsample)
    
    data = data.astype(np.float32)
    n_rows, n_cols = data.shape
    print(f"Data shape: {data.shape}")
    
    # Random sample 100k for first batch (build), keep rest in original order
    build_size = 100_000
    rng = np.random.default_rng(42)
    all_indices = np.arange(n_rows)
    build_indices = rng.choice(n_rows, size=min(build_size, n_rows), replace=False)
    build_indices_set = set(build_indices)
    remaining_indices = np.array([i for i in all_indices if i not in build_indices_set])
    
    # Reorder: [random 100k for build] + [remaining in original order]
    reordered_indices = np.concatenate([build_indices, remaining_indices])
    data = data[reordered_indices]
    print(f"Data reordered: first {len(build_indices):,} randomly sampled (seed=42), remaining {len(remaining_indices):,} in original order")
    
    
    # Generate queries and GT
    queries, gt_indices, gt_distances = generate_queries_and_gt(
        data, args.n_queries, args.k
    )
    
    if args.algorithm == "cagra":
        # CAGRA configuration
        param_list = [32, 64, 128, 256]
        param_name = "itopk"
        
        build_params = cagra.IndexParams(
            intermediate_graph_degree=256,
            graph_degree=128,
            build_algo="nn_descent",
        )
        
        build_streaming_func = build_streaming_cagra
        build_single_func = build_single_cagra
        sweep_func = sweep_cagra
        
    else:  # ivfpq
        # IVF-PQ configuration
        param_list = [16, 32, 64, 128, 256]
        param_name = "n_probes"
        
        n_lists = min(1024, n_rows // 100)
        # pq_dim: higher = better recall, lower = more compression
        # Must divide n_cols evenly. Common choices: 64, 96, 128, 192
        pq_dim = args.pq_dim if args.pq_dim > 0 else min(n_cols, 128)
        build_params = ivf_pq.IndexParams(
            n_lists=n_lists,
            pq_dim=pq_dim,
            pq_bits=8,
        )
        print(f"IVF-PQ params: n_lists={n_lists}, pq_dim={pq_dim} (dims_per_code={n_cols//pq_dim})")
        
        build_streaming_func = build_streaming_ivfpq
        build_single_func = build_single_ivfpq
        sweep_func = sweep_ivfpq
    
    print(f"\nAlgorithm: {args.algorithm.upper()}")
    
    # Streaming build
    print(f"\n[Building with streaming (batch_size={args.batch_size:,})]")
    stream_start = time.perf_counter()
    stream_index = build_streaming_func(data, args.batch_size, build_params)
    stream_time = time.perf_counter() - stream_start
    print(f"  Streaming build time: {stream_time:.2f}s")
    
    # Single build
    print(f"\n[Building with single call]")
    single_start = time.perf_counter()
    single_index = build_single_func(data, build_params)
    single_time = time.perf_counter() - single_start
    print(f"  Single build time: {single_time:.2f}s")
    
    # Compare recalls
    stream_results = sweep_func(
        stream_index, queries, gt_indices, gt_distances, 
        args.k, param_list, "Streaming Index"
    )
    
    single_results = sweep_func(
        single_index, queries, gt_indices, gt_distances,
        args.k, param_list, "Single Build Index"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: Streaming vs Single Build ({args.algorithm.upper()})")
    print("=" * 60)
    print(f"  Build time - Streaming: {stream_time:.2f}s, Single: {single_time:.2f}s")
    print(f"\n  {param_name:>8} {'Stream':>10} {'Single':>10} {'Diff':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for sr, si in zip(stream_results, single_results):
        diff = sr['recall'] - si['recall']
        print(f"  {sr['param']:>8} {sr['recall']:>10.4f} {si['recall']:>10.4f} {diff:>+10.4f}")
