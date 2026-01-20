"""
Check if vectors in an fbin file are normalized (unit L2 norm).
Optionally normalize and save to a new fbin file.
"""

import numpy as np
import struct
import argparse
import os

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def load_fbin(filepath: str, max_rows: int = -1) -> np.ndarray:
    """Load fbin file with header [n_rows, n_dim] as uint32."""
    with open(filepath, 'rb') as f:
        n_rows, n_dim = struct.unpack('II', f.read(8))
        print(f"File: {filepath}")
        print(f"Shape: ({n_rows:,}, {n_dim})")
        
        if max_rows > 0 and max_rows < n_rows:
            print(f"Loading first {max_rows:,} rows...")
            data = np.fromfile(f, dtype=np.float32, count=max_rows * n_dim)
            data = data.reshape(max_rows, n_dim)
        else:
            data = np.fromfile(f, dtype=np.float32)
            data = data.reshape(n_rows, n_dim)
    
    return data


def write_fbin(filepath: str, data: np.ndarray):
    """Write data to fbin file with header [n_rows, n_dim] as uint32."""
    n_rows, n_dim = data.shape
    with open(filepath, 'wb') as f:
        f.write(struct.pack('II', n_rows, n_dim))
        data.astype(np.float32).tofile(f)
    print(f"\nWritten to: {filepath}")
    print(f"Shape: ({n_rows:,}, {n_dim})")


def normalize_data(data: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Normalize vectors to unit L2 norm."""
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        print("  Using GPU (CuPy) for normalization...")
        data_gpu = cp.asarray(data)
        norms = cp.linalg.norm(data_gpu, axis=1, keepdims=True)
        norms = cp.maximum(norms, 1e-10)
        result = data_gpu / norms
        return cp.asnumpy(result)
    else:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return data / norms


def standardscaler(data: np.ndarray, use_gpu: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply StandardScaler: (x - mean) / std per feature (column).
    
    Args:
        data: Input array (n_samples, n_dims)
        use_gpu: Use CuPy for GPU acceleration
    
    Returns:
        Tuple of (scaled_data, means, stds)
    """
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        print("  Using GPU (CuPy) for StandardScaler...")
        data_gpu = cp.asarray(data)
        means = cp.mean(data_gpu, axis=0)
        stds = cp.std(data_gpu, axis=0)
        # Avoid division by zero
        stds = cp.maximum(stds, 1e-10)
        result = (data_gpu - means) / stds
        return cp.asnumpy(result), cp.asnumpy(means), cp.asnumpy(stds)
    else:
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        # Avoid division by zero
        stds = np.maximum(stds, 1e-10)
        result = (data - means) / stds
        return result, means, stds


def normalize_data_subspace(data: np.ndarray, subspace_size: int, use_gpu: bool = False) -> np.ndarray:
    """Normalize each subspace slice to unit L2 norm independently.
    
    This is useful for data that will be used with subspace clustering (PQ-style),
    where each subspace should be normalized for better KMeans performance.
    
    Args:
        data: Input array (n_samples, n_dims)
        subspace_size: Size of each subspace (n_dims must be divisible by this)
        use_gpu: Use CuPy for GPU acceleration
    
    Returns:
        Array with each subspace normalized to unit norm
    """
    n_samples, n_dims = data.shape
    
    if n_dims % subspace_size != 0:
        raise ValueError(f"n_dims ({n_dims}) must be divisible by subspace_size ({subspace_size})")
    
    n_subspaces = n_dims // subspace_size
    print(f"  Normalizing {n_subspaces} subspaces of {subspace_size}D each...")
    
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        print("  Using GPU (CuPy) for subspace normalization...")
        data_gpu = cp.asarray(data)
        result_gpu = cp.zeros_like(data_gpu)
        
        for i in range(n_subspaces):
            start = i * subspace_size
            end = start + subspace_size
            subspace = data_gpu[:, start:end]
            norms = cp.linalg.norm(subspace, axis=1, keepdims=True)
            norms = cp.maximum(norms, 1e-10)
            result_gpu[:, start:end] = subspace / norms
        
        return cp.asnumpy(result_gpu)
    else:
        result = np.zeros_like(data)
        
        for i in range(n_subspaces):
            start = i * subspace_size
            end = start + subspace_size
            subspace = data[:, start:end]
            norms = np.linalg.norm(subspace, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            result[:, start:end] = subspace / norms
        
        return result


def check_normalized(data: np.ndarray, tolerance: float = 1e-5, use_gpu: bool = False):
    """Check if vectors are normalized (unit L2 norm)."""
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        print("  Using GPU (CuPy) for checking...")
        data_gpu = cp.asarray(data)
        norms_gpu = cp.linalg.norm(data_gpu, axis=1)
        
        # Compute stats on GPU
        min_val = float(cp.min(norms_gpu))
        max_val = float(cp.max(norms_gpu))
        mean_val = float(cp.mean(norms_gpu))
        std_val = float(cp.std(norms_gpu))
        median_val = float(cp.median(norms_gpu))
        
        is_unit_norm = bool(cp.allclose(norms_gpu, 1.0, atol=tolerance))
        within_tolerance = int(cp.sum(cp.abs(norms_gpu - 1.0) < tolerance))
        
        # Transfer norms back to CPU for detailed stats if needed
        norms = cp.asnumpy(norms_gpu)
    else:
        norms = np.linalg.norm(data, axis=1)
        min_val = norms.min()
        max_val = norms.max()
        mean_val = norms.mean()
        std_val = norms.std()
        median_val = np.median(norms)
        is_unit_norm = np.allclose(norms, 1.0, atol=tolerance)
        within_tolerance = np.sum(np.abs(norms - 1.0) < tolerance)
    
    pct_normalized = within_tolerance / len(norms) * 100
    
    print(f"\n{'='*60}")
    print("L2 Norm Statistics")
    print(f"{'='*60}")
    print(f"  Min:    {min_val:.6f}")
    print(f"  Max:    {max_val:.6f}")
    print(f"  Mean:   {mean_val:.6f}")
    print(f"  Std:    {std_val:.6f}")
    print(f"  Median: {median_val:.6f}")
    
    print(f"\n{'='*60}")
    print("Normalization Check")
    print(f"{'='*60}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Vectors within tolerance of 1.0: {within_tolerance:,} / {len(norms):,} ({pct_normalized:.2f}%)")
    
    if is_unit_norm:
        print(f"\n  ✅ Data IS NORMALIZED (all vectors have unit L2 norm)")
    else:
        print(f"\n  ❌ Data is NOT normalized")
        
        # Show distribution of deviations (use CPU norms for detailed analysis)
        deviations = np.abs(norms - 1.0)
        print(f"\n  Deviation from unit norm:")
        print(f"    Max deviation: {deviations.max():.6f}")
        print(f"    Mean deviation: {deviations.mean():.6f}")
        
        # Show some examples
        if deviations.max() > tolerance:
            worst_idx = np.argmax(deviations)
            print(f"\n  Worst case: index {worst_idx}, norm = {norms[worst_idx]:.6f}")
            
            # Show histogram of norms
            print(f"\n  Norm distribution:")
            percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
            for p in percentiles:
                val = np.percentile(norms, p)
                print(f"    p{p:3d}: {val:.6f}")
    
    return is_unit_norm, norms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if fbin vectors are normalized")
    parser.add_argument("file", type=str, help="Path to .fbin file")
    parser.add_argument("--max_rows", type=int, default=-1,
                        help="Max rows to load (-1 for all)")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="Tolerance for considering norm as 1.0 (default: 1e-5)")
    parser.add_argument("--normalize", "-n", action="store_true",
                        help="Normalize the data and write to <input>_normalized.fbin")
    parser.add_argument("--subspace_size", type=int, default=0,
                        help="If > 0, normalize each subspace independently (for PQ-style clustering)")
    parser.add_argument("--first_dims", type=int, default=0,
                        help="If > 0, only keep the first N dimensions (truncate)")
    parser.add_argument("--standardscaler", "-s", action="store_true",
                        help="Apply StandardScaler (mean=0, std=1 per feature) instead of L2 normalization")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CuPy) for checking and normalization")
    
    args = parser.parse_args()
    
    data = load_fbin(args.file, args.max_rows)
    base, ext = os.path.splitext(args.file)
    
    # Truncate to first N dimensions if requested
    if args.first_dims > 0:
        if args.first_dims > data.shape[1]:
            raise ValueError(f"--first_dims ({args.first_dims}) exceeds data dims ({data.shape[1]})")
        print(f"\n{'='*60}")
        print(f"Truncating to first {args.first_dims} dimensions")
        print(f"{'='*60}")
        print(f"  Original: {data.shape}")
        data = data[:, :args.first_dims]
        print(f"  Truncated: {data.shape}")
        
        # Update base for subsequent operations
        base = f"{base}_first{args.first_dims}"
        
        # Save truncated data (only if not normalizing, otherwise we save the final result)
        if not args.normalize:
            output_file = f"{base}{ext}"
            write_fbin(output_file, data)
    
    is_normalized, norms = check_normalized(data, args.tolerance, use_gpu=args.gpu)
    
    if args.normalize or args.standardscaler:
        print(f"\n{'='*60}")
        print("Normalizing Data")
        print(f"{'='*60}")
        
        if args.standardscaler:
            # StandardScaler: (x - mean) / std per feature
            output_file = f"{base}_standardscaled{ext}"
            print(f"  Mode: StandardScaler (mean=0, std=1 per feature)")
            normalized_data, means, stds = standardscaler(data, use_gpu=args.gpu)
            print(f"  Mean range: [{means.min():.6f}, {means.max():.6f}]")
            print(f"  Std range:  [{stds.min():.6f}, {stds.max():.6f}]")
        elif args.subspace_size > 0:
            # Per-subspace normalization
            output_file = f"{base}_separated{args.subspace_size}_normalized{ext}"
            print(f"  Mode: Per-subspace normalization (subspace_size={args.subspace_size})")
            normalized_data = normalize_data_subspace(data, args.subspace_size, use_gpu=args.gpu)
        else:
            # Global normalization
            output_file = f"{base}_normalized{ext}"
            print(f"  Mode: Global normalization (unit L2 norm)")
            normalized_data = normalize_data(data, use_gpu=args.gpu)
        
        write_fbin(output_file, normalized_data)
        
        # Verify the normalization
        print("\nVerifying normalized data...")
        if args.standardscaler:
            # Verify StandardScaler: check mean ~0 and std ~1 per feature
            # Use looser tolerance (1e-3) since floating point accumulation on large datasets
            out_means = np.mean(normalized_data, axis=0)
            out_stds = np.std(normalized_data, axis=0)
            mean_ok = np.allclose(out_means, 0.0, atol=1e-3)
            std_ok = np.allclose(out_stds, 1.0, atol=1e-3)
            print(f"  Output mean range: [{out_means.min():.6f}, {out_means.max():.6f}] (should be ~0)")
            print(f"  Output std range:  [{out_stds.min():.6f}, {out_stds.max():.6f}] (should be ~1)")
            if mean_ok and std_ok:
                print(f"\n✅ StandardScaler applied successfully (mean=0, std=1 per feature)!")
            else:
                print(f"\n⚠️ StandardScaler verification failed (mean_ok={mean_ok}, std_ok={std_ok})")
        elif args.subspace_size > 0:
            # Check each subspace
            n_dims = data.shape[1]
            n_subspaces = n_dims // args.subspace_size
            print(f"\nChecking {n_subspaces} subspaces...")
            all_ok = True
            for i in range(n_subspaces):
                start = i * args.subspace_size
                end = start + args.subspace_size
                subspace_data = normalized_data[:, start:end]
                subspace_norms = np.linalg.norm(subspace_data, axis=1)
                is_ok = np.allclose(subspace_norms, 1.0, atol=args.tolerance)
                status = "✅" if is_ok else "❌"
                print(f"  Subspace {i} (dims {start}-{end-1}): {status} norm range [{subspace_norms.min():.6f}, {subspace_norms.max():.6f}]")
                if not is_ok:
                    all_ok = False
            if all_ok:
                print(f"\n✅ All subspaces normalized to unit norm!")
        else:
            check_normalized(normalized_data, args.tolerance, use_gpu=args.gpu)
