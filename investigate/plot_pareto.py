"""
Plot Pareto curves comparing:
1. Original data
2. Mock data (previous diagonal covariance approach)
3. Mock data (new PCA low-rank approach)

Usage:
    python plot_pareto.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset and cluster stats info (for title)
ORIGINAL_DATA = "clothes (32M)"
CLUSTER_STATS = "10K clusters (10M subsample)"
ALGORITHM = "CAGRA + IVFPQ (pq_dim=128)"
PCA_NCOMP = 128

# ============================================================================
# PARSER FUNCTION
# ============================================================================

def parse_results(text: str, n_queries: int = 1000) -> list:
    """
    Parse results from compare_full_builds.py output.
    
    Input format:
        [Sweeping search params]
          param=32: recall=0.8493, QPS=192,181, times=[6.68ms, 4.31ms, ...]
          param=64: recall=0.9138, QPS=163,131, times=[...]
          ...
    
    QPS is computed from times:
        - Remove min and max outliers
        - Average the remaining 3 times
        - QPS = n_queries / avg_time
    
    Returns:
        List of dicts: [{'param': 32, 'recall': 0.8493, 'qps': 192181}, ...]
    """
    results = []
    
    # Pattern to match: param=X: recall=Y, QPS=Z, times=[...]
    pattern = r'param=(\d+):\s*recall=([\d.]+),\s*QPS=[\d,]+,\s*times=\[([\d.,ms\s]+)\]'
    
    for match in re.finditer(pattern, text):
        param = int(match.group(1))
        recall = float(match.group(2))
        
        # Parse times: "6.68ms, 4.31ms, 6.62ms, 4.14ms, 4.28ms"
        times_str = match.group(3)
        times_ms = [float(t.replace('ms', '').strip()) for t in times_str.split(',')]
        
        # Remove min and max outliers, average the rest
        if len(times_ms) >= 3:
            times_ms_sorted = sorted(times_ms)
            # Remove smallest and largest
            middle_times = times_ms_sorted[1:-1]
            avg_time_ms = sum(middle_times) / len(middle_times)
        else:
            avg_time_ms = sum(times_ms) / len(times_ms)
        
        # Compute QPS: n_queries / avg_time_seconds
        avg_time_sec = avg_time_ms / 1000.0
        qps = int(n_queries / avg_time_sec)
        
        results.append({'param': param, 'recall': recall, 'qps': qps})
    
    return results


# ============================================================================
# PASTE YOUR RESULTS HERE (as raw strings)
# ============================================================================

# Original data results
original_text = """
[Sweeping search params]
  param=32: recall=0.8434, QPS=248,150, times=[3.79ms, 4.10ms, 4.08ms, 4.09ms, 4.09ms]
  param=64: recall=0.8768, QPS=133,516, times=[8.92ms, 5.63ms, 10.33ms, 7.00ms, 5.57ms]
  param=128: recall=0.8962, QPS=110,003, times=[9.00ms, 9.02ms, 9.02ms, 8.99ms, 9.42ms]
  param=256: recall=0.9273, QPS=64,280, times=[18.63ms, 15.33ms, 14.50ms, 14.63ms, 14.70ms]
"""

# Mock data - Previous approach (diagonal covariance / pca_scale=1.0)
mock_diagonal_text = """
[Sweeping search params]
  param=32: recall=0.3090, QPS=184,449, times=[4.46ms, 4.28ms, 4.61ms, 6.65ms, 7.10ms]
  param=64: recall=0.4429, QPS=141,704, times=[6.21ms, 6.21ms, 6.20ms, 6.36ms, 10.31ms]
  param=128: recall=0.5572, QPS=97,938, times=[10.07ms, 10.00ms, 10.00ms, 10.02ms, 10.96ms]
  param=256: recall=0.6582, QPS=44,858, times=[20.28ms, 22.65ms, 27.54ms, 20.61ms, 20.37ms]
"""

# Mock data - New PCA approach (pca_scale=0.8)
mock_pca_text = """
[Sweeping search params]
  param=32: recall=0.8493, QPS=192,181, times=[6.68ms, 4.31ms, 6.62ms, 4.14ms, 4.28ms]
  param=64: recall=0.9138, QPS=163,131, times=[6.12ms, 6.13ms, 6.13ms, 6.14ms, 6.13ms]
  param=128: recall=0.9441, QPS=103,179, times=[9.67ms, 9.68ms, 9.68ms, 9.69ms, 9.74ms]
  param=256: recall=0.9713, QPS=51,020, times=[19.28ms, 21.18ms, 19.04ms, 19.12ms, 19.38ms]
"""

# Parse the text into results
original_results = parse_results(original_text)
mock_diagonal_results = parse_results(mock_diagonal_text)
mock_pca_results = parse_results(mock_pca_text)

# ============================================================================
# PLOTTING
# ============================================================================

def plot_three_curves(original_results, mock_diagonal_results, mock_pca_results, 
                      original_name="clothes", cluster_stats="10K clusters", 
                      algorithm="CAGRA", pca_scale=-1, pca_ncomp=128, output_path=None):
    """Plot Pareto curves comparing original, mock diagonal, and mock PCA."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    color_original = '#e74c3c'    # Red
    color_diagonal = '#95a5a6'    # Gray
    color_pca = '#2ecc71'         # Green
    
    # Extract data
    def extract(results):
        return ([r['recall'] for r in results], 
                [r['qps'] for r in results],
                [r['param'] for r in results])
    
    orig_recall, orig_qps, orig_params = extract(original_results)
    diag_recall, diag_qps, diag_params = extract(mock_diagonal_results)
    pca_recall, pca_qps, pca_params = extract(mock_pca_results)
    
    # Build PCA label
    if pca_scale == -1:
        pca_label = f'Mock (PCA ncomp={pca_ncomp})'
    else:
        pca_label = f'Mock (PCA ncomp={pca_ncomp}, scale={pca_scale})'
    
    # Plot curves
    ax.plot(orig_recall, orig_qps, 's-', color=color_original, linewidth=2.5, 
            markersize=12, label='Original Data', zorder=3)
    ax.plot(diag_recall, diag_qps, '^--', color=color_diagonal, linewidth=2, 
            markersize=10, label='Mock (previous appraoch)', alpha=0.8, zorder=2)
    ax.plot(pca_recall, pca_qps, 'o-', color=color_pca, linewidth=2.5, 
            markersize=12, label=pca_label, zorder=3)
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('QPS (log scale)', fontsize=14)
    
    # Title with dataset, cluster stats, and itopk values
    itopk_values = orig_params  # Use params from original (should be same for all)
    title = f'{algorithm}: Recall vs QPS Pareto Curves\n'
    title += f'Original: {original_name} | Cluster Stats: {cluster_stats}\n'
    title += f'itopk={itopk_values}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Set axis limits with some padding
    all_recalls = orig_recall + diag_recall + pca_recall
    ax.set_xlim(min(all_recalls) - 0.03, max(all_recalls) + 0.02)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()
    
    return fig, ax


def print_comparison_table(original_results, mock_diagonal_results, mock_pca_results):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("RECALL COMPARISON")
    print("=" * 80)
    print(f"  {'itopk':>8}  {'original':>12}  {'mock_diag':>12}  {'mock_pca':>12}  {'pca_diff':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    
    for o, d, p in zip(original_results, mock_diagonal_results, mock_pca_results):
        diff = p['recall'] - o['recall']
        print(f"  {o['param']:>8}  {o['recall']:>12.4f}  {d['recall']:>12.4f}  {p['recall']:>12.4f}  {diff:>+10.4f}")
    
    print("\n" + "=" * 80)
    print("QPS COMPARISON")
    print("=" * 80)
    print(f"  {'itopk':>8}  {'original':>12}  {'mock_diag':>12}  {'mock_pca':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    
    for o, d, p in zip(original_results, mock_diagonal_results, mock_pca_results):
        print(f"  {o['param']:>8}  {o['qps']:>12,}  {d['qps']:>12,}  {p['qps']:>12,}")


if __name__ == "__main__":
    print_comparison_table(original_results, mock_diagonal_results, mock_pca_results)
    
    plot_three_curves(
        original_results, 
        mock_diagonal_results,
        mock_pca_results,
        original_name=ORIGINAL_DATA,
        cluster_stats=CLUSTER_STATS,
        algorithm=ALGORITHM,
        pca_scale=0.7,
        pca_ncomp=128,
        output_path="pareto_curves/pareto_three_curves.png"
    )
