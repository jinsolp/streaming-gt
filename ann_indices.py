"""
ANN Index Wrappers

Contains:
- CagraIndex: CAGRA graph-based ANN index wrapper
- IvfPqIndex: IVF-PQ quantization-based ANN index wrapper  
"""

import numpy as np
from cuvs.neighbors import cagra, ivf_pq
from pylibraft.common import device_ndarray
from typing import Tuple, Optional


class CagraIndex:
    """Wrapper for CAGRA index with build, extend, and search functionality."""
    
    def __init__(
        self,
        build_params: Optional[cagra.IndexParams] = None,
        extend_params: Optional[cagra.ExtendParams] = None,
        search_params: Optional[cagra.SearchParams] = None,
    ):
        """
        Initialize CagraIndex with optional pre-configured params.
        
        Args:
            build_params: cagra.IndexParams for building. If None, uses defaults.
            extend_params: cagra.ExtendParams for extending. If None, uses defaults.
            search_params: cagra.SearchParams for searching. If None, uses defaults.
        """
        self.build_params = build_params if build_params is not None else cagra.IndexParams()
        self.extend_params = extend_params if extend_params is not None else cagra.ExtendParams()
        self.search_params = search_params if search_params is not None else cagra.SearchParams()
        self.index = None
    
    def build(self, dataset: np.ndarray):
        """Build the CAGRA index from dataset."""
        # dataset_device = device_ndarray(dataset.astype(np.float32))
        self.index = cagra.build(self.build_params, dataset)
    
    def extend(self, dataset: np.ndarray):
        """Extend the CAGRA index with additional data."""
        if self.index is None:
            raise RuntimeError("Index must be built before extending. Call build() first.")
        # dataset_device = device_ndarray(dataset.astype(np.float32))
        self.index = cagra.extend(self.extend_params, self.index, dataset)
    
    def build_or_extend(self, dataset: np.ndarray):
        """Build the index if it doesn't exist, otherwise extend it."""
        if self.index is None:
            self.build(dataset)
        else:
            self.extend(dataset)
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for k nearest neighbors."""
        queries_device = device_ndarray(queries.astype(np.float32))
        out_idx = np.zeros((queries.shape[0], k), dtype=np.uint32)
        out_dist = np.zeros((queries.shape[0], k), dtype=np.float32)
        out_idx_device = device_ndarray(out_idx)
        out_dist_device = device_ndarray(out_dist)
        
        cagra.search(
            self.search_params,
            self.index,
            queries_device,
            k,
            neighbors=out_idx_device,
            distances=out_dist_device,
        )

        print(f"out_idx_device: {out_idx_device.copy_to_host()}")
        
        return out_idx_device.copy_to_host(), out_dist_device.copy_to_host()


class IvfPqIndex:
    """Wrapper for IVF-PQ index with build, extend, and search functionality."""
    
    def __init__(
        self,
        build_params: Optional[ivf_pq.IndexParams] = None,
        search_params: Optional[ivf_pq.SearchParams] = None,
    ):
        """
        Initialize IvfPqIndex with optional pre-configured params.
        
        Args:
            build_params: ivf_pq.IndexParams for building. If None, uses defaults.
            search_params: ivf_pq.SearchParams for searching. If None, uses defaults.
        """
        self.build_params = build_params if build_params is not None else ivf_pq.IndexParams()
        self.search_params = search_params if search_params is not None else ivf_pq.SearchParams()
        self.index = None
        self.n_indexed = 0  # Track total indexed for extend
    
    def build(self, dataset: np.ndarray):
        """Build the IVF-PQ index from dataset."""
        dataset_device = device_ndarray(dataset.astype(np.float32))
        self.index = ivf_pq.build(self.build_params, dataset_device)
        self.n_indexed = len(dataset)
    
    def extend(self, dataset: np.ndarray):
        """Extend the IVF-PQ index with additional data."""
        if self.index is None:
            raise RuntimeError("Index must be built before extending. Call build() first.")
        
        # IVF-PQ extend requires explicit indices
        indices = np.arange(self.n_indexed, self.n_indexed + len(dataset), dtype=np.int64)
        dataset_device = device_ndarray(dataset.astype(np.float32))
        indices_device = device_ndarray(indices)
        self.index = ivf_pq.extend(self.index, dataset_device, indices_device)
        self.n_indexed += len(dataset)
    
    def build_or_extend(self, dataset: np.ndarray):
        """Build the index if it doesn't exist, otherwise extend it."""
        if self.index is None:
            self.build(dataset)
        else:
            self.extend(dataset)
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for k nearest neighbors."""
        queries_device = device_ndarray(queries.astype(np.float32))
        out_idx = np.zeros((queries.shape[0], k), dtype=np.int64)
        out_dist = np.zeros((queries.shape[0], k), dtype=np.float32)
        out_idx_device = device_ndarray(out_idx)
        out_dist_device = device_ndarray(out_dist)
        
        ivf_pq.search(
            self.search_params,
            self.index,
            queries_device,
            k,
            neighbors=out_idx_device,
            distances=out_dist_device,
        )
        
        return out_idx_device.copy_to_host(), out_dist_device.copy_to_host()
