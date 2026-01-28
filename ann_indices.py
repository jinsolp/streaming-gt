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
        self.index_mapping = None  # Maps internal CAGRA indices to global indices
    
    def build(self, dataset: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the CAGRA index from dataset.
        
        Args:
            dataset: Vectors to index
            indices: Optional explicit global indices for each vector. If provided,
                     search results will be translated back to these indices.
        """
        self.index = cagra.build(self.build_params, dataset)
        
        # Store index mapping if provided
        if indices is not None:
            self.index_mapping = indices.astype(np.int64).copy()
        else:
            self.index_mapping = None
    
    def extend(self, dataset: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Extend the CAGRA index with additional data.
        
        Args:
            dataset: Vectors to add
            indices: Optional explicit global indices for each vector.
        """
        if self.index is None:
            raise RuntimeError("Index must be built before extending. Call build() first.")
        
        self.index = cagra.extend(self.extend_params, self.index, dataset)
        
        # Update index mapping
        if indices is not None:
            if self.index_mapping is None:
                raise RuntimeError("Cannot extend with indices when build was called without indices.")
            self.index_mapping = np.concatenate([self.index_mapping, indices.astype(np.int64)])
        elif self.index_mapping is not None:
            raise RuntimeError("Cannot extend without indices when build was called with indices.")
        
    
    def build_or_extend(self, dataset: np.ndarray, indices: Optional[np.ndarray] = None):
        """Build the index if it doesn't exist, otherwise extend it."""
        if self.index is None:
            self.build(dataset, indices=indices)
        else:
            self.extend(dataset, indices=indices)
    
    def search(self, queries: np.ndarray, k: int, itopk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for k nearest neighbors.
        
        Args:
            queries: Query vectors
            k: Number of nearest neighbors to return
            itopk: Internal top-k size for search (larger = better recall, slower)
                   If None, uses the value from search_params (default usually 64)
        """
        queries_device = device_ndarray(queries.astype(np.float32))
        out_idx = np.zeros((queries.shape[0], k), dtype=np.uint32)
        out_dist = np.zeros((queries.shape[0], k), dtype=np.float32)
        out_idx_device = device_ndarray(out_idx)
        out_dist_device = device_ndarray(out_dist)
        
        # Use custom itopk if provided, otherwise use default search_params
        if itopk is not None:
            search_params = cagra.SearchParams(itopk_size=itopk)
        else:
            search_params = self.search_params
        
        cagra.search(
            search_params,
            self.index,
            queries_device,
            k,
            neighbors=out_idx_device,
            distances=out_dist_device,
        )
        
        result_indices = out_idx_device.copy_to_host()
        result_distances = out_dist_device.copy_to_host()
        
        # Translate internal indices to global indices if mapping exists
        if self.index_mapping is not None:
            result_indices = self.index_mapping[result_indices]
        
        return result_indices, result_distances


class IvfPqIndex:
    """Wrapper for IVF-PQ index with build, extend, and search functionality."""
    
    def __init__(
        self,
        build_params: Optional[ivf_pq.IndexParams] = None,
        search_params: Optional[ivf_pq.SearchParams] = None,
        store_vectors: bool = False,
    ):
        """
        Initialize IvfPqIndex with optional pre-configured params.
        
        Args:
            build_params: ivf_pq.IndexParams for building. If None, uses defaults.
            search_params: ivf_pq.SearchParams for searching. If None, uses defaults.
            store_vectors: If True, stores original vectors for refinement. Required for search_with_refine().
        """
        self.build_params = build_params if build_params is not None else ivf_pq.IndexParams()
        self.search_params = search_params if search_params is not None else ivf_pq.SearchParams()
        self.index = None
        self.n_indexed = 0  # Track total indexed for extend
        self.store_vectors = store_vectors
        self._vectors = {}  # Maps index -> vector (only used if store_vectors=True)
    
    def build(self, dataset: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the IVF-PQ index from dataset.
        
        Args:
            dataset: Vectors to index
            indices: Optional explicit indices for each vector. If None, uses 0 to n-1.
        """
        dataset_device = device_ndarray(dataset.astype(np.float32))
        
        if indices is not None:
            # Train quantizers only (don't add data), then extend with correct indices
            train_params = ivf_pq.IndexParams(
                n_lists=self.build_params.n_lists,
                pq_dim=self.build_params.pq_dim,
                pq_bits=self.build_params.pq_bits,
                add_data_on_build=False,  # Train only, don't add data
            )
            self.index = ivf_pq.build(train_params, dataset_device)
            
            # Now extend with correct indices
            indices = indices.astype(np.int64)
            indices_device = device_ndarray(indices)
            self.index = ivf_pq.extend(self.index, dataset_device, indices_device)
            
            # Store vectors for refinement if enabled
            if self.store_vectors:
                for i, idx in enumerate(indices):
                    self._vectors[int(idx)] = dataset[i].astype(np.float32)
        else:
            # Normal build with default indices (0 to n-1)
            self.index = ivf_pq.build(self.build_params, dataset_device)
            
            # Store vectors for refinement if enabled
            if self.store_vectors:
                for i in range(len(dataset)):
                    self._vectors[i] = dataset[i].astype(np.float32)
        
        self.n_indexed = len(dataset)
    
    def extend(self, dataset: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Extend the IVF-PQ index with additional data.
        
        Args:
            dataset: Vectors to add
            indices: Optional explicit indices for each vector. If None, uses sequential indices.
        """
        if self.index is None:
            raise RuntimeError("Index must be built before extending. Call build() first.")
        
        # Use explicit indices if provided, otherwise sequential
        if indices is None:
            indices = np.arange(self.n_indexed, self.n_indexed + len(dataset), dtype=np.int64)
        else:
            indices = indices.astype(np.int64)
        
        dataset_device = device_ndarray(dataset.astype(np.float32))
        indices_device = device_ndarray(indices)
        self.index = ivf_pq.extend(self.index, dataset_device, indices_device)
        
        # Store vectors for refinement if enabled
        if self.store_vectors:
            for i, idx in enumerate(indices):
                self._vectors[int(idx)] = dataset[i].astype(np.float32)
        
        self.n_indexed += len(dataset)
    
    def build_or_extend(self, dataset: np.ndarray):
        """Build the index if it doesn't exist, otherwise extend it."""
        if self.index is None:
            self.build(dataset)
        else:
            self.extend(dataset)
    
    def search(self, queries: np.ndarray, k: int, nprobes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for k nearest neighbors."""
        queries_device = device_ndarray(queries.astype(np.float32))
        out_idx = np.zeros((queries.shape[0], k), dtype=np.int64)
        out_dist = np.zeros((queries.shape[0], k), dtype=np.float32)
        out_idx_device = device_ndarray(out_idx)
        out_dist_device = device_ndarray(out_dist)
        
        if nprobes is not None:
            search_params = ivf_pq.SearchParams(n_probes=nprobes)
        else:
            search_params = self.search_params
        
        ivf_pq.search(
            search_params,
            self.index,
            queries_device,
            k,
            neighbors=out_idx_device,
            distances=out_dist_device,
        )
        
        return out_idx_device.copy_to_host(), out_dist_device.copy_to_host()
    
    def search_with_refine(
        self, 
        queries: np.ndarray, 
        k: int, 
        refine_k: Optional[int] = None,
        nprobes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with refinement: retrieve more candidates, compute exact distances, reorder.
        
        Args:
            queries: Query vectors
            k: Number of final nearest neighbors to return
            refine_k: Number of candidates to retrieve before refinement (default: 2*k)
            nprobes: Number of probes for IVF search
            
        Returns:
            Tuple of (indices, distances) with shape (n_queries, k)
        """
        if not self.store_vectors:
            raise RuntimeError("search_with_refine requires store_vectors=True in __init__")
        
        if refine_k is None:
            refine_k = 2 * k
        
        # Get more candidates than needed
        candidate_indices, _ = self.search(queries, refine_k, nprobes=nprobes)
        
        n_queries = queries.shape[0]
        final_indices = np.zeros((n_queries, k), dtype=np.int64)
        final_distances = np.zeros((n_queries, k), dtype=np.float32)
        
        # Refine each query
        for q_idx in range(n_queries):
            query = queries[q_idx]
            cand_idx = candidate_indices[q_idx]
            
            # Gather candidate vectors
            cand_vectors = np.array([self._vectors[int(idx)] for idx in cand_idx])
            
            # Compute exact L2 distances
            diff = cand_vectors - query
            exact_dists = np.sum(diff * diff, axis=1).astype(np.float32)
            
            # Sort by exact distance and take top k
            sorted_order = np.argsort(exact_dists)[:k]
            final_indices[q_idx] = cand_idx[sorted_order]
            final_distances[q_idx] = exact_dists[sorted_order]
        
        return final_indices, final_distances