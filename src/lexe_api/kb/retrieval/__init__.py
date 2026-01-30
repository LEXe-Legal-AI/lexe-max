"""
LEXE Knowledge Base - Retrieval Engine

Moduli per ricerca ibrida su KB massimari:
- dense: Ricerca vettoriale (pgvector HNSW)
- sparse: BM25 (pg_search) + trgm typo-catch
- hybrid: RRF fusion 3-way
- reranker: bge-reranker-v2-m3
- graph: Graph expansion pesato (Apache AGE)
- temporal: Temporal ranking con data_decisione
"""

from .dense import (
    DenseSearchConfig,
    DenseSearchResult,
    dense_search,
    dense_search_multi_model,
    estimate_mrr,
    estimate_recall,
)
from .graph import (
    GraphEdge,
    GraphExpansionResult,
    GraphNode,
    build_same_principle_edges,
    get_related_massime,
    graph_expand_weighted,
)
from .hybrid import (
    HybridSearchResult,
    calculate_precision_at_k,
    hybrid_search,
    hybrid_search_multi_model,
    reciprocal_rank_fusion,
)
from .reranker import (
    BGEReranker,
    RerankedResult,
    calculate_rerank_lift,
    rerank_results,
)
from .sparse import (
    SparseSearchConfig,
    SparseSearchResult,
    bm25_search,
    keyword_boost_search,
    trgm_search,
)
from .temporal import (
    TemporalBoostConfig,
    apply_temporal_boost,
    apply_temporal_to_results,
    calculate_recency_score,
    extract_temporal_from_query,
    fetch_massima_dates,
)

__all__ = [
    # dense
    "DenseSearchConfig",
    "DenseSearchResult",
    "dense_search",
    "dense_search_multi_model",
    "estimate_mrr",
    "estimate_recall",
    # sparse
    "SparseSearchConfig",
    "SparseSearchResult",
    "bm25_search",
    "trgm_search",
    "keyword_boost_search",
    # hybrid
    "HybridSearchResult",
    "hybrid_search",
    "hybrid_search_multi_model",
    "reciprocal_rank_fusion",
    "calculate_precision_at_k",
    # reranker
    "BGEReranker",
    "RerankedResult",
    "rerank_results",
    "calculate_rerank_lift",
    # graph
    "GraphNode",
    "GraphEdge",
    "GraphExpansionResult",
    "graph_expand_weighted",
    "build_same_principle_edges",
    "get_related_massime",
    # temporal
    "TemporalBoostConfig",
    "calculate_recency_score",
    "apply_temporal_boost",
    "apply_temporal_to_results",
    "fetch_massima_dates",
    "extract_temporal_from_query",
]
