"""
LEXE Knowledge Base - GraphRAG Reranker

Integrazione graph nel retrieval pipeline:
1. Hybrid search → top N candidates
2. Graph expand da top K seeds
3. Boost candidates graph-connected
4. Re-rank → final results

v1.0: SQL-based expansion (kb.graph_edges), no AGE dependency.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog

from .hybrid import HybridSearchResult

logger = structlog.get_logger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class GraphExpandedResult:
    """Risultato con graph expansion info."""

    massima_id: UUID
    original_score: float
    original_rank: int
    graph_boost: float
    final_score: float
    final_rank: int

    # Graph connection info
    is_graph_connected: bool = False
    connected_to_seeds: list[UUID] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    total_edge_weight: float = 0.0
    min_path_length: int = 0

    # Original component scores
    dense_score: float | None = None
    bm25_score: float | None = None


@dataclass
class GraphRAGConfig:
    """Configurazione GraphRAG reranking."""

    # Expansion settings
    seed_count: int = 10  # Top N results as seeds
    expansion_depth: int = 2  # Max hops
    min_edge_weight: float = 0.5  # Min weight for expansion
    max_expanded: int = 100  # Max expanded nodes

    # Boost settings
    graph_boost_factor: float = 0.15  # Additive boost for graph-connected
    depth_decay: float = 0.7  # Decay per hop (0.7^depth)
    edge_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "CITES": 1.0,
            "CONFIRMS": 1.2,
            "OVERRULES": 0.8,
            "DISTINGUISHES": 0.6,
        }
    )


# ============================================================
# GRAPH EXPANSION (SQL-based)
# ============================================================


async def graph_expand_sql(
    seed_ids: list[UUID],
    conn: Any,
    config: GraphRAGConfig,
) -> dict[UUID, dict]:
    """
    Expand from seed nodes using SQL (kb.graph_edges).

    Returns dict: massima_id -> {
        seeds: [connected seed IDs],
        edge_types: [edge types used],
        total_weight: sum of edge weights,
        min_depth: shortest path length
    }
    """
    if not seed_ids:
        return {}

    expanded: dict[UUID, dict] = {}

    # Depth 1: Direct connections from seeds
    depth1_query = """
    SELECT
        e.target_id as related_id,
        e.source_id as seed_id,
        e.edge_type,
        COALESCE(e.relation_subtype, e.edge_type) as relation,
        e.weight
    FROM kb.graph_edges e
    WHERE e.source_id = ANY($1::uuid[])
      AND e.weight >= $2
      AND e.run_id = (SELECT id FROM kb.graph_runs WHERE is_active ORDER BY id DESC LIMIT 1)

    UNION ALL

    SELECT
        e.source_id as related_id,
        e.target_id as seed_id,
        e.edge_type,
        COALESCE(e.relation_subtype, e.edge_type) as relation,
        e.weight
    FROM kb.graph_edges e
    WHERE e.target_id = ANY($1::uuid[])
      AND e.weight >= $2
      AND e.run_id = (SELECT id FROM kb.graph_runs WHERE is_active ORDER BY id DESC LIMIT 1)
    """

    seed_id_list = [str(sid) for sid in seed_ids]
    seed_set = set(seed_ids)

    rows = await conn.fetch(depth1_query, seed_id_list, config.min_edge_weight)

    for row in rows:
        related_id = row["related_id"]
        if related_id in seed_set:
            continue  # Skip seeds themselves

        if related_id not in expanded:
            expanded[related_id] = {
                "seeds": [],
                "edge_types": [],
                "total_weight": 0.0,
                "min_depth": 1,
            }

        expanded[related_id]["seeds"].append(row["seed_id"])
        expanded[related_id]["edge_types"].append(row["relation"])
        expanded[related_id]["total_weight"] += row["weight"]

    # Depth 2: If configured
    if config.expansion_depth >= 2 and expanded:
        depth1_ids = list(expanded.keys())[:50]  # Limit for performance

        depth2_query = """
        SELECT
            e.target_id as related_id,
            e.source_id as via_id,
            e.edge_type,
            COALESCE(e.relation_subtype, e.edge_type) as relation,
            e.weight
        FROM kb.graph_edges e
        WHERE e.source_id = ANY($1::uuid[])
          AND e.weight >= $2
          AND e.target_id NOT IN (SELECT unnest($3::uuid[]))
          AND e.run_id = (SELECT id FROM kb.graph_runs WHERE is_active ORDER BY id DESC LIMIT 1)

        UNION ALL

        SELECT
            e.source_id as related_id,
            e.target_id as via_id,
            e.edge_type,
            COALESCE(e.relation_subtype, e.edge_type) as relation,
            e.weight
        FROM kb.graph_edges e
        WHERE e.target_id = ANY($1::uuid[])
          AND e.weight >= $2
          AND e.source_id NOT IN (SELECT unnest($3::uuid[]))
          AND e.run_id = (SELECT id FROM kb.graph_runs WHERE is_active ORDER BY id DESC LIMIT 1)
        """

        depth1_str = [str(d) for d in depth1_ids]
        exclude_str = seed_id_list + depth1_str

        rows2 = await conn.fetch(depth2_query, depth1_str, config.min_edge_weight, exclude_str)

        for row in rows2:
            related_id = row["related_id"]
            if related_id in seed_set:
                continue

            if related_id not in expanded:
                expanded[related_id] = {
                    "seeds": [],
                    "edge_types": [],
                    "total_weight": 0.0,
                    "min_depth": 2,
                }

            # Find which seed this connects to via depth1
            via_id = row["via_id"]
            if via_id in expanded:
                for seed in expanded[via_id]["seeds"]:
                    if seed not in expanded[related_id]["seeds"]:
                        expanded[related_id]["seeds"].append(seed)

            expanded[related_id]["edge_types"].append(row["relation"])
            expanded[related_id]["total_weight"] += row["weight"] * config.depth_decay

    logger.debug(
        "Graph expansion completed",
        seeds=len(seed_ids),
        expanded=len(expanded),
    )

    return expanded


# ============================================================
# GRAPH-BOOSTED RERANKING
# ============================================================


def compute_graph_boost(
    expansion_info: dict,
    config: GraphRAGConfig,
) -> float:
    """
    Compute boost score for a graph-connected result.

    Factors:
    - Number of seed connections
    - Edge types (CONFIRMS > CITES > OVERRULES)
    - Total edge weight
    - Path depth
    """
    if not expansion_info:
        return 0.0

    n_seeds = len(expansion_info.get("seeds", []))
    edge_types = expansion_info.get("edge_types", [])
    total_weight = expansion_info.get("total_weight", 0.0)
    min_depth = expansion_info.get("min_depth", 1)

    # Seed connection bonus (diminishing returns)
    seed_bonus = min(n_seeds * 0.3, 1.0)

    # Edge type score
    edge_score = 0.0
    for et in edge_types:
        edge_score += config.edge_type_weights.get(et, 0.8)
    edge_score = min(edge_score / max(len(edge_types), 1), 1.5)

    # Weight contribution (normalized)
    weight_score = min(total_weight / 2.0, 1.0)

    # Depth decay
    depth_multiplier = config.depth_decay ** (min_depth - 1)

    # Final boost
    boost = (
        config.graph_boost_factor
        * (0.4 * seed_bonus + 0.3 * edge_score + 0.3 * weight_score)
        * depth_multiplier
    )

    return boost


async def rerank_with_graph(
    hybrid_results: list[HybridSearchResult],
    conn: Any,
    config: GraphRAGConfig | None = None,
) -> list[GraphExpandedResult]:
    """
    Rerank hybrid results with graph boost.

    Pipeline:
    1. Take top K seeds from hybrid results
    2. Graph expand from seeds
    3. Find which candidates are graph-connected
    4. Boost connected candidates
    5. Re-sort by final_score

    Args:
        hybrid_results: Results from hybrid_search()
        conn: Database connection
        config: GraphRAG configuration

    Returns:
        Reranked results with graph info
    """
    if config is None:
        config = GraphRAGConfig()

    if not hybrid_results:
        return []

    # Step 1: Extract seeds (top K)
    seeds = [r.massima_id for r in hybrid_results[: config.seed_count]]

    # Step 2: Graph expand
    expanded = await graph_expand_sql(seeds, conn, config)

    # Step 3: Compute boosted scores
    results = []
    seed_set = set(seeds)

    for r in hybrid_results:
        is_connected = r.massima_id in expanded
        is_seed = r.massima_id in seed_set

        expansion_info = expanded.get(r.massima_id, {})
        graph_boost = compute_graph_boost(expansion_info, config) if is_connected else 0.0

        # Seeds get a small inherent boost
        if is_seed:
            graph_boost = max(graph_boost, config.graph_boost_factor * 0.5)

        final_score = r.rrf_score + graph_boost

        results.append(
            GraphExpandedResult(
                massima_id=r.massima_id,
                original_score=r.rrf_score,
                original_rank=r.final_rank,
                graph_boost=graph_boost,
                final_score=final_score,
                final_rank=0,  # Will be set after sorting
                is_graph_connected=is_connected or is_seed,
                connected_to_seeds=expansion_info.get("seeds", [])
                if is_connected
                else ([r.massima_id] if is_seed else []),
                edge_types=expansion_info.get("edge_types", []),
                total_edge_weight=expansion_info.get("total_weight", 0.0),
                min_path_length=expansion_info.get("min_depth", 0) if is_connected else 0,
                dense_score=r.dense_score,
                bm25_score=r.bm25_score,
            )
        )

    # Step 4: Re-sort by final score
    results.sort(key=lambda x: x.final_score, reverse=True)

    # Step 5: Assign final ranks
    for i, r in enumerate(results, start=1):
        r.final_rank = i

    # Log stats
    n_boosted = sum(1 for r in results if r.graph_boost > 0)
    n_reordered = sum(1 for r in results if r.final_rank != r.original_rank)
    avg_boost = sum(r.graph_boost for r in results) / len(results) if results else 0

    logger.info(
        "GraphRAG reranking completed",
        total=len(results),
        boosted=n_boosted,
        reordered=n_reordered,
        avg_boost=f"{avg_boost:.4f}",
    )

    return results


# ============================================================
# FULL PIPELINE
# ============================================================


async def hybrid_search_with_graph(
    query: str,
    query_embedding: list[float],
    hybrid_config: Any,
    graph_config: GraphRAGConfig | None,
    conn: Any,
) -> list[GraphExpandedResult]:
    """
    Full hybrid + graph search pipeline.

    Args:
        query: Query text
        query_embedding: Query embedding vector
        hybrid_config: HybridSearchConfig
        graph_config: GraphRAGConfig (None = skip graph)
        conn: Database connection

    Returns:
        GraphRAG-reranked results
    """
    from .hybrid import hybrid_search

    # Step 1: Hybrid search
    hybrid_results = await hybrid_search(
        query,
        query_embedding,
        hybrid_config,
        conn,
    )

    if not hybrid_results:
        return []

    # Step 2: Graph rerank (if enabled)
    if graph_config is not None:
        return await rerank_with_graph(hybrid_results, conn, graph_config)

    # No graph: convert to GraphExpandedResult format
    return [
        GraphExpandedResult(
            massima_id=r.massima_id,
            original_score=r.rrf_score,
            original_rank=r.final_rank,
            graph_boost=0.0,
            final_score=r.rrf_score,
            final_rank=r.final_rank,
            is_graph_connected=False,
            dense_score=r.dense_score,
            bm25_score=r.bm25_score,
        )
        for r in hybrid_results
    ]


# ============================================================
# METRICS
# ============================================================


def calculate_graph_hit_rate(results: list[GraphExpandedResult], k: int = 10) -> float:
    """Calculate % of top-K results that are graph-connected."""
    if not results or k == 0:
        return 0.0
    top_k = results[:k]
    connected = sum(1 for r in top_k if r.is_graph_connected)
    return connected / len(top_k)


def calculate_rank_change(results: list[GraphExpandedResult]) -> dict:
    """Calculate rank change statistics."""
    if not results:
        return {"avg": 0, "improved": 0, "worsened": 0, "unchanged": 0}

    changes = [r.original_rank - r.final_rank for r in results]
    improved = sum(1 for c in changes if c > 0)
    worsened = sum(1 for c in changes if c < 0)
    unchanged = sum(1 for c in changes if c == 0)

    return {
        "avg": sum(changes) / len(changes),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }
