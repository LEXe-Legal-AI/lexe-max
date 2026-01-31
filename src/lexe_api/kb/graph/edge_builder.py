"""
LEXE Knowledge Base - Edge Builder

Dual-write edge builder for citation graph:
- SQL: kb.graph_edges table (for low-latency retrieval)
- AGE: lexe_jurisprudence graph (for exploration)

Features:
- Batch inserts for performance
- Run versioning with graph_runs
- Cache invalidation trigger
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

from lexe_api.kb.graph.citation_extractor import ResolvedCitation

logger = structlog.get_logger(__name__)


@dataclass
class GraphRunMetrics:
    """Metrics for a graph build run."""

    run_id: int
    run_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"

    # Counts
    total_massime_processed: int = 0
    total_mentions_extracted: int = 0
    total_resolved: int = 0
    total_unresolved: int = 0
    total_deduped: int = 0
    total_edges_created: int = 0

    # Rates
    resolution_rate: float = 0.0
    dedup_rate: float = 0.0

    # By resolver
    by_resolver: dict = None

    def __post_init__(self):
        if self.by_resolver is None:
            self.by_resolver = {}

    def to_dict(self) -> dict:
        return {
            "total_massime_processed": self.total_massime_processed,
            "total_mentions_extracted": self.total_mentions_extracted,
            "total_resolved": self.total_resolved,
            "total_unresolved": self.total_unresolved,
            "total_deduped": self.total_deduped,
            "total_edges_created": self.total_edges_created,
            "resolution_rate": self.resolution_rate,
            "dedup_rate": self.dedup_rate,
            "by_resolver": self.by_resolver,
        }


# ============================================================
# GRAPH RUN MANAGEMENT
# ============================================================

async def create_graph_run(
    conn,
    run_type: str,
    config: Optional[dict] = None,
) -> int:
    """
    Create a new graph run record.

    Returns run_id.
    """
    run_id = await conn.fetchval(
        """
        INSERT INTO kb.graph_runs (run_type, config, status)
        VALUES ($1, $2, 'running')
        RETURNING id
        """,
        run_type,
        json.dumps(config or {}),
    )
    logger.info("Graph run created", run_id=run_id, run_type=run_type)
    return run_id


async def complete_graph_run(
    conn,
    run_id: int,
    metrics: GraphRunMetrics,
    set_active: bool = True,
) -> None:
    """
    Mark graph run as completed and optionally set as active.
    """
    await conn.execute(
        """
        UPDATE kb.graph_runs
        SET completed_at = NOW(),
            status = 'completed',
            metrics = $2
        WHERE id = $1
        """,
        run_id,
        json.dumps(metrics.to_dict()),
    )

    if set_active:
        # Use helper function to set this run as active
        await conn.execute("SELECT kb.set_active_run($1, $2)", run_id, metrics.run_type)

    logger.info(
        "Graph run completed",
        run_id=run_id,
        edges=metrics.total_edges_created,
        resolution_rate=f"{metrics.resolution_rate:.1%}",
    )


async def fail_graph_run(conn, run_id: int, error_message: str) -> None:
    """Mark graph run as failed."""
    await conn.execute(
        """
        UPDATE kb.graph_runs
        SET completed_at = NOW(),
            status = 'failed',
            error_message = $2
        WHERE id = $1
        """,
        run_id,
        error_message,
    )
    logger.error("Graph run failed", run_id=run_id, error=error_message)


# ============================================================
# SQL EDGE INSERTION
# ============================================================

async def insert_edges_sql(
    conn,
    edges: list[ResolvedCitation],
    run_id: int,
) -> int:
    """
    Insert edges into kb.graph_edges table.

    Uses batch insert for performance.
    Returns number of edges inserted.
    """
    if not edges:
        return 0

    # Prepare values for batch insert
    values = [
        (
            edge.source_id,
            edge.target_id,
            edge.relation_type,
            edge.relation_subtype,
            edge.confidence,
            edge.weight,
            json.dumps(edge.evidence),
            edge.context_span[:500] if edge.context_span else None,  # Limit context length
            run_id,
        )
        for edge in edges
    ]

    # Batch insert with ON CONFLICT DO NOTHING (dedup constraint handles it)
    result = await conn.executemany(
        """
        INSERT INTO kb.graph_edges
            (source_id, target_id, edge_type, relation_subtype, confidence, weight, evidence, context_span, run_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
        ON CONFLICT (source_id, target_id, edge_type, relation_subtype, run_id) DO NOTHING
        """,
        values,
    )

    logger.debug("SQL edges inserted", count=len(edges), run_id=run_id)
    return len(edges)


# ============================================================
# AGE EDGE INSERTION
# ============================================================

async def insert_edges_age(
    conn,
    edges: list[ResolvedCitation],
    run_id: int,
) -> int:
    """
    Insert edges into Apache AGE graph.

    Creates :Massima nodes if they don't exist and connects with edges.
    Returns number of edges inserted.
    """
    if not edges:
        return 0

    # Prepare AGE environment
    await conn.execute("SET search_path TO ag_catalog, kb, public")
    await conn.execute("LOAD 'age'")

    inserted = 0
    for edge in edges:
        try:
            # Create or match nodes and create edge
            # Note: AGE requires string UUIDs
            cypher = f"""
            SELECT * FROM cypher('lexe_jurisprudence', $$
                MERGE (s:Massima {{id: '{edge.source_id}'}})
                MERGE (t:Massima {{id: '{edge.target_id}'}})
                MERGE (s)-[r:{edge.relation_type} {{
                    subtype: '{edge.relation_subtype or ""}',
                    confidence: {edge.confidence},
                    weight: {edge.weight},
                    run_id: {run_id}
                }}]->(t)
                RETURN r
            $$) as (r agtype)
            """
            await conn.execute(cypher)
            inserted += 1
        except Exception as e:
            # Log but don't fail on individual edge errors
            logger.warning(
                "AGE edge insert failed",
                source=str(edge.source_id)[:8],
                target=str(edge.target_id)[:8],
                error=str(e),
            )

    logger.debug("AGE edges inserted", count=inserted, run_id=run_id)
    return inserted


# ============================================================
# DUAL-WRITE
# ============================================================

async def insert_edges_dual(
    conn,
    edges: list[ResolvedCitation],
    run_id: int,
    skip_age: bool = False,
) -> tuple[int, int]:
    """
    Dual-write edges to SQL and AGE.

    Args:
        conn: Database connection
        edges: List of resolved citations
        run_id: Graph run ID
        skip_age: If True, only write to SQL (faster for testing)

    Returns:
        (sql_count, age_count)
    """
    sql_count = await insert_edges_sql(conn, edges, run_id)

    age_count = 0
    if not skip_age:
        age_count = await insert_edges_age(conn, edges, run_id)

    return sql_count, age_count


# ============================================================
# BATCH FETCHING
# ============================================================

async def fetch_massime_batch(
    conn,
    batch_size: int = 1000,
    offset: int = 0,
) -> list[dict]:
    """
    Fetch a batch of massime for processing.

    Returns list of {id, testo} dicts.
    """
    rows = await conn.fetch(
        """
        SELECT id, testo
        FROM kb.massime
        WHERE is_active = TRUE
        AND testo IS NOT NULL
        AND length(testo) > 50
        -- Guardrail: solo massime "plausibili" (v3.2.3)
        AND (
            rv IS NOT NULL
            OR (numero IS NOT NULL AND anno IS NOT NULL)
            OR testo_normalizzato ~* 'rv\\s*\\.?\\s*\\d{5,7}'
            -- Rescue: n./anno nel testo
            OR testo_normalizzato ~* 'n\\s*\\.?\\s*\\d+\\s*/\\s*(19|20)\\d{2}'
            OR testo_normalizzato ~* 'n\\s*\\.?\\s*\\d+\\s+del\\s+(19|20)\\d{2}'
            -- Rescue finale: numero plausibile (v3.2.3)
            -- Solo 1-5 cifre, non confondibile con pagina
            OR (
                numero IS NOT NULL
                AND numero ~ '^\\d{1,5}$'
                AND (pagina_inizio IS NULL OR numero::int <> pagina_inizio)
            )
        )
        ORDER BY id
        LIMIT $1 OFFSET $2
        """,
        batch_size,
        offset,
    )
    return [{"id": row["id"], "testo": row["testo"]} for row in rows]


async def count_active_massime(conn) -> int:
    """Count total active massime (with guardrail for plausible massime)."""
    return await conn.fetchval(
        """
        SELECT COUNT(*)
        FROM kb.massime
        WHERE is_active = TRUE
        AND testo IS NOT NULL
        AND length(testo) > 50
        -- Guardrail: solo massime "plausibili" (v3.2.3)
        AND (
            rv IS NOT NULL
            OR (numero IS NOT NULL AND anno IS NOT NULL)
            OR testo_normalizzato ~* 'rv\\s*\\.?\\s*\\d{5,7}'
            -- Rescue: n./anno nel testo
            OR testo_normalizzato ~* 'n\\s*\\.?\\s*\\d+\\s*/\\s*(19|20)\\d{2}'
            OR testo_normalizzato ~* 'n\\s*\\.?\\s*\\d+\\s+del\\s+(19|20)\\d{2}'
            -- Rescue finale: numero plausibile (v3.2.3)
            -- Solo 1-5 cifre, non confondibile con pagina
            OR (
                numero IS NOT NULL
                AND numero ~ '^\\d{1,5}$'
                AND (pagina_inizio IS NULL OR numero::int <> pagina_inizio)
            )
        )
        """
    )


# ============================================================
# CLEANUP
# ============================================================

async def delete_edges_for_run(conn, run_id: int) -> int:
    """Delete all edges for a specific run (for rollback)."""
    result = await conn.execute(
        "DELETE FROM kb.graph_edges WHERE run_id = $1",
        run_id,
    )
    # Parse DELETE count from result
    count = int(result.split()[-1]) if result else 0
    logger.info("Edges deleted for run", run_id=run_id, count=count)
    return count


async def get_edge_stats(conn, run_id: Optional[int] = None) -> dict:
    """Get edge statistics, optionally for a specific run."""
    where_clause = "WHERE run_id = $1" if run_id else ""
    params = [run_id] if run_id else []

    stats = await conn.fetchrow(
        f"""
        SELECT
            COUNT(*) as total_edges,
            COUNT(DISTINCT source_id) as unique_sources,
            COUNT(DISTINCT target_id) as unique_targets,
            COUNT(*) FILTER (WHERE relation_subtype IS NOT NULL) as edges_with_subtype,
            AVG(weight) as avg_weight,
            COUNT(*) FILTER (WHERE weight >= 0.6) as valid_weight_count
        FROM kb.graph_edges
        {where_clause}
        """,
        *params,
    )

    return dict(stats) if stats else {}
