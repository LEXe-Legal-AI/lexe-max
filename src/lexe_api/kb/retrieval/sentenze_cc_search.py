"""Sentenze Corte Costituzionale retrieval — sparse + optional dense.

Phase 1: BM25 (tsv_italian) + metadata filters (anno, tipo).
Phase 2: Dense (sentenze_cc_embedding HNSW) + RRF fusion when embeddings exist.

Pattern follows normativa_hybrid.py but simplified: sentenze_cc doesn't have
a chunk/work hierarchy — each row IS the full document.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

import asyncpg
import httpx
import structlog

logger = structlog.get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def get_sentenze_cc_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        dsn = os.environ.get(
            "LEXE_MAX_DATABASE_URL",
            "postgresql://lexe_kb:lexe_kb_secret@lexe-max:5432/lexe_kb",
        )
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    return _pool


class SentenzeSearchMode(str, Enum):
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class SentenzaCCResult:
    id: str
    tipo: str
    numero: int
    anno: int
    presidente: str | None
    relatore: str | None
    data_deposito: str | None
    dispositivo: str | None
    text_preview: str
    sparse_score: float
    dense_score: float | None = None
    rrf_score: float | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SPARSE_SQL = """
SELECT
    s.id::text,
    s.tipo,
    s.numero,
    s.anno,
    s.presidente,
    s.relatore,
    s.data_deposito::text,
    s.dispositivo,
    ts_headline('italian', LEFT(s.testo_integrale, 3000), plainto_tsquery('italian', $1),
        'MaxWords=60, MinWords=20, StartSel=**, StopSel=**') AS text_preview,
    ts_rank_cd(s.tsv_italian, plainto_tsquery('italian', $1)) AS sparse_score,
    s.source_url
FROM kb.sentenze_cc s
WHERE s.tsv_italian @@ plainto_tsquery('italian', $1)
  {anno_filter}
  {tipo_filter}
ORDER BY sparse_score DESC
LIMIT $2
"""

DENSE_SQL = """
SELECT
    s.id::text,
    s.tipo,
    s.numero,
    s.anno,
    s.presidente,
    s.relatore,
    s.data_deposito::text,
    s.dispositivo,
    LEFT(e.testo_chunk, 300) AS text_preview,
    1 - (e.embedding <=> $1::vector(1536)) AS dense_score,
    s.source_url
FROM kb.sentenze_cc_embedding e
JOIN kb.sentenze_cc s ON s.id = e.sentenza_id
{anno_filter}
{tipo_filter}
ORDER BY e.embedding <=> $1::vector(1536)
LIMIT $2
"""


async def search_sentenze_cc(
    query: str,
    pool: asyncpg.Pool,
    *,
    query_embedding: list[float] | None = None,
    top_k: int = 5,
    year_min: int | None = None,
    year_max: int | None = None,
    ruling_type: str | None = None,
    mode: SentenzeSearchMode = SentenzeSearchMode.SPARSE,
) -> tuple[list[SentenzaCCResult], float]:
    """Search sentenze CC with sparse (BM25) or hybrid (BM25 + dense RRF).

    Phase 1: sparse mode (no embedding needed).
    Phase 2: hybrid mode requires query_embedding.
    """
    start = time.perf_counter()

    # Build dynamic filters
    params: list[Any] = [query, top_k]
    anno_parts: list[str] = []
    tipo_filter = ""
    param_idx = 3

    if year_min is not None:
        anno_parts.append(f"s.anno >= ${param_idx}")
        params.append(year_min)
        param_idx += 1
    if year_max is not None:
        anno_parts.append(f"s.anno <= ${param_idx}")
        params.append(year_max)
        param_idx += 1
    if ruling_type and ruling_type != "any":
        tipo_filter = f"AND s.tipo = ${param_idx}"
        params.append(ruling_type)
        param_idx += 1

    anno_filter = "AND " + " AND ".join(anno_parts) if anno_parts else ""

    async with pool.acquire() as conn:
        if mode == SentenzeSearchMode.HYBRID and query_embedding:
            results = await _hybrid_rrf(
                conn, query, query_embedding, top_k,
                anno_filter, tipo_filter, params,
            )
        else:
            sql = SPARSE_SQL.format(anno_filter=anno_filter, tipo_filter=tipo_filter)
            rows = await conn.fetch(sql, *params)
            results = [
                SentenzaCCResult(
                    id=r[0], tipo=r[1], numero=r[2], anno=r[3],
                    presidente=r[4], relatore=r[5], data_deposito=r[6],
                    dispositivo=r[7], text_preview=r[8],
                    sparse_score=float(r[9]), source_url=r[10],
                )
                for r in rows
            ]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "sentenze_cc search",
        mode=mode.value, results=len(results), top_k=top_k,
        year_min=year_min, year_max=year_max, ruling_type=ruling_type,
        time_ms=round(elapsed_ms, 1),
    )
    return results, elapsed_ms


async def _hybrid_rrf(
    conn: asyncpg.Connection,
    query: str,
    query_embedding: list[float],
    top_k: int,
    anno_filter: str,
    tipo_filter: str,
    sparse_params: list[Any],
) -> list[SentenzaCCResult]:
    """RRF fusion of sparse + dense search."""
    k = 60  # RRF parameter

    # Sparse pass
    sparse_sql = SPARSE_SQL.format(anno_filter=anno_filter, tipo_filter=tipo_filter)
    sparse_rows = await conn.fetch(sparse_sql, *sparse_params)

    # Dense pass — needs embedding
    dense_filter = anno_filter.replace("s.anno", "s.anno")
    dense_tipo = tipo_filter.replace("s.tipo", "s.tipo")
    # For dense, $1 = embedding, $2 = limit, then filters
    dense_params: list[Any] = [query_embedding, top_k * 3]
    dense_idx = 3
    if "s.anno >=" in anno_filter:
        for p in sparse_params[2:]:
            dense_params.append(p)
    dense_sql_raw = DENSE_SQL.format(
        anno_filter="WHERE 1=1 " + anno_filter if anno_filter else "",
        tipo_filter=tipo_filter,
    )
    try:
        dense_rows = await conn.fetch(dense_sql_raw, *dense_params)
    except Exception as exc:
        logger.warning("dense search failed (embeddings missing?): %s", exc)
        dense_rows = []

    # RRF merge
    scores: dict[str, dict[str, Any]] = {}

    for rank, r in enumerate(sparse_rows, 1):
        sid = r[0]
        scores[sid] = {
            "data": SentenzaCCResult(
                id=r[0], tipo=r[1], numero=r[2], anno=r[3],
                presidente=r[4], relatore=r[5], data_deposito=r[6],
                dispositivo=r[7], text_preview=r[8],
                sparse_score=float(r[9]), source_url=r[10],
            ),
            "rrf": 1.0 / (k + rank),
        }

    for rank, r in enumerate(dense_rows, 1):
        sid = r[0]
        rrf_add = 1.0 / (k + rank)
        if sid in scores:
            scores[sid]["rrf"] += rrf_add
            scores[sid]["data"].dense_score = float(r[9])
        else:
            scores[sid] = {
                "data": SentenzaCCResult(
                    id=r[0], tipo=r[1], numero=r[2], anno=r[3],
                    presidente=r[4], relatore=r[5], data_deposito=r[6],
                    dispositivo=r[7], text_preview=r[8],
                    sparse_score=0.0, dense_score=float(r[9]), source_url=r[10],
                ),
                "rrf": rrf_add,
            }

    for v in scores.values():
        v["data"].rrf_score = v["rrf"]

    ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
    return [v["data"] for v in ranked]


async def get_sentenze_cc_stats(pool: asyncpg.Pool) -> dict[str, Any]:
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM kb.sentenze_cc")
        by_tipo = await conn.fetch(
            "SELECT tipo, COUNT(*) FROM kb.sentenze_cc GROUP BY tipo"
        )
        by_anno = await conn.fetch(
            "SELECT anno, COUNT(*) FROM kb.sentenze_cc GROUP BY anno ORDER BY anno DESC LIMIT 10"
        )
        edges = await conn.fetchval("SELECT COUNT(*) FROM kb.sentenze_cc_edges")
        embeddings = await conn.fetchval("SELECT COUNT(*) FROM kb.sentenze_cc_embedding")
    return {
        "total_sentenze": total,
        "by_tipo": {r[0]: r[1] for r in by_tipo},
        "recent_anni": {r[0]: r[1] for r in by_anno},
        "total_edges": edges,
        "total_embeddings": embeddings,
    }
