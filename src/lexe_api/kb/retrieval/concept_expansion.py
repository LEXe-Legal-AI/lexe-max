"""Concept hierarchy expansion for legal code structure.

Exploits the natural hierarchy of Italian law codes
(Codice->Libro->Titolo->Capo->Sezione->Articolo) to improve retrieval
by finding structurally adjacent articles.

Inspired by H-CMR (cascading concept interventions) and
SAT-Graph RAG (Work/Expression temporal distinction).
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Hierarchy levels
LEVELS = {
    1: "Codice",
    2: "Libro",
    3: "Titolo",
    4: "Capo",
    5: "Sezione",
    6: "Articolo",
}

# Max siblings per expansion to control latency
MAX_SIBLINGS = 5
# Max results to expand (top N from initial search)
MAX_EXPAND_RESULTS = 3
# RRF boost for structural adjacency
ADJACENCY_BOOST = 0.05


async def get_concept_path(
    conn,
    normativa_id: UUID,
) -> list[str] | None:
    """Get concept path for an article.

    Returns path like ["CC", "Libro IV", "Titolo IX", "Capo I"].
    """
    row = await conn.fetchrow(
        "SELECT concept_path FROM kb.normativa WHERE id = $1",
        normativa_id,
    )
    if row and row["concept_path"]:
        return list(row["concept_path"])
    return None


async def get_parent_concepts(
    conn,
    normativa_id: UUID,
) -> list[dict[str, Any]]:
    """Get ascending parent chain for an article.

    Returns list of {level, label, structure_id} from article to root.
    """
    rows = await conn.fetch(
        """
        WITH RECURSIVE parents AS (
            SELECT s.id, s.level, s.label, s.parent_id
            FROM kb.normativa_structure s
            JOIN kb.normativa_structure_link l ON l.structure_id = s.id
            WHERE l.normativa_id = $1

            UNION ALL

            SELECT s.id, s.level, s.label, s.parent_id
            FROM kb.normativa_structure s
            JOIN parents p ON p.parent_id = s.id
        )
        SELECT id, level, label FROM parents ORDER BY level ASC
        """,
        normativa_id,
    )
    return [{"level": r["level"], "label": r["label"], "structure_id": r["id"]} for r in rows]


async def expand_concept_path(
    conn,
    concept_path: list[str],
    work_id: UUID | None = None,
    exclude_ids: set[UUID] | None = None,
) -> list[UUID]:
    """Find sibling articles in the same structural section.

    Given a concept path, finds other articles in the same
    parent section (one level up).

    Returns list of normativa IDs (max MAX_SIBLINGS).
    """
    if not concept_path or len(concept_path) < 2:
        return []

    exclude = exclude_ids or set()

    # Find the parent structure node
    parent_label = concept_path[-2] if len(concept_path) >= 2 else concept_path[0]
    parent_level = len(concept_path) - 1

    query = """
        SELECT DISTINCT l.normativa_id
        FROM kb.normativa_structure_link l
        JOIN kb.normativa_structure s ON s.id = l.structure_id
        JOIN kb.normativa_structure parent ON parent.id = s.parent_id
        WHERE parent.label = $1
          AND parent.level = $2
    """
    params: list[Any] = [parent_label, parent_level]

    if work_id:
        query += " AND s.work_id = $3"
        params.append(work_id)

    query += f" LIMIT {MAX_SIBLINGS + len(exclude)}"

    rows = await conn.fetch(query, *params)

    result = []
    for row in rows:
        nid = row["normativa_id"]
        if nid not in exclude and len(result) < MAX_SIBLINGS:
            result.append(nid)

    return result


async def hierarchical_search(
    conn,
    initial_results: list[dict[str, Any]],
    expand_concepts: bool = True,
) -> list[dict[str, Any]]:
    """Enhance search results with hierarchical expansion.

    1. Take top MAX_EXPAND_RESULTS from initial results
    2. For each, get concept_path and find siblings
    3. Re-rank with ADJACENCY_BOOST for structural proximity

    Args:
        conn: Database connection
        initial_results: List of search results with 'id' and 'score'
        expand_concepts: Whether to perform expansion

    Returns:
        Enhanced results list with concept_path added
    """
    if not expand_concepts or not initial_results:
        return initial_results

    # Add concept_path to initial results
    for result in initial_results:
        if "id" in result:
            path = await get_concept_path(conn, result["id"])
            result["concept_path"] = path

    # Expand top results
    existing_ids = {r["id"] for r in initial_results if "id" in r}
    expanded: list[dict[str, Any]] = []

    for result in initial_results[:MAX_EXPAND_RESULTS]:
        if not result.get("concept_path"):
            continue

        sibling_ids = await expand_concept_path(
            conn,
            result["concept_path"],
            exclude_ids=existing_ids,
        )

        for sid in sibling_ids:
            existing_ids.add(sid)
            # Fetch basic info for siblings
            row = await conn.fetchrow(
                """
                SELECT id, act_type, article, rubrica, concept_path
                FROM kb.normativa WHERE id = $1
                """,
                sid,
            )
            if row:
                expanded.append({
                    "id": row["id"],
                    "act_type": row["act_type"],
                    "article": row["article"],
                    "title": row.get("rubrica", ""),
                    "concept_path": list(row["concept_path"]) if row["concept_path"] else None,
                    "score": (result.get("score", 0.0) or 0.0) * 0.8 + ADJACENCY_BOOST,
                    "source": "concept_expansion",
                })

    # Merge and sort by score
    all_results = initial_results + expanded
    all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    return all_results
