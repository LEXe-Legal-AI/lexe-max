"""
LEXE Knowledge Base - Graph Expansion

Graph augmentation pesato con Apache AGE.
Espande risultati seguendo relazioni semantiche nel knowledge graph.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from ..config import GraphExpansionConfig

logger = structlog.get_logger(__name__)


@dataclass
class GraphNode:
    """Nodo nel knowledge graph."""

    massima_id: UUID
    sezione: str | None = None
    numero: str | None = None
    anno: int | None = None
    testo_breve: str | None = None


@dataclass
class GraphEdge:
    """Arco pesato nel graph."""

    source_id: UUID
    target_id: UUID
    edge_type: str  # CITES, APPLIES, SAME_PRINCIPLE, etc.
    weight: float
    metadata: dict[str, Any] | None = None


@dataclass
class GraphExpansionResult:
    """Risultato espansione graph."""

    massima_id: UUID
    expansion_score: float
    path_length: int
    edge_types: list[str]
    total_weight: float
    is_seed: bool = False


async def graph_expand_weighted(
    seed_ids: list[UUID],
    config: GraphExpansionConfig,
    db_pool: Any,
) -> list[GraphExpansionResult]:
    """
    Espandi risultati con edge weights.

    Non espansione cieca: seleziona solo top-N per peso.

    Args:
        seed_ids: ID massime seed (da hybrid search)
        config: Configurazione espansione
        db_pool: Connection pool

    Returns:
        Lista massime espanse con score
    """
    if not seed_ids:
        return []

    # Query Cypher per AGE con edge weights
    # Trova massime collegate entro N hop con peso >= soglia
    cypher_query = f"""
    SELECT * FROM cypher('lexe_jurisprudence', $$
        MATCH (m:Massima)-[r*1..{config.expansion_depth}]-(related:Massima)
        WHERE m.id IN $seed_ids
        AND ALL(rel IN r WHERE rel.weight >= {config.min_weight})
        WITH related,
             REDUCE(s = 0.0, rel IN r | s + rel.weight) as total_weight,
             length(r) as path_len,
             [rel IN r | type(rel)] as edge_types
        WHERE related.id NOT IN $seed_ids
        RETURN DISTINCT
            related.id as massima_id,
            total_weight,
            path_len,
            edge_types
        ORDER BY total_weight DESC
        LIMIT {config.top_n_per_seed * len(seed_ids)}
    $$, $1) as (
        massima_id agtype,
        total_weight agtype,
        path_len agtype,
        edge_types agtype
    )
    """

    try:
        async with db_pool.acquire() as conn:
            # Verifica se AGE e' disponibile
            try:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, kb, public")
            except Exception:
                logger.warning("Apache AGE not available, skipping graph expansion")
                return []

            # Esegui query
            rows = await conn.fetch(
                cypher_query,
                {"seed_ids": [str(sid) for sid in seed_ids]},
            )

        results = []

        # Aggiungi seed come risultati con is_seed=True
        for seed_id in seed_ids:
            results.append(
                GraphExpansionResult(
                    massima_id=seed_id,
                    expansion_score=1.0,
                    path_length=0,
                    edge_types=[],
                    total_weight=0.0,
                    is_seed=True,
                )
            )

        # Aggiungi nodi espansi
        for row in rows:
            # Parse AGE types
            massima_id = UUID(str(row["massima_id"]).strip('"'))
            total_weight = float(str(row["total_weight"]))
            path_len = int(str(row["path_len"]))
            edge_types = eval(str(row["edge_types"]))  # AGE ritorna come stringa

            # Calcola expansion score normalizzato
            # Decadimento per path length, boost per peso
            expansion_score = total_weight / (1 + path_len * 0.3)

            results.append(
                GraphExpansionResult(
                    massima_id=massima_id,
                    expansion_score=expansion_score,
                    path_length=path_len,
                    edge_types=edge_types,
                    total_weight=total_weight,
                    is_seed=False,
                )
            )

        logger.info(
            "Graph expansion completed",
            seeds=len(seed_ids),
            expanded=len(results) - len(seed_ids),
            total=len(results),
        )

        return results

    except Exception as e:
        logger.error("Graph expansion failed", error=str(e))
        # Fallback: ritorna solo seed
        return [
            GraphExpansionResult(
                massima_id=sid,
                expansion_score=1.0,
                path_length=0,
                edge_types=[],
                total_weight=0.0,
                is_seed=True,
            )
            for sid in seed_ids
        ]


async def build_same_principle_edges(
    db_pool: Any,
    embedding_similarity_threshold: float = 0.85,
    norm_overlap_min: int = 2,
) -> int:
    """
    Costruisce archi SAME_PRINCIPLE tra massime correlate.

    Criteri:
    - Overlap di norme citate >= 2
    - Embedding similarity >= threshold
    - Stessa sezione Cassazione (boost)

    Args:
        db_pool: Connection pool
        embedding_similarity_threshold: Soglia cosine similarity
        norm_overlap_min: Minimo overlap norme

    Returns:
        Numero archi creati
    """
    # Step 1: Trova coppie con overlap norme
    overlap_query = """
    SELECT
        c1.massima_id as m1,
        c2.massima_id as m2,
        COUNT(*) as norm_overlap,
        CASE WHEN m1.sezione = m2.sezione THEN true ELSE false END as same_section,
        ABS(m1.anno - m2.anno) as year_diff
    FROM kb.citations c1
    JOIN kb.citations c2 ON c1.articolo = c2.articolo
                        AND c1.codice = c2.codice
                        AND c1.massima_id < c2.massima_id
    JOIN kb.massime m1 ON c1.massima_id = m1.id
    JOIN kb.massime m2 ON c2.massima_id = m2.id
    WHERE c1.tipo = 'norma' AND c2.tipo = 'norma'
    GROUP BY c1.massima_id, c2.massima_id, m1.sezione, m2.sezione, m1.anno, m2.anno
    HAVING COUNT(*) >= $1
    """

    try:
        async with db_pool.acquire() as conn:
            pairs = await conn.fetch(overlap_query, norm_overlap_min)

            edges_created = 0

            for pair in pairs:
                m1, m2 = pair["m1"], pair["m2"]
                norm_overlap = pair["norm_overlap"]
                same_section = pair["same_section"]
                year_diff = pair["year_diff"]

                # Calcola embedding similarity
                emb_sim = await conn.fetchval(
                    """
                    SELECT 1 - (e1.embedding <=> e2.embedding)
                    FROM kb.embeddings e1
                    JOIN kb.embeddings e2 ON e1.model = e2.model
                                          AND e1.channel = e2.channel
                    WHERE e1.massima_id = $1
                      AND e2.massima_id = $2
                      AND e1.model = 'qwen3'
                      AND e1.channel = 'testo'
                    LIMIT 1
                    """,
                    m1, m2,
                )

                if emb_sim is None or emb_sim < embedding_similarity_threshold:
                    continue

                # Calcola peso finale
                # weight = 0.3*norm + 0.4*emb_sim + 0.2*temporal + 0.1*same_section
                norm_score = min(norm_overlap / 5, 1.0)  # Normalize to max 5
                temporal_score = 1.0 / (1 + year_diff * 0.1)
                section_score = 1.0 if same_section else 0.0

                weight = (
                    0.3 * norm_score +
                    0.4 * emb_sim +
                    0.2 * temporal_score +
                    0.1 * section_score
                )

                # Inserisci in edge_weights table
                await conn.execute(
                    """
                    INSERT INTO kb.edge_weights
                    (source_id, target_id, edge_type, weight,
                     norm_overlap, embedding_similarity, temporal_proximity, same_section)
                    VALUES ($1, $2, 'SAME_PRINCIPLE', $3, $4, $5, $6, $7)
                    ON CONFLICT (source_id, target_id, edge_type)
                    DO UPDATE SET weight = EXCLUDED.weight
                    """,
                    m1, m2, weight, norm_overlap, emb_sim, year_diff, same_section,
                )

                # Crea edge in AGE graph
                try:
                    await conn.execute(
                        """
                        SELECT * FROM cypher('lexe_jurisprudence', $$
                            MATCH (m1:Massima {id: $1}), (m2:Massima {id: $2})
                            MERGE (m1)-[r:SAME_PRINCIPLE]->(m2)
                            SET r.weight = $3
                        $$) as (r agtype)
                        """,
                        str(m1), str(m2), weight,
                    )
                except Exception:
                    pass  # AGE non disponibile

                edges_created += 1

            logger.info(
                "SAME_PRINCIPLE edges built",
                edges_created=edges_created,
            )

            return edges_created

    except Exception as e:
        logger.error("Failed to build edges", error=str(e))
        raise


async def get_related_massime(
    massima_id: UUID,
    db_pool: Any,
    edge_types: list[str] | None = None,
    min_weight: float = 0.3,
    limit: int = 10,
) -> list[tuple[UUID, str, float]]:
    """
    Ottieni massime correlate via graph.

    Args:
        massima_id: ID massima sorgente
        db_pool: Connection pool
        edge_types: Tipi edge da seguire (None = tutti)
        min_weight: Peso minimo
        limit: Max risultati

    Returns:
        Lista (massima_id, edge_type, weight)
    """
    edge_filter = ""
    if edge_types:
        types_str = ", ".join(f"'{t}'" for t in edge_types)
        edge_filter = f"AND ew.edge_type IN ({types_str})"

    query = f"""
    SELECT
        CASE WHEN ew.source_id = $1 THEN ew.target_id ELSE ew.source_id END as related_id,
        ew.edge_type,
        ew.weight
    FROM kb.edge_weights ew
    WHERE (ew.source_id = $1 OR ew.target_id = $1)
      AND ew.weight >= $2
      {edge_filter}
    ORDER BY ew.weight DESC
    LIMIT $3
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, massima_id, min_weight, limit)

    return [(row["related_id"], row["edge_type"], row["weight"]) for row in rows]
