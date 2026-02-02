"""Legal tools API endpoints."""

from fastapi import APIRouter, HTTPException

from lexe_api.config import settings
from lexe_api.models.schemas import (
    EurLexRequest,
    EurLexResponse,
    InfoLexRequest,
    InfoLexResponse,
    KBMassimaResult,
    KBSearchRequest,
    KBSearchResponse,
    NormattivaRequest,
    NormattivaResponse,
    VigenzaResponse,
)
from lexe_api.tools.base import CircuitOpenError
from lexe_api.tools.eurlex import eurlex_tool
from lexe_api.tools.health_monitor import health_monitor
from lexe_api.tools.infolex import infolex_tool
from lexe_api.tools.normattiva import normattiva_tool

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


# =============================================================================
# Normattiva Endpoints
# =============================================================================


@router.post("/normattiva/search", response_model=NormattivaResponse)
async def normattiva_search(request: NormattivaRequest) -> NormattivaResponse:
    """Search Italian legislation on Normattiva.it.

    Examples:
    - {"act_type": "legge", "date": "1990-08-07", "act_number": "241", "article": "1"}
    - {"act_type": "codice civile", "article": "1321"}
    - {"act_type": "decreto legislativo", "date": "2003", "act_number": "196"}
    """
    if not settings.ff_normattiva_enabled:
        raise HTTPException(
            status_code=503,
            detail="Normattiva tool is disabled",
        )

    try:
        result = await normattiva_tool.search(
            act_type=request.act_type,
            date=request.date,
            act_number=request.act_number,
            article=request.article,
            version=request.version.value,
        )
        return NormattivaResponse(**result)

    except CircuitOpenError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Tool temporarily unavailable: {e}",
        ) from e
    except Exception as e:
        await health_monitor.report_failure("normattiva", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/normattiva/vigenza", response_model=VigenzaResponse)
async def normattiva_vigenza(request: NormattivaRequest) -> VigenzaResponse:
    """Quick check if an article is still in force.

    Faster than full search - returns only vigenza status.
    """
    if not settings.ff_normattiva_enabled:
        raise HTTPException(
            status_code=503,
            detail="Normattiva tool is disabled",
        )

    try:
        result = await normattiva_tool.verify_vigenza(
            act_type=request.act_type,
            date=request.date,
            act_number=request.act_number,
            article=request.article,
        )
        return VigenzaResponse(**result)

    except CircuitOpenError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Tool temporarily unavailable: {e}",
        ) from e
    except Exception as e:
        await health_monitor.report_failure("normattiva", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# EUR-Lex Endpoints
# =============================================================================


@router.post("/eurlex/search", response_model=EurLexResponse)
async def eurlex_search(request: EurLexRequest) -> EurLexResponse:
    """Search European legislation on EUR-Lex.

    Examples:
    - {"act_type": "regolamento", "year": 2016, "number": 679} - GDPR
    - {"act_type": "direttiva", "year": 2019, "number": 790} - Copyright
    - {"act_type": "regolamento", "year": 2022, "number": 2065} - DSA
    """
    if not settings.ff_eurlex_enabled:
        raise HTTPException(
            status_code=503,
            detail="EUR-Lex tool is disabled",
        )

    try:
        result = await eurlex_tool.search(
            act_type=request.act_type.value,
            year=request.year,
            number=request.number,
            article=request.article,
            language=request.language,
        )
        return EurLexResponse(**result)

    except CircuitOpenError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Tool temporarily unavailable: {e}",
        ) from e
    except Exception as e:
        await health_monitor.report_failure("eurlex", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# InfoLex (Brocardi) Endpoints
# =============================================================================


@router.post("/infolex/search", response_model=InfoLexResponse)
async def infolex_search(request: InfoLexRequest) -> InfoLexResponse:
    """Search case law and commentary on Brocardi.it.

    Examples:
    - {"act_type": "codice civile", "article": "1321"} - Nozione di contratto
    - {"act_type": "codice penale", "article": "575"} - Omicidio
    - {"act_type": "codice civile", "article": "2043"} - Responsabilità extracontrattuale
    """
    if not settings.ff_infolex_enabled:
        raise HTTPException(
            status_code=503,
            detail="InfoLex tool is disabled",
        )

    try:
        result = await infolex_tool.search(
            act_type=request.act_type,
            article=request.article,
            include_massime=request.include_massime,
            include_relazioni=request.include_relazioni,
            include_footnotes=request.include_footnotes,
        )
        return InfoLexResponse(**result)

    except CircuitOpenError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Tool temporarily unavailable: {e}",
        ) from e
    except Exception as e:
        await health_monitor.report_failure("infolex", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Combined Search (convenience endpoint)
# =============================================================================


# =============================================================================
# KB Massime Search Endpoint
# =============================================================================


@router.post("/kb/search", response_model=KBSearchResponse)
async def kb_search(request: KBSearchRequest) -> KBSearchResponse:
    """Search the KB massime (case law summaries) using hybrid search.

    Uses dense (embedding) + sparse (BM25) search with RRF fusion.

    Examples:
    - {"query": "danno ingiusto art. 2043 c.c."}
    - {"query": "responsabilità contrattuale inadempimento", "top_k": 5}
    - {"query": "divorzio assegno", "filters": {"materia": "famiglia"}}
    """
    try:
        from lexe_api.kb.retrieval.hybrid import HybridSearchConfig, hybrid_search
        from lexe_api.kb.config import EmbeddingChannel, EmbeddingModel
        from lexe_api.database import get_kb_pool
        import httpx

        # Get embedding for the query using LiteLLM
        query_embedding = await _get_query_embedding(request.query)
        if not query_embedding:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate query embedding",
            )

        # Configure hybrid search
        config = HybridSearchConfig(
            dense_limit=request.top_k * 5,  # Get more candidates for fusion
            bm25_limit=request.top_k * 5,
            trgm_limit=request.top_k * 3,
            rrf_k=60,
            final_limit=request.top_k,
            min_similarity=request.min_score,
            model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
            channel=EmbeddingChannel.TESTO,
        )

        # Get KB database pool
        pool = await get_kb_pool()

        # Execute hybrid search
        search_results = await hybrid_search(
            query=request.query,
            query_embedding=query_embedding,
            config=config,
            db_pool=pool,
            filters=request.filters,
        )

        # Fetch full massima data for results
        results = await _enrich_massime_results(search_results, pool)

        return KBSearchResponse(
            success=True,
            query=request.query,
            total_results=len(results),
            results=results,
            search_mode="hybrid",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _get_query_embedding(query: str) -> list[float] | None:
    """Get embedding for a query using LiteLLM."""
    import httpx

    litellm_url = settings.litellm_api_base or "http://lexe-litellm:4000"
    litellm_key = settings.litellm_api_key or ""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{litellm_url}/embeddings",
                headers={"Authorization": f"Bearer {litellm_key}"},
                json={
                    "model": "text-embedding-3-small",
                    "input": query,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["embedding"]
            return None
    except Exception:
        return None


async def _enrich_massime_results(
    search_results: list,
    pool,
) -> list[KBMassimaResult]:
    """Fetch full massima data and build response objects."""
    if not search_results:
        return []

    import asyncpg

    massima_ids = [str(r.massima_id) for r in search_results]

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, testo, sezione, numero, anno, rv, materia
            FROM kb.massime
            WHERE id = ANY($1::uuid[])
            """,
            massima_ids,
        )

    # Create lookup
    massima_lookup = {str(row["id"]): row for row in rows}

    results = []
    for search_result in search_results:
        massima_id = str(search_result.massima_id)
        row = massima_lookup.get(massima_id)
        if row:
            results.append(
                KBMassimaResult(
                    massima_id=search_result.massima_id,
                    testo=row["testo"],
                    sezione=row["sezione"],
                    numero=row["numero"],
                    anno=row["anno"],
                    rv=row["rv"],
                    materia=row.get("materia"),
                    score=search_result.rrf_score,
                    dense_score=search_result.dense_score,
                    sparse_score=search_result.bm25_score,
                    rank=search_result.final_rank,
                )
            )

    return results


# =============================================================================
# Tool Status
# =============================================================================


@router.get("/status")
async def tools_status() -> dict:
    """Get status of all legal tools."""
    health = await health_monitor.get_all_health()
    return {
        "tools": {
            "normattiva": {
                "enabled": settings.ff_normattiva_enabled,
                "healthy": health["normattiva"].state == "healthy",
                "circuit": health["normattiva"].circuit_state.value,
            },
            "eurlex": {
                "enabled": settings.ff_eurlex_enabled,
                "healthy": health["eurlex"].state == "healthy",
                "circuit": health["eurlex"].circuit_state.value,
            },
            "infolex": {
                "enabled": settings.ff_infolex_enabled,
                "healthy": health["infolex"].state == "healthy",
                "circuit": health["infolex"].circuit_state.value,
            },
            "kb": {
                "enabled": True,
                "healthy": True,
                "circuit": "closed",
            },
        }
    }
