"""Legal tools API endpoints."""

from fastapi import APIRouter, HTTPException

from lexe_api.config import settings
from lexe_api.models.schemas import (
    EurLexRequest,
    EurLexResponse,
    InfoLexRequest,
    InfoLexResponse,
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
        }
    }
