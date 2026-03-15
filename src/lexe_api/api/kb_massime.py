"""LEXE Knowledge Base - Massime URL Write-back API Router.

PATCH endpoint for persisting validated source URLs to kb.massime.
Called by lexe-core after SearXNG resolve + HTTP HEAD + LLM validation.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from lexe_api.database import get_kb_pool

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/kb/massime",
    tags=["kb-massime"],
)


class SourceUrlUpdate(BaseModel):
    source_url: str


class SourceUrlResponse(BaseModel):
    massima_id: str
    source_url: str
    updated: bool


@router.patch(
    "/{massima_id}/source-url",
    response_model=SourceUrlResponse,
    summary="Update massima source URL",
    description="Persist a validated external URL for a massima. "
    "Called after SearXNG resolution + HTTP HEAD + LLM content validation.",
)
async def update_source_url(
    massima_id: UUID,
    body: SourceUrlUpdate,
) -> SourceUrlResponse:
    """Write validated source_url to kb.massime."""
    if not body.source_url or not body.source_url.startswith("http"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_url must be a valid HTTP(S) URL",
        )

    try:
        pool = await get_kb_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE kb.massime
                SET source_url = $1
                WHERE id = $2 AND is_active = true
                RETURNING id
                """,
                body.source_url,
                massima_id,
            )

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Massima {massima_id} not found or inactive",
                )

            logger.info(
                "massima.source_url.updated",
                massima_id=str(massima_id),
                source_url=body.source_url[:120],
            )

            return SourceUrlResponse(
                massima_id=str(massima_id),
                source_url=body.source_url,
                updated=True,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update source_url", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}",
        )
