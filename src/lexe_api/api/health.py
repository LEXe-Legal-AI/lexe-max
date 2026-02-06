"""Health check endpoints for lexe-max (KB service)."""

from fastapi import APIRouter, HTTPException

from lexe_api import __version__
from lexe_api.cache import cache
from lexe_api.database import db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    """Liveness probe - service is running."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness() -> dict:
    """Readiness probe - service can accept traffic."""
    try:
        # Check database
        async with db.acquire() as conn:
            await conn.fetchval("SELECT 1")

        # Check cache
        await cache.set("health_check", "ok", ttl_seconds=10)
        await cache.get("health_check")

        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/status")
async def health_status() -> dict:
    """Full health status."""
    db_healthy = True
    cache_healthy = True

    # Check database
    try:
        async with db.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception:
        db_healthy = False

    # Check cache
    try:
        await cache.set("health_check", "ok", ttl_seconds=10)
        await cache.get("health_check")
    except Exception:
        cache_healthy = False

    overall = "healthy" if db_healthy and cache_healthy else "unhealthy"

    return {
        "status": overall,
        "version": __version__,
        "service": "lexe-max-kb",
        "database": db_healthy,
        "cache": cache_healthy,
    }
