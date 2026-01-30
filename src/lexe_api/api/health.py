"""Health check endpoints."""

from fastapi import APIRouter, HTTPException

from lexe_api import __version__
from lexe_api.cache import cache
from lexe_api.database import db
from lexe_api.models.schemas import HealthStatus, ToolHealthResponse
from lexe_api.tools.health_monitor import health_monitor

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


@router.get("/status", response_model=HealthStatus)
async def health_status() -> HealthStatus:
    """Full health status including all tools."""
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

    # Check tools
    tools_health = await health_monitor.get_all_health()
    tools_status = {
        name: health.state == "healthy" for name, health in tools_health.items()
    }

    overall = "healthy"
    if not db_healthy or not cache_healthy:
        overall = "unhealthy"
    elif not all(tools_status.values()):
        overall = "degraded"

    return HealthStatus(
        status=overall,
        version=__version__,
        database=db_healthy,
        cache=cache_healthy,
        tools=tools_status,
    )


@router.get("/tools", response_model=dict[str, ToolHealthResponse])
async def tools_health() -> dict[str, ToolHealthResponse]:
    """Get health status for all legal tools."""
    return await health_monitor.get_all_health()


@router.get("/tools/{tool_name}", response_model=ToolHealthResponse)
async def tool_health(tool_name: str) -> ToolHealthResponse:
    """Get health status for a specific tool."""
    if tool_name not in health_monitor.TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    return await health_monitor.get_tool_health(tool_name)


@router.post("/tools/{tool_name}/reset")
async def reset_tool(tool_name: str) -> dict:
    """Reset a tool to healthy state (admin only)."""
    if tool_name not in health_monitor.TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    await health_monitor.reset_tool(tool_name)
    return {"status": "ok", "message": f"Tool {tool_name} reset to healthy"}
