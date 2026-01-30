"""LEXe API - Legal Tools Service.

FastAPI application for Italian and European legal document search.
"""

# Configure structured logging
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexe_api import __version__
from lexe_api.api.health import router as health_router
from lexe_api.api.tools import router as tools_router
from lexe_api.cache import cache
from lexe_api.config import settings
from lexe_api.database import db

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        LOG_LEVELS.get(settings.log_level.upper(), logging.INFO)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(
        "Starting LEXe API",
        version=__version__,
        host=settings.api_host,
        port=settings.api_port,
    )

    # Connect to database
    await db.connect()
    logger.info("Database connected")

    # Connect to cache
    await cache.connect()
    logger.info("Cache connected")

    yield

    # Shutdown
    logger.info("Shutting down LEXe API")
    await cache.disconnect()
    await db.disconnect()


# Create FastAPI app
app = FastAPI(
    title="LEXe API",
    description=(
        "Legal Tools Service for Italian and European legislation.\n\n"
        "## Tools\n"
        "- **Normattiva**: Italian legislation (leggi, decreti, codici)\n"
        "- **EUR-Lex**: European legislation (regolamenti, direttive)\n"
        "- **InfoLex**: Case law and commentary (Brocardi.it)\n\n"
        "## Features\n"
        "- Vigenza verification\n"
        "- Cache-first lookup with database persistence\n"
        "- Circuit breaker for external source resilience\n"
        "- Health monitoring and alerting"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(tools_router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LEXe API",
        "version": __version__,
        "description": "Legal Tools Service",
        "docs": "/docs",
        "health": "/health/status",
        "tools": {
            "normattiva": {
                "search": "POST /api/v1/tools/normattiva/search",
                "vigenza": "POST /api/v1/tools/normattiva/vigenza",
            },
            "eurlex": {
                "search": "POST /api/v1/tools/eurlex/search",
            },
            "infolex": {
                "search": "POST /api/v1/tools/infolex/search",
            },
        },
    }


# For running with `python -m lexe_api.main`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "lexe_api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
