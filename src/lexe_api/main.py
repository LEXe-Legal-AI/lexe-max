"""LEXe Max - Knowledge Base Service.

FastAPI application for legal knowledge base (massimari, normativa).
"""

# Configure structured logging
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexe_api import __version__
from lexe_api.api.health import router as health_router
from lexe_api.api.kb_retrieval import router as kb_normativa_router
from lexe_api.cache import cache
from lexe_api.config import settings
from lexe_api.database import db
from lexe_api.kb.retrieval.normativa_hybrid import close_normativa_pool

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
        "Starting LEXe Max KB",
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
    logger.info("Shutting down LEXe Max KB")
    await close_normativa_pool()
    await cache.disconnect()
    await db.disconnect()


# Create FastAPI app
app = FastAPI(
    title="LEXe Max - Knowledge Base",
    description=(
        "Legal Knowledge Base Service.\n\n"
        "## Features\n"
        "- **Massimari**: Cassazione case law summaries (38K+ massime)\n"
        "- **Normativa**: Italian legislation from Altalex PDFs\n"
        "- **Hybrid Search**: Dense + Sparse + RRF fusion\n"
        "- **Citation Graph**: Cross-reference navigation\n"
        "- **Norm Graph**: Article citation lookup\n\n"
        "## Note\n"
        "Legal tools (normattiva, eurlex, infolex) are served by lexe-tools-it:8021"
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
app.include_router(kb_normativa_router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LEXe Max - Knowledge Base",
        "version": __version__,
        "description": "Legal Knowledge Base Service",
        "docs": "/docs",
        "health": "/health/status",
        "note": "Legal tools served by lexe-tools-it:8021",
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
