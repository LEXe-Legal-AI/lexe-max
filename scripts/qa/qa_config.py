"""
QA Protocol - Shared Configuration

Centralizes DB, PDF path, and API URLs.
Override via environment variables for different environments.

Local defaults (Windows dev):
    DB: lexe_kb on localhost:5434
    PDFs: C:/PROJECTS/lexe-genesis/raccolta
    Unstructured: localhost:8500

Staging (91.99.229.111):
    export QA_DB_URL="postgresql://lexe:stage_postgres_2026_secure@localhost:5433/lexe"
    export QA_PDF_DIR="/opt/lexe-platform/lexe-max/data/massimari"
"""

import os
from pathlib import Path

# ── Database ─────────────────────────────────────────────────────
DB_URL = os.getenv(
    "QA_DB_URL",
    "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb",
)

# ── PDF directory ────────────────────────────────────────────────
PDF_DIR = Path(
    os.getenv(
        "QA_PDF_DIR",
        "C:/PROJECTS/lexe-genesis/data/raccolta",
    )
)

# ── Unstructured API ─────────────────────────────────────────────
UNSTRUCTURED_URL = os.getenv(
    "QA_UNSTRUCTURED_URL",
    "http://localhost:8500/general/v0/general",
)

# ── OpenRouter API ───────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

# LLM model for reasoning (Phase 9)
LLM_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"

# Embedding models for benchmark
EMBED_MISTRAL = "mistralai/mistral-embed-2312"
EMBED_GEMINI = "google/gemini-embedding-001"
EMBED_DIM = 1024  # Fixed dimension for both
