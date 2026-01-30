# lexe-api - Claude Context

## Overview
LEXe Legal Tools API - Separate service for Italian and European legal document search.

## Tech Stack
- Python 3.12, FastAPI, Pydantic v2
- PostgreSQL 17 (simple tables, no vector for Phase 1)
- Valkey (Redis-compatible) for caching
- BeautifulSoup4 for HTML parsing
- HTTPX for async HTTP requests
- Tenacity for retry logic

## Architecture

```
LEXe Services (completely separate from LEO)
├── lexe-api:8020      ← FastAPI service with legal tools
├── lexe-postgres:5433 ← Document storage (separate from LEO)
└── lexe-valkey:6380   ← Cache (separate from LEO)
```

## Commands
```bash
# Development
cd lexe-api
uv sync
pytest -v

# Linting
ruff check src/
ruff format src/

# Run server
uvicorn lexe_api.main:app --reload --port 8020

# Docker (from leo-infra)
docker compose -f docker-compose.lexe.yml --env-file .env.lexe up -d
```

## Module Structure
```
src/lexe_api/
├── main.py             # FastAPI application
├── config.py           # Settings (env vars)
├── database.py         # PostgreSQL client
├── cache.py            # Valkey client
├── api/
│   ├── health.py       # Health endpoints
│   └── tools.py        # Legal tools endpoints
├── tools/
│   ├── base.py         # BaseLegalTool (cache, circuit breaker)
│   ├── normattiva.py   # Italian legislation
│   ├── eurlex.py       # European legislation
│   ├── infolex.py      # Brocardi case law
│   └── health_monitor.py # Alerting
├── scrapers/
│   ├── http_client.py  # Throttled HTTP with retries
│   └── selectors.py    # CSS selectors (centralized)
└── models/
    └── schemas.py      # Pydantic models
```

## API Endpoints

### Tools
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tools/normattiva/search` | POST | Search Italian legislation |
| `/api/v1/tools/normattiva/vigenza` | POST | Quick vigenza check |
| `/api/v1/tools/eurlex/search` | POST | Search EU legislation |
| `/api/v1/tools/infolex/search` | POST | Search Brocardi case law |
| `/api/v1/tools/status` | GET | All tools status |

### Health
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/health/status` | GET | Full status |
| `/health/tools` | GET | All tools health |
| `/health/tools/{name}` | GET | Single tool health |
| `/health/tools/{name}/reset` | POST | Reset tool to healthy |

## Database Schema (Phase 1)

```sql
-- Legal documents
documents (
    id, source, urn, act_type, act_number, article,
    title, content, html_raw, is_vigente, metadata,
    scraped_at, expires_at
)

-- Case law summaries
massime (
    id, document_id, autorita, numero, data,
    testo, keywords, brocardi_url
)

-- Tool health monitoring
tool_health (
    tool_name, state, circuit_state, failure_count,
    last_success_at, last_failure_at, last_error_message
)
```

## Ports and Services

| Service | Port | Note |
|---------|------|------|
| lexe-api | 8020 | Legal tools API |
| lexe-postgres | 5433 | Separate from LEO (5432) |
| lexe-valkey | 6380 | Separate from LEO (6379) |

## Environment Variables

```bash
# Database
LEXE_DATABASE_URL=postgresql://lexe:pass@lexe-postgres:5432/lexe
LEXE_REDIS_URL=redis://:pass@lexe-valkey:6379/0

# API
LEXE_API_PORT=8020
LEXE_LOG_LEVEL=INFO

# Cache
LEXE_CACHE_TTL_HOURS=24
LEXE_DOCUMENT_EXPIRE_DAYS=30

# Health
LEXE_HEALTH_FAILURE_THRESHOLD=5
LEXE_ADMIN_EMAILS=admin@example.com
LEXE_ALERT_WEBHOOK_URL=https://hooks.slack.com/...

# Feature Flags
FF_LEXE_NORMATTIVA_ENABLED=true
FF_LEXE_EURLEX_ENABLED=true
FF_LEXE_INFOLEX_ENABLED=true
```

## Integration with LEO

leo-orchestrator calls lexe-api via HTTP:

```python
from leo_orchestrator.clients import LexeClient

client = LexeClient()
result = await client.normattiva_search(
    act_type="legge",
    date="1990-08-07",
    act_number="241",
    article="1"
)
```

## KB Massimari (NEW - 2026-01-27)

Knowledge Base per i massimari della Corte di Cassazione.

### Infrastruttura

| Container | Porta | Descrizione |
|-----------|-------|-------------|
| lexe-kb | 5434 | PostgreSQL 17 + pgvector + AGE + pg_search |
| unstructured-api | 8500 | PDF extraction API |

### Estensioni PostgreSQL

- **pgvector** 0.7.4 - HNSW vector search
- **pg_search** 0.21.4 - BM25 native (ParadeDB)
- **Apache AGE** 1.6.0 - Graph database
- **pg_trgm** 1.6 - Fuzzy search

### Schema KB

```sql
kb.documents      -- PDF sorgente
kb.massime        -- Massime estratte (166+ test)
kb.embeddings     -- Vettori multi-modello
kb.citations      -- Citazioni normative
kb.sections       -- Sezioni documento
```

### Comandi KB

```bash
# Start KB infrastructure
docker compose -f docker-compose.kb.yml up -d

# Start Unstructured API
docker run -p 8500:8000 -d --name unstructured-api \
  downloads.unstructured.io/unstructured-io/unstructured-api:latest

# Test ingestion
uv run python scripts/test_ingestion.py

# Test retrieval
uv run python scripts/test_retrieval.py

# Compare PyMuPDF vs Unstructured
uv run python scripts/test_unstructured.py
```

### Moduli KB

```
src/lexe_api/kb/
├── config.py, models.py
├── ingestion/   # 8 moduli (extractor, cleaner, parser, etc.)
└── retrieval/   # 6 moduli (dense, sparse, hybrid, etc.)
```

### Risultati Test

| Metodo | Massime Estratte | Tempo |
|--------|------------------|-------|
| PyMuPDF | 1,267 | ~0.3s |
| **Unstructured** | **4,547** | ~25s |

**Documentazione completa:** `docs/KB-MASSIMARI-ARCHITECTURE.md`

---

## Future Phases

**Phase 2: Vectors + Mini-RAG** ✅ IMPLEMENTED (KB Module)
- pgvector extension ✅
- HNSW indexes ✅
- Multi-model embeddings (Qwen3, E5, BGE, Legal-BERT)

**Phase 3: Knowledge Graph** ✅ IMPLEMENTED (KB Module)
- Apache AGE extension ✅
- lexe_jurisprudence graph ✅
- CITA, INTERPRETA, CONFERMA edges

**Phase 4: Production**
- API endpoints for KB search
- Integration with LEO TRIDENT
- Benchmark S1-S5 configurations

## Alerting

When tools fail:
1. Circuit breaker opens after 5 failures
2. Admin email sent (if configured)
3. Webhook notification (Slack/Discord)
4. Tool marked as degraded
5. Auto-retry after 5 minutes

---
*Created: 2026-01-13*
*Updated: 2026-01-27*
*Status: Phase 1 (Tools) + KB Massimari MVP*
