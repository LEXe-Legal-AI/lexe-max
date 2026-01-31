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

## KB Massimari — PRODUCTION READY (2026-01-31)

Knowledge Base per i massimari della Corte di Cassazione.

### Status

| Metric | Value |
|--------|-------|
| **Active Massime** | 38,718 |
| **Embeddings** | 41,437 (text-embedding-3-small) |
| **Citation Graph Edges** | 58,737 |
| **RV Coverage** | 99.0% |
| **Recall@10** | 97.5% |
| **MRR** | 0.756 |

### Infrastruttura

| Container | Porta | Descrizione |
|-----------|-------|-------------|
| lexe-kb | 5434 | PostgreSQL 17 + pgvector 0.7.4 |

### Retrieval Architecture

```
Query → Router → Citation? → Direct Lookup (RV/Sez/Num/Anno)
                    ↓ NO
              Hybrid Search
              ├─ Dense (vector, top-50)
              ├─ Sparse (tsvector, top-50)
              └─ RRF Fusion → Top-K
```

### Citation Graph (v3.2.3)

| Metric | Value |
|--------|-------|
| Edges | 58,737 |
| Unique sources | 26,415 |
| Unique targets | 20,868 |
| Resolution rate | 44.8% |
| rv_exact | 47.5% |

**Resolver cascade:**
1. `rv_exact` - RV column match (47.5%)
2. `sez_num_anno` - Sez+Num+Anno match (25.0%)
3. `rv_text_fallback` - RV in text (14.5%)
4. `sez_num_anno_raw` - Raw numero (8.3%)
5. `num_anno` - Num+Anno only (4.7%)

**Guardrail v3.2.3:** Solo massime "plausibili" (rv OR numero+anno OR patterns in text)

### Quick Commands

```bash
# Start infrastructure
docker compose -f docker-compose.kb.yml up -d

# Run retrieval evaluation
OPENROUTER_API_KEY="sk-or-..." uv run python scripts/qa/run_retrieval_eval.py \
  --top-k 10 --mode hybrid --log-results

# Generate embeddings for new massime
uv run python scripts/qa/generate_openai_embeddings.py --batch-size 100 --commit

# Regenerate golden set
uv run python scripts/qa/generate_golden_set.py --count 200 --commit

# Build citation graph
uv run python scripts/graph/build_citation_graph.py --commit --skip-age
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `run_retrieval_eval.py` | Hybrid search + eval + JSONL/CSV logging |
| `generate_openai_embeddings.py` | Embeddings via OpenRouter |
| `generate_golden_set.py` | Auto-generate test queries |
| `build_citation_graph.py` | Build citation graph with dual-write |

### Schema KB

```sql
kb.massime (id, testo, sezione, numero, anno, rv, is_active, quality_flags, tsv_italian)
kb.embeddings (massima_id, model_name, embedding vector(1536))
kb.graph_edges (source_id, target_id, edge_type, relation_subtype, confidence, weight, run_id)
kb.graph_runs (id, run_type, status, metrics, config)
kb.golden_queries (query_text, query_type, expected_massima_id)
```

**Full documentation:** `docs/KB-HANDOFF.md`

---

## Phases Status

**Phase 1: Legal Tools** ✅ COMPLETE
- Normattiva, EUR-Lex, Infolex scrapers
- Circuit breaker, caching, health monitoring

**Phase 2: KB Vectors** ✅ COMPLETE
- pgvector 0.7.4 + HNSW indexes
- text-embedding-3-small (1536 dim)
- Hybrid search (dense + sparse + RRF)

**Phase 3: QA Protocol** ✅ COMPLETE
- Golden set auto-generation
- Retrieval evaluation pipeline
- JSONL/CSV logging

**Phase 4: Citation Graph** ✅ COMPLETE
- Citation extraction with resolver cascade
- Graph edges (58,737) with confidence/weight
- Query router with direct lookup
- Guardrail v3.2.3 for data quality

**Phase 5: Production API** 🔜 NEXT
- FastAPI endpoints for KB search
- Integration with LEO TRIDENT
- Streaming/SSE support

## Alerting

When tools fail:
1. Circuit breaker opens after 5 failures
2. Admin email sent (if configured)
3. Webhook notification (Slack/Discord)
4. Tool marked as degraded
5. Auto-retry after 5 minutes

---
*Created: 2026-01-13*
*Updated: 2026-01-31*
*Status: Phase 1-4 Complete | Citation Graph + Router*
*Repo: https://github.com/LEO-ITC/lexe-max*
