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
LEXe Services (completely separate from legacy)
├── lexe-api:8020      ← FastAPI service with legal tools
├── lexe-max:5436      ← KB Legal database (normativa, embeddings)
├── lexe-postgres:5435 ← Sistema database (Logto, core)
└── lexe-valkey:6380   ← Cache (separate from legacy)
```

## ⚠️ DATABASE - IMPORTANTE!

| Container | Porta | Scopo | User/DB |
|-----------|-------|-------|---------|
| **lexe-max** | 5436 | **KB LEGAL** (usa questo!) | lexe_max/lexe_max |
| lexe-postgres | 5435 | Sistema (Logto, core) | lexe/lexe |

**REGOLA:** Per KB Legal (normativa, massime, embeddings) → **lexe-max:5436**

```bash
# STAGING - KB queries
ssh -i ~/.ssh/id_stage_new root@91.99.229.111
docker exec lexe-max psql -U lexe_max -d lexe_max -c "SELECT * FROM kb.normativa LIMIT 5;"

# LOCALE - KB queries
docker exec lexe-max psql -U lexe_kb -d lexe_kb -c "SELECT * FROM kb.normativa LIMIT 5;"
```

## KB Database Content (lexe-max)

| Tabella | Staging | Descrizione |
|---------|---------|-------------|
| kb.work | 69 | Codici/leggi (CC, CP, CPC, CPP, COST, TUB...) |
| kb.normativa | 6,335 | Articoli (da Brocardi) |
| kb.normativa_chunk | 10,246 | Chunks per retrieval |
| kb.normativa_chunk_embeddings | ~3,300 | Embeddings (text-embedding-3-small) |
| kb.annotation | 13,281 | Note Brocardi |
| kb.annotation_embeddings | 13,180 | Embeddings annotations |

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

# Docker (from lexe-infra)
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
| lexe-postgres | 5433 | Separate from legacy (5432) |
| lexe-valkey | 6380 | Separate from legacy (6379) |

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

## Integration with LEXE Platform

lexe-orchestrator calls lexe-api via HTTP:

```python
from lexe_orchestrator.clients import LexeClient

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
| **Embeddings** | 41,437 total (38,718 active) |
| **Citation Graph Edges** | 58,737 |
| **Norm Graph Edges** | 42,338 |
| **Unique Norms** | 4,128 |
| **Norm Coverage** | 60.3% (23,365 massime) |
| **Recall@10** | 97.5% |
| **MRR** | 0.756 |

### Infrastruttura

| Container | Porta | Descrizione |
|-----------|-------|-------------|
| lexe-max | 5434 | PostgreSQL 17 + pgvector 0.7.4 |

### Retrieval Architecture

```
Query --> Router --> Citation? --> Direct Lookup (RV/Sez/Num/Anno)
                |         |
                |    Norm? --> Norm Lookup (CC:2043, LEGGE:241:1990)
                |         |
                +-- Semantic --> Hybrid Search
                                 |-- Dense (vector, top-50)
                                 |-- Sparse (tsvector, top-50)
                                 +-- RRF Fusion --> Top-K
                                       |
                                 Norm Boost (if norm in query)
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

### Norm Graph (v3.3.0)

Grafo delle norme citate nelle massime per lookup diretto.

| Metric | Value |
|--------|-------|
| Unique norms | 4,128 |
| Total edges | 42,338 |
| Massime with norms | 23,365 (60.3%) |
| Top norm | D.Lgs. 165/2001 (587 citations) |

**Supported norm types:**
- Codici: CC, CPC, CP, CPP, COST
- Testi unici: TUB, TUF, CAD
- Leggi: LEGGE, DLGS, DPR, DL

**Query routing:**
```
Query "art. 2043 c.c." -> RouteType.NORM -> norm_lookup() -> massime citing CC:2043
Query "danno ingiusto art. 2043 c.c." -> hybrid_search + norm_boost
```

**Scripts:**
```bash
# Build norm graph
uv run python scripts/graph/build_norm_graph.py --batch-size 500

# Sanity checks
uv run python scripts/graph/sanity_check_norm_graph.py

# Test router
uv run python scripts/test_norm_router.py
```

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
| `build_norm_graph.py` | Extract norms and build norm graph |
| `sanity_check_norm_graph.py` | Validate norm graph data |
| `test_norm_router.py` | Test norm detection and lookup |

### Schema KB

```sql
kb.massime (id, testo, sezione, numero, anno, rv, is_active, quality_flags, tsv_italian)
kb.embeddings (massima_id, model_name, embedding vector(1536))
kb.graph_edges (source_id, target_id, edge_type, relation_subtype, confidence, weight, run_id)
kb.graph_runs (id, run_type, status, metrics, config)
kb.norms (id, code, article, suffix, number, year, full_ref, citation_count)
kb.massima_norms (massima_id, norm_id, context_span, run_id)
kb.golden_queries (query_text, query_type, expected_massima_id)
```

**Full documentation:** `docs/KB-HANDOFF.md`

---

## KB Normativa (Altalex) — BATCH READY (2026-02-06)

Knowledge Base per codici e leggi italiane da PDF Altalex.

### Pipeline Architecture (4 Fonti)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PDF ALTALEX (primaria)                        │
│           Docling + LLM extraction → JSON                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONFRONTO OFFLINE (parallelo)                       │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │ STUDIO CATALDI  │         │ BROCARDI LOCAL  │               │
│   │   (24 codici)   │         │ (CC,CP,CPC,CPP) │               │
│   └────────┬────────┘         └────────┬────────┘               │
│            └──────────┬────────────────┘                        │
│                       ▼                                          │
│              SIMILARITY CHECK                                    │
│         ┌─────────────────────────┐                             │
│         │ sim >= 0.70 → CONFIRM   │                             │
│         │ 0.40 <= sim < 0.70 →    │                             │
│         │    PARTIAL (usa migliore)│                             │
│         │ sim < 0.40 → CONFLICT   │─────┐                       │
│         └─────────────────────────┘     │                       │
└─────────────────────────────────────────│───────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FALLBACK ONLINE                               │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │ BROCARDI ONLINE │         │   NORMATTIVA    │               │
│   │  (rate limited) │         │  (API ufficiale)│               │
│   └─────────────────┘         └─────────────────┘               │
│         Solo per: CONFLICT, EMPTY, watchlist critica            │
└─────────────────────────────────────────────────────────────────┘
```

### Article Classification (2 Assi Indipendenti)

**ArticleIdentityClass** (per ordinamento, range, dedup):
| Classe | Descrizione | Esempio |
|--------|-------------|---------|
| `BASE` | Articolo numerico puro | 2043, 1, 360 |
| `SUFFIX` | Con suffisso latino | 2635-ter, 360-bis |
| `SPECIAL` | Fuori schema | disp. transitorie, allegati |

**ArticleQualityClass** (per validi/deboli/enrichment):
| Classe | Criteri | Mappatura |
|--------|---------|-----------|
| `VALID_STRONG` | Testo presente, struttura coerente, >150 chars | validi |
| `VALID_SHORT` | Corto ma semanticamente pieno, definizioni | validi |
| `WEAK` | Incompleto, segnali taglio, rumore | deboli |
| `EMPTY` | Vuoto, "abrogato", "omissis" | deboli |
| `INVALID` | Estrazione rotta, heading catturato | invalidi |

**Suffissi latini supportati:**
```
bis, ter, quater, quinquies, sexies, septies, octies,
novies, nonies, decies, undecies, duodecies, terdecies,
quaterdecies, quinquiesdecies, sexiesdecies, septiesdecies, octiesdecies
```

### Tabella Batch Standard

| Colonna | Descrizione |
|---------|-------------|
| `documento` | Nome/codice documento |
| `dal` | Primo articolo (sort_key) |
| `al` | Ultimo articolo (sort_key) |
| `totale` | Articoli unici estratti (inclusi SUFFIX) |
| `validi` | VALID_STRONG + VALID_SHORT |
| `deboli` | WEAK + EMPTY |
| `invalidi` | INVALID (contati a parte) |
| `sospetti` | Articoli con warning critici |
| `coverage_pct` | % copertura vs expected set |

### Risultati Gold (2 campioni)

| Documento | Articoli | Validi | Deboli | Invalidi | Sospetti |
|-----------|----------|--------|--------|----------|----------|
| Codice Civile | 3,208 | 3,201 | 3 | 4 | 0 |
| Codice Crisi Impresa | 415 | 414 | 1 | 0 | 0 |

### Enrichment Policy

**Trigger enrichment quando:**
1. Articolo `WEAK` o `EMPTY`
2. Articolo in **watchlist** (226 articoli chiave: CC, CP, CCI, TUB, TUF)
3. Mismatch hash tra fonti
4. Buchi di coverage (articoli attesi non trovati)

**Fonti offline disponibili:**
| Fonte | Codici | Path |
|-------|--------|------|
| Studio Cataldi | TUB, TUF, TUIR, TUE, TUEL, TUI, TUSL, +17 altri | `C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa` |
| Brocardi Local | CC (in download), CP, CPC, CPP | `C:/Mie pagine Web/broc-civ/www.brocardi.it` |

### Quick Commands - BATCH ESTRAZIONE

```bash
# ══════════════════════════════════════════════════════════════════
# STEP 1: Estrazione PDF con Docling + LLM
# ══════════════════════════════════════════════════════════════════

# Singolo documento con OCR e GPU
cd C:/PROJECTS/lexe-genesis/lexe-max
uv run python scripts/llm_assisted_extraction.py \
  "C:/PROJECTS/lexe-genesis/altalex pdf/Costituzione e 4 codici/codice-civile-30-dicembre-2025-DEF pdf.pdf" \
  --codice CC --ocr

# Batch tutti i PDF
uv run python scripts/llm_assisted_extraction.py --batch --ocr

# ══════════════════════════════════════════════════════════════════
# STEP 2: Quality Report
# ══════════════════════════════════════════════════════════════════

# Genera report qualità per tutti i JSON estratti
uv run python scripts/generate_quality_report.py

# Output: altalex pdf/lexe_quality_report.json e .csv

# ══════════════════════════════════════════════════════════════════
# STEP 3: Ingestion Pipeline (4 fonti)
# ══════════════════════════════════════════════════════════════════

# Test singolo documento (dry-run)
uv run python scripts/ingestion_pipeline.py --doc CC --dry-run --no-online

# Processa con enrichment offline
uv run python scripts/ingestion_pipeline.py --doc CC

# Batch tutti (quando fonti offline complete)
uv run python scripts/ingestion_pipeline.py --all

# ══════════════════════════════════════════════════════════════════
# STEP 4: Test Parser Offline
# ══════════════════════════════════════════════════════════════════

# Test Studio Cataldi parser
uv run python scripts/studio_cataldi_parser.py

# Test Brocardi parser
uv run python scripts/brocardi_parser.py
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/llm_assisted_extraction.py` | PDF → JSON con Docling + LLM |
| `scripts/article_classification.py` | Classification v2 (2 assi) |
| `scripts/generate_quality_report.py` | Quality report batch |
| `scripts/ingestion_pipeline.py` | Pipeline 4 fonti |
| `scripts/studio_cataldi_parser.py` | Parser HTML Studio Cataldi |
| `scripts/brocardi_parser.py` | Parser HTML Brocardi offline |
| `scripts/enrichment_policy.py` | Policy e trigger enrichment |

### Schema Extension (v2)

```sql
-- Columns per normalizzazione articolo
articolo_num_norm INTEGER      -- 2043 per "2043-bis"
articolo_suffix TEXT           -- "bis", "ter", "quater"...
articolo_sort_key TEXT         -- "002043.02" per sort naturale
global_key TEXT UNIQUE         -- "altalex:cc:2043:bis"

-- Provenance e enrichment
testo_context TEXT             -- Overlap ±200 chars
canonical_source VARCHAR(50)   -- 'pdf_altalex'
mirror_source VARCHAR(50)      -- 'brocardi' | 'studio_cataldi'
validation_status VARCHAR(20)  -- 'pending', 'verified', 'content_diff'
```

### Status Batch

- [x] Docling extraction with OCR+GPU - **Tested CC, CCI**
- [x] Article classification v2 (2 assi) - **Ready**
- [x] Quality report generator - **Ready**
- [x] Studio Cataldi parser - **280 articoli TUB testati**
- [x] Brocardi parser - **384 articoli CC (download in corso)**
- [x] Ingestion pipeline 4 fonti - **Ready**
- [ ] Brocardi download completo - **In progress**
- [ ] Batch 69 PDF - **Next**
- [ ] Fallback online - **Pending**

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

**Phase 5: KB Normativa** 🔄 IN PROGRESS
- [x] PDF extraction pipeline (Docling + LLM)
- [x] Article classification v2 (2 assi: identity + quality)
- [x] 4-source ingestion architecture
- [x] Offline parsers (Studio Cataldi, Brocardi)
- [ ] Batch 69 PDF extraction
- [ ] Embeddings generation
- [ ] Integration with retrieval layer

**Phase 6: Production API** 🔜 NEXT
- FastAPI endpoints for KB search
- Integration with LEXE TRIDENT
- Streaming/SSE support

---

## Documentation

| File | Descrizione |
|------|-------------|
| `docs/SCHEMA_KB_OVERVIEW.md` | Schema PostgreSQL+pgvector completo |
| `docs/KB-HANDOFF.md` | Handoff KB Massimari |
| `docs/ARTICLE_EXTRACTION_STRATEGIES.md` | Strategie estrazione articoli |

## Alerting

When tools fail:
1. Circuit breaker opens after 5 failures
2. Admin email sent (if configured)
3. Webhook notification (Slack/Discord)
4. Tool marked as degraded
5. Auto-retry after 5 minutes

---
*Created: 2026-01-13*
*Updated: 2026-02-04*
*Status: Phase 1-4 Complete | KB Normativa in progress*
*Repo: https://github.com/LEXe-Legal-AI/lexe-max*
