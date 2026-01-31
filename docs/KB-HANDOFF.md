# KB Massimari - Handoff Document

> Knowledge Base per i Massimari della Corte di Cassazione
> Status: **Production Ready** | Last Updated: 2026-01-31

---

## Quick Start

```bash
# 1. Start infrastructure
docker compose -f docker-compose.kb.yml up -d

# 2. Generate golden set (first time only)
uv run python scripts/qa/generate_golden_set.py --count 200 --commit

# 3. Run retrieval eval
OPENROUTER_API_KEY="sk-or-..." uv run python scripts/qa/run_retrieval_eval.py \
  --top-k 10 --mode hybrid --log-results --log-dir retrieval_logs --tag prod_v1
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ──▶ Router ──▶ Citation Lookup? ──YES──▶ Direct DB     │
│                │                                                │
│                NO                                               │
│                ▼                                                │
│         ┌─────────────┐                                        │
│         │   HYBRID    │                                        │
│         ├─────────────┤                                        │
│         │ Dense (50)  │──┐                                     │
│         │ Sparse (50) │──┼──▶ RRF Fusion ──▶ Top-K Results    │
│         └─────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | PostgreSQL 17 + pgvector | Storage + Vector search |
| **Embeddings** | text-embedding-3-small | 1536-dim, cosine similarity |
| **Sparse Search** | tsvector (Italian) | BM25-style full-text |
| **Fusion** | RRF (k=60) | Reciprocal Rank Fusion |

---

## Database Schema

### Core Tables

```sql
-- Main massime table
kb.massime (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES kb.documents(id),
    testo TEXT NOT NULL,
    sezione VARCHAR(20),      -- "1", "2", "L", "U"
    numero VARCHAR(20),       -- "00208", "12345"
    anno INTEGER,             -- 2020, 2021
    rv VARCHAR(30),           -- "639966", "639966-01"
    is_active BOOLEAN,        -- Latest batch wins
    ingest_batch_id INTEGER,
    tsv_italian TSVECTOR,     -- Full-text search (GIN indexed)
    content_hash TEXT,        -- Dedup
    created_at TIMESTAMPTZ
)

-- Embeddings (multi-model ready)
kb.embeddings (
    id UUID PRIMARY KEY,
    massima_id UUID REFERENCES kb.massime(id),
    model_name TEXT,          -- 'openai/text-embedding-3-small'
    embedding vector(1536),   -- HNSW indexed
    is_normalized BOOLEAN
)

-- Evaluation
kb.golden_queries (query_text, query_type, expected_massima_id)
kb.retrieval_logs (query_text, retrieval_mode, results, metrics)
```

### Key Indexes

```sql
-- Vector search (HNSW)
idx_embeddings_hnsw_cosine ON kb.embeddings USING hnsw (embedding vector_cosine_ops)

-- Citation lookup
idx_massime_rv_active ON kb.massime (rv) WHERE is_active = TRUE
idx_massime_sez_num_anno_active ON kb.massime (sezione, numero, anno) WHERE is_active = TRUE

-- Full-text
idx_massime_tsv ON kb.massime USING gin (tsv_italian)
```

---

## Scripts Reference

### Ingestion

| Script | Purpose | Usage |
|--------|---------|-------|
| `reingest_civile_citation_anchored.py` | Re-extract massime with citation anchoring | `--batch-size 10 --commit` |
| `massivo_write_civile.py` | Bulk write after dry-run validation | `--commit` |
| `canary_write_civile.py` | Test write on single document | `--doc-id <uuid>` |

### Embeddings

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_openai_embeddings.py` | Generate embeddings via OpenRouter | `--batch-size 100 --commit` |
| `generate_gemini_embeddings.py` | Alternative: Gemini embeddings | `--batch-size 50 --commit` |

### Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_golden_set.py` | Auto-generate test queries | `--count 200 --commit` |
| `run_retrieval_eval.py` | Run retrieval evaluation | `--mode hybrid --log-results` |

---

## Retrieval API

### Python Usage

```python
import asyncpg
import httpx

# 1. Get embedding for query
async def get_embedding(query: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": "openai/text-embedding-3-small", "input": [query]}
        )
        return response.json()["data"][0]["embedding"]

# 2. Hybrid search
async def search(conn, query: str, embedding: list[float], top_k: int = 10):
    # Try citation lookup first
    citation = parse_citation(query)  # Extract Rv./Sez./n. patterns
    if citation:
        results = await citation_lookup(conn, citation, top_k)
        if results:
            return results, "citation_lookup"

    # Hybrid: dense + sparse + RRF
    dense = await dense_search(conn, embedding, 50)
    sparse = await sparse_search(conn, query, 50)
    fused = rrf_fusion(dense, sparse, top_k)
    return fused, "hybrid_rrf"
```

### SQL Examples

```sql
-- Citation lookup (direct)
SELECT id, testo, sezione, numero, anno, rv
FROM kb.massime
WHERE is_active = TRUE AND rv = '639966';

-- Dense search (vector similarity)
SELECT m.id, 1 - (e.embedding <=> $1::vector) as score
FROM kb.embeddings e
JOIN kb.massime m ON m.id = e.massima_id
WHERE m.is_active = TRUE
ORDER BY e.embedding <=> $1::vector
LIMIT 10;

-- Sparse search (full-text)
SELECT id, ts_rank_cd(tsv_italian, query) as score
FROM kb.massime, plainto_tsquery('italian', 'responsabilità civile') query
WHERE is_active = TRUE AND tsv_italian @@ query
ORDER BY score DESC
LIMIT 10;
```

---

## Current Metrics (2026-01-31)

### Data Volume

| Metric | Count |
|--------|-------|
| Active massime | 41,437 |
| Embeddings | 41,437 |
| RV populated | 16,002 (38.6%) |
| Documents (PDF) | 63 |

### Retrieval Performance

| Metric | Value | Target |
|--------|-------|--------|
| **Recall@10** | 97.5% | ≥75% ✅ |
| **MRR** | 0.756 | ≥0.55 ✅ |
| **Self Recall** | 95.0% | ≥75% ✅ |
| **Citation Recall** | 100.0% | ≥65% ✅ |
| **Latency p95** | 78ms | ≤500ms ✅ |

### Search Mode Distribution

| Mode | Usage |
|------|-------|
| citation_lookup | 54% |
| hybrid_rrf | 46% |

---

## Maintenance Procedures

### 1. Adding New Documents

```bash
# 1. Place PDFs in staging directory
# 2. Run ingestion
uv run python scripts/qa/reingest_civile_citation_anchored.py \
  --input-dir /path/to/pdfs --batch-size 10 --dry-run

# 3. Review dry-run results, then commit
uv run python scripts/qa/reingest_civile_citation_anchored.py \
  --input-dir /path/to/pdfs --batch-size 10 --commit

# 4. Generate embeddings for new massime
uv run python scripts/qa/generate_openai_embeddings.py --batch-size 100 --commit

# 5. Run eval to verify
uv run python scripts/qa/run_retrieval_eval.py --mode hybrid
```

### 2. Backfill Missing RV

```sql
-- Check current state
SELECT
  COUNT(*) FILTER (WHERE rv IS NOT NULL) AS rv_populated,
  COUNT(*) FILTER (WHERE rv IS NULL AND testo ~ 'Rv\.?[\s\u00a0]+\d{5,7}') AS rv_in_text
FROM kb.massime WHERE is_active = TRUE;

-- Backfill single-RV massime (safe)
UPDATE kb.massime m
SET rv = (regexp_match(m.testo, 'Rv\.?[\s\u00a0]+(\d{5,7})'))[1]
WHERE m.is_active = TRUE
  AND m.rv IS NULL
  AND m.testo ~ 'Rv\.?[\s\u00a0]+\d{5,7}'
  AND (SELECT COUNT(*) FROM regexp_matches(m.testo, 'Rv\.?[\s\u00a0]+\d{5,7}', 'g')) = 1;
```

### 3. Regenerate Golden Set

```bash
# Deactivates old queries, generates new ones
uv run python scripts/qa/generate_golden_set.py --count 500 --commit
```

### 4. Monitor Retrieval Quality

```bash
# Weekly eval with logging
uv run python scripts/qa/run_retrieval_eval.py \
  --top-k 10 --mode hybrid \
  --log-results --log-dir retrieval_logs \
  --tag weekly_$(date +%Y%m%d)
```

---

## Troubleshooting

### Low Citation Recall

1. Check RV column population: `SELECT COUNT(*) FROM kb.massime WHERE rv IS NOT NULL`
2. Run RV backfill (see above)
3. Verify indexes exist on `rv` and `(sezione, numero, anno)`

### Low Self Recall

1. Check embeddings exist: `SELECT COUNT(*) FROM kb.embeddings`
2. Verify HNSW index: `\di+ idx_embeddings_hnsw_cosine`
3. Check embedding model consistency

### High Latency

1. Check HNSW index health
2. Increase `ef_search` for better recall (tradeoff: slower)
3. Check connection pooling

### NO-BREAK SPACE Issues

Some PDFs have U+00A0 (NBSP) instead of regular space. Use pattern:
```sql
testo ~ 'Rv\.?[\s\u00a0]+\d{5,7}'  -- Handles both
```

---

## Configuration

### Environment Variables

```bash
# Database
LEXE_KB_DATABASE_URL=postgresql://lexe_kb:password@localhost:5434/lexe_kb

# Embeddings API
OPENROUTER_API_KEY=sk-or-v1-...

# Model
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536
```

### Tuning Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `top_k` | 10 | Final results returned |
| `dense_k` | 50 | Candidates from vector search |
| `sparse_k` | 50 | Candidates from text search |
| `rrf_k` | 60 | RRF fusion constant |
| `HNSW m` | 16 | Index connectivity |
| `HNSW ef_construction` | 64 | Build-time quality |

---

## File Structure

```
lexe-api/
├── src/lexe_api/kb/
│   ├── ingestion/
│   │   ├── massima_extractor.py    # Citation-anchored extraction
│   │   └── cut_validator.py        # Oversized massime handling
│   └── retrieval/                  # (future: API endpoints)
├── scripts/qa/
│   ├── generate_golden_set.py      # Test query generation
│   ├── generate_openai_embeddings.py
│   ├── run_retrieval_eval.py       # Hybrid search + eval
│   ├── migrations/                 # SQL schema changes
│   └── retrieval_logs/             # JSONL + CSV outputs
└── docs/
    ├── KB-MASSIMARI-ARCHITECTURE.md
    └── KB-HANDOFF.md               # This file
```

---

## Next Steps (Roadmap)

1. **API Endpoints**: Expose retrieval via FastAPI `/api/v1/kb/search`
2. **Reranking**: Add cross-encoder reranking for top-20
3. **Streaming**: SSE for long-running searches
4. **Multi-model**: A/B test Gemini vs OpenAI embeddings
5. **Graph**: Citation graph via Apache AGE

---

## Contacts

- **Repo**: https://github.com/LEO-ITC/lexe-max
- **Branch**: `stage` (current), `main` (production)

---

*Generated: 2026-01-31 | QA Protocol v3.2*
