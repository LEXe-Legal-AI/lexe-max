# LEXE KB Normativa - Development Timeline

> **Complete chronological development log**
> Last Updated: 2026-02-07 18:30 UTC

---

## Executive Summary

Development of a hybrid retrieval system for Italian legal codes (Codice Civile, Codice Penale, etc.) with dense vector search + sparse FTS + RRF fusion.

| Phase | Period | Status |
|-------|--------|--------|
| Data Ingestion | 2026-02-04 to 2026-02-06 | Completed |
| Chunking | 2026-02-06 | Completed |
| Embedding Generation | 2026-02-06 to 2026-02-07 | Completed |
| Hybrid Search Implementation | 2026-02-07 | Completed |
| Evaluation | 2026-02-07 | Completed |

---

## Timeline

### 2026-02-04 — Data Source Setup

**Objective:** Configure KB database and identify data sources

- Identified Brocardi as primary source for legal codes
- Configured `lexe-max` PostgreSQL container on staging (port 5436)
- Schema: `kb.work`, `kb.normativa`, `kb.annotation`

**Database Containers:**
| Container | Port | Purpose |
|-----------|------|---------|
| lexe-postgres | 5435 | Sistema (Logto, core) |
| lexe-max | 5436 | KB Legal (normativa, embeddings) |

---

### 2026-02-05 — Brocardi Ingestion

**Objective:** Ingest Italian legal codes from Brocardi

**Work items created:**
| Code | Full Name | Source |
|------|-----------|--------|
| CC | Codice Civile | Brocardi |
| CP | Codice Penale | Brocardi |
| CPC | Codice Procedura Civile | Brocardi |
| CPP | Codice Procedura Penale | Brocardi |
| COST | Costituzione | Brocardi |

**Ingestion results:**
| Code | Articles | Annotations |
|------|----------|-------------|
| CC | 3,221 | ~5,000 |
| CPC | 1,055 | ~2,500 |
| CP | 962 | ~2,000 |
| CPP | 958 | ~2,500 |
| COST | 139 | ~1,200 |
| **TOTAL** | **6,335** | **~13,200** |

---

### 2026-02-06 — Chunking Pipeline

**Objective:** Create semantic chunks for retrieval

**Chunking parameters:**
- Target size: ~1000 characters
- Overlap: 150 characters
- Preserves article boundaries

**Script:** `scripts/chunk_remote.py`

```python
# Chunking logic
MAX_CHUNK = 1000
OVERLAP = 150
```

**Results:**
| Code | Articles | Chunks | Avg Chunks/Article |
|------|----------|--------|-------------------|
| CC | 3,221 | 4,054 | 1.26 |
| CPC | 1,055 | 2,192 | 2.08 |
| CP | 962 | 1,310 | 1.36 |
| CPP | 958 | 2,494 | 2.60 |
| COST | 139 | 196 | 1.41 |
| **TOTAL** | **6,335** | **10,246** | **1.62** |

**FTS Index created:**
```sql
CREATE TABLE kb.normativa_chunk_fts (
    chunk_id INTEGER PRIMARY KEY,
    tsv_it tsvector
);
CREATE INDEX idx_chunk_fts_tsv ON kb.normativa_chunk_fts USING GIN(tsv_it);
```

---

### 2026-02-07 — Embedding Generation

**Objective:** Generate dense embeddings for all chunks

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Model | text-embedding-3-small |
| Dimensions | 1536 |
| Provider | OpenRouter API |
| Batch size | 50 |

**Script:** `scripts/embed_lexe_max_staging.py`

**Execution log:**
```
Connecting to lexe-max staging...
  Host: localhost:5436
  Database: lexe_max

Chunks senza embeddings:
  CC: 754
  COST: 196
  CP: 1310
  CPC: 2192
  CPP: 2494
  TOTALE: 6946

Stima costo: $0.0347 (1,738 tokens avg)

Generando embeddings...
  50/6946 (0.7%)
  100/6946 (1.4%)
  ...
  6946/6946 (100.0%)

DONE! Embeddings generati: 6946
```

**Final coverage:**
| Code | Chunks | Embeddings | Coverage |
|------|--------|------------|----------|
| CC | 4,054 | 4,054 | 100% |
| CPC | 2,192 | 2,192 | 100% |
| CP | 1,310 | 1,310 | 100% |
| CPP | 2,494 | 2,494 | 100% |
| COST | 196 | 196 | 100% |
| **TOTAL** | **10,246** | **10,246** | **100%** |

---

### 2026-02-07 — Hybrid Search Implementation

**Objective:** Implement Dense + Sparse + RRF fusion search

**Architecture:**
```
Query
  │
  ├─► Dense Search (pgvector, top-50)
  │     ORDER BY embedding <=> query_emb
  │
  ├─► Sparse Search (tsvector, top-50)
  │     WHERE tsv_it @@ plainto_tsquery('italian', query)
  │
  └─► RRF Fusion (k=60)
        score = 1/(k+rank_dense) + 1/(k+rank_sparse)
```

**Scripts created:**
| Script | Purpose |
|--------|---------|
| `hybrid_search_staging.py` | Full hybrid search |
| `hybrid_test.py` | Test with source indicator (BOTH/DENSE/SPARSE) |

**RRF Formula:**
```
RRF(d) = Σ 1/(k + rank_i(d))

where:
  k = 60 (smoothing constant)
  rank_i = rank in retrieval method i
```

---

### 2026-02-07 — Evaluation Tests

**Objective:** Validate hybrid search across all codes

#### Test 1: Penale Query
```
Query: "sequestro preventivo beni"

Results:
Code  Art        Source RRF     Dense  Sparse
CPP   104-bis    BOTH   0.0293  0.606  0.0048
CPP   104-bis    BOTH   0.0273  0.577  0.0100
CPC   670        DENSE  0.0164  0.635  0.0000
CPP   104-bis    SPARSE 0.0161  0.000  0.0100
CPC   679        DENSE  0.0161  0.634  0.0000
```

**Analysis:** Cross-code retrieval works (CPP + CPC together)

#### Test 2: Constitutional Query
```
Query: "diritto di voto"

Results:
Code  Art        Source RRF     Dense  Sparse
CC    2351       BOTH   0.0320  0.543  0.4333
COST  48         BOTH   0.0318  0.637  0.1085
CC    2370       BOTH   0.0300  0.595  0.0534
```

**Analysis:** COST art. 48 (electoral rights) found at rank 2

#### Test 3: Civil Law Query
```
Query: "responsabilità extracontrattuale danno ingiusto"

Results:
Code  Art        Source RRF     Dense  Sparse
CC    1338       DENSE  0.0164  0.629  0.0000
CC    1218       DENSE  0.0161  0.626  0.0000
...
CC    2043       DENSE  0.0141  0.582  0.0000
```

**Analysis:** Art. 2043 CC (tort liability) found at rank 11

---

## Verification Summary

### Final Database State (2026-02-07 18:00 UTC)

```sql
-- Verified via SSH to staging
SELECT w.code,
       COUNT(DISTINCT n.id) as articoli,
       COUNT(DISTINCT c.id) as chunks,
       COUNT(DISTINCT e.chunk_id) as embeddings
FROM kb.work w
LEFT JOIN kb.normativa n ON n.work_id = w.id
LEFT JOIN kb.normativa_chunk c ON c.work_id = w.id
LEFT JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
GROUP BY w.code
ORDER BY COUNT(DISTINCT n.id) DESC;

 code  | articoli | chunks | embeddings
-------+----------+--------+------------
 CC    |     3221 |   4054 |       4054
 CPC   |     1055 |   2192 |       2192
 CP    |      962 |   1310 |       1310
 CPP   |      958 |   2494 |       2494
 COST  |      139 |    196 |        196
```

### Annotations (Brocardi Notes)

```sql
SELECT 'annotation' as tipo, COUNT(*) as total FROM kb.annotation
UNION ALL
SELECT 'annotation_embeddings', COUNT(*) FROM kb.annotation_embeddings;

            tipo            | total
----------------------------+-------
 annotation                 | 13281
 annotation_embeddings      | 13180
```

---

## Technical Specifications

### Embedding Model

| Property | Value |
|----------|-------|
| Model | text-embedding-3-small |
| Provider | OpenAI via OpenRouter |
| Dimensions | 1536 |
| Cost | ~$0.00002/1K tokens |

### PostgreSQL Extensions

| Extension | Version | Purpose |
|-----------|---------|---------|
| pgvector | 0.7.4 | Vector similarity |
| pg_trgm | - | Trigram matching |

### Index Configuration

```sql
-- Dense index (HNSW)
CREATE INDEX idx_chunk_emb_hnsw
ON kb.normativa_chunk_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Sparse index (GIN)
CREATE INDEX idx_chunk_fts_tsv
ON kb.normativa_chunk_fts
USING GIN(tsv_it);
```

---

## Links to Related Documents

| Document | Path |
|----------|------|
| KB Schema | [../SCHEMA_KB_OVERVIEW.md](../SCHEMA_KB_OVERVIEW.md) |
| Massimari KB | [../KB-HANDOFF.md](../KB-HANDOFF.md) |
| Norm Graph | [../KB-NORM-GRAPH-HANDOFF.md](../KB-NORM-GRAPH-HANDOFF.md) |
| CLAUDE.md | [../../CLAUDE.md](../../CLAUDE.md) |

---

## Next Steps

1. **API Endpoint** — Create FastAPI retrieval endpoint
2. **Reranking** — Add cross-encoder reranking layer
3. **Production Deploy** — Deploy to production server
4. **Remaining Codes** — Ingest 64 additional Altalex PDFs

---

*Generated: 2026-02-07 18:30 UTC*
*Author: LEXE Development Team*
