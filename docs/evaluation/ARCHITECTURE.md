# LEXE KB Normativa - System Architecture

> **Technical Architecture Documentation**
> Version: 1.0 | Date: 2026-02-07

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LEXE PLATFORM                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐             │
│  │  lexe-webchat  │    │  lexe-core     │    │ lexe-orchestrator│            │
│  │   (Frontend)   │───►│  (API Gateway) │───►│   (ORCHIDEA)    │            │
│  └────────────────┘    └────────────────┘    └───────┬────────┘             │
│                                                       │                      │
│                                                       ▼                      │
│                              ┌────────────────────────────────────┐          │
│                              │          LEXE KB LAYER             │          │
│                              │  ┌──────────────────────────────┐  │          │
│                              │  │     KB Normativa (this)      │  │          │
│                              │  │  ┌─────────┐  ┌──────────┐   │  │          │
│                              │  │  │ Dense   │  │ Sparse   │   │  │          │
│                              │  │  │(pgvector│  │(tsvector)│   │  │          │
│                              │  │  └────┬────┘  └────┬─────┘   │  │          │
│                              │  │       └─────┬──────┘         │  │          │
│                              │  │             ▼                │  │          │
│                              │  │        RRF Fusion            │  │          │
│                              │  └──────────────────────────────┘  │          │
│                              │  ┌──────────────────────────────┐  │          │
│                              │  │     KB Massimari             │  │          │
│                              │  │  (Case Law - 38K massime)    │  │          │
│                              │  └──────────────────────────────┘  │          │
│                              └────────────────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Architecture

### 2.1 Container Layout

| Container | Port | Database | Purpose |
|-----------|------|----------|---------|
| `lexe-postgres` | 5435 | lexe | Sistema (Logto, core, memory) |
| `lexe-max` | 5436 | lexe_max | **KB Legal** (normativa, massimari) |

### 2.2 Schema: `kb`

```sql
-- Work (Codici/Leggi)
CREATE TABLE kb.work (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,  -- 'CC', 'CP', 'CPC', 'CPP', 'COST'
    name TEXT,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Normativa (Articoli)
CREATE TABLE kb.normativa (
    id SERIAL PRIMARY KEY,
    work_id INTEGER REFERENCES kb.work(id),
    articolo VARCHAR(50) NOT NULL,
    rubrica TEXT,
    testo TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks (Semantic Units)
CREATE TABLE kb.normativa_chunk (
    id SERIAL PRIMARY KEY,
    normativa_id INTEGER REFERENCES kb.normativa(id),
    work_id INTEGER REFERENCES kb.work(id),
    chunk_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Embeddings (Dense Vectors)
CREATE TABLE kb.normativa_chunk_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES kb.normativa_chunk(id),
    model VARCHAR(100) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    dims INTEGER NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id, model, channel, dims)
);

-- FTS Index (Sparse Vectors)
CREATE TABLE kb.normativa_chunk_fts (
    chunk_id INTEGER PRIMARY KEY REFERENCES kb.normativa_chunk(id),
    tsv_it tsvector NOT NULL
);

-- Annotations (Brocardi Notes)
CREATE TABLE kb.annotation (
    id SERIAL PRIMARY KEY,
    normativa_id INTEGER REFERENCES kb.normativa(id),
    tipo VARCHAR(50),
    testo TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE kb.annotation_embeddings (
    id SERIAL PRIMARY KEY,
    annotation_id INTEGER REFERENCES kb.annotation(id),
    model VARCHAR(100),
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2.3 Indexes

```sql
-- HNSW index for dense search
CREATE INDEX idx_chunk_emb_hnsw
ON kb.normativa_chunk_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for sparse search
CREATE INDEX idx_chunk_fts_tsv
ON kb.normativa_chunk_fts
USING GIN(tsv_it);

-- B-tree indexes for joins
CREATE INDEX idx_chunk_normativa ON kb.normativa_chunk(normativa_id);
CREATE INDEX idx_chunk_work ON kb.normativa_chunk(work_id);
CREATE INDEX idx_normativa_work ON kb.normativa(work_id);
```

---

## 3. Data Pipeline

### 3.1 Ingestion Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │     │   Parse     │     │   Chunk     │     │   Embed     │
│  (Brocardi) │────►│  Articles   │────►│  (~1000ch)  │────►│  (OpenAI)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Store     │◄────│   Index     │◄────│   FTS       │◄────│  tsvector   │
│  PostgreSQL │     │   HNSW      │     │   GIN       │     │  (italian)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 3.2 Pipeline Scripts

| Step | Script | Purpose |
|------|--------|---------|
| 1. Ingest | `ingest_staging.py` | Parse articles from Brocardi |
| 2. Chunk | `chunk_remote.py` | Create ~1000 char chunks |
| 3. Embed | `embed_lexe_max_staging.py` | Generate embeddings |
| 4. FTS | `create_fts_index.sql` | Build tsvector index |

---

## 4. Retrieval Architecture

### 4.1 Hybrid Search Flow

```
                        ┌──────────────────────────────────────┐
                        │            User Query                 │
                        │  "responsabilità extracontrattuale"  │
                        └──────────────────┬───────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            │
   ┌────────────────────┐       ┌────────────────────┐                  │
   │   OpenRouter API   │       │   PostgreSQL FTS   │                  │
   │  text-embedding-3  │       │  plainto_tsquery   │                  │
   │      -small        │       │    ('italian')     │                  │
   └─────────┬──────────┘       └─────────┬──────────┘                  │
             │                            │                             │
             ▼                            ▼                             │
   ┌────────────────────┐       ┌────────────────────┐                  │
   │   Dense Search     │       │   Sparse Search    │                  │
   │   pgvector HNSW    │       │   GIN tsvector     │                  │
   │   ORDER BY <=>     │       │   WHERE @@ query   │                  │
   │   LIMIT 50         │       │   LIMIT 50         │                  │
   └─────────┬──────────┘       └─────────┬──────────┘                  │
             │                            │                             │
             │         rank_dense         │         rank_sparse         │
             │                            │                             │
             └────────────┬───────────────┘                             │
                          │                                             │
                          ▼                                             │
               ┌────────────────────┐                                   │
               │     RRF Fusion     │                                   │
               │                    │                                   │
               │  score = Σ 1/(k+r) │                                   │
               │      k = 60        │                                   │
               └─────────┬──────────┘                                   │
                         │                                              │
                         ▼                                              │
               ┌────────────────────┐                                   │
               │   Final Results    │                                   │
               │   code, article,   │                                   │
               │   chunk, text      │                                   │
               └────────────────────┘                                   │
```

### 4.2 Distance Metrics

| Search Type | Metric | Operator |
|-------------|--------|----------|
| Dense | Cosine Distance | `<=>` |
| Sparse | TS Rank CD | `ts_rank_cd()` |

### 4.3 RRF Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k` | 60 | Smoothing constant |
| `top_dense` | 50 | Dense candidates |
| `top_sparse` | 50 | Sparse candidates |
| `final_k` | 15 | Final results |

---

## 5. API Endpoints (Planned)

### 5.1 Retrieval API

```
POST /api/v1/kb/normativa/search
Content-Type: application/json

{
  "query": "responsabilità extracontrattuale",
  "top_k": 10,
  "codes": ["CC", "CP"],  // optional filter
  "mode": "hybrid"        // hybrid | dense | sparse
}
```

**Response:**

```json
{
  "results": [
    {
      "code": "CC",
      "article": "2043",
      "chunk_no": 0,
      "source": "BOTH",
      "rrf_score": 0.0293,
      "dense_score": 0.606,
      "sparse_score": 0.0048,
      "text": "Qualunque fatto doloso o colposo..."
    }
  ],
  "query_time_ms": 45,
  "total_chunks": 10246
}
```

---

## 6. ORCHIDEA Integration

### 6.1 Tool Definition

```python
@tool("kb_normativa_search")
async def search_normativa(
    query: str,
    top_k: int = 5,
    codes: list[str] | None = None
) -> list[NormativaResult]:
    """
    Search Italian legal codes using hybrid retrieval.

    Args:
        query: Natural language legal query
        top_k: Number of results to return
        codes: Optional filter by code (CC, CP, CPC, CPP, COST)

    Returns:
        List of relevant article chunks with metadata
    """
    results = await hybrid_search(query, top_k, codes)
    return [
        NormativaResult(
            code=r.code,
            article=r.article,
            text=r.text,
            relevance=r.rrf_score
        )
        for r in results
    ]
```

### 6.2 Context Injection

```python
# In ORCHIDEA pipeline
legal_context = await kb_normativa_search(
    query=user_question,
    top_k=5
)

prompt = f"""
Based on the following legal articles:

{format_context(legal_context)}

Answer the user's question: {user_question}
"""
```

---

## 7. Performance Characteristics

### 7.1 Query Latency

| Component | Latency (p50) | Latency (p99) |
|-----------|---------------|---------------|
| Embedding API | ~200ms | ~500ms |
| Dense Search | ~10ms | ~50ms |
| Sparse Search | ~5ms | ~20ms |
| RRF Fusion | ~1ms | ~5ms |
| **Total** | ~220ms | ~600ms |

### 7.2 Resource Usage

| Resource | Value |
|----------|-------|
| Embeddings Storage | ~60MB (10K × 1536 × 4 bytes) |
| FTS Index | ~20MB |
| Total DB Size | ~150MB |

---

## 8. Deployment

### 8.1 Staging Environment

```bash
# SSH to staging
ssh -i ~/.ssh/id_stage_new root@91.99.229.111

# Access KB database
docker exec -it lexe-max psql -U lexe_max -d lexe_max

# Check status
SELECT COUNT(*) FROM kb.normativa_chunk;
SELECT COUNT(*) FROM kb.normativa_chunk_embeddings;
```

### 8.2 Local Development

```bash
# Create SSH tunnel
ssh -i ~/.ssh/id_stage_new -L 5436:localhost:5436 root@91.99.229.111

# Connect locally
psql -h localhost -p 5436 -U lexe_max -d lexe_max
```

---

## 9. Future Enhancements

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| Reranking | High | Cross-encoder for top-20 results |
| Query Expansion | Medium | Legal synonym expansion |
| Norm Boosting | Medium | Boost canonical articles |
| Caching | Low | Redis cache for frequent queries |

---

*Architecture Document v1.0*
*Generated: 2026-02-07*
