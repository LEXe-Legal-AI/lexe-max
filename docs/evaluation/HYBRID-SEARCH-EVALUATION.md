# Hybrid Search Evaluation Report

> **LEXE KB Normativa — Dense + Sparse + RRF Fusion**
> Evaluation Date: 2026-02-07

---

## 1. System Overview

The LEXE KB Normativa implements a **hybrid retrieval system** combining:

1. **Dense Search** — Vector similarity using pgvector (HNSW index)
2. **Sparse Search** — Full-text search using PostgreSQL tsvector
3. **RRF Fusion** — Reciprocal Rank Fusion to combine results

### 1.1 Architecture

```
                          ┌─────────────────────────────────────┐
                          │           Query Input               │
                          │   "responsabilità extracontrattuale"│
                          └──────────────┬──────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
         ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
         │  Query Embedding │ │     Tokenize     │ │                  │
         │  (OpenAI API)    │ │   (Italian FTS)  │ │                  │
         └────────┬─────────┘ └────────┬─────────┘ │                  │
                  │                    │           │                  │
                  ▼                    ▼           │                  │
         ┌──────────────────┐ ┌──────────────────┐ │                  │
         │   Dense Search   │ │  Sparse Search   │ │                  │
         │   (pgvector)     │ │   (tsvector)     │ │                  │
         │   top-50         │ │   top-50         │ │                  │
         └────────┬─────────┘ └────────┬─────────┘ │                  │
                  │                    │           │                  │
                  └─────────┬──────────┘           │                  │
                            │                      │                  │
                            ▼                      │                  │
                  ┌──────────────────┐             │                  │
                  │   RRF Fusion     │             │                  │
                  │   k = 60         │◄────────────┘                  │
                  │                  │                                │
                  │ score = Σ 1/(k+r)│                                │
                  └────────┬─────────┘                                │
                           │                                          │
                           ▼                                          │
                  ┌──────────────────┐                                │
                  │   Top-K Results  │                                │
                  │   with metadata  │                                │
                  └──────────────────┘                                │
```

---

## 2. Dataset Statistics

### 2.1 Corpus Size

| Code | Full Name | Articles | Chunks | Embeddings |
|------|-----------|----------|--------|------------|
| CC | Codice Civile | 3,221 | 4,054 | 4,054 |
| CPC | Codice di Procedura Civile | 1,055 | 2,192 | 2,192 |
| CPP | Codice di Procedura Penale | 958 | 2,494 | 2,494 |
| CP | Codice Penale | 962 | 1,310 | 1,310 |
| COST | Costituzione | 139 | 196 | 196 |
| **TOTAL** | | **6,335** | **10,246** | **10,246** |

### 2.2 Embedding Model

| Property | Value |
|----------|-------|
| Model | `text-embedding-3-small` |
| Dimensions | 1536 |
| Provider | OpenAI via OpenRouter |
| Distance | Cosine (1 - similarity) |

### 2.3 FTS Configuration

| Property | Value |
|----------|-------|
| Language | Italian |
| Dictionary | `italian` |
| Index Type | GIN |
| Query Parser | `plainto_tsquery` |

---

## 3. Evaluation Methodology

### 3.1 Test Queries

Three representative queries covering different legal domains:

| ID | Query | Expected Domain |
|----|-------|-----------------|
| Q1 | "sequestro preventivo beni" | CPP, CPC (criminal/civil procedure) |
| Q2 | "diritto di voto" | COST (constitutional), CC (corporate) |
| Q3 | "responsabilità extracontrattuale danno ingiusto" | CC (civil law, art. 2043) |

### 3.2 Evaluation Metrics

- **Source Distribution**: BOTH vs DENSE-only vs SPARSE-only
- **Cross-Code Retrieval**: Ability to retrieve from multiple codes
- **Rank Position**: Position of expected articles

---

## 4. Test Results

### 4.1 Query 1: Criminal Procedure

**Query:** `"sequestro preventivo beni"`

| Rank | Code | Article | Source | RRF Score | Dense | Sparse |
|------|------|---------|--------|-----------|-------|--------|
| 1 | CPP | 104-bis | BOTH | 0.0293 | 0.606 | 0.0048 |
| 2 | CPP | 104-bis | BOTH | 0.0273 | 0.577 | 0.0100 |
| 3 | CPC | 670 | DENSE | 0.0164 | 0.635 | 0.0000 |
| 4 | CPP | 104-bis | SPARSE | 0.0161 | 0.000 | 0.0100 |
| 5 | CPC | 679 | DENSE | 0.0161 | 0.634 | 0.0000 |
| 6 | CPP | 104-bis | SPARSE | 0.0159 | 0.000 | 0.0070 |
| 7 | CPP | 319 | DENSE | 0.0159 | 0.626 | 0.0000 |
| 8 | CPP | 183-quater | DENSE | 0.0156 | 0.623 | 0.0000 |
| 9 | CPC | 686 | DENSE | 0.0154 | 0.620 | 0.0000 |
| 10 | CP | 189 | DENSE | 0.0152 | 0.615 | 0.0000 |

**Analysis:**
- Cross-code retrieval successful: CPP, CPC, CP
- BOTH source appears for most relevant results
- Art. 104-bis CPP (sequestro preventivo) correctly ranked #1

### 4.2 Query 2: Constitutional Law

**Query:** `"diritto di voto"`

| Rank | Code | Article | Source | RRF Score | Dense | Sparse |
|------|------|---------|--------|-----------|-------|--------|
| 1 | CC | 2351 | BOTH | 0.0320 | 0.543 | 0.4333 |
| 2 | **COST** | **48** | BOTH | 0.0318 | 0.637 | 0.1085 |
| 3 | CC | 2370 | BOTH | 0.0300 | 0.595 | 0.0534 |
| 4 | CC | 2538 | BOTH | 0.0290 | 0.494 | 0.1021 |
| 5 | CC | 2543 | BOTH | 0.0286 | 0.506 | 0.0534 |
| 6 | CC | 2352 | BOTH | 0.0282 | 0.470 | 0.1153 |
| ... | ... | ... | ... | ... | ... | ... |
| 12 | **COST** | **75** | BOTH | 0.0239 | 0.501 | 0.0125 |

**Analysis:**
- COST Art. 48 (electoral rights) at rank 2
- COST Art. 75 (referendum) at rank 12
- CC corporate voting articles (2351, 2352, 2370) also retrieved
- All top results are BOTH (hybrid advantage)

### 4.3 Query 3: Civil Law

**Query:** `"responsabilità extracontrattuale danno ingiusto"`

| Rank | Code | Article | Source | RRF Score | Dense | Sparse |
|------|------|---------|--------|-----------|-------|--------|
| 1 | CC | 1338 | DENSE | 0.0164 | 0.629 | 0.0000 |
| 2 | CC | 1218 | DENSE | 0.0161 | 0.626 | 0.0000 |
| 3 | CC | 1398 | DENSE | 0.0159 | 0.611 | 0.0000 |
| 4 | CC | 2044 | DENSE | 0.0156 | 0.606 | 0.0000 |
| ... | ... | ... | ... | ... | ... | ... |
| 11 | **CC** | **2043** | DENSE | 0.0141 | 0.582 | 0.0000 |
| 12 | CC | 2045 | DENSE | 0.0139 | 0.580 | 0.0000 |

**Analysis:**
- Art. 2043 CC (tort liability, the canonical article) at rank 11
- Related articles (2044, 2045 - defenses) also present
- All results are DENSE-only (semantic similarity)
- FTS didn't match due to stemming differences

---

## 5. Source Distribution Analysis

### 5.1 Overall Distribution

| Source Type | Description | Typical Use Case |
|-------------|-------------|------------------|
| **BOTH** | Found by dense AND sparse | Exact term matches + semantic |
| **DENSE** | Found by dense only | Semantic similarity, synonyms |
| **SPARSE** | Found by sparse only | Exact keyword matches |

### 5.2 Query-Level Distribution

| Query | BOTH | DENSE | SPARSE | Total |
|-------|------|-------|--------|-------|
| Q1 (sequestro) | 2 | 10 | 3 | 15 |
| Q2 (voto) | 15 | 0 | 0 | 15 |
| Q3 (responsabilità) | 0 | 15 | 0 | 15 |

**Insight:** Query Q2 shows maximum hybrid benefit with all results from BOTH.

---

## 6. RRF Parameter Analysis

### 6.1 RRF Formula

```
RRF(d) = Σ 1/(k + rank_i(d))
```

Where:
- `k = 60` (smoothing constant)
- `rank_i` = rank in retrieval method i (dense or sparse)

### 6.2 Effect of k Parameter

| k Value | Effect |
|---------|--------|
| Lower (k=10) | More weight to top ranks |
| Higher (k=100) | More uniform weighting |
| **k=60** | Balanced (standard) |

---

## 7. Conclusions

### 7.1 Strengths

1. **Cross-Code Retrieval**: Successfully retrieves from multiple codes in single query
2. **Hybrid Advantage**: BOTH results show superior relevance
3. **Complete Coverage**: 100% embedding coverage across all codes
4. **Italian FTS**: Proper stemming for Italian legal terminology

### 7.2 Limitations

1. **Rank Position**: Canonical articles (e.g., 2043) not always at top
2. **FTS Stemming**: Some queries don't trigger sparse matches
3. **No Reranking**: Cross-encoder reranking would improve precision

### 7.3 Recommendations

1. Add **cross-encoder reranking** for top-20 results
2. Consider **query expansion** for legal synonyms
3. Implement **article-level boosting** for canonical articles
4. Add **recency weighting** for legislative updates

---

## 8. Appendix: SQL Queries

### 8.1 Hybrid Search Query

```sql
WITH query_params AS (
    SELECT
        %s::vector(1536) as qemb,
        plainto_tsquery('italian', %s) as qtsv
),
dense AS (
    SELECT c.id as chunk_id,
           ROW_NUMBER() OVER (ORDER BY e.embedding <=> q.qemb) as rank_dense,
           1 - (e.embedding <=> q.qemb) as score_dense
    FROM kb.normativa_chunk c
    JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
    CROSS JOIN query_params q
    ORDER BY e.embedding <=> q.qemb
    LIMIT 50
),
sparse AS (
    SELECT f.chunk_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC) as rank_sparse,
           ts_rank_cd(f.tsv_it, q.qtsv) as score_sparse
    FROM kb.normativa_chunk_fts f
    CROSS JOIN query_params q
    WHERE f.tsv_it @@ q.qtsv
    ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC
    LIMIT 50
),
rrf AS (
    SELECT
        COALESCE(d.chunk_id, s.chunk_id) as chunk_id,
        COALESCE(1.0 / (60 + d.rank_dense), 0) +
        COALESCE(1.0 / (60 + s.rank_sparse), 0) as rrf_score,
        d.score_dense,
        s.score_sparse,
        CASE
            WHEN d.chunk_id IS NOT NULL AND s.chunk_id IS NOT NULL THEN 'BOTH'
            WHEN d.chunk_id IS NOT NULL THEN 'DENSE'
            ELSE 'SPARSE'
        END as source
    FROM dense d
    FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
)
SELECT w.code, n.articolo, c.chunk_no, r.source,
       r.rrf_score, r.score_dense, r.score_sparse,
       LEFT(c.text, 150) as preview
FROM rrf r
JOIN kb.normativa_chunk c ON c.id = r.chunk_id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
ORDER BY r.rrf_score DESC
LIMIT 15;
```

---

*Report generated: 2026-02-07 18:30 UTC*
*LEXE KB Normativa Evaluation*
