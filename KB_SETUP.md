# LEXE Knowledge Base - Setup Guide

## Overview

Knowledge Base per massimari giurisprudenziali della Cassazione con:
- **pgvector** - Vector similarity search (HNSW indexes)
- **Apache AGE** - Graph database per relazioni
- **pg_search (ParadeDB)** - BM25 full-text search
- **pg_trgm** - Fuzzy matching e typo catch

## Quick Start

### 1. Avvia Docker Desktop

Assicurati che Docker Desktop sia in esecuzione.

### 2. Build dell'immagine

```bash
cd lexe-api
docker build -f Dockerfile.kb -t lexe-kb:latest .
```

### 3. Avvia il database

```bash
# Con password di default (solo sviluppo!)
docker compose -f docker-compose.kb.yml up -d

# Oppure con password custom
LEXE_KB_PASSWORD=your_secure_password docker compose -f docker-compose.kb.yml up -d
```

### 4. Verifica

```bash
# Check container running
docker ps | grep lexe-kb

# Check logs
docker logs lexe-kb

# Connect e test
docker exec -it lexe-kb psql -U lexe_kb -d lexe_kb -c "SELECT extname, extversion FROM pg_extension;"
```

## Connessione

| Parametro | Valore |
|-----------|--------|
| Host | localhost |
| Port | 5434 |
| Database | lexe_kb |
| User | lexe_kb |
| Password | (vedi LEXE_KB_PASSWORD) |

```bash
# psql
psql -h localhost -p 5434 -U lexe_kb -d lexe_kb

# Connection string
postgresql://lexe_kb:password@localhost:5434/lexe_kb
```

## Schema

Il database usa lo schema `kb` con le seguenti tabelle:

| Tabella | Descrizione |
|---------|-------------|
| `kb.documents` | PDF sorgente con metriche OCR |
| `kb.sections` | Struttura gerarchica (parte/capitolo/sezione) |
| `kb.massime` | Massime atomiche (Chunk A + B) |
| `kb.embeddings` | Vettori multi-modello multi-canale |
| `kb.citations` | Citazioni e norme estratte |
| `kb.duplicates` | Tracking near-duplicates |
| `kb.edge_weights` | Pesi archi grafo |
| `kb.ingestion_jobs` | Job tracking |
| `kb.benchmark_runs` | Risultati benchmark |

## Embedding Models

Schema flessibile con partial HNSW indexes:

| Modello | Dimensioni | Channel |
|---------|-----------|---------|
| qwen3 | 1536 | testo, tema, contesto |
| e5-large | 1024 | testo, tema |
| bge-m3 | 1024 | testo, tema |
| legal-bert-it | 768 | testo |

## BM25 Search

Se `pg_search` è installato correttamente:

```sql
-- BM25 search diretto
SELECT m.id, paradedb.score(m.id) as score
FROM kb.massime m
WHERE m.id @@@ paradedb.search(
    query => paradedb.parse('competenza territorio'),
    index => 'massime_bm25_idx'
)
ORDER BY score DESC
LIMIT 10;

-- Funzione wrapper con fallback
SELECT * FROM kb.bm25_search('competenza territorio', 10);

-- Hybrid search (dense + sparse + RRF)
SELECT * FROM kb.hybrid_search(
    'competenza territorio',
    '[0.1, 0.2, ...]'::vector,  -- query embedding
    'qwen3',                     -- model
    'testo',                     -- channel
    20,                          -- limit
    60                           -- rrf_k
);
```

## Graph (Apache AGE)

```sql
-- Carica AGE
LOAD 'age';
SET search_path TO ag_catalog, kb, public;

-- Query Cypher
SELECT * FROM cypher('lexe_jurisprudence', $$
    MATCH (m:Massima)-[r:SAME_PRINCIPLE]->(m2:Massima)
    WHERE r.weight > 0.5
    RETURN m.id, m2.id, r.weight
    LIMIT 10
$$) as (m1 agtype, m2 agtype, weight agtype);
```

## Troubleshooting

### pg_search non funziona

Se vedi warning "pg_search extension not available", il sistema usa automaticamente il fallback tsvector:

```sql
-- Verifica mode
SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search');

-- Se false, le query usano tsvector
SELECT m.id, ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', 'competenza')) as score
FROM kb.massime m
WHERE m.tsv_italian @@ plainto_tsquery('italian', 'competenza')
ORDER BY score DESC;
```

### Rebuild indexes

```sql
-- HNSW rebuild (dopo molte insert)
REINDEX INDEX kb.idx_emb_qwen3_testo;

-- Vacuum per performance
VACUUM ANALYZE kb.massime;
VACUUM ANALYZE kb.embeddings;
```

### Reset database

```bash
# WARNING: cancella tutti i dati!
docker compose -f docker-compose.kb.yml down -v
docker compose -f docker-compose.kb.yml up -d
```

## Licensing

- **pgvector**: PostgreSQL License
- **Apache AGE**: Apache 2.0
- **pg_search (ParadeDB)**: AGPLv3 (solo uso interno)
- **pg_trgm**: PostgreSQL License

**IMPORTANTE**: pg_search ha licenza AGPLv3. Se LEXE diventa prodotto commerciale, valutare alternative (vedi BACKLOG P0).
