# LEXE Knowledge Base - Schema PostgreSQL + pgvector

> Versione: 2026-02-06
> Database: PostgreSQL 17 + pgvector 0.7.4 + Apache AGE

---

## Indice

1. [Architettura Generale](#architettura-generale)
2. [Extensions](#extensions)
3. [Schema Core - Massimari](#schema-core---massimari)
4. [Schema Graph](#schema-graph)
5. [Schema Normativa](#schema-normativa)
6. [Schema Brocardi/Dizionario](#schema-brocardidizionario)
7. [Evoluzione e Motivazioni](#evoluzione-e-motivazioni)

---

## Architettura Generale

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEXE KNOWLEDGE BASE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   MASSIMARI  │  │    GRAPH     │  │  NORMATIVA   │  │  GLOSSARIO   │ │
│  │  (38k docs)  │  │  (58k edges) │  │  (69 codici) │  │  (brocardi)  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │          │
│         └─────────────────┼─────────────────┼─────────────────┘          │
│                           │                 │                            │
│  ┌────────────────────────┴─────────────────┴────────────────────────┐  │
│  │                      EMBEDDINGS LAYER                              │  │
│  │  pgvector HNSW (1536, 1024, 768 dims) + FTS + pg_trgm              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      RETRIEVAL LAYER                               │  │
│  │  Hybrid Search (Dense + Sparse + RRF) + Graph Expansion            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Extensions

```sql
-- 001_init_extensions.sql

CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector per similarity search
CREATE EXTENSION IF NOT EXISTS age;           -- Apache AGE per graph database
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- Fuzzy matching e typo catch
CREATE EXTENSION IF NOT EXISTS pg_search;     -- ParadeDB BM25 (opzionale)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- UUID generation
CREATE EXTENSION IF NOT EXISTS unaccent;      -- Ricerca senza accenti
```

**Motivazione**: Stack completo per RAG legale con vector search (dense), BM25 (sparse), fuzzy matching (typo tolerance), e graph traversal.

---

## Schema Core - Massimari

### Tabelle Principali

```sql
-- kb.documents: PDF sorgente dei massimari
CREATE TABLE kb.documents (
    id UUID PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_hash VARCHAR(64) UNIQUE,     -- SHA256 per dedup
    anno INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    tipo VARCHAR(10),                   -- 'civile' | 'penale'

    -- OCR Quality metrics
    ocr_quality_score FLOAT,
    ocr_valid_chars_ratio FLOAT,
    ocr_italian_tokens_ratio FLOAT
);

-- kb.massime: Unità atomica di retrieval (38,718 attive)
CREATE TABLE kb.massime (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES kb.documents(id),

    -- Contenuto
    testo TEXT NOT NULL,                -- Chunk A: Massima pulita
    testo_con_contesto TEXT,            -- Chunk B: Con OCR attigui
    testo_normalizzato TEXT NOT NULL,   -- Per dedup e trgm
    content_hash VARCHAR(64) NOT NULL,

    -- Citazione normalizzata
    sezione VARCHAR(20),                -- "Sez. U", "Sez. 1"
    numero VARCHAR(20),
    anno INTEGER,
    data_decisione DATE,
    rv VARCHAR(30),                     -- "Rv. 123456-01"

    -- FTS (fallback se pg_search non disponibile)
    tsv_simple TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', testo)) STORED,
    tsv_italian TSVECTOR GENERATED ALWAYS AS (to_tsvector('italian', testo)) STORED
);

-- kb.embeddings: Multi-modello, multi-canale
CREATE TABLE kb.embeddings (
    id UUID PRIMARY KEY,
    massima_id UUID REFERENCES kb.massime(id),
    model VARCHAR(50) NOT NULL,         -- 'qwen3', 'e5-large', 'bge-m3'
    channel VARCHAR(20) NOT NULL,       -- 'testo', 'tema', 'contesto'
    embedding vector NOT NULL,
    dims INTEGER NOT NULL               -- 1536, 1024, 768
);
```

### HNSW Indexes per Modello

```sql
-- Qwen3 (1536 dims)
CREATE INDEX idx_emb_qwen3_testo ON kb.embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE model = 'qwen3' AND channel = 'testo';

-- E5-Large (1024 dims)
CREATE INDEX idx_emb_e5_testo ON kb.embeddings
USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE model = 'e5-large' AND channel = 'testo';

-- Legal-BERT-IT (768 dims)
CREATE INDEX idx_emb_legal_testo ON kb.embeddings
USING hnsw ((embedding::vector(768)) vector_cosine_ops)
WHERE model = 'legal-bert-it' AND channel = 'testo';
```

**Motivazione**: Partial indexes permettono di usare modelli diversi per casi d'uso diversi (e5-large per italiano, legal-bert per dominio specifico) senza duplicare dati.

---

## Schema Graph

### Citation Graph (58,737 edges)

```sql
-- kb.graph_runs: Versioning per idempotency
CREATE TABLE kb.graph_runs (
    id SERIAL PRIMARY KEY,
    run_type VARCHAR(50) NOT NULL,      -- 'citation_extraction', 'norm_extraction'
    status VARCHAR(20) DEFAULT 'running',
    is_active BOOLEAN DEFAULT FALSE,    -- Solo 1 run attivo per tipo
    metrics JSONB DEFAULT '{}'          -- {resolution_rate, edge_count}
);

-- kb.graph_edges: Edges SQL per low-latency reranking
CREATE TABLE kb.graph_edges (
    id SERIAL PRIMARY KEY,
    source_id UUID REFERENCES kb.massime(id),
    target_id UUID REFERENCES kb.massime(id),
    edge_type VARCHAR(30) NOT NULL,     -- 'CITES'
    relation_subtype VARCHAR(30),       -- 'CONFIRMS', 'DISTINGUISHES', 'OVERRULES'

    -- Scoring
    confidence FLOAT DEFAULT 1.0,       -- Quanto siamo sicuri del match
    weight FLOAT DEFAULT 1.0,           -- Per pruning (keep if >= 0.6)

    -- Evidence trail
    evidence JSONB DEFAULT '{}',        -- {pattern, resolver, raw_span}
    context_span TEXT,
    run_id INTEGER REFERENCES kb.graph_runs(id)
);

-- Partial index per retrieval (solo edges validi)
CREATE INDEX idx_graph_edges_weight_valid
ON kb.graph_edges(weight DESC) WHERE weight >= 0.6;
```

### Norm Graph (42,338 edges, 4,128 norme uniche)

```sql
-- kb.norms: Norme canonicalizzate
CREATE TABLE kb.norms (
    id TEXT PRIMARY KEY,                -- "CC:2043", "LEGGE:241:1990"
    code TEXT NOT NULL,                 -- CC, CPC, CP, LEGGE, DLGS
    article TEXT,                       -- Per codici: 2043, 360
    suffix TEXT,                        -- bis, ter, quater
    number TEXT,                        -- Per leggi: 241, 50
    year INT,                           -- 1990, 2016
    full_ref TEXT NOT NULL,             -- "art. 2043 c.c."
    citation_count INT DEFAULT 0
);

-- kb.massima_norms: Edges massima -> norma
CREATE TABLE kb.massima_norms (
    massima_id UUID REFERENCES kb.massime(id),
    norm_id TEXT REFERENCES kb.norms(id),
    context_span TEXT,
    PRIMARY KEY (massima_id, norm_id)
);
```

**Motivazione**: Il Norm Graph permette query dirette tipo "trova tutte le massime che citano art. 2043 c.c." senza embedding search, con O(1) lookup.

### Categories e Turning Points

```sql
-- kb.categories: 6 macro L1 + gerarchiche L2/L3
CREATE TABLE kb.categories (
    id VARCHAR(50) PRIMARY KEY,         -- 'CIVILE', 'PENALE'
    name VARCHAR(100) NOT NULL,
    level INTEGER CHECK (level IN (1, 2, 3)),
    parent_id VARCHAR(50) REFERENCES kb.categories(id),
    keywords TEXT[] DEFAULT '{}',
    centroid vector(1536)               -- Media embeddings massime seed
);

-- kb.turning_points: Sezioni Unite e mutamenti
CREATE TABLE kb.turning_points (
    id SERIAL PRIMARY KEY,
    massima_id UUID REFERENCES kb.massime(id),
    overruled_massima_id UUID REFERENCES kb.massime(id),
    turning_point_type VARCHAR(30),     -- 'SEZ_UNITE', 'CONTRASTO_RISOLTO'
    is_turning_point BOOLEAN DEFAULT TRUE,
    rationale_span TEXT,
    confidence FLOAT
);
```

---

## Schema Normativa

### Articoli di Codice (URN:NIR)

```sql
-- kb.normativa: Articoli con cross-validation
CREATE TABLE kb.normativa (
    id UUID PRIMARY KEY,

    -- Identificazione URN:NIR standard
    urn_nir VARCHAR(200) UNIQUE,        -- urn:nir:stato:legge:1942-03-16;262:art2043
    codice VARCHAR(50) NOT NULL,        -- 'CC', 'CP', 'CPC'
    articolo VARCHAR(20) NOT NULL,      -- '2043', '360-bis'
    comma VARCHAR(10),

    -- Gerarchia
    libro VARCHAR(150),
    titolo VARCHAR(150),
    capo VARCHAR(150),
    sezione VARCHAR(150),

    -- Contenuto
    rubrica TEXT,                       -- "Risarcimento per fatto illecito"
    testo TEXT NOT NULL,
    testo_normalizzato TEXT,

    -- Cross-validation (cintura e bretelle)
    canonical_source VARCHAR(50),       -- 'normattiva' | 'gazzetta'
    canonical_hash VARCHAR(64),
    mirror_source VARCHAR(50),          -- 'studiocataldi' | 'brocardi'
    mirror_hash VARCHAR(64),
    validation_status VARCHAR(20),      -- 'pending', 'verified', 'content_diff'

    -- Multivigenza
    data_vigenza_da DATE,
    data_vigenza_a DATE,
    is_current BOOLEAN DEFAULT TRUE,
    nota_modifica TEXT,
    previous_version_id UUID REFERENCES kb.normativa(id)
);
```

### Altalex PDF Extension

```sql
-- kb.normativa_altalex: Estensione per PDF Altalex
ALTER TABLE kb.normativa_altalex ADD COLUMN
    articolo_num_norm INTEGER,          -- 2043 per "2043-bis"
    articolo_suffix TEXT,               -- 'bis', 'ter'
    articolo_sort_key TEXT,             -- '002043.bis' per sort
    global_key TEXT UNIQUE,             -- 'altalex:cc:2043:bis'
    testo_context TEXT,                 -- Overlap ±200 chars
    commi JSONB,                        -- [{num: 1, testo: "..."}]
    riferimenti_parsed JSONB,           -- ["CC:1218", "CC:2059"]
    testo_tsv tsvector;                 -- FTS con unaccent

-- Multi-dim embeddings
CREATE TABLE kb.altalex_embeddings (
    altalex_id UUID REFERENCES kb.normativa_altalex(id),
    model TEXT NOT NULL,
    channel TEXT CHECK (channel IN ('testo', 'rubrica', 'contesto')),
    dims INTEGER CHECK (dims IN (384, 768, 1024, 1536)),
    embedding vector NOT NULL
);
```

### Number-Anchored Knowledge Graph

```sql
-- kb.legal_numbers: Master index numeri legali
CREATE TABLE kb.legal_numbers (
    id UUID PRIMARY KEY,
    canonical_id VARCHAR(100) UNIQUE,   -- "CC:2043", "L:241:1990", "CASS:12345:2020"
    number_type VARCHAR(20),            -- 'article', 'law', 'sentence'
    codice VARCHAR(30),
    numero VARCHAR(50),
    anno INTEGER,
    citation_count INTEGER DEFAULT 0
);

-- kb.number_citations: Chi cita cosa
CREATE TABLE kb.number_citations (
    source_type VARCHAR(20),            -- 'massima', 'normativa', 'sentenza'
    source_id UUID,
    target_number_id UUID REFERENCES kb.legal_numbers(id),
    target_canonical VARCHAR(100),      -- Denormalizzato per performance
    confidence FLOAT DEFAULT 1.0
);
```

**Motivazione**: Il Number-Anchored Knowledge Graph permette traversal cross-dominio: "Trova tutte le massime, articoli, e sentenze che citano art. 2043 c.c."

---

## Schema Brocardi/Dizionario

```sql
-- kb.brocardi: Massime latine
CREATE TABLE kb.brocardi (
    id UUID PRIMARY KEY,
    latino TEXT NOT NULL,               -- "Ad impossibilia nemo tenetur"
    italiano TEXT,                      -- "Nessuno è tenuto all'impossibile"
    significato TEXT,                   -- Spiegazione operativa
    tags TEXT[],
    categoria VARCHAR(50),              -- 'principio', 'massima', 'adagio'
    area_diritto VARCHAR(50)            -- 'civile', 'penale'
);

-- kb.dizionario: Vocabolario giuridico
CREATE TABLE kb.dizionario (
    id UUID PRIMARY KEY,
    voce VARCHAR(200) NOT NULL,         -- "Capacità giuridica"
    voce_normalizzata VARCHAR(200),     -- Per search
    definizione TEXT NOT NULL,
    definizione_breve TEXT,             -- One-liner
    sinonimi TEXT[],
    area VARCHAR(50),
    livello VARCHAR(20)                 -- 'base', 'tecnico'
);

-- Vista unificata per ricerca
CREATE VIEW kb.legal_glossary AS
SELECT 'brocardi' as tipo, id, latino as termine, significato as definizione
FROM kb.brocardi
UNION ALL
SELECT 'dizionario' as tipo, id, voce as termine, definizione
FROM kb.dizionario;
```

---

## Evoluzione e Motivazioni

### Fase 1: Core Massimari (v1.0)

| Decisione | Motivazione |
|-----------|-------------|
| `content_hash` per dedup | Massimari hanno duplicati cross-volume (stessa massima in anni diversi) |
| `testo_normalizzato` | Permette confronto fuzzy ignorando spazi/maiuscole |
| Multi-model embeddings | Flessibilità: qwen3 per accuracy, e5 per velocità |
| Partial HNSW indexes | Evita rebuild completo quando si aggiunge modello |

### Fase 2: Hybrid Search (v2.0)

| Decisione | Motivazione |
|-----------|-------------|
| BM25 via pg_search | Keyword esatte ("art. 2043") battono semantic search |
| FTS fallback (tsvector) | pg_search è opzionale, serve fallback |
| RRF fusion | Combina dense + sparse senza tuning manuale |
| pg_trgm | Typo tolerance: "resposabilità" trova "responsabilità" |

### Fase 3: Citation Graph (v3.0)

| Decisione | Motivazione |
|-----------|-------------|
| Dual-write SQL + AGE | SQL per low-latency lookup, AGE per traversal complesso |
| `graph_runs` versioning | Idempotency: re-run non duplica edges |
| `is_active` flag | Cache invalidation quando cambia run attivo |
| `weight >= 0.6` pruning | Rimuove edges rumorosi da graph expansion |
| `evidence` JSONB | Debug trail: quale pattern ha matchato |

### Fase 4: Norm Graph (v3.3)

| Decisione | Motivazione |
|-----------|-------------|
| Norme canonicalizzate | "art. 2043 c.c." = "CC:2043" = stesso nodo |
| `citation_count` | Hot norms per ranking (art. 2043 > art. 2044) |
| Resolver cascade | rv_exact → sez_num_anno → rv_text_fallback |
| 60.3% coverage | 23,365 massime con almeno 1 norma estratta |

### Fase 5: Normativa Altalex (v4.0)

| Decisione | Motivazione |
|-----------|-------------|
| URN:NIR standard | Interoperabilità con Normattiva |
| Cross-validation | 4 fonti (PDF, Cataldi, Brocardi, Normattiva) per accuracy |
| `articolo_sort_key` | Sort naturale: 1, 2, ..., 10, 11 (non 1, 10, 11, 2) |
| `global_key` | Lookup O(1) cross-documento |
| `testo_context` overlap | Migliora retrieval con context window |
| `commi` JSONB | Granularità comma per citazioni precise |
| Latin suffix support | bis, ter, quater... fino a octiesdecies (18°) |

### Fase 6: Legal Knowledge Graph (v5.0)

| Decisione | Motivazione |
|-----------|-------------|
| `legal_numbers` master | Single source of truth per tutti i numeri |
| Cross-domain edges | massima→norma, norma→norma, sentenza→norma |
| `citation_count` stats | Ranking basato su importanza effettiva |
| Materialized view | `top_cited_numbers` per dashboard |

---

## Funzioni Chiave

```sql
-- Hybrid search con RRF
kb.hybrid_search(query, embedding, model, channel, limit, rrf_k)

-- BM25 con fallback FTS
kb.bm25_search(query, limit)

-- Graph neighbors
kb.get_neighbors(massima_id, depth, min_weight)

-- Norm citation counts
kb.recompute_norm_citation_counts()

-- Legal number stats
kb.refresh_legal_number_stats()
```

---

## Metriche Attuali

| Metrica | Valore |
|---------|--------|
| Massime attive | 38,718 |
| Embeddings | 41,437 |
| Citation Graph edges | 58,737 |
| Norm Graph edges | 42,338 |
| Norme uniche | 4,128 |
| Norm coverage | 60.3% |
| Recall@10 | 97.5% |
| MRR | 0.756 |

---

*Ultimo aggiornamento: 2026-02-06*
