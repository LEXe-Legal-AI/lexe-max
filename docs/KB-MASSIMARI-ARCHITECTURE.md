# KB Massimari - Architettura Sistema

> Knowledge Base per Massimari Giurisprudenziali della Corte di Cassazione
>
> Data: 2026-01-27 | Status: **MVP Funzionante**

---

## Overview

Il modulo KB Massimari è un sistema di ingestion e retrieval per i massimari della Corte di Cassazione italiana. Supporta:

- **Ingestion** di PDF massimari (civili e penali)
- **Estrazione** automatica di massime con pattern matching
- **Retrieval ibrido**: FTS (BM25) + Vector Search (HNSW) + Graph (Cypher)
- **Multi-model embeddings**: Qwen3, E5-Large, BGE-M3, Legal-BERT-IT

---

## Stack Tecnologico

| Componente | Tecnologia | Versione | Porta |
|------------|------------|----------|-------|
| Database | PostgreSQL | 17.7 | 5432 |
| Vector Search | pgvector | 0.7.4 | - |
| BM25 Search | pg_search (ParadeDB) | 0.21.4 | - |
| Graph DB | Apache AGE | 1.6.0 | - |
| Fuzzy Search | pg_trgm | 1.6 | - |
| PDF Extraction | Unstructured API | latest | 8500 |
| PDF Extraction (alt) | PyMuPDF | 1.26.7 | - |

---

## Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                        lexe-api                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Ingestion  │    │  Retrieval  │    │    API      │         │
│  │  Pipeline   │    │   Engine    │    │  Endpoints  │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              lexe-postgres (PostgreSQL 17)          │       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │       │
│  │  │pgvector │ │pg_search│ │  AGE    │ │ pg_trgm │   │       │
│  │  │ (HNSW)  │ │ (BM25)  │ │ (Graph) │ │ (Fuzzy) │   │       │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Unstructured API   │ ← PDF Extraction (OCR + Layout)
│     (port 8500)     │
└─────────────────────┘
```

---

## Infrastruttura Docker

### Container `lexe-postgres`

```yaml
# docker-compose.kb.yml
services:
  lexe-postgres:
    build:
      context: .
      dockerfile: Dockerfile.kb
    container_name: lexe-postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: lexe
      POSTGRES_PASSWORD: lexe_dev_password
      POSTGRES_DB: lexe
    volumes:
      - lexe_data:/var/lib/postgresql/data
    command: >
      postgres
      -c shared_preload_libraries=age,pg_search
      -c search_path=ag_catalog,kb,public
      -c shared_buffers=256MB
      -c effective_cache_size=768MB
```

### Container `unstructured-api`

```bash
docker run -p 8500:8000 -d --name unstructured-api \
  downloads.unstructured.io/unstructured-io/unstructured-api:latest \
  --port 8000 --host 0.0.0.0
```

---

## Schema Database

### Schema `kb`

```
kb.documents            # Documenti PDF sorgente
kb.sections             # Sezioni/capitoli estratti
kb.massime              # Massime giurisprudenziali
kb.citations            # Citazioni normative
kb.embeddings           # Vettori multi-modello
kb.duplicates           # Deduplicazione
kb.edge_weights         # Pesi grafi
kb.benchmark_runs       # Benchmark retrieval
kb.ingestion_jobs       # Job di ingestion
kb.ingestion_profiles   # Profili configurazione ingestion (NEW)
```

### Tabella `documents`

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | UUID | PK |
| source_path | TEXT | Path PDF originale |
| source_hash | VARCHAR(64) | SHA256 del filename |
| anno | INTEGER | Anno del massimario |
| volume | INTEGER | Numero volume |
| tipo | VARCHAR(10) | 'civile' o 'penale' |
| titolo | TEXT | Titolo completo |
| pagine | INTEGER | Numero pagine |
| ocr_quality_score | FLOAT | Score qualità OCR |

### Tabella `massime`

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | UUID | PK |
| document_id | UUID | FK → documents |
| testo | TEXT | Testo massima |
| testo_normalizzato | TEXT | Testo normalizzato per search |
| content_hash | VARCHAR(64) | SHA256 per dedup |
| sezione | VARCHAR(20) | Es. "4", "Un." |
| numero | VARCHAR(20) | Numero sentenza |
| anno | INTEGER | Anno decisione |
| data_decisione | DATE | Data decisione |
| materia | VARCHAR(100) | Civile/Penale |
| tsv_simple | TSVECTOR | FTS simple (generated) |
| tsv_italian | TSVECTOR | FTS italian (generated) |

### Tabella `embeddings`

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | UUID | PK |
| massima_id | UUID | FK → massime |
| model | VARCHAR(50) | qwen3/e5-large/bge-m3/legal-bert-it |
| channel | VARCHAR(20) | testo/tema/contesto |
| embedding | VECTOR(1536) | Vettore (dim varia per modello) |

### Indici

```sql
-- HNSW per vector search (partial indexes per modello/canale)
CREATE INDEX idx_emb_qwen3_testo ON kb.embeddings
  USING hnsw (embedding vector_cosine_ops)
  WHERE model = 'qwen3' AND channel = 'testo';

-- FTS con tsvector
CREATE INDEX idx_massime_tsv_italian ON kb.massime USING gin(tsv_italian);

-- Trigram per fuzzy
CREATE INDEX idx_massime_trgm ON kb.massime
  USING gin(testo_normalizzato gin_trgm_ops);
```

### Graph (Apache AGE)

```sql
-- Grafo giurisprudenziale
SELECT * FROM ag_catalog.create_graph('lexe_jurisprudence');

-- Labels: Massima, Norma, Concetto
-- Edges: CITA, INTERPRETA, CONFERMA, CONTRASTA
```

---

## Moduli Python

### Struttura

```
src/lexe_api/kb/
├── __init__.py           # Export pubblici
├── config.py             # Settings, enums, costanti
├── models.py             # Pydantic models
│
├── ingestion/
│   ├── __init__.py
│   ├── extractor.py      # PDF → raw text
│   ├── cleaner.py        # Pulizia OCR
│   ├── parser.py         # Parsing sezioni
│   ├── massima_extractor.py  # Estrazione massime
│   ├── citation_parser.py    # Parsing citazioni
│   ├── deduplicator.py   # SimHash dedup
│   ├── embedder.py       # Generazione vettori
│   └── pipeline.py       # Orchestrazione
│
└── retrieval/
    ├── __init__.py
    ├── dense.py          # Vector search (HNSW)
    ├── sparse.py         # BM25/FTS
    ├── hybrid.py         # RRF fusion
    ├── reranker.py       # Cross-encoder
    ├── graph.py          # Cypher queries
    └── temporal.py       # Filtri temporali
```

### Configurazioni Sistema (S1-S5)

| Config | Descrizione | Componenti |
|--------|-------------|------------|
| S1 | Atomico | Solo BM25 |
| S2 | Hybrid | BM25 + HNSW + RRF |
| S3 | Rerank | S2 + Cross-encoder |
| S4 | Graph | S3 + Cypher expansion |
| S5 | Full | S4 + Multi-model ensemble |

### Modelli Embedding

| Modello | Dimensioni | Uso |
|---------|------------|-----|
| Qwen3 | 1536 | Default, multilingue |
| E5-Large | 1024 | Retrieval ottimizzato |
| BGE-M3 | 1024 | Multilingue, dense+sparse |
| Legal-BERT-IT | 768 | Dominio legale italiano |

---

## Script di Test

### `scripts/test_ingestion.py`

Estrae massime da PDF usando PyMuPDF e inserisce nel database.

```bash
cd lexe-api
uv run python scripts/test_ingestion.py
```

### `scripts/test_retrieval.py`

Testa il sistema di retrieval FTS/BM25.

```bash
cd lexe-api
uv run python scripts/test_retrieval.py
```

### `scripts/test_unstructured.py`

Confronta estrazione PyMuPDF vs Unstructured API.

```bash
cd lexe-api
uv run python scripts/test_unstructured.py
```

---

## Risultati Test (2026-01-27)

### Confronto Estrazione PDF

| PDF | PyMuPDF | Unstructured | Ratio |
|-----|---------|--------------|-------|
| 2021 PENALE Vol.1 | 645 | **2114** | 3.3x |
| 2023 PENALE Vol.1 | 433 | **1633** | 3.8x |
| 2018 CIVILE Vol.1 | 189 | **800** | 4.2x |
| **TOTALE** | 1267 | **4547** | **3.6x** |

**Conclusione:** Unstructured estrae **3.6x più massime** grazie a:
- Migliore gestione layout PDF
- OCR integrato
- Estrazione completa del documento (non solo prime N pagine)

### Timing

| Metodo | Tempo medio |
|--------|-------------|
| PyMuPDF | ~0.3s |
| Unstructured (fast) | ~25s |
| Unstructured (hi_res) | ~60s |

### Retrieval FTS

```sql
-- Query test
SELECT * FROM kb.bm25_search('colpa', 5);
-- Risultati: 5 massime con score 0.1-0.3

SELECT * FROM kb.bm25_search('sezioni unite', 5);
-- Risultati: 5 massime con score 0.2-0.25
```

---

## Pattern Estrazione Massime

### Pattern Space-Tolerant (per Unstructured)

```python
# Sez. 4, n. 6513 del 27/01/2021
re.compile(
    r"Sez\s*\.?\s*(\d+)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+"
    r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
    re.IGNORECASE
)

# Sez. Un., n. 12345 del 27/01/2021
re.compile(
    r"Sez\s*\.?\s*(Un\.?|Unite)\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+"
    r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
    re.IGNORECASE
)

# Simple: Sez X n Y (senza data)
re.compile(
    r"Sez\s*\.?\s*(\d+)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d{3,})",
    re.IGNORECASE
)
```

---

## Funzioni SQL Custom

### `kb.bm25_search(query, limit)`

```sql
-- Prova BM25 nativo (pg_search), fallback a FTS
CREATE FUNCTION kb.bm25_search(p_query TEXT, p_limit INT)
RETURNS TABLE(massima_id UUID, score FLOAT)
```

### `kb.hybrid_search(query, embedding, model, channel, limit, rrf_k)`

```sql
-- Ricerca ibrida: BM25 + HNSW con RRF fusion
CREATE FUNCTION kb.hybrid_search(...)
RETURNS TABLE(massima_id UUID, rrf_score FLOAT, dense_rank INT, sparse_rank INT)
```

---

## Comandi Utili

### Avvio Infrastruttura

```bash
# Start lexe-postgres
cd lexe-api
docker compose -f docker-compose.kb.yml up -d

# Start Unstructured API
docker run -p 8500:8000 -d --name unstructured-api \
  downloads.unstructured.io/unstructured-io/unstructured-api:latest

# Verifica
docker ps --filter name=lexe-postgres
docker ps --filter name=unstructured
curl http://localhost:8500/healthcheck
```

### Database

```bash
# Connessione
docker exec -it lexe-postgres psql -U lexe -d lexe

# Stats
SELECT COUNT(*) FROM kb.documents;
SELECT COUNT(*) FROM kb.massime;
SELECT materia, COUNT(*) FROM kb.massime GROUP BY materia;
```

### Ingestion

```bash
cd lexe-api
uv run python scripts/test_ingestion.py
```

---

## Estrazione Indici (2026-01-27)

Lo script `extract_index.py` estrae la struttura gerarchica (TOC) dai PDF:

### Risultati Estrazione

| PDF | Parti | Capitoli | Sezioni | Totale |
|-----|-------|----------|---------|--------|
| 2018 CIVILE | 8 | 35 | 179 | **224** |
| 2021 PENALE | 3 | 9 | 81 | **95** |
| 2023 PENALE | 5 | 7 | 5 | **17** |
| **TOTALE** | **16** | **51** | **265** | **336** |

### Copertura Pagine Sezioni (dopo Backfill)

| PDF | Sezioni | Con Pagina | Coverage |
|-----|---------|------------|----------|
| 2018 CIVILE | 224 | 209 | **93.3%** |
| 2021 PENALE | 95 | 95 | **100%** |
| 2023 PENALE | 17 | 17 | **100%** |

**Backfill** cerca i titoli nel testo PDF (dopo area indice) con match fuzzy.

### Funzione SQL per Match Sezioni

```sql
-- Trova la sezione per una data pagina
SELECT kb.find_section_for_page(document_id, page_number);
```

### Pipeline End-to-End (kb_pipeline.py)

Script unificato che esegue:
1. Estrazione massime con Unstructured (page_number incluso)
2. Content hash standardizzato (SHA256 completo)
3. Calcolo page_end per sezioni
4. Upsert massime con deduplica
5. Link a sezioni via pagina + fallback FTS
6. Report QA automatico

### Risultati Pipeline (2026-01-27) - Con Gate Policy

| PDF | Massime | Linked | Short | Out of Range | Coverage |
|-----|---------|--------|-------|--------------|----------|
| 2018 CIVILE | 492 | 492 | 0% | 0 | **100%** |
| 2021 PENALE | 102 | 102 | 0% | 0 | **100%** |
| 2023 PENALE | 178 | 178 | 0% | 0 | **100%** |
| **TOTALE** | **772** | **772** | **0%** | **0** | **100%** |

**Gate Policy** ha filtrato il 52% dei falsi positivi (liste citazioni, frammenti).

**SKIP_PAGES dinamico** evita matching nel TOC usando la prima pagina delle massime come riferimento.

---

---

## Sistema Profili Ingestion (2026-01-27)

Ogni documento usa un profilo di configurazione che definisce parametri di estrazione, gate policy, e soglie QA.

### Tabella `kb.ingestion_profiles`

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | UUID | PK |
| name | VARCHAR(100) | Nome profilo univoco |
| doc_type | VARCHAR(50) | 'civile', 'penale', NULL=tutti |
| anno_min/max | INTEGER | Range anni applicabile |
| config | JSONB | Configurazione completa |
| is_default | BOOLEAN | Profilo fallback |

### Profili Definiti

| Profilo | Tipo | Anni | Caratteristiche |
|---------|------|------|-----------------|
| `massimario_default_v1` | tutti | tutti | SKIP_PAGES dinamico, gate standard |
| `massimario_penale_2021_2023` | penale | 2021-2023 | TOC pulito, soglie QA strette |
| `massimario_civile_toc_collision` | civile | 2015-2020 | Normalizzazione aggressiva, gestione collisioni |

### Struttura Config JSONB

```json
{
  "extraction": {
    "strategy": "fast",
    "skip_pages_dynamic": true,
    "content_start_buffer": 10
  },
  "gate_policy": {
    "min_length": 150,
    "max_citation_ratio": 0.03,
    "required_keywords_short": true
  },
  "section_backfill": {
    "strategy": "title_search",
    "fuzzy_threshold": 0.6,
    "page_end_buffer_single": 5
  },
  "qa_thresholds": {
    "max_short_pct": 5.0,
    "max_out_of_range": 10,
    "min_linking_pct": 90.0
  }
}
```

### Funzioni Helper

```sql
-- Trova profilo migliore per documento
SELECT kb.find_profile_for_document('penale', 2023);

-- Estrai config specifica
SELECT kb.get_profile_config(profile_id, 'gate_policy.min_length');
```

---

## Sistema QA (2026-01-27)

### View QA

| View | Descrizione |
|------|-------------|
| `kb.qa_document_report` | Report completo: short%, linked%, out_of_range |
| `kb.qa_page_collisions` | Sezioni con stessa pagina (anomalia TOC) |
| `kb.qa_hierarchy_anomalies` | Figli con pagina prima del padre |

### Query Report QA

```sql
SELECT anno, tipo, profile_name,
       total_massime, pct_short, pct_linked,
       out_of_range_count
FROM kb.qa_document_report;
```

### Metriche Monitoraggio

| Metrica | Soglia | Azione se superata |
|---------|--------|-------------------|
| pct_short | < 5% | Review gate policy |
| out_of_range | < 10 | Check section pages |
| pct_linked | > 90% | OK |
| page_collisions | < 3 per livello | Backfill conservativo |

---

## Prossimi Step

1. ~~**Estrazione Indici**~~ ✅ Completato (336 sezioni)
2. ~~**Re-ingestion Massime**~~ ✅ Completato con Gate Policy (772 massime)
3. ~~**Collegamento Massime-Sezioni**~~ ✅ Completato (100% linked)
4. ~~**Backfill Sezioni 2023**~~ ✅ Completato (100% coverage)
5. ~~**Sistema Profili**~~ ✅ Completato (`kb.ingestion_profiles`)
6. ~~**Sistema QA**~~ ✅ Completato (views + metriche)
7. **Embeddings** - Benchmark `dlicari/distil-ita-legal-bert` vs BGE-M3
8. **API REST** - Endpoint `/api/v1/kb/search`
9. **Benchmark** - Confronto S1-S5 su query set
10. **Integrazione LEXE** - Tool TRIDENT per ricerca massime

---

## File Creati

```
lexe-api/
├── Dockerfile.kb                    # PostgreSQL 17 custom
├── docker-compose.kb.yml            # Container config
├── migrations/
│   ├── 001_init_schema.sql          # Schema base
│   ├── 002_init_vectors.sql         # pgvector + HNSW
│   ├── 003_init_bm25.sql            # pg_search + FTS
│   └── 003_ingestion_profiles.sql   # Profili + QA views (NEW)
├── scripts/
│   ├── test_ingestion.py            # Test ingestion
│   ├── test_retrieval.py            # Test retrieval
│   ├── test_unstructured.py         # Confronto extraction
│   ├── extract_index.py             # Estrazione TOC/indici
│   ├── link_massime_sections.py     # Collegamento massime-sezioni (legacy)
│   ├── kb_pipeline.py               # Pipeline end-to-end con Gate Policy
│   └── backfill_section_pages.py    # Backfill dinamico pagine sezioni (NEW)
├── src/lexe_api/kb/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── ingestion/ (8 moduli)
│   └── retrieval/ (6 moduli)
└── docs/
    └── KB-MASSIMARI-ARCHITECTURE.md # Questo documento
```

---

*Ultimo aggiornamento: 2026-01-27 (Ingestion Profiles + QA System)*
*Autore: Claude Code + Francesco*
*Repo: https://github.com/LEXe-Legal-AI/lexe-max*
