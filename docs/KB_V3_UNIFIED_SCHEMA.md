# KB V3 Unified Schema - LEXE Legal Knowledge Base

> **Version:** 3.0.0
> **Date:** 2026-02-06
> **Migration:** `migrations/kb/050_kb_v3_unified_schema.sql`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [URN:NIR Alignment](#urnnir-alignment)
4. [Schema Diagram](#schema-diagram)
5. [Tables Reference](#tables-reference)
6. [Functions & Triggers](#functions--triggers)
7. [Views](#views)
8. [Indexes Strategy](#indexes-strategy)
9. [Graph Integration](#graph-integration)
10. [Migration Guide](#migration-guide)
11. [Usage Examples](#usage-examples)

---

## Overview

KB V3 è lo schema unificato per la Knowledge Base legale LEXE. Unifica:

- **Normativa** (articoli di codici, leggi, decreti)
- **Massime** (giurisprudenza, integrazione con KB esistente)
- **Sentenze** (decisioni complete)
- **Annotazioni** (note, commenti, dottrina, brocardi)

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **work → normativa hierarchy** | Un "atto" (work) contiene molti articoli (normativa) |
| **nir_mapping enforcement** | Garantisce coerenza tra code interno e URN:NIR |
| **Multi-source tracking** | Ogni entità traccia canonical + mirror sources |
| **Quality enum** | Classification 2-assi integrata nello schema |
| **Embeddings multi-dim** | Support per 768/1024/1536 dims con partial indexes |
| **Apache AGE optional** | Graph queries opzionali, fallback a join SQL |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KB V3 UNIFIED SCHEMA                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ source_type  │      │source_system │      │  nir_mapping │              │
│  │ COSTITUZIONE │      │ ALTALEX_PDF  │      │ CC → URN     │              │
│  │ CODICE       │      │ BROCARDI_*   │      │ CP → URN     │              │
│  │ TESTO_UNICO  │      │ NORMATTIVA   │      │ CCII → URN   │              │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘              │
│         │                     │                      │                      │
│         └─────────────────────┼──────────────────────┘                      │
│                               │                                             │
│                     ┌─────────▼─────────┐                                   │
│                     │       work        │                                   │
│                     │ (atto normativo)  │                                   │
│                     │  code, title,     │                                   │
│                     │  nir_base         │                                   │
│                     └─────────┬─────────┘                                   │
│                               │                                             │
│            ┌──────────────────┼──────────────────┐                          │
│            │                  │                  │                          │
│  ┌─────────▼─────────┐ ┌──────▼──────┐ ┌────────▼────────┐                  │
│  │work_source_link   │ │ work_topic  │ │    normativa    │                  │
│  │(provenance)       │ │(M2M topics) │ │   (articoli)    │                  │
│  └───────────────────┘ └─────────────┘ │ codice, art,    │                  │
│                                        │ testo, quality  │                  │
│                                        │ urn_nir         │                  │
│                                        └────────┬────────┘                  │
│                                                 │                           │
│            ┌────────────────────────────────────┼──────────────┐            │
│            │                     │              │              │            │
│  ┌─────────▼─────────┐ ┌────────▼───────┐ ┌────▼────┐ ┌───────▼───────┐    │
│  │normativa_altalex  │ │normativa_emb   │ │ norms   │ │annotation_link│    │
│  │(Altalex-specific) │ │(vectors 1536)  │ │(refs)   │ └───────┬───────┘    │
│  └───────────────────┘ └────────────────┘ └─────────┘         │            │
│                                                               │            │
│                                                     ┌─────────▼─────────┐  │
│                                                     │    annotation     │  │
│                                                     │ note, commenti,   │  │
│                                                     │ massime, brocardi │  │
│                                                     └─────────┬─────────┘  │
│                                                               │            │
│                                                     ┌─────────▼─────────┐  │
│                                                     │ annotation_emb    │  │
│                                                     └───────────────────┘  │
│                                                                            │
│  ┌───────────────────┐                    ┌────────────────────────────┐   │
│  │     sentenza      │───────────────────▶│    sentenza_embeddings     │   │
│  │ (giurisprudenza)  │                    └────────────────────────────┘   │
│  └───────────────────┘                                                     │
│                                                                            │
│  ┌───────────────────┐   ┌───────────────────┐                             │
│  │  ingestion_run    │──▶│  ingestion_event  │                             │
│  │ (batch tracking)  │   │  (logs/errors)    │                             │
│  └───────────────────┘   └───────────────────┘                             │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## URN:NIR Alignment

### Standard URN:NIR

LEXE allinea i codici interni allo standard **URN:NIR** definito da AgID:

- **Guida Normattiva:** https://www.normattiva.it/staticPage/utilita
- **Specifiche AgID:** https://www.agid.gov.it/sites/agid/files/2024-06/Linee_guida_marcatura_documenti_normativi.pdf

### Struttura URN

```
urn:nir:<autorità>:<tipo_atto>:<data>;<numero>[~<partizione>]

Esempi:
  urn:nir:stato:regio.decreto:1942-03-16;262           # Codice Civile (atto)
  urn:nir:stato:regio.decreto:1942-03-16;262~art2043  # Articolo 2043
  urn:nir:stato:regio.decreto:1942-03-16;262~art2043-com1  # Comma 1
```

### Gold Mappings

La tabella `kb.nir_mapping` contiene i mapping verificati:

| code | nir_base | nome_completo |
|------|----------|---------------|
| `COST` | `urn:nir:stato:costituzione:1947-12-27` | Costituzione della Repubblica Italiana |
| `CC` | `urn:nir:stato:regio.decreto:1942-03-16;262` | Codice Civile |
| `CP` | `urn:nir:stato:regio.decreto:1930-10-19;1398` | Codice Penale |
| `CPC` | `urn:nir:stato:regio.decreto:1940-10-28;1443` | Codice di Procedura Civile |
| `CPP` | `urn:nir:stato:decreto.presidente.repubblica:1988-09-22;447` | Codice di Procedura Penale |
| `CCII` | `urn:nir:stato:decreto.legislativo:2019-01-12;14` | Codice della Crisi d'Impresa |
| `TUB` | `urn:nir:stato:decreto.legislativo:1993-09-01;385` | Testo Unico Bancario |
| `TUF` | `urn:nir:stato:decreto.legislativo:1998-02-24;58` | Testo Unico della Finanza |
| `GDPR` | `urn:nir:unione.europea:regolamento:2016-04-27;2016-679` | Regolamento Generale Protezione Dati |

### Enforcement

Il campo `work.code` è FK verso `nir_mapping.code`, garantendo che:
1. Ogni work ha un URN:NIR valido
2. Non si possono creare work con codici non mappati
3. `work.nir_base` è auto-popolato da trigger

```sql
-- Validation function
SELECT kb.validate_nir_code('CC');  -- TRUE
SELECT kb.validate_nir_code('INVALID');  -- FALSE

-- Get NIR base
SELECT kb.get_nir_base('CC');
-- 'urn:nir:stato:regio.decreto:1942-03-16;262'

-- Build article URN
SELECT kb.build_article_urn('CC', '2043', '1');
-- 'urn:nir:stato:regio.decreto:1942-03-16;262~art2043-com1'
```

---

## Schema Diagram

### Entity-Relationship (Simplified)

```
                    source_type                    source_system
                        │                               │
                        └───────────┬───────────────────┘
                                    │
    topic ◄─────── work_topic ─────►│
                                    │
                              ┌─────▼─────┐
    nir_mapping ──────────────►   work    │◄─────── work_source_link
                              └─────┬─────┘
                                    │
                                    │ 1:N
                                    │
                              ┌─────▼─────┐
                              │ normativa │
                              └─────┬─────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        │               │           │           │               │
        ▼               ▼           ▼           ▼               ▼
  normativa_       normativa_   normativa_   annotation_    norms
  altalex          embeddings   norms        link
                                               │
                                               ▼
                                          annotation
                                               │
                                               ▼
                                       annotation_embeddings


    sentenza ──────────► sentenza_embeddings
        │
        └──────────────► annotation_link


    ingestion_run ──────► ingestion_event
```

---

## Tables Reference

### Lookup Tables

#### `kb.source_type`
Tipologie di fonti normative.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | COSTITUZIONE, CODICE, TESTO_UNICO, etc. |
| label | TEXT | Display name |
| priority | INT | Ranking (100=highest) |
| description | TEXT | Detailed description |

#### `kb.source_system`
Sistemi sorgente per acquisizione.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | ALTALEX_PDF, BROCARDI_OFFLINE, NORMATTIVA_ONLINE |
| label | TEXT | Display name |
| is_canonical | BOOLEAN | TRUE if authoritative source |
| is_offline | BOOLEAN | TRUE if local mirror available |
| rate_limit_per_sec | NUMERIC | Request rate limit |
| base_url | TEXT | Base URL or path |

#### `kb.nir_mapping`
Mapping code → URN:NIR con enforcement.

| Column | Type | Description |
|--------|------|-------------|
| code | TEXT PK | Internal code: CC, CP, CCII |
| nir_base | TEXT UNIQUE | URN base: urn:nir:stato:... |
| nome_completo | TEXT | Full Italian name |
| nome_breve | TEXT | Short display name |
| abbreviazioni | TEXT[] | Common abbreviations |
| source_type_id | TEXT FK | Reference to source_type |
| atto_tipo | TEXT | regio.decreto, decreto.legislativo |
| atto_data | DATE | Publication date |
| atto_numero | TEXT | Act number |
| normattiva_url | TEXT | Direct Normattiva URL |

### Main Tables

#### `kb.work`
Atti normativi (documents/acts).

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | Auto-generated |
| code | TEXT FK | → nir_mapping.code |
| title | TEXT | Full title |
| title_short | TEXT | Short title for display |
| source_type_id | TEXT FK | → source_type.id |
| is_abrogated | BOOLEAN | TRUE if no longer in force |
| is_consolidated | BOOLEAN | TRUE if consolidated text |
| edition_date | DATE | Edition/consolidation date |
| nir_base | TEXT | Auto-populated from nir_mapping |
| article_count | INT | Number of articles |
| meta | JSONB | Additional metadata |

#### `kb.normativa`
Articoli di normativa.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | Auto-generated |
| work_id | UUID FK | → work.id |
| urn_nir | VARCHAR(250) UNIQUE | Auto-computed URN |
| codice | TEXT | CC, CP, CCII (denormalized) |
| articolo | TEXT | '2043', '360-bis' |
| comma | TEXT | '1', '2', 'bis' |
| articolo_num | INT | Parsed number: 2043 |
| articolo_suffix | TEXT | 'bis', 'ter' |
| articolo_sort_key | TEXT | '002043.02' for sorting |
| libro | TEXT | Hierarchy level 1 |
| titolo | TEXT | Hierarchy level 2 |
| capo | TEXT | Hierarchy level 3 |
| sezione | TEXT | Hierarchy level 4 |
| rubrica | TEXT | Article title |
| testo | TEXT | Full article text |
| testo_normalizzato | TEXT | Lowercase, normalized |
| canonical_source | TEXT | 'NORMATTIVA_ONLINE' |
| canonical_hash | VARCHAR(64) | SHA256 of canonical text |
| mirror_source | TEXT | 'BROCARDI_OFFLINE' |
| mirror_hash | VARCHAR(64) | SHA256 of mirror text |
| validation_status | TEXT | pending, verified, content_diff |
| data_vigenza_da | DATE | In force from |
| data_vigenza_a | DATE | In force until (NULL=current) |
| is_current | BOOLEAN | TRUE if current version |
| quality | article_quality | VALID_STRONG, WEAK, etc. |
| warnings | JSONB | Array of warnings |

#### `kb.normativa_altalex`
Altalex-specific extension for normativa.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| normativa_id | UUID FK UNIQUE | → normativa.id |
| global_key | TEXT UNIQUE | altalex:cc:2043:bis |
| testo_context | TEXT | Overlap ±200 chars |
| commi | JSONB | [{num: 1, testo: "..."}, ...] |
| riferimenti_parsed | JSONB | ["CC:1218", "CC:2059"] |
| riferimenti_raw | TEXT[] | Raw unparsed references |
| page_start | INT | PDF page start |
| page_end | INT | PDF page end |
| testo_tsv | TSVECTOR | Full-text search Italian |

### Embeddings Tables

#### `kb.normativa_embeddings`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| normativa_id | UUID FK | → normativa.id |
| model | TEXT | 'text-embedding-3-small' |
| channel | TEXT | 'testo', 'rubrica', 'combined' |
| dims | INT | 768, 1024, 1536 |
| embedding | vector | The embedding vector |

**HNSW Indexes:** Partial indexes per (dims, channel):
- `idx_norm_emb_1536_testo`
- `idx_norm_emb_1024_testo`
- `idx_norm_emb_768_testo`
- `idx_norm_emb_1536_rubrica`

### Annotation Tables

#### `kb.annotation`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| annotation_type | TEXT | 'nota', 'commento', 'massima', 'brocardo', 'ratio' |
| source_system_id | TEXT FK | Source of annotation |
| author | TEXT | Author name |
| title | TEXT | Annotation title |
| text | TEXT | Full text |
| text_normalized | TEXT | Normalized for search |
| meta | JSONB | Additional data |

#### `kb.annotation_link`
Many-to-many between annotation and normativa/sentenza.

| Column | Type | Description |
|--------|------|-------------|
| annotation_id | UUID FK | → annotation.id |
| normativa_id | UUID FK | → normativa.id (nullable) |
| sentenza_id | UUID FK | → sentenza.id (nullable) |
| relevance | DOUBLE | Relevance score |
| span | JSONB | {start: 100, end: 200} |

### Tracking Tables

#### `kb.ingestion_run`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| started_at | TIMESTAMPTZ | Run start |
| finished_at | TIMESTAMPTZ | Run end |
| source_system_id | TEXT FK | Source being ingested |
| work_code | TEXT | Target work code |
| status | TEXT | running, completed, failed, partial |
| stats | JSONB | {total: 100, inserted: 95, errors: 5} |

#### `kb.ingestion_event`

| Column | Type | Description |
|--------|------|-------------|
| run_id | UUID FK | → ingestion_run.id |
| entity_type | TEXT | 'normativa', 'annotation', 'embedding' |
| entity_id | UUID | Affected entity |
| action | TEXT | 'insert', 'update', 'skip', 'error' |
| severity | TEXT | debug, info, warning, error |
| message | TEXT | Human-readable message |

---

## Functions & Triggers

### NIR Functions

```sql
-- Validate code exists in nir_mapping
kb.validate_nir_code(p_code TEXT) → BOOLEAN

-- Get URN base from code
kb.get_nir_base(p_code TEXT) → TEXT

-- Build full article URN
kb.build_article_urn(p_code TEXT, p_articolo TEXT, p_comma TEXT) → TEXT
```

### Auto-computed Triggers

#### `trg_work_nir_base`
Auto-populates `work.nir_base` from `nir_mapping` when inserting/updating work.

#### `trg_normativa_computed`
On normativa insert/update:
- Parses `articolo_num` and `articolo_suffix` from `articolo`
- Computes `articolo_sort_key` (NNNNNN.SS format)
- Builds `urn_nir` from work.nir_base + article
- Normalizes `testo_normalizzato`

#### `trg_altalex_computed`
On normativa_altalex insert/update:
- Builds `global_key` from parent normativa
- Updates `testo_tsv` for full-text search

---

## Views

### `kb.v_normativa_current`
Current articles with work info.

```sql
SELECT * FROM kb.v_normativa_current
WHERE codice = 'CC' AND articolo = '2043';
```

### `kb.v_quality_stats`
Quality statistics per work/codice.

```sql
SELECT * FROM kb.v_quality_stats;
```

| codice | total | validi | deboli | invalidi | dal | al | coverage_pct |
|--------|-------|--------|--------|----------|-----|-----|--------------|
| CC | 3208 | 3201 | 3 | 4 | 000001.00 | 002969.00 | 99.8 |
| CCII | 415 | 414 | 1 | 0 | 000001.00 | 000391.00 | 99.8 |

### `kb.v_normativa_annotations`
Annotation counts per article.

```sql
SELECT * FROM kb.v_normativa_annotations
WHERE codice = 'CC' AND note_count > 0
LIMIT 10;
```

---

## Indexes Strategy

### Lookup Indexes
- `idx_normativa_lookup_current` - Partial index for current articles
- `idx_normativa_work` - FK index
- `idx_normativa_sort` - Sorting by codice + sort_key

### Search Indexes
- `idx_normativa_trgm` - GIN trigram for fuzzy search
- `idx_normativa_fts` - GIN tsvector for Italian full-text
- `idx_altalex_tsv` - GIN tsvector with unaccent

### Vector Indexes (HNSW)
Partial indexes per dimension and channel:

```sql
-- 1536 dims (OpenAI text-embedding-3-small)
idx_norm_emb_1536_testo WHERE dims = 1536 AND channel = 'testo'

-- 1024 dims (e5, bge models)
idx_norm_emb_1024_testo WHERE dims = 1024 AND channel = 'testo'

-- 768 dims (legal-bert, multilingual)
idx_norm_emb_768_testo WHERE dims = 768 AND channel = 'testo'
```

---

## Graph Integration

### Apache AGE (Optional)

Se Apache AGE è installato, viene creato il graph `kb_graph`:

```sql
-- Create graph (done by migration)
SELECT ag_catalog.create_graph('kb_graph');

-- Example: Create vertices for normativa
SELECT * FROM cypher('kb_graph', $$
    CREATE (n:Normativa {
        id: 'CC:2043',
        codice: 'CC',
        articolo: '2043',
        rubrica: 'Risarcimento per fatto illecito'
    })
    RETURN n
$$) as (n agtype);

-- Example: Create edge for citation
SELECT * FROM cypher('kb_graph', $$
    MATCH (a:Normativa {id: 'CC:2043'})
    MATCH (b:Normativa {id: 'CC:2059'})
    CREATE (a)-[:CITA {context: 'danno non patrimoniale'}]->(b)
$$) as (r agtype);
```

### Fallback SQL

Se AGE non è disponibile, usare le tabelle `normativa_norms` e `annotation_link` per le relazioni:

```sql
-- Articles citing CC:2043
SELECT DISTINCT n.codice, n.articolo
FROM kb.normativa n
JOIN kb.normativa_norms nn ON n.id = nn.normativa_id
WHERE nn.norm_id = 'CC:2043';
```

---

## Migration Guide

### From Existing Schema

Se hai già dati in `kb.normativa` (v1/v2):

```sql
-- 1. Backup existing data
CREATE TABLE kb.normativa_backup AS SELECT * FROM kb.normativa;

-- 2. Create work entries for each distinct codice
INSERT INTO kb.work (code, title, edition_date)
SELECT DISTINCT
    codice,
    nm.nome_completo,
    '2026-02-06'::date
FROM kb.normativa n
JOIN kb.nir_mapping nm ON n.codice = nm.code
ON CONFLICT DO NOTHING;

-- 3. Update normativa.work_id
UPDATE kb.normativa n
SET work_id = w.id
FROM kb.work w
WHERE n.codice = w.code;

-- 4. Verify triggers computed fields
UPDATE kb.normativa SET updated_at = now();

-- 5. Check URN generation
SELECT codice, articolo, urn_nir
FROM kb.normativa
WHERE urn_nir IS NOT NULL
LIMIT 10;
```

### Fresh Install

```bash
# Connect to database
psql -h localhost -p 5434 -U lexe -d lexe_kb

# Run migration
\i migrations/kb/050_kb_v3_unified_schema.sql
```

---

## Usage Examples

### Insert a new work

```sql
INSERT INTO kb.work (code, title, edition_date)
VALUES ('CC', 'Codice Civile', '2026-02-06')
RETURNING id, code, nir_base;
-- nir_base auto-populated!
```

### Insert an article

```sql
INSERT INTO kb.normativa (work_id, codice, articolo, rubrica, testo, quality)
VALUES (
    'uuid-of-cc-work',
    'CC',
    '2043',
    'Risarcimento per fatto illecito',
    'Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto...',
    'VALID_STRONG'
)
RETURNING id, urn_nir, articolo_sort_key;
-- urn_nir = urn:nir:stato:regio.decreto:1942-03-16;262~art2043
-- articolo_sort_key = 002043.00
```

### Hybrid Search (vector + FTS)

```sql
-- Dense search
WITH query_embedding AS (
    SELECT embedding
    FROM kb.normativa_embeddings
    WHERE normativa_id = 'known-article-id'
    AND model = 'text-embedding-3-small'
    AND channel = 'testo'
)
SELECT
    n.codice,
    n.articolo,
    n.rubrica,
    e.embedding <=> q.embedding AS distance
FROM kb.normativa n
JOIN kb.normativa_embeddings e ON n.id = e.normativa_id
CROSS JOIN query_embedding q
WHERE e.dims = 1536 AND e.channel = 'testo'
AND n.is_current = TRUE
ORDER BY distance
LIMIT 10;

-- Sparse search (FTS)
SELECT codice, articolo, rubrica
FROM kb.normativa
WHERE to_tsvector('italian', testo) @@ plainto_tsquery('italian', 'danno ingiusto')
AND is_current = TRUE
LIMIT 10;
```

### Quality Report

```sql
SELECT * FROM kb.v_quality_stats;
```

### Track Ingestion

```sql
-- Start run
INSERT INTO kb.ingestion_run (source_system_id, work_code)
VALUES ('ALTALEX_PDF', 'CC')
RETURNING id;

-- Log events
INSERT INTO kb.ingestion_event (run_id, entity_type, action, severity, message)
VALUES
    ('run-uuid', 'normativa', 'insert', 'info', 'Inserted art. 2043'),
    ('run-uuid', 'normativa', 'error', 'error', 'Failed art. 2044: encoding issue');

-- Complete run
UPDATE kb.ingestion_run
SET finished_at = now(),
    status = 'completed',
    stats = '{"total": 3208, "inserted": 3200, "errors": 8}'::jsonb
WHERE id = 'run-uuid';
```

---

## Appendix: Sort Key Algorithm

Il `articolo_sort_key` usa il formato `NNNNNN.SS`:

- `NNNNNN`: numero articolo padded a 6 cifre
- `SS`: ordine suffisso (00 = base, 02 = bis, 03 = ter, ...)

| articolo | articolo_num | articolo_suffix | sort_key |
|----------|--------------|-----------------|----------|
| 1 | 1 | NULL | 000001.00 |
| 2043 | 2043 | NULL | 002043.00 |
| 2043-bis | 2043 | bis | 002043.02 |
| 2043-ter | 2043 | ter | 002043.03 |
| 360-quinquies | 360 | quinquies | 000360.05 |

Suffissi supportati (fino a 18°):

| # | Suffisso | Order |
|---|----------|-------|
| 2 | bis | 02 |
| 3 | ter | 03 |
| 4 | quater | 04 |
| 5 | quinquies | 05 |
| 6 | sexies | 06 |
| 7 | septies | 07 |
| 8 | octies | 08 |
| 9 | novies/nonies | 09 |
| 10 | decies | 10 |
| 11 | undecies | 11 |
| 12 | duodecies | 12 |
| 13 | terdecies | 13 |
| 14 | quaterdecies | 14 |
| 15 | quinquiesdecies | 15 |
| 16 | sexiesdecies | 16 |
| 17 | septiesdecies | 17 |
| 18 | octiesdecies | 18 |

---

*Created: 2026-02-06*
*Last updated: 2026-02-06*
*Author: LEXE Team*
