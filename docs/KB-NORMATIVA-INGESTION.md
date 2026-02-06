# KB Normativa Ingestion - Documentazione Completa

> **Data**: 2026-02-04
> **Autore**: Claude Code Session
> **Status**: CC e CP ingestiti, cross-validation pending

---

## Overview

Ingestion dei codici italiani nella Knowledge Base LEXE con approccio **FORENSE**:

- Fonti editoriali come **CARBURANTE** (Brocardi.it)
- Fonti ufficiali come **ANCORA** (Normattiva.it) per cross-validation
- Verifiche incrociate automatiche come **CINTURA E BRETELLE**

---

## Architettura

### Gerarchia Fonti (Trust Levels)

```
TIER A - CANONICHE (Fonte di Verità)
├── Normattiva.it - Riferimento UFFICIALE
└── Gazzetta Ufficiale - Second opinion

TIER B - EDITORIALI (Arricchimento)
├── Brocardi.it - Ottima struttura, aggiornato
└── StudioCataldi - Mirror locale

TIER C - OPZIONALI
└── Altalex (bloccato robots.txt)
```

### Number-Anchored Knowledge Graph

I numeri legali sono **ANCHOR POINTS** deterministici:

- Articoli: `CC:2043`, `CP:575`
- Leggi: `L:241:1990`, `DLGS:165:2001`
- Sentenze: `CASS:12345:2020`

Questo permette collegamento **deterministico** tra documenti.

---

## Database Schema

### Tabelle Principali

```sql
-- Articoli di codice
kb.normativa (
    id UUID PRIMARY KEY,
    urn_nir VARCHAR(200) UNIQUE,      -- URN:NIR standard
    codice VARCHAR(50),                -- 'CC', 'CP', etc.
    articolo VARCHAR(20),              -- '2043', '575'
    rubrica TEXT,                      -- Titolo articolo
    testo TEXT,                        -- Testo completo

    -- Cross-validation
    canonical_source VARCHAR(50),      -- 'normattiva'
    canonical_hash VARCHAR(64),        -- SHA256
    validation_status VARCHAR(20),     -- 'pending', 'verified', etc.
    validated_at TIMESTAMPTZ,          -- Data validazione

    -- Metadata
    mirror_source VARCHAR(50),         -- 'brocardi'
    mirror_hash VARCHAR(64),
    created_at TIMESTAMPTZ
)

-- Number-Anchored Graph
kb.legal_numbers (
    id UUID PRIMARY KEY,
    canonical_id VARCHAR(100) UNIQUE,  -- 'CC:2043'
    number_type VARCHAR(20),           -- 'article', 'law', 'sentence'
    codice VARCHAR(30),
    numero VARCHAR(50),
    citation_count INTEGER
)

-- Graph Edges
kb.number_citations (
    source_type VARCHAR(20),           -- 'massima', 'normativa'
    source_id UUID,
    target_number_id UUID,
    target_canonical VARCHAR(100)      -- Denormalized for perf
)
```

### Tabelle Supporto

```sql
kb.normativa_embeddings    -- Vector embeddings (pgvector)
kb.normativa_citations     -- Cross-references tra articoli
kb.normativa_updates       -- Tracking modifiche settimanali
kb.codice_lookup           -- Mapping codici → URN
kb.top_cited_numbers       -- Materialized view numeri più citati
```

### Migration File

```
lexe-max/migrations/kb/010_normativa_schema.sql
```

---

## Scripts Creati

### 1. fetch_cc_brocardi.py

Scarica articoli da Brocardi.it con rate limiting.

```bash
cd lexe-max

# Fetch Codice Civile (3191 articoli, ~18 min)
uv run python scripts/fetch_cc_brocardi.py --codice CC --rps 3

# Fetch Codice Penale (976 articoli, ~6 min)
uv run python scripts/fetch_cc_brocardi.py --codice CP --rps 3

# Fetch con limite (test)
uv run python scripts/fetch_cc_brocardi.py --codice CC --limit 50 --rps 3

# Codici disponibili
# CC, CP, CPC, CPP, COST, CDS, CCONS, CDPR
```

**Output**: `scripts/{codice}_brocardi_results.json`

### 2. ingest_normativa_to_db.py

Carica JSON nel database PostgreSQL.

```bash
cd lexe-max

# Ingest CC
uv run python scripts/ingest_normativa_to_db.py \
    --input scripts/cc_brocardi_results.json \
    --source brocardi

# Ingest CP
uv run python scripts/ingest_normativa_to_db.py \
    --input scripts/cp_brocardi_results.json \
    --source brocardi

# Dry run (no changes)
uv run python scripts/ingest_normativa_to_db.py \
    --input scripts/cc_brocardi_results.json \
    --dry-run
```

### 3. cross_validate_via_api.py (RECOMMENDED)

Cross-validation via LEXE Tools API (usa Normattiva internamente).

```bash
cd lexe-max

# Da server Hetzner (lexe-tools accessibile su localhost:8021)
ssh root@49.12.85.92
cd /opt/lexe-platform/lexe-max
uv run python scripts/cross_validate_via_api.py \
    --codice CC --sample 0.05 \
    --api-url http://localhost:8021

# Oppure via SSH tunnel (da locale)
# Terminal 1: apri tunnel
ssh -L 8021:localhost:8021 root@49.12.85.92

# Terminal 2: esegui validazione
uv run python scripts/cross_validate_via_api.py \
    --codice CC --limit 100 \
    --api-url http://localhost:8021
```

### 3b. cross_validate_normattiva.py (DEPRECATED)

Scraping diretto - non funziona da IP residenziali (bloccato).
Usare solo se API non disponibile.

### 4. ingest_cds_test.py

Test ingestion Codice della Strada da mirror locale.

```bash
uv run python scripts/ingest_cds_test.py
```

---

## Adapter: BrocardiAdapter

### Location

```
lexe-max/src/lexe_api/kb/sources/brocardi_adapter.py
```

### Usage

```python
from lexe_api.kb.sources.brocardi_adapter import BrocardiAdapter

async with BrocardiAdapter(requests_per_second=3.0) as adapter:
    # Lista codici disponibili
    codici = await adapter.list_codici()
    # ['CC', 'CP', 'CPC', 'CPP', 'COST', 'CDS', 'CCONS', 'CDPR']

    # Fetch singolo articolo
    article = await adapter.fetch_article('CC', '2043')
    print(article.testo)

    # Stream intero codice (memory efficient)
    async for article in adapter.stream_codice('CC', progress_callback=my_cb):
        process(article)
```

### ArticleExtract Contract

```python
@dataclass
class ArticleExtract:
    codice: str              # 'CC'
    articolo: str            # '2043'
    rubrica: str | None      # 'Risarcimento per fatto illecito'
    testo: str               # Testo completo
    urn_nir: str | None      # URN:NIR standard
    content_hash: str        # SHA256 normalizzato

    libro: str | None        # Gerarchia
    titolo: str | None

    source: str              # 'brocardi'
    source_url: str | None
    retrieved_at: datetime

    citations_raw: list[str] | None  # ['CC:2044', 'L:241:1990']
```

---

## Stato Attuale Database

### Statistiche (2026-02-04)

| Codice     | Articoli | Avg Chars | Status  |
| ---------- | -------- | --------- | ------- |
| CC         | 3170     | 332       | pending |
| CP         | 947      | 365       | pending |
| **Totale** | **4117** | -         | -       |

### Query Verifica

```sql
-- Conta articoli per codice
SELECT codice, COUNT(*) as articles,
       AVG(LENGTH(testo))::int as avg_chars,
       validation_status
FROM kb.normativa
GROUP BY codice, validation_status;

-- Articoli famosi
SELECT articolo, rubrica, LEFT(testo, 100)
FROM kb.normativa
WHERE codice = 'CC' AND articolo IN ('2043', '2059', '1218');

-- Status validazione
SELECT validation_status, COUNT(*)
FROM kb.normativa
GROUP BY validation_status;
```

---

## Cross-Validation Flow

### Livello 1: Hash Comparison (Deterministico)

```
Mirror (Brocardi) ──► Normalize ──► SHA256 ──┐
                                              ├──► Compare
Canonical (Normattiva) ──► Normalize ──► SHA256 ──┘
                                              │
                                              ▼
                              MATCH? ──► Yes: status = 'verified'
                                    └──► No: Livello 2
```

### Livello 2: Semantic Diff (LLM Tier 2)

Solo per hash mismatch (~10-15% dei casi):

- Analizza diff testuale
- Classifica: `formatting` | `minor` | `substantive`
- Se `substantive` → flag per review manuale

### Validation Status Values

| Status          | Significato                   |
| --------------- | ----------------------------- |
| `pending`       | Non ancora validato           |
| `verified`      | Hash match esatto             |
| `format_diff`   | Solo differenze formattazione |
| `content_diff`  | Differenze contenuto (minori) |
| `review_needed` | Richiede review manuale       |

---

## Performance Metriche

### Fetch da Brocardi

| Codice | Articoli | Tempo    | Rate     |
| ------ | -------- | -------- | -------- |
| CC     | 3191     | 18.8 min | 2.83/sec |
| CP     | 976      | 5.8 min  | 2.80/sec |

### Database Ingestion

| Codice | Inseriti | Errori | Tempo | Rate    |
| ------ | -------- | ------ | ----- | ------- |
| CC     | 3170     | 21     | 11s   | 288/sec |
| CP     | 947      | 29     | 4s    | 247/sec |

**Errori**: Duplicati URN (articoli bis/ter con stesso numero base)

---

## Prossimi Passi

### Fase 1: Completare Ingestion

```bash
# Da server Hetzner
ssh root@49.12.85.92
cd /opt/lexe-platform/lexe-max

# Fetch altri codici
uv run python scripts/fetch_cc_brocardi.py --codice CPC --rps 3
uv run python scripts/fetch_cc_brocardi.py --codice CPP --rps 3
uv run python scripts/fetch_cc_brocardi.py --codice COST --rps 3

# Ingest
uv run python scripts/ingest_normativa_to_db.py --input scripts/cpc_brocardi_results.json
```

### Fase 2: Cross-Validation

```bash
# Da Hetzner (Normattiva raggiungibile)
uv run python scripts/cross_validate_normattiva.py --codice CC --sample 0.05 --rps 1
uv run python scripts/cross_validate_normattiva.py --codice CP --sample 0.05 --rps 1
```

### Fase 3: Embeddings

```bash
# Genera embeddings per semantic search
uv run python scripts/generate_normativa_embeddings.py --codice CC --model text-embedding-3-small
```

### Fase 4: API Endpoints

Creare endpoints in `lexe-max/src/lexe_api/api/normativa.py`:

- `GET /api/v1/normativa/{codice}/{articolo}`
- `POST /api/v1/normativa/search`
- `GET /api/v1/normativa/{codice}/{articolo}/citations`

---

## Connessione Database

### Dev Locale

```bash
# Container lexe-max su porta 5434
docker exec -it lexe-max psql -U lexe_max -d lexe_max
```

### Connection String

```
postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max
```

### Python

```python
import asyncpg

conn = await asyncpg.connect(
    'postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max'
)
```

---

## Files Creati/Modificati

```
lexe-max/
├── migrations/kb/
│   └── 010_normativa_schema.sql      # Schema completo
├── src/lexe_api/kb/sources/
│   ├── base_adapter.py               # Interface + CrossValidator
│   ├── models.py                     # ArticleExtract, ValidationResult
│   └── brocardi_adapter.py           # Brocardi.it adapter
├── scripts/
│   ├── fetch_cc_brocardi.py          # Fetch da Brocardi
│   ├── ingest_normativa_to_db.py     # Load JSON → PostgreSQL
│   ├── cross_validate_normattiva.py  # Cross-validation
│   ├── ingest_cds_test.py            # Test CdS
│   ├── cc_brocardi_results.json      # Output CC fetch
│   ├── cp_brocardi_results.json      # Output CP fetch
│   └── cds_ingestion_results.json    # Output CdS test
└── docs/
    └── KB-NORMATIVA-INGESTION.md     # Questa documentazione
```

---

## Troubleshooting

### Normattiva non raggiungibile

```
Error: All connection attempts failed
```

**Causa**: Normattiva.it blocca alcuni IP range:

- IP residenziali non-italiani
- Alcuni ISP/VPN
- IP senza reverse DNS valido

**Diagnosi**:

```bash
# Se questo fallisce, il sito blocca il tuo IP
ping www.normattiva.it
curl -I https://www.normattiva.it --max-time 10
```

**Soluzione**: Eseguire da server con IP italiano o datacenter (Hetzner funziona).

### Duplicati URN

```
duplicate key value violates unique constraint "normativa_urn_nir_key"
```

**Causa**: Articoli bis/ter/quater con stesso URN base.
**Soluzione**: Normale, ignorare. Lo script usa ON CONFLICT DO UPDATE.

### Container non trovato

```
Error: Cannot connect to database
```

**Soluzione**:

```bash
docker ps | grep lexe-max
# Se non running:
cd lexe-infra && docker compose up -d lexe-max
```

---

## Riferimenti

- **Piano completo**: `C:\Users\Fra\.claude\plans\compressed-forging-moon.md`
- **URN:NIR spec**: https://www.normattiva.it/static/URN.html
- **Brocardi.it**: https://www.brocardi.it
- **Normattiva.it**: https://www.normattiva.it

---

*Documentazione generata automaticamente - 2026-02-04*
