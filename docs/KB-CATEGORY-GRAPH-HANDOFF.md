# KB Category Graph - Handoff Documentation

> **Version:** v3.4.0
> **Date:** 2026-01-31
> **Status:** Production Ready

---

## Overview

Il Category Graph classifica le massime in categorie gerarchiche per:
1. **Topic filtering** - Filtrare risultati per area giuridica
2. **Topic boost** - Boosting leggero per query topic-specific (future)
3. **Analytics** - Distribuzione massime per materia

---

## Metriche Finali

### Category Graph Stats

| Metric | Value |
|--------|-------|
| **L1 Categories** | 8 |
| **L2 Categories** | 43 |
| **Total Assignments** | 45,332 |
| **L1 Assignments** | 37,646 |
| **L2 Assignments** | 7,686 |
| **L1 Coverage** | 97.2% |
| **Avg Confidence** | 0.575 |

### Distribuzione L1

| Category | Assignments | % |
|----------|-------------|---|
| PROCESSUALE_CIVILE | 20,639 | 54.8% |
| CIVILE | 8,724 | 23.2% |
| LAVORO | 4,521 | 12.0% |
| PROCESSUALE_PENALE | 857 | 2.3% |
| TRIBUTARIO | 789 | 2.1% |
| PENALE | 722 | 1.9% |
| AMMINISTRATIVO | 707 | 1.9% |
| FALLIMENTARE_CRISI | 687 | 1.8% |

### L2 by Parent

| Parent | L2 Assignments |
|--------|----------------|
| PROCESSUALE_CIVILE | 3,465 |
| CIVILE | 3,160 |
| LAVORO | 736 |
| FALLIMENTARE_CRISI | 126 |
| PENALE | 82 |
| TRIBUTARIO | 44 |
| AMMINISTRATIVO | 38 |
| PROCESSUALE_PENALE | 35 |

---

## Schema Database

### kb.categories

```sql
CREATE TABLE kb.categories (
    id TEXT PRIMARY KEY,           -- CIVILE, CIVILE_RESP_CIV
    name TEXT NOT NULL,            -- "Diritto Civile"
    description TEXT,              -- Description for UI
    level INT NOT NULL,            -- 1=macro, 2=sub, 3=specific
    parent_id TEXT REFERENCES kb.categories(id),
    keywords TEXT[] DEFAULT '{}',  -- Keywords for matching
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);
```

### kb.category_assignments

```sql
CREATE TABLE kb.category_assignments (
    massima_id UUID REFERENCES kb.massime(id),
    category_id TEXT REFERENCES kb.categories(id),
    confidence FLOAT NOT NULL,     -- 0-1
    method TEXT NOT NULL,          -- 'keyword', 'embedding', 'hybrid', 'manual'
    evidence_terms TEXT[],         -- Keywords that triggered match
    run_id INT,                    -- Classification run
    created_at TIMESTAMPTZ,
    PRIMARY KEY (massima_id, category_id, run_id)
);
```

---

## Gerarchia Categorie

### L1 Macro (8)

| ID | Name | Description |
|----|------|-------------|
| CIVILE | Diritto Civile | Obbligazioni, contratti, proprietà, famiglia |
| LAVORO | Diritto del Lavoro | Rapporto di lavoro, licenziamento, previdenza |
| PROCESSUALE_CIVILE | Diritto Processuale Civile | Procedura civile, impugnazioni, esecuzione |
| PENALE | Diritto Penale | Reati, pene, circostanze |
| PROCESSUALE_PENALE | Diritto Processuale Penale | Procedura penale, indagini, cautelare |
| AMMINISTRATIVO | Diritto Amministrativo | PA, appalti, urbanistica |
| TRIBUTARIO | Diritto Tributario | Imposte, accertamento, contenzioso |
| FALLIMENTARE_CRISI | Fallimentare e Crisi | Fallimento, concordato, liquidazione |

### L2 Subcategories (43)

#### CIVILE (10)
- CIVILE_OBBLIGAZIONI - Obbligazioni e Contratti
- CIVILE_RESP_CIVILE - Responsabilità Civile
- CIVILE_PROPRIETA - Proprietà e Diritti Reali
- CIVILE_FAMIGLIA - Diritto di Famiglia
- CIVILE_SUCCESSIONI - Successioni e Donazioni
- CIVILE_SOCIETA - Diritto Societario
- CIVILE_ASSICURAZIONE - Assicurazioni
- CIVILE_LOCAZIONI - Locazioni
- CIVILE_CONSUMATORE - Tutela del Consumatore
- CIVILE_BANCA_FINANZA - Banca e Finanza

#### LAVORO (5)
- LAVORO_RAPPORTO - Rapporto di Lavoro
- LAVORO_LICENZIAMENTO - Licenziamento
- LAVORO_PUBBLICO - Pubblico Impiego
- LAVORO_PREVIDENZA - Previdenza e Assistenza
- LAVORO_SINDACALE - Diritto Sindacale

#### PROCESSUALE_CIVILE (5)
- PROC_CIV_COGNIZIONE - Processo di Cognizione
- PROC_CIV_IMPUGNAZIONI - Impugnazioni Civili
- PROC_CIV_ESECUZIONE - Esecuzione Forzata
- PROC_CIV_CAUTELARE - Tutela Cautelare
- PROC_CIV_SPECIALI - Procedimenti Speciali

#### PENALE (5)
- PENALE_PERSONA - Reati contro la Persona
- PENALE_PATRIMONIO - Reati contro il Patrimonio
- PENALE_PA - Reati contro la PA
- PENALE_STUPEFACENTI - Stupefacenti
- PENALE_TRIBUTARIO - Reati Tributari

#### PROCESSUALE_PENALE (5)
- PROC_PEN_INDAGINI - Indagini Preliminari
- PROC_PEN_CAUTELARI - Misure Cautelari
- PROC_PEN_RITI - Riti Speciali
- PROC_PEN_IMPUGNAZIONI - Impugnazioni Penali
- PROC_PEN_ESECUZIONE - Esecuzione Penale

#### AMMINISTRATIVO (5)
- AMM_ATTO - Atto Amministrativo
- AMM_PROCEDIMENTO - Procedimento Amministrativo
- AMM_APPALTI - Appalti Pubblici
- AMM_URBANISTICA - Urbanistica ed Edilizia
- AMM_CONTENZIOSO - Contenzioso Amministrativo

#### TRIBUTARIO (4)
- TRIB_ACCERTAMENTO - Accertamento Tributario
- TRIB_RISCOSSIONE - Riscossione
- TRIB_CONTENZIOSO - Contenzioso Tributario
- TRIB_IVA - IVA

#### FALLIMENTARE_CRISI (4)
- FALL_FALLIMENTO - Fallimento / Liquidazione Giudiziale
- FALL_STATO_PASSIVO - Stato Passivo
- FALL_REVOCATORIA - Revocatoria
- FALL_CONCORDATO - Concordato

---

## Architettura Classificatore

```
Massima Text
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  classify_massima()                      │
│                                                          │
│  1. Match L1 keywords → Select best L1                  │
│  2. Match L2 keywords under best L1                     │
│  3. Calculate confidence scores                         │
│  4. Return L1 (always) + L2 (if conf >= 0.50)          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
CategoryMatch(category_id, level, confidence, evidence_terms)
```

### Confidence Formula

**L1:**
```
base = match_count / min(total_keywords, 20) + 0.20
confidence = min(base * 1.5, 0.95)
```

**L2:**
```
base = match_count / min(total_keywords, 8)
confidence = min(base * 1.5, 0.95)
```

L2 is more lenient (smaller denominator) to ensure subcategory assignments.

---

## Scripts

### Seed Categories

```bash
uv run python scripts/graph/seed_categories.py
uv run python scripts/graph/seed_categories.py --clear  # reseed
```

### Build Category Graph

```bash
uv run python scripts/graph/build_category_graph.py
uv run python scripts/graph/build_category_graph.py --batch-size 500
uv run python scripts/graph/build_category_graph.py --clear  # rebuild
uv run python scripts/graph/build_category_graph.py --dry-run  # preview
```

### Sanity Checks

```bash
uv run python scripts/graph/sanity_check_category_graph.py
```

Output:
- Schema integrity (3 checks)
- Category definitions (3 checks)
- Assignment volume (2 checks)
- Coverage (1 check)
- Distribution balance (1 check)
- Confidence distribution (1 check)
- Referential integrity (2 checks)
- Run tracking (1 check)

---

## Views and Functions

### kb.category_stats

```sql
SELECT * FROM kb.category_stats;
-- id, name, level, parent_id, assignment_count, avg_confidence
```

### kb.massime_with_category

```sql
SELECT * FROM kb.massime_with_category WHERE l1_category = 'CIVILE';
-- massima_id, rv, sezione, numero, anno, l1_category, l1_name, l1_confidence
```

### kb.get_massima_categories(uuid)

```sql
SELECT * FROM kb.get_massima_categories('abc-123-...');
-- category_id, category_name, level, parent_id, confidence, method
```

### kb.get_massima_l1(uuid)

```sql
SELECT kb.get_massima_l1('abc-123-...');
-- Returns: 'CIVILE'
```

---

## API Usage (Future)

```python
# Filter search by category
results = await search_kb(
    query="danno da incidente",
    filters={"category": "CIVILE_RESP_CIVILE"}
)

# Topic boost
results = await search_kb(
    query="responsabilità civile custodia",
    topic_boost={"CIVILE_RESP_CIVILE": 0.10}
)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `migrations/kb/007_category_graph.sql` | Schema |
| `migrations/kb/008_golden_category_queries.sql` | Golden set schema |
| `src/lexe_api/kb/graph/categories.py` | Category definitions (51) |
| `src/lexe_api/kb/graph/category_classifier.py` | Keyword classifier |
| `scripts/graph/seed_categories.py` | Seed definitions |
| `scripts/graph/build_category_graph.py` | Batch classification |
| `scripts/graph/sanity_check_category_graph.py` | Validation |
| `scripts/qa/generate_golden_category_set.py` | Test queries |
| `scripts/qa/run_category_eval.py` | Evaluation |

---

## Known Limitations

1. **Keyword-only**: Classification based solely on keyword matching
2. **L2 threshold**: Low L2 coverage (~20%) due to keyword sparsity
3. **Single L1**: Each massima assigned only one L1 category
4. **No embedding**: Future improvement with embedding-based classification

---

## Next Steps

1. **Topic Filter API** - Add category filter to search endpoint
2. **Topic Boost** - Light boosting for category-matching results
3. **Embedding Classifier** - Use embeddings for better L2 coverage
4. **Multi-L1** - Allow multiple L1 categories per massima

---

*Last updated: 2026-01-31*
