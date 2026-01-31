# KB Norm Graph - Handoff Documentation

> **Version:** v3.3.0
> **Date:** 2026-01-31
> **Status:** Production Ready

---

## Overview

Il Norm Graph estrae riferimenti normativi dalle massime e li indicizza per:
1. **Lookup diretto** - Query tipo "art. 2043 c.c." → massime che citano quella norma
2. **Norm boost** - Query miste "danno ingiusto art. 2043" → hybrid search + boost

---

## Metriche Finali

### Norm Graph Stats

| Metric | Value |
|--------|-------|
| **Unique Norms** | 4,128 |
| **Total Edges** | 42,338 |
| **Massime with Norms** | 23,365 (60.3%) |
| **Lookup Latency** | < 10ms |

### Distribuzione per Codice

| Code | Norms | Citations | Avg/Norm |
|------|-------|-----------|----------|
| LEGGE | 1,257 | 11,826 | 9.4 |
| CPC | 520 | 8,740 | 16.8 |
| CC | 995 | 7,722 | 7.8 |
| DLGS | 412 | 7,244 | 17.6 |
| DL | 229 | 1,917 | 8.4 |
| DPR | 140 | 1,461 | 10.4 |
| COST | 58 | 1,420 | 24.5 |
| CPP | 298 | 1,149 | 3.9 |
| CP | 192 | 778 | 4.1 |
| TUF | 16 | 52 | 3.3 |
| TUB | 11 | 29 | 2.6 |

### Top 10 Norme Citate

| Rank | Norm | Citations |
|------|------|-----------|
| 1 | D.Lgs. n. 165/2001 | 587 |
| 2 | art. 111 Cost. | 518 |
| 3 | D.Lgs. n. 546/1992 | 438 |
| 4 | L. n. 689/1981 | 417 |
| 5 | D.Lgs. n. 150/2011 | 384 |
| 6 | D.Lgs. n. 286/1998 | 329 |
| 7 | D.Lgs. n. 109/2006 | 271 |
| 8 | L. n. 92/2012 | 271 |
| 9 | art. 327 c.p.c. | 254 |
| 10 | L. n. 69/2009 | 250 |

---

## Golden Set Evaluation

### Test Set

| Class | Count | Description |
|-------|-------|-------------|
| `pure_norm` | 60 | Solo riferimento normativo |
| `mixed` | 40 | Semantico + norma |
| **Total** | **100** | |

### Risultati (Target vs Actual)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **pure_norm Recall@10** | ≥ 98% | **100%** | ✅ PASS |
| **pure_norm MRR** | ≥ 0.90 | **1.0** | ✅ PASS |
| **pure_norm Router Accuracy** | ≥ 98% | **100%** | ✅ PASS |
| **mixed Norm Hit Rate** | ≥ 70% | **97.5%** | ✅ PASS |
| **mixed Norm MRR** | - | **0.975** | ✅ |

---

## Schema Database

### kb.norms

```sql
CREATE TABLE kb.norms (
    id TEXT PRIMARY KEY,           -- CC:2043, LEGGE:241:1990
    code TEXT NOT NULL,            -- CC, CPC, LEGGE, DLGS...
    article TEXT,                  -- 2043, 360 (for codes)
    suffix TEXT,                   -- bis, ter, quater...
    number TEXT,                   -- 241, 165 (for laws)
    year INT,                      -- 1990, 2001 (for laws)
    full_ref TEXT NOT NULL,        -- "art. 2043 c.c."
    citation_count INT DEFAULT 0
);
```

### kb.massima_norms

```sql
CREATE TABLE kb.massima_norms (
    massima_id UUID REFERENCES kb.massime(id),
    norm_id TEXT REFERENCES kb.norms(id),
    context_span TEXT,             -- snippet around citation
    run_id INT,
    PRIMARY KEY (massima_id, norm_id)
);
```

### kb.golden_norm_queries

```sql
CREATE TABLE kb.golden_norm_queries (
    id UUID PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    query_class VARCHAR(20),       -- 'pure_norm' | 'mixed'
    expected_norm_id VARCHAR(50),  -- CC:2043
    is_active BOOLEAN DEFAULT TRUE
);
```

---

## Architettura Retrieval

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                    classify_query()                      │
│  1. Citation RV? ──────────────► CITATION_RV            │
│  2. Citation Sez/Num/Anno? ────► CITATION_SEZ_NUM_ANNO  │
│  3. Norm? ─────────────────────► NORM                   │
│  4. Citation Num/Anno? ────────► CITATION_NUM_ANNO      │
│  5. else ──────────────────────► SEMANTIC               │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                     route_query()                        │
│                                                          │
│  NORM ──────► norm_lookup() ──► massime via kb.norms    │
│  SEMANTIC ──► hybrid_search() + norm_boost (optional)   │
└─────────────────────────────────────────────────────────┘
```

### Norm Lookup Flow

1. Parse query → `ParsedNorm` con canonical ID
2. Lookup `kb.norms` per verificare esistenza
3. Join `kb.massima_norms` → `kb.massime`
4. Order by `anno DESC` (recency)
5. Return top-K con score 0.85-0.95

### Norm Boost Flow (query miste)

1. Hybrid search normale → RRF results
2. Detect norm in query
3. Trova massime che citano quella norma
4. Applica boost additivo +0.10 (cap 0.20)
5. Re-sort by final_score

---

## Tipi di Norme Supportati

### Codici (article-based)

| Code | Pattern Examples | Canonical ID |
|------|------------------|--------------|
| CC | art. 2043 c.c., 2043 cc | CC:2043 |
| CPC | art. 360 c.p.c., 360 cpc | CPC:360 |
| CP | art. 640 c.p. | CP:640 |
| CPP | art. 384 c.p.p. | CPP:384 |
| COST | art. 111 Cost. | COST:111 |
| TUB | art. 117 t.u.b. | TUB:117 |
| TUF | art. 21 t.u.f. | TUF:21 |
| CAD | art. 5 c.a.d. | CAD:5 |

### Leggi (number/year-based)

| Code | Pattern Examples | Canonical ID |
|------|------------------|--------------|
| LEGGE | L. 241/1990, legge n. 241 del 1990 | LEGGE:241:1990 |
| DLGS | D.Lgs. 165/2001, dlgs 165 2001 | DLGS:165:2001 |
| DPR | D.P.R. 445/2000 | DPR:445:2000 |
| DL | D.L. 18/2020 | DL:18:2020 |

### Suffissi Articolo

bis, ter, quater, quinquies, sexies, septies, octies, novies, decies

Esempio: art. 360 bis c.p.c. → `CPC:360:bis`

---

## Scripts

### Build Norm Graph

```bash
uv run python scripts/graph/build_norm_graph.py --batch-size 500
uv run python scripts/graph/build_norm_graph.py --dry-run  # preview
uv run python scripts/graph/build_norm_graph.py --clear    # rebuild
```

### Sanity Checks

```bash
uv run python scripts/graph/sanity_check_norm_graph.py
```

Output:
- Schema integrity
- Data volume
- Distribution by code
- Top cited norms
- Citation count consistency
- Lookup performance
- Orphan edges

### Golden Set

```bash
# Generate 100 test queries
uv run python scripts/qa/generate_golden_norm_set.py

# Run evaluation
uv run python scripts/qa/run_norm_eval.py --top-k 10 --log-results
```

### Test Router

```bash
uv run python scripts/test_norm_router.py
```

---

## Grafana Panels

File: `docs/grafana/norm_graph_panels.sql`

| Panel | Query |
|-------|-------|
| Total Norms | `SELECT COUNT(*) FROM kb.norms` |
| Coverage % | Massime with norms / Total active |
| By Code (Pie) | Group by code |
| Top 20 Cited | Order by citation_count DESC |
| Eval Results | From kb.norm_eval_runs |

---

## Dirty Query Handling

Il parser gestisce query "sporche" senza punteggiatura:

| Input | Normalized | Result |
|-------|------------|--------|
| `art 2043 cc` | `art. 2043 c.c.` | CC:2043 |
| `2043 cc` | `2043 c.c.` | CC:2043 |
| `dlgs 165 2001` | `d.lgs. 165/2001` | DLGS:165:2001 |
| `111 cost` | `art. 111 cost.` | COST:111 |

---

## Known Limitations

1. **Suffix matching**: "art. 360 bis c.p.c." richiede match esatto del suffix
2. **Multi-norm queries**: Solo la prima norma viene usata per lookup
3. **Range articles**: "artt. 2043-2045" estrae solo il primo articolo

---

## Next Steps

1. **Category Graph L1** - 6-8 macro aree (Civile, Penale, Proc, Amm...)
2. **Golden Set Categorie** - 50 query per validare topic routing
3. **Topic Boost** - Leggero, solo dopo validazione

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/lexe_api/kb/graph/norm_extractor.py` | Extraction + parsing |
| `src/lexe_api/kb/retrieval/router.py` | Query routing |
| `src/lexe_api/kb/retrieval/norm_booster.py` | Hybrid + norm boost |
| `migrations/kb/005_norm_graph.sql` | Schema |
| `migrations/kb/006_golden_norm_queries.sql` | Golden set schema |
| `scripts/graph/build_norm_graph.py` | Batch builder |
| `scripts/graph/sanity_check_norm_graph.py` | Validation |
| `scripts/qa/generate_golden_norm_set.py` | Test queries |
| `scripts/qa/run_norm_eval.py` | Evaluation |

---

*Last updated: 2026-01-31*
