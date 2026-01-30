# Checklist Chirurgica Rielaborazione Civile

**Data:** 2026-01-30
**Obiettivo:** Aumentare massime estratte e coverage per documenti Civile critici

---

## Executive Summary

| Categoria | Documenti | Azione Prioritaria |
|-----------|-----------|-------------------|
| Non elaborati (0 massime) | 5 | Fix classification + re-ingest |
| Coverage < 15% | 5 | Citation-anchored extraction |
| Coverage 15-40% | 7 | TOC skip + gate tuning |
| Coverage 40-60% | 10 | Gate tuning only |
| High collision (>40%) | 6 | Split per citazione |

---

## FASE 1: DOCUMENTI NON ELABORATI (Priorità CRITICA)

Questi 5 documenti hanno `anno=0`, `tipo=N/A`, 0 massime.
**Root cause:** Classification fallita, quindi guided_ingestion li ha saltati.

### 1.1 Volume III_2016_Approfond_Tematici.pdf

| Campo | Valore |
|-------|--------|
| Pages | 144 |
| Massime | 0 |
| Problema | Non classificato |

**Diagnosi:** "Approfondimenti Tematici" è un doc_type diverso, probabilmente `commentary_only` o `list_only`.

**Azione:**
```
[ ] 1. Classificare manualmente: doc_type = "commentary_only", profile = "baseline_toc_filter"
[ ] 2. Verificare se contiene massime citabili (pattern Rv., Sez.)
[ ] 3. Se sì: citation_anchored extraction con window 3+1
[ ] 4. Se no: skip (è puro commento, non massimario)
[ ] 5. Re-ingest con nuovo batch_id
```

**Gate settings:**
```python
{
    "min_length": 120,  # più basso per commentary
    "citation_ratio_max": 0.08,  # alto, è citation-dense
    "toc_skip_pages": 10,
    "profile": "baseline_toc_filter"
}
```

---

### 1.2 Volume III_2017_Approfond_Tematici.pdf

| Campo | Valore |
|-------|--------|
| Pages | 152 |
| Massime | 0 |
| Problema | Non classificato |

**Diagnosi:** Stesso pattern di 2016, stessa collana.

**Azione:** Identica a 1.1

---

### 1.3 RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ 2021

| Campo | Valore |
|-------|--------|
| Pages | 362 |
| Massime | 0 |
| Problema | Non classificato |

**Diagnosi:** "Rassegna" è un formato diverso - contiene massime MA con struttura narrativa.

**Azione:**
```
[ ] 1. Classificare: doc_type = "mixed", profile = "mixed_hybrid"
[ ] 2. Prescan per toc_page_ratio (probabilmente alto)
[ ] 3. TOC skip aggressivo (prime 15-20 pagine)
[ ] 4. Citation-anchored extraction
[ ] 5. Re-ingest
```

**Gate settings:**
```python
{
    "min_length": 150,
    "citation_ratio_max": 0.05,
    "toc_skip_pages": 20,
    "profile": "mixed_hybrid"
}
```

---

### 1.4 & 1.5 Rassegna della giurisprudenza di legittimità (442 e 504 pages)

**Diagnosi:** Stessa collana di 1.3, volumi più grandi.

**Azione:** Identica a 1.3, ma con `toc_skip_pages`: 25 per il volume da 504 pagine.

---

## FASE 2: CIVILE CON COVERAGE < 15% (Priorità ALTA)

Questi documenti hanno poche massime estratte (28-47) su centinaia di pagine.
**Root cause:** L'estrattore attuale non riconosce le massime integrate nel testo.

### 2.1 Volume I_2016_Massimario_Civile_1_372.pdf

| Campo | Valore |
|-------|--------|
| Pages | 408 |
| Massime estratte | 43 |
| Coverage | 7.8% |
| Ref units | 395 |
| Collision rate | 33.3% |

**Diagnosi:**
- 408 pagine → dovrebbe avere 300-500 massime
- Solo 43 → estrattore sta tagliando il 90%+
- Coverage 7.8% → quasi tutto il contenuto è "unmatched"

**Pattern identificato:** Volume I Civile 2016 ha indice generale esteso + massime con citazione integrata tipo "Sez. L, n. 1234, Rv. 123456".

**Azione:**
```
[ ] 1. Analisi TOC: identificare pagine indice (stima: 1-30)
[ ] 2. Prescan citazioni: contare occorrenze Rv., Sez., n. per pagina
[ ] 3. Implementare citation_anchored_extraction:
      - Per ogni match di pattern citazione
      - Estrai window: 2 frasi prima + citazione + 1 frase dopo
      - Se multiple citazioni in paragrafo, split per citazione
[ ] 4. Gate policy: citation_dense (ratio max 0.06)
[ ] 5. Re-ingest con nuovo batch_id
[ ] 6. Verifica: target 200+ massime (5x attuale)
```

**Gate settings:**
```python
{
    "min_length": 120,
    "citation_ratio_max": 0.06,
    "toc_skip_pages": 30,
    "toc_skip_pattern": r"^(INDICE|Indice|SOMMARIO|Capitolo)",
    "profile": "structured_parent_child",
    "extraction_mode": "citation_anchored",
    "citation_window": {"before": 2, "after": 1}
}
```

---

### 2.2 2014 Mass civile Vol 1 pagg 408.pdf

| Campo | Valore |
|-------|--------|
| Pages | 408 |
| Massime estratte | 47 |
| Coverage | 8.2% |
| Collision rate | 40.6% |

**Diagnosi:** Stesso pattern del 2016 Vol I. Collision alta (40.6%) suggerisce che le poche massime estratte matchano multiple ref units → estrattore sta aggregando troppo.

**Azione:**
```
[ ] 1. TOC skip: prime 25-30 pagine
[ ] 2. Citation-anchored extraction con split per citazione
[ ] 3. Gate: citation_dense
[ ] 4. Target: 200+ massime
```

**Gate settings:** Come 2.1

---

### 2.3 Volume I_2017_Massimario_Civile_1_372.pdf

| Campo | Valore |
|-------|--------|
| Pages | 376 |
| Massime estratte | 41 |
| Coverage | 9.9% |

**Diagnosi:** Stessa collana, stesso problema.

**Azione:** Come 2.1

---

### 2.4 Volume II_2024_Massimario_Civile

| Campo | Valore |
|-------|--------|
| Pages | 273 |
| Massime estratte | 28 |
| Coverage | 13.5% |
| Collision rate | **52.8%** (peggiore!) |

**Diagnosi:**
- Solo 28 massime su 273 pagine = 0.1 massime/pagina (dovrebbe essere ~1)
- Collision 52.8% = metà dei match sono duplicati → estrattore crea chunks troppo grandi

**Azione:**
```
[ ] 1. Split aggressivo per citazione
[ ] 2. Ridurre max_chunk_length a 1500 chars
[ ] 3. Citation-anchored con window stretto (1+1)
[ ] 4. Target: 150+ massime
```

**Gate settings:**
```python
{
    "min_length": 100,
    "max_length": 1500,  # ridotto!
    "citation_ratio_max": 0.07,
    "toc_skip_pages": 15,
    "extraction_mode": "citation_anchored",
    "citation_window": {"before": 1, "after": 1},
    "split_on_multiple_citations": True
}
```

---

### 2.5 Volume II_2023_Massimario_Civile

| Campo | Valore |
|-------|--------|
| Pages | 268 |
| Massime estratte | 35 |
| Coverage | 14.1% |
| Collision rate | 43.2% |

**Diagnosi:** Stesso pattern 2024.

**Azione:** Come 2.4

---

## FASE 3: CIVILE CON COVERAGE 15-40% (Priorità MEDIA)

Questi hanno più massime ma ancora sotto-estratti.

### 3.1 Volume I_2024_Massimario_Civile

| Coverage | Massime | Target |
|----------|---------|--------|
| 23.8% | 68 | 150+ |

**Gate settings:**
```python
{
    "min_length": 120,
    "citation_ratio_max": 0.06,
    "toc_skip_pages": 20,
    "extraction_mode": "citation_anchored"
}
```

---

### 3.2 Volume II_2018_Massimario_Civile

| Coverage | Massime | Collision | Target |
|----------|---------|-----------|--------|
| 25.5% | 70 | 43.8% | 180+ |

**Azione speciale:** Collision alta → split per citazione prioritario.

---

### 3.3 Volume II_2022_Massimario_Civile

| Coverage | Massime | Target |
|----------|---------|--------|
| 26.9% | 72 | 150+ |

---

### 3.4 Volume I_2023_Massimario_Civile

| Coverage | Massime | Collision | Target |
|----------|---------|-----------|--------|
| 28.0% | 68 | 46.0% | 150+ |

---

### 3.5 Volume I_2018_Massimario_Civile

| Coverage | Massime | Target | Note |
|----------|---------|--------|------|
| 36.7% | **611** | 800+ | Ha GIA molte massime! |

**Diagnosi speciale:** Questo volume ha 611 massime (il più alto tra i Civile!) ma coverage solo 36.7%.

**Interpretazione:** L'estrattore funziona QUI, ma le massime estratte non matchano le reference units → possibile problema di **segmentazione reference** o **normalizzazione**.

**Azione:**
```
[ ] 1. NON cambiare extraction (funziona!)
[ ] 2. Verificare normalizzazione delle 611 massime
[ ] 3. Verificare se reference units per questo doc sono corrette
[ ] 4. Possibile: reference troppo grandi (page-based) vs massime piccole
```

---

### 3.6 2015 approfondimenti tematici Volume 3

| Coverage | Massime | Collision | Target |
|----------|---------|-----------|--------|
| 38.2% | 74 | 47.1% | 150+ |

**Azione:** Citation-anchored + split per citazione (collision alta).

---

### 3.7 Volume I_2022_Massimario_Civile

| Coverage | Massime | Collision | Target |
|----------|---------|-----------|--------|
| 42.6% | 123 | 39.9% | 200+ |

---

## FASE 4: CIVILE CON COVERAGE 40-60% (Priorità BASSA)

Questi richiedono solo tuning leggero dei gate.

| Documento | Coverage | Massime | Azione |
|-----------|----------|---------|--------|
| 2014 Mass civile Vol 2 | 46.1% | 94 | Gate tuning |
| rassegna civile 2020 vol_IV | 47.5% | 107 | Gate tuning |
| 2015 principi diritto proc. | 48.6% | 97 | Gate tuning |
| Volume II_2017_Civile | 50.4% | 102 | Gate tuning |
| Volume II_2016_Civile | 50.5% | 90 | Gate tuning |
| Volume III_2023_Civile | 53.2% | 118 | Gate tuning |
| Volume III_2018_Civile | 54.1% | 180 | Gate tuning |
| Volume III_2024_Civile | 57.6% | 110 | Gate tuning |

**Gate settings comuni:**
```python
{
    "min_length": 130,
    "citation_ratio_max": 0.05,
    "toc_skip_pages": 15
}
```

---

## IMPLEMENTAZIONE: Ordine Operativo

### Step 1: Fix Classification (5 non-elaborati)
```bash
# Script da creare: fix_unclassified_civile.py
uv run python scripts/qa/fix_unclassified_civile.py
```

### Step 2: Implementare Citation-Anchored Extraction
```bash
# Modifica: src/lexe_api/kb/ingestion/massima_extractor.py
# Aggiungere modalità "citation_anchored"
```

### Step 3: Re-ingest Civile Critici (coverage < 15%)
```bash
# Script da creare: reingest_civile_critical.py
uv run python scripts/qa/reingest_civile_critical.py --batch-name civile_fix_v1
```

### Step 4: Re-ingest Civile Medium (coverage 15-40%)
```bash
uv run python scripts/qa/reingest_civile_medium.py --batch-name civile_fix_v1
```

### Step 5: Verificare Alignment
```bash
uv run python scripts/qa/s6_reference_alignment_v2.py
uv run python scripts/qa/verify_alignment_fix.py
```

### Step 6: Report Finale
```bash
uv run python scripts/qa/qa_full_report.py > docs/QA_REPORT_POST_CIVILE_FIX.md
```

---

## Metriche Target Post-Fix

| Metrica | Attuale | Target | Stretch |
|---------|---------|--------|---------|
| Coverage media Civile | 62.6% | **80%** | 90% |
| Coverage media globale | 66.7% | **75%** | 85% |
| Documenti < 60% coverage | 22 | **< 10** | < 5 |
| Documenti < 15% coverage | 5 | **0** | 0 |
| Massime totali | 10,093 | **15,000** | 20,000 |
| Collision rate media | 27.4% | **< 20%** | < 15% |

---

## Checklist Finale Pre-Deploy

```
[ ] Tutti i 5 non-elaborati hanno massime > 0
[ ] Nessun documento con coverage < 15%
[ ] Coverage media Civile >= 75%
[ ] Coverage media globale >= 70%
[ ] Alignment trust >= 95%
[ ] Embedding % < 5%
[ ] Collision rate media < 25%
[ ] Backfill testo_normalizzato completato
[ ] Nuovo batch_id tracciabile
[ ] Report QA generato e salvato
```

---

*Generato da QA Protocol v3.2 - 2026-01-30*
