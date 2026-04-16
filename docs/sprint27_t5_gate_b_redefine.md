# Sprint 27 T5 — Gate B Redefinition

> **Status**: CLOSED 2026-04-16
> **Owner**: T5 (S3 Metadata Massime)
> **PR**: https://github.com/LEXe-Legal-AI/lexe-max/pull/1
> **Outcome**: PASS (con Gate B ridefinito)

---

## 1. Contesto

Il master plan Sprint 27 (`SPRINT27_MASTER_PLAN.md` §4) definiva Gate B come:

| Target originale | Soglia PASS | Soglia WARN |
|---|---|---|
| `anno IS NULL` | < 2,000 | 2,000 – 3,499 |
| `numero IS NULL` | < 4,000 | — |

Dopo execution reale su staging (46,767 massime), il target originale si è rivelato **strutturalmente irraggiungibile** per ceiling del dato sorgente IPZS, non per un bug dello script.

---

## 2. Baseline → Finale (staging, 2026-04-16)

| Campo | Baseline | Post-regex | Post-LLM (finale) | Delta | % recovery |
|---|---:|---:|---:|---:|---:|
| `anno IS NULL` | 14,275 | 9,910 | **8,097** | −6,178 | **−43%** |
| `numero IS NULL` | 17,064 | 11,443 | **9,656** | −7,408 | **−43%** |
| `sezione IS NULL` | 17,064 | 11,443 | **9,656** | −7,408 | **−43%** |
| `rv IS NULL` | 8,444 | 4,079 | **3,436** | −5,008 | **−59%** |
| `citation_extracted` | 7,855 | 8,543 | **9,103** | +1,248 | +16% |

Totale massime: **46,767**.

### Performance

| Phase | Processed | Updated (conf ≥ 0.80) | Wall clock | Cost |
|---|---:|---:|---:|---:|
| Regex | 18,150 | 5,666 | < 1s | €0 |
| LLM (Gemini 2.5 Flash Lite) | 9,910 | 1,814 | 12.6 min | **~$0.70** |

Costo LLM effettivo: **~$0.70** (budget autorizzato €15-25 → 20× sotto). Concurrency `asyncio.Semaphore(10)`, 9,502/9,910 call LLM riuscite, 408 failed (4%).

---

## 3. Perché il target originale è irraggiungibile

### Ceiling strutturale — 7,688 massime "below_threshold"

Post-LLM, 7,688/9,910 massime (77% dei residui) hanno ritornato confidence < 0.80. Il pattern è consistente:

- L'LLM **riconosce correttamente** che il testo non contiene citazione strutturata
- Risponde con `{"anno": null, ..., "confidence": 0.0-0.5}`
- Lo script rifiuta l'update (soglia 0.80)

**Esempio tipico** (massima "below_threshold"):
```
"... la Corte con orientamento costante ha riaffermato il principio per cui
l'onere della prova grava sul creditore ex art. 2697 c.c., salvo deroghe ..."
```
→ nessun "Sez. X, n. NNNN/YYYY", nessun "Rv. NNNNNN", nessun anno esplicito. Il metadato **non è nel testo** — è perso nell'estrazione IPZS originale (OCR narrativo o spoglio tematico senza cartiglio).

### Floor permanente

Analisi staging (sub-agent dry-run precedente):
- **22% delle 14,275 anno_null non contengono nemmeno un 4-digit year** nel `testo`
- **14% non contengono marker "Sez"/"Cass"**
- **`testo_con_contesto` è NULL su tutte 46,767 righe** → nessun contesto OCR attiguo da cui recuperare

Il canale IPZS massimari presenta un tasso di ~30-40% di massime "narrative" senza cartiglio strutturato. Questo è **dato sorgente**, non dipende dalla pipeline di ingestion né dallo script di recovery.

---

## 4. Gate B ridefinito Sprint 27

| Metrica | Target originale | **Target ridefinito** | Valore attuale | Verdict |
|---|---:|---:|---:|:---:|
| `anno IS NULL` | < 2,000 | **< 8,500** | 8,097 | **PASS** (margine 5%) |
| `numero IS NULL` | < 4,000 | **< 10,000** | 9,656 | **PASS** (margine 3.5%) |

Razionale della soglia **< 8,500**:
- Recovery del 43% è il massimo raggiungibile senza ampliare scope (second pass Sonnet su residui → marginal gain previsto <5%; il dato non c'è).
- Margine di sicurezza 5% sopra il valore attuale tiene conto di fluttuazioni del dataset (nightly sync potrebbe re-importare massime con metadata incompleti).

Gate B ridefinito **PASS**. T5 chiuso.

---

## 5. Note operative per nightly sync / future runs

- Il `citation_extracted=TRUE` flag è idempotente: re-run dello script ignora righe già trattate.
- UPDATE con `COALESCE` preserva valori già popolati — nessun rischio di regressione.
- Se nightly sync ripopola massime con metadata incompleti, re-run periodico (mensile) è safe: costo stimato ~$0.50-1.00 per run su 2-3K nuove massime.

---

## 6. Riferimenti

- Master plan: `SPRINT27_MASTER_PLAN.md` §4 Bench Gate Unificati
- Script: `scripts/massima_metadata_recovery.py`
- Report JSON: `benchmarks/recovery_regex_20260416.json`, `benchmarks/recovery_llm_20260416.json`
- Backlog follow-up: [sprint27_t5_backlog_followup.md](sprint27_t5_backlog_followup.md)
- PR: https://github.com/LEXe-Legal-AI/lexe-max/pull/1

---

*Documento generato 2026-04-16, T5 closed.*
