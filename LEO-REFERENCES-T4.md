# Riferimenti LEO Residui - T4 Repos

> Generato: 2026-02-01
> Repos analizzati: lexe-max, lexe-tools-it, lexe-tools-br
> Pattern cercato: `leo` (case insensitive)

---

## Riepilogo

| Repository | Riferimenti Totali | Critici (P1) | Documentazione | Scripts/Config |
|------------|-------------------|--------------|----------------|----------------|
| **lexe-max** | 90+ | 1 | 40+ | 50+ |
| **lexe-tools-it** | 0 | 0 | 0 | 0 |
| **lexe-tools-br** | 0 | 0 | 0 | 0 |

---

## lexe-max

### P1 - CRITICI (Codice Sorgente)

| File | Linea | Contenuto | Azione |
|------|-------|-----------|--------|
| ~~`src/lexe_api/tools/health_monitor.py`~~ | ~~144~~ | ~~`leo-core`~~ | ✅ FIXATO → `lexe-core` |

### P2 - CONFIG (Docker/Infra) ✅ COMPLETATO

| File | Status |
|------|--------|
| `docker-compose.kb.yml` | ✅ `leo-platform-network` → `lexe-platform-network` |
| `Dockerfile.kb` | ✅ maintainer → `LEXE Platform <dev@lexe.pro>` |

### P2 - SCRIPTS (Hardcoded Paths) ✅ COMPLETATO

| File | Status |
|------|--------|
| `scripts/fix_approfondimenti.py` | ✅ DB + path aggiornati |
| `scripts/ingest_staging.py` | ✅ DB + path aggiornati |
| `scripts/ingest_recover_anno.py` | ✅ DB + path aggiornati |
| `scripts/generate_embeddings_staging.py` | ✅ DB aggiornato |
| `scripts/test_retrieval_staging.py` | ✅ DB aggiornato |
| `scripts/benchmark_r1_r2_complete.py` | ✅ path aggiornato |
| `scripts/benchmark_rerankers.py` | ✅ path aggiornato |
| `scripts/run_retrieval_benchmark.py` | ✅ path aggiornato |
| `scripts/test_openrouter_embeddings.py` | ✅ path aggiornato |
| `scripts/extract_index.py` | ✅ path aggiornato |

### P2 - SCRIPTS QA (Hardcoded Paths) ✅ COMPLETATO

| File | Status |
|------|--------|
| `scripts/qa/qa_config.py` | ✅ DB + paths aggiornati |
| `scripts/qa/README.md` | ✅ paths + comandi aggiornati |
| `scripts/qa/run_qa_protocol.sh` | ✅ path aggiornato |
| `scripts/qa/LOCAL_TEST_STATUS.md` | ✅ path aggiornato |
| `scripts/qa/smart_pdf_split.py` | ✅ path aggiornato |
| `scripts/qa/smart_split_llm.py` | ✅ path aggiornato |
| `scripts/qa/test_segmentation_fix.py` | ✅ path aggiornato |
| `scripts/qa/test_cloud_single.py` | ✅ path aggiornato |
| `scripts/qa/s0_*.py` - `scripts/qa/s10_*.py` (20 file) | ✅ tutti aggiornati |

### P3 - DOCUMENTAZIONE (Descrittiva)

| File | Note |
|------|------|
| `CLAUDE.md` | 11 riferimenti - descrivono separazione da LEO, integration examples |
| `docs/KB-HANDOFF.md` | 1 riferimento - repo URL `LEO-ITC` |
| `docs/GRAPH-IMPLEMENTATION-PLAN.md` | 3 riferimenti - `leo-postgres`, `leo-grafana`, `leo-frontend` |
| `docs/KB-MASSIMARI-COMPLETE.md` | 8 riferimenti - paths e comandi legacy |
| `docs/KB-MASSIMARI-ARCHITECTURE.md` | 1 riferimento - integrazione LEO |
| `docs/KB-MASSIMARI-BENCHMARK-REPORT.md` | 2 riferimenti - project name, integration |
| `docs/KB-MASSIMARI-STAGING-DEPLOY.md` | 7 riferimenti - paths e comandi staging |
| `mappa-lexe-max.md` | 6 riferimenti - sezione "Riferimenti LEO" (documentazione generata) |

### P4 - EXPORT/DATI (Non modificare)

| File | Note |
|------|------|
| `ChatGPT-Strategie per pipeline KB.json` | ~15 riferimenti - export ChatGPT, contiene "Carleo" (nome persona) |
| `scripts/qa/sample_100_massime.json` | 1 riferimento - "Carleo" (nome giudice in massima) |

---

## lexe-tools-it

**NESSUN RIFERIMENTO LEO** nel codice sorgente.

I 4 match trovati sono solo nella mappa generata (`mappa-lexe-tools-it.md`) che documenta l'assenza di riferimenti.

---

## lexe-tools-br

**REPOSITORY VUOTO** - Nessun file da analizzare (solo .git/).

---

## Azioni Consigliate

### Priorita 1 (Immediata)
1. [x] `health_monitor.py:144` - ✅ Sostituito `leo-core` → `lexe-core`

### Priorita 2 (Prossimo Sprint) ✅ COMPLETATO
1. [x] ✅ Docker configs aggiornati (Dockerfile.kb, docker-compose.kb.yml)
2. [x] ✅ 12 scripts principali aggiornati (DB + paths)
3. [x] ✅ 20+ scripts QA aggiornati (paths nei commenti)
4. [x] ✅ Tutti i riferimenti `leo` → `lexe` (41 file totali)

### Priorita 3 (Backlog - Documentazione)
> **STATUS: BACKLOG** - Bassa priorita, riferimenti descrittivi

1. [ ] Aggiornare `CLAUDE.md` (11 riferimenti LEO)
2. [ ] Aggiornare `docs/KB-*.md` (20+ riferimenti)
3. [ ] Aggiornare `docs/GRAPH-IMPLEMENTATION-PLAN.md` (3 riferimenti)

### Priorita 4 (Backlog - Export/Dati)
> **STATUS: BACKLOG** - Non modificare, dati storici

1. [ ] `ChatGPT-Strategie per pipeline KB.json` - export ChatGPT storico
2. [ ] `scripts/qa/sample_100_massime.json` - contiene "Carleo" (nome giudice)

---

## Statistiche Finali T4

| Categoria | Originali | Fixati | Rimasti |
|-----------|-----------|--------|---------|
| Codice sorgente (P1) | 1 | ✅ 1 | 0 |
| Config/Docker (P2) | 2 | ✅ 2 | 0 |
| Scripts (P2) | 39 | ✅ 39 | 0 |
| Documentazione (P3) | 40+ | - | 40+ |
| Export/Dati (P4) | 15+ | - | 15+ |
| **Totale fixati** | - | **42** | - |
| **lexe-tools-it** | **0** | - | **0** |
| **lexe-tools-br** | **0** | - | **0** |

> **P3/P4 non fixati**: Documentazione descrittiva e export ChatGPT - bassa priorita

---

*Generato da T4 - 2026-02-01*
