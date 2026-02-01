# QA Protocol - Local Test Status

**Data:** 2026-01-29 (updated)
**Database:** lexe-kb (localhost:5434)
**PDFs:** 63 in C:\PROJECTS\lexe-genesis\data\raccolta

---

## Phases Completed

| Phase | Script | Status | Risultato |
|-------|--------|--------|-----------|
| 0 | `s0_build_manifest.py` | OK | 63 PDFs, 3 doc_id linkati |
| 1 | `s1_page_extraction_stats.py` | OK | 20,005 pages, 471 empty, 165 TOC |
| 1 | `s1_year_resolution.py` | OK | 63 resolved, 63 conflicts |
| 1 | `s1_health_flags.py` | OK | 64 flags (1 empty_seq + 63 year_conflict) |
| 2 | `s2_extraction_quality.py` | OK | 63 docs, tutti grade A (0.90-0.97) |
| 2 | `s2_noise_detection.py` | OK | 3,306 noise markers |
| 3 | `s3_gate_policy_audit.py` | OK | 40,991 decisions, 7.9% accepted |
| 4 | `s4_chunk_analysis.py` | OK | 772 chunks (3 docs linkati) |
| 5 | `s5_silver_labeling.py` | OK | 772 = 100% massima, 0 LLM calls |
| 7a | `s7_generate_query_set.py` | OK | 125 queries generated |
| 7b | `s7_run_retrieval_eval.py` | OK | R@5 52%, R@10 52%, MRR 0.52 |
| 8 | `s8_assign_profiles.py` | OK | 63 clean_standard |
| 9 | `s9_llm_ambiguous_year.py` | OK | 61 LLM calls, $0.0037 |
| 9 | `s9_llm_borderline.py` | OK | 0 borderline chunks |
| 9 | `s9_llm_boundary_repair.py` | OK | 0 fragmented units |
| 10 | `s10_generate_reports.py` | OK | 6 D + 57 C |
| 10 | `s10_recommended_actions.py` | OK | 6 docs at risk |

---

## Embedding Benchmark Results

**Models Tested:**
- `mistralai/mistral-embed-2312` (existing 772 embeddings)
- `google/gemini-embedding-001` (new 772 embeddings created)

| Metrica | Mistral | Gemini | Delta |
|---------|---------|--------|-------|
| Recall@5 | 98.5% | **100%** | +1.5% |
| Recall@10 | 100% | 100% | = |
| MRR | 0.937 | **0.977** | +4.0% |

**Conclusione:** Gemini performa leggermente meglio di Mistral per questa KB.

---

## Retrieval Eval Results

| Metodo | R@5 | R@10 | MRR | Noise@10 | Latency |
|--------|-----|------|-----|----------|---------|
| R1_hybrid | 52.0% | 52.0% | 0.520 | 0% | 52ms |
| dense_only | 51.2% | 52.0% | 0.487 | 0% | 53ms |
| sparse_only | 52.0% | 52.0% | 0.520 | 0% | 0ms |

**Note:** 125 queries, 65 with expected results.

---

## LLM Costs Summary

| Script | Model | Calls | Costo |
|--------|-------|-------|-------|
| Phase 9 year | Gemini Flash | 61 | $0.0037 |
| Phase 9 borderline | - | 0 | $0.00 |
| Phase 9 boundary | - | 0 | $0.00 |
| **Totale** | | **61** | **$0.0037** |

---

## Bug Fixati

1. **`asyncpg.Range`** in `s1_health_flags.py`
   - Problema: INT4RANGE richiede `asyncpg.Range` object, non stringhe
   - Fix: `from asyncpg import Range` + `Range(start, end+1)`

2. **UUID `'None'`** in `s10_generate_reports.py`
   - Problema: `str(None)` produce `'None'` stringa invece di NULL
   - Fix: Filter `[r["doc_id"] for r in top_risk if r["doc_id"] is not None]`

3. **Mistral Embed API** in `benchmark_embeddings.py`
   - Problema: `dimensions` parameter causa 404 per Mistral (fixed a 1024)
   - Fix: Solo Gemini riceve `dimensions`, Mistral no

4. **Column name mismatch** in `benchmark_embeddings.py`
   - Problema: `expected_massima_id` non esiste
   - Fix: Usare `source_massima_id` e `ground_truth_ids`

---

## Phases Remaining

### BLOCKED / VERY SLOW

| Phase | Script | Problema | Stima |
|-------|--------|----------|-------|
| 0 | `s0_extract_reference_units.py` | hi_res extraction | 5-10 ore |
| 6 | `s6_reference_alignment.py` | Dipende da Phase 0 | Blocked |

---

## Database Summary

```
kb.pdf_manifest                  :     63 rows
kb.page_extraction_stats         : 20,005 rows
kb.pdf_year_resolution           :     63 rows
kb.pdf_health_flags              :     64 rows
kb.pdf_extraction_quality        :     63 rows
kb.gate_decisions                : 40,991 rows
kb.chunk_features                :    772 rows
kb.chunk_labels                  :    772 rows
kb.qa_ingestion_profiles         :     63 rows
kb.qa_document_reports           :     63 rows
kb.qa_global_reports             :      1 rows
kb.emb_mistral                   :    772 rows
kb.emb_gemini                    :    772 rows (NEW)
kb.retrieval_eval_queries        :    125 rows
kb.retrieval_eval_results        :    375 rows (125 x 3 methods)
kb.llm_decisions                 :     61 rows
kb.qa_reference_units            :      0 rows
kb.reference_alignment           :      0 rows
```

---

## Key Insights

### Gate Policy Analysis
- **40,991 total decisions**
- **7.9% acceptance rate** (3,248 accepted)
- **93.7% rejected as header_footer** (35,365)
- **4.9% rejected as too_citation_dense** (1,831)
- **1.4% rejected as too_short** (547)

### Noise Detection
- **3,306 total noise markers**
- Top PDFs: 2018 volumes (646, 563, 319 markers)
- 2018 layout ha significativamente più dotted lines

### Risk Distribution
- **6 docs grade D** (risk 0.66-0.71): 2014-2015 PDFs + Rassegna 2021
- **57 docs grade C** (risk 0.47-0.50): tutti gli altri

### Year Conflicts (RESOLVED)
- **63/63 documenti** avevano conflict tra filename/content/metadata
- **Tutti risolti** via LLM (Gemini Flash) con confidence 0.90-1.00

---

## Config Updates

### qa_config.py
```python
# Models (updated)
LLM_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
EMBED_MISTRAL = "mistralai/mistral-embed-2312"
EMBED_GEMINI = "google/gemini-embedding-001"
EMBED_DIM = 1024
```

---

## Recommended Next Steps

1. **Local Phase 0** (optional) - Start reference extraction in background (5-10 ore)
2. **After Phase 0** - Run Phase 6 (reference alignment)
3. **Staging** - Deploy su server con tutti i 63 PDFs
4. **Production** - Ingestion guidata basata sui profili

---

*Generated: 2026-01-29*
