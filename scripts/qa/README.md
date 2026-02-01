# QA Protocol - KB Massimari

Data dictionary and execution guide for the QA protocol.

## Quick Start

```bash
# On staging server (91.99.229.111)
cd /opt/lexe-platform/lexe-max

# 1. Run migration
psql -U lexe -d lexe -f scripts/qa/migrations/001_qa_protocol_tables.sql

# 2. Run all phases
export OPENROUTER_API_KEY='sk-or-...'
bash scripts/qa/run_qa_protocol.sh

# 3. Run single phase
bash scripts/qa/run_qa_protocol.sh --phase 0
bash scripts/qa/run_qa_protocol.sh --phase 1-3
bash scripts/qa/run_qa_protocol.sh --phase guided
```

## File Map

| File | Phase | Description |
|------|-------|-------------|
| `migrations/001_qa_protocol_tables.sql` | - | Schema (17 tables, 3 ENUMs, 1 view) |
| `s0_build_manifest.py` | 0 | Build PDF manifest + qa_run |
| `s0_extract_reference_units.py` | 0 | Independent ground-truth extraction (hi_res) |
| `s1_page_extraction_stats.py` | 1 | Per-page stats |
| `s1_year_resolution.py` | 1 | Year resolution from 3 sources |
| `s1_health_flags.py` | 1 | Health flags (depends on s1_page + s1_year) |
| `s2_extraction_quality.py` | 2 | Per-document extraction quality |
| `s2_noise_detection.py` | 2 | Noise pattern detection |
| `s3_gate_policy_audit.py` | 3 | Gate policy audit with structured logging |
| `s4_chunk_analysis.py` | 4 | Chunk features (TOC score, citation score) |
| `s5_silver_labeling.py` | 5 | 3-step silver labeling (heur + cheap + LLM) |
| `s6_reference_alignment.py` | 6 | Reference alignment (exact/partial/split/merged) |
| `s7_generate_query_set.py` | 7 | Generate 200+ eval queries |
| `s7_run_retrieval_eval.py` | 7 | Run retrieval eval (R@5, R@10, MRR, noise) |
| `s8_assign_profiles.py` | 8 | Assign ingestion profiles |
| `s9_llm_ambiguous_year.py` | 9 | LLM: resolve year conflicts |
| `s9_llm_borderline.py` | 9 | LLM: classify borderline chunks |
| `s9_llm_boundary_repair.py` | 9 | LLM: boundary repair for split units |
| `s10_generate_reports.py` | 10 | Risk reports per document + global |
| `s10_recommended_actions.py` | 10 | Smoke tests + recommendations |
| `guided_ingestion.py` | 11 | Profile-based re-ingestion |
| `run_qa_protocol.sh` | - | Orchestrator shell |

## Library Module

`src/lexe_api/kb/ingestion/gate_policy.py` - Gate policy with structured logging.

## Data Dictionary

### ENUM Types

| Type | Values |
|------|--------|
| `kb.qa_decision` | accepted, rejected |
| `kb.qa_quality_grade` | A, B, C, D |
| `kb.qa_match_type` | exact, partial, split, merged, unmatched |

### Tables

#### Phase 0

**`kb.qa_runs`** - QA run versioning
| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | PK |
| run_name | TEXT | Run identifier |
| git_sha | TEXT | Git commit hash |
| config_json | JSONB | Gate params, lib versions, PDF source |
| status | TEXT | running/completed/failed |

**`kb.ingest_batches`** - Ingestion batch tracking
| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | PK |
| batch_name | TEXT | UNIQUE (standard_v1, guided_v1) |
| pipeline | TEXT | Pipeline identifier |

**`kb.pdf_manifest`** - PDF inventory (63 files)
| Column | Type | Description |
|--------|------|-------------|
| doc_id | UUID | FK to kb.documents |
| sha256 | TEXT | UNIQUE file hash |
| pages | INT | Page count (PyMuPDF) |
| bytes | BIGINT | File size |

**`kb.qa_reference_units`** - Ground truth (hi_res extraction)
| Column | Type | Description |
|--------|------|-------------|
| manifest_id | BIGINT | FK to pdf_manifest |
| content_hash | TEXT | From stable normalization |
| extraction_method | TEXT | unstructured_hi_res |

#### Phase 1

**`kb.page_extraction_stats`** - Per-page stats
| Column | Type | Description |
|--------|------|-------------|
| is_toc_candidate | BOOLEAN | TOC heuristic flag |
| non_alnum_ratio | FLOAT | Non-alphanumeric ratio |
| valid_chars_ratio | FLOAT | Valid character ratio |

**`kb.pdf_year_resolution`** - Year from 3 sources
| Column | Type | Description |
|--------|------|-------------|
| anno_from_filename | INT | From filename regex |
| anno_from_content | INT | From first pages text |
| anno_from_metadata | INT | From PDF metadata |
| has_conflict | BOOLEAN | At least 2 sources differ |

**`kb.pdf_health_flags`** - Health issues
| Column | Type | Description |
|--------|------|-------------|
| flag_type | TEXT | empty_page_sequence, low_ocr_quality, etc. |
| severity | INT | 1-5 (CHECK constraint) |
| page_range | INT4RANGE | Affected pages |

#### Phase 2

**`kb.pdf_extraction_quality`** - Per-doc quality
| Column | Type | Description |
|--------|------|-------------|
| quality_grade | qa_quality_grade | A/B/C/D ENUM |
| overall_quality_score | FLOAT | 0-1 composite |

#### Phase 3

**`kb.gate_decisions`** - Gate policy log
| Column | Type | Description |
|--------|------|-------------|
| decision | qa_decision | accepted/rejected ENUM |
| rejection_reason | TEXT | too_short, too_citation_dense, etc. |
| rejection_details | JSONB | Numeric details |

#### Phase 4

**`kb.chunk_features`** - Chunk analysis
| Column | Type | Description |
|--------|------|-------------|
| toc_infiltration_score | FLOAT | 0-1, CHECK constraint |
| citation_list_score | FLOAT | 0-1, CHECK constraint |

#### Phase 5

**`kb.chunk_labels`** - Silver labels
| Column | Type | Description |
|--------|------|-------------|
| heur_label | TEXT | Step 1 heuristic |
| llm_label | TEXT | Step 3 LLM (nullable) |
| final_label | TEXT | Resolved label |
| label_method | TEXT | heuristic/cheap_heuristic/llm |

#### Phase 6

**`kb.reference_alignment`** - Unit alignment
| Column | Type | Description |
|--------|------|-------------|
| match_type | qa_match_type | exact/partial/split/merged/unmatched |
| fragment_count | INT | Number of fragments (split) |
| fusion_count | INT | Number of fused refs (merged) |

**`kb.reference_alignment_summary`** - Per-doc summary
| Column | Type | Description |
|--------|------|-------------|
| coverage_pct | FLOAT | 0-100 |
| fragmentation_score | FLOAT | Avg fragments per split |

#### Phase 7

**`kb.retrieval_eval_results`** - Per-query results
| Column | Type | Description |
|--------|------|-------------|
| noise_rate_at_10 | FLOAT | % noise in top-10 |
| recall_at_5 / recall_at_10 | FLOAT | Recall metrics |

#### Phase 8-10

**`kb.qa_ingestion_profiles`** - Profile per PDF
| Column | Type | Description |
|--------|------|-------------|
| profile | TEXT | clean_standard, ocr_needed, etc. |

**`kb.qa_document_reports`** - Risk reports
| Column | Type | Description |
|--------|------|-------------|
| composite_risk_score | FLOAT | 0-1 weighted formula |
| risk_grade | TEXT | A/B/C/D/F |

### View

**`kb.pdf_rankings_by_risk`** - PDFs ordered by risk score DESC.

## Risk Score Formula

```
risk = (1 - extraction_quality) * 0.25
     + gate_penalty * 0.20
     + (100 - coverage_pct)/100 * 0.30
     + (1 - recall_5) * 0.15
     + health_flags * 0.02 (cap 0.10)
```

Grades: A (0-0.2), B (0.2-0.4), C (0.4-0.6), D (0.6-0.8), F (0.8-1.0)

## Ingestion Profiles

| Profile | Criteria | Actions |
|---------|----------|---------|
| clean_standard | grade=A, no conflicts | Standard pipeline |
| legacy_layout_2010_2013 | anno 2010-2013, low quality | min_length=120 |
| toc_heavy | >10% toc chunks | Skip TOC pages |
| citation_dense | >20% citation chunks | citation_ratio=5% |
| ocr_needed | quality<0.6 | hi_res extraction |

## LLM Budget

| Trigger | Calls | Cost |
|---------|-------|------|
| Silver labeling | ~300 | $0.05 |
| Query embedding | ~200 | $0.02 |
| Ambiguous year | ~5 | $0.003 |
| Borderline classify | ~150 | $0.025 |
| Boundary repair | ~50 | $0.025 |
| **Total** | **~705** | **~$0.12** |

## Verification Queries

```sql
-- Post Phase 0
SELECT count(*) FROM kb.pdf_manifest;  -- 63
SELECT count(DISTINCT sha256) FROM kb.pdf_manifest;  -- 63

-- Post Phase 1
SELECT count(DISTINCT manifest_id) FROM kb.page_extraction_stats;  -- 63

-- Post Phase 3
SELECT decision, count(*) FROM kb.gate_decisions GROUP BY decision;

-- Post Phase 5
SELECT final_label, count(*) FROM kb.chunk_labels GROUP BY final_label;

-- Post Phase 6
SELECT avg(coverage_pct) FROM kb.reference_alignment_summary;  -- >= 85%

-- Post Phase 7
SELECT method, avg(recall_at_10), avg(noise_rate_at_10)
FROM kb.retrieval_eval_summary GROUP BY method;

-- Post Phase 10
SELECT risk_grade, count(*) FROM kb.pdf_rankings_by_risk GROUP BY risk_grade;
```
