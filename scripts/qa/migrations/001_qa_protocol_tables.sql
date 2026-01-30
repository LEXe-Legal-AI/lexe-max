-- KB Massimari QA Protocol - Schema Migration
-- Version: 1.0
-- Date: 2026-01-28
-- Schema: kb (extends existing)
--
-- Creates 17 tables + 3 ENUM types + 1 view for the QA protocol.
-- All tables include qa_run_id for full traceability.

BEGIN;

CREATE SCHEMA IF NOT EXISTS kb;

-- ============================================================
-- ENUM TYPES
-- ============================================================

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'qa_decision') THEN
    CREATE TYPE kb.qa_decision AS ENUM ('accepted', 'rejected');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'qa_quality_grade') THEN
    CREATE TYPE kb.qa_quality_grade AS ENUM ('A', 'B', 'C', 'D');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'qa_match_type') THEN
    CREATE TYPE kb.qa_match_type AS ENUM ('exact', 'partial', 'split', 'merged', 'unmatched');
  END IF;
END$$;

-- ============================================================
-- FASE 0: QA Runs + Ingest Batches + Manifest + Reference Units
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.qa_runs (
  id BIGSERIAL PRIMARY KEY,
  run_name TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'running',
  git_sha TEXT,
  pipeline TEXT,
  notes TEXT,
  config_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_qa_runs_started_at ON kb.qa_runs(started_at);

CREATE TABLE IF NOT EXISTS kb.ingest_batches (
  id BIGSERIAL PRIMARY KEY,
  batch_name TEXT NOT NULL UNIQUE,
  pipeline TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'running'
);

CREATE INDEX IF NOT EXISTS ix_ingest_batches_started_at ON kb.ingest_batches(started_at);

CREATE TABLE IF NOT EXISTS kb.pdf_manifest (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  doc_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE RESTRICT,
  filename TEXT NOT NULL,
  filename_norm TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  pages INT NOT NULL CHECK (pages >= 1),
  bytes BIGINT NOT NULL CHECK (bytes >= 0),
  anno INT,
  tipo TEXT,
  volume TEXT,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_pdf_manifest_doc_id ON kb.pdf_manifest(doc_id);
CREATE INDEX IF NOT EXISTS ix_pdf_manifest_ingest_batch ON kb.pdf_manifest(ingest_batch_id);
CREATE INDEX IF NOT EXISTS ix_pdf_manifest_filename_norm ON kb.pdf_manifest(filename_norm);

CREATE TABLE IF NOT EXISTS kb.qa_reference_units (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  unit_index INT NOT NULL CHECK (unit_index >= 0),
  testo TEXT NOT NULL,
  testo_norm TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  char_count INT NOT NULL CHECK (char_count >= 0),
  page_start INT CHECK (page_start >= 0),
  page_end INT CHECK (page_end >= 0),
  has_citation BOOLEAN NOT NULL DEFAULT false,
  extraction_method TEXT NOT NULL,
  UNIQUE(manifest_id, unit_index)
);

CREATE INDEX IF NOT EXISTS ix_ref_units_manifest ON kb.qa_reference_units(manifest_id);
CREATE INDEX IF NOT EXISTS ix_ref_units_content_hash ON kb.qa_reference_units(content_hash);

-- ============================================================
-- FASE 1: Page Stats + Year Resolution + Health Flags
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.page_extraction_stats (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  page_number INT NOT NULL CHECK (page_number >= 0),
  char_count INT NOT NULL CHECK (char_count >= 0),
  word_count INT NOT NULL CHECK (word_count >= 0),
  line_count INT,
  element_count INT NOT NULL CHECK (element_count >= 0),
  has_narrative_text BOOLEAN NOT NULL DEFAULT false,
  has_title BOOLEAN NOT NULL DEFAULT false,
  has_table BOOLEAN NOT NULL DEFAULT false,
  is_empty BOOLEAN NOT NULL DEFAULT false,
  is_ocr_candidate BOOLEAN NOT NULL DEFAULT false,
  is_toc_candidate BOOLEAN NOT NULL DEFAULT false,
  valid_chars_ratio DOUBLE PRECISION,
  italian_tokens_ratio DOUBLE PRECISION,
  non_alnum_ratio DOUBLE PRECISION,
  UNIQUE(manifest_id, page_number)
);

CREATE INDEX IF NOT EXISTS ix_page_stats_manifest ON kb.page_extraction_stats(manifest_id);

CREATE TABLE IF NOT EXISTS kb.pdf_year_resolution (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL UNIQUE REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  anno_from_filename INT,
  anno_from_content INT,
  anno_from_metadata INT,
  anno_resolved INT,
  resolution_method TEXT,
  has_conflict BOOLEAN NOT NULL DEFAULT false,
  conflict_details TEXT
);

CREATE TABLE IF NOT EXISTS kb.pdf_health_flags (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  flag_type TEXT NOT NULL,
  severity INT NOT NULL CHECK (severity BETWEEN 1 AND 5),
  page_range INT4RANGE,
  details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_health_flags_manifest ON kb.pdf_health_flags(manifest_id);
CREATE INDEX IF NOT EXISTS ix_health_flags_type ON kb.pdf_health_flags(flag_type);

-- ============================================================
-- FASE 2: Extraction Quality
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.pdf_extraction_quality (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL UNIQUE REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  total_chars INT NOT NULL CHECK (total_chars >= 0),
  total_words INT NOT NULL CHECK (total_words >= 0),
  total_elements INT NOT NULL CHECK (total_elements >= 0),
  noise_markers_count INT NOT NULL DEFAULT 0 CHECK (noise_markers_count >= 0),
  parsing_artifacts_count INT NOT NULL DEFAULT 0 CHECK (parsing_artifacts_count >= 0),
  page_number_only_count INT NOT NULL DEFAULT 0 CHECK (page_number_only_count >= 0),
  valid_chars_ratio DOUBLE PRECISION,
  italian_tokens_ratio DOUBLE PRECISION,
  citation_regex_success BOOLEAN,
  overall_quality_score DOUBLE PRECISION CHECK (overall_quality_score BETWEEN 0 AND 1),
  quality_grade kb.qa_quality_grade
);

-- ============================================================
-- FASE 3: Gate Decisions
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.gate_decisions (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  element_index INT NOT NULL CHECK (element_index >= 0),
  page_number INT CHECK (page_number >= 0),
  char_count INT NOT NULL CHECK (char_count >= 0),
  word_count INT NOT NULL CHECK (word_count >= 0),
  decision kb.qa_decision NOT NULL,
  rejection_reason TEXT,
  rejection_details JSONB NOT NULL DEFAULT '{}'::jsonb,
  element_category TEXT,
  text_preview TEXT
);

CREATE INDEX IF NOT EXISTS ix_gate_decisions_manifest_batch ON kb.gate_decisions(manifest_id, ingest_batch_id);
CREATE INDEX IF NOT EXISTS ix_gate_decisions_decision ON kb.gate_decisions(decision);
CREATE INDEX IF NOT EXISTS ix_gate_decisions_rejection_reason ON kb.gate_decisions(rejection_reason);

-- ============================================================
-- FASE 4: Chunk Features
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.chunk_features (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  massima_id UUID REFERENCES kb.massime(id) ON DELETE SET NULL,
  chunk_index INT NOT NULL CHECK (chunk_index >= 0),
  char_count INT NOT NULL CHECK (char_count >= 0),
  word_count INT NOT NULL CHECK (word_count >= 0),
  sentence_count INT,
  page_start INT CHECK (page_start >= 0),
  page_end INT CHECK (page_end >= 0),
  is_short BOOLEAN NOT NULL DEFAULT false,
  is_very_long BOOLEAN NOT NULL DEFAULT false,
  toc_infiltration_score DOUBLE PRECISION CHECK (toc_infiltration_score BETWEEN 0 AND 1),
  citation_list_score DOUBLE PRECISION CHECK (citation_list_score BETWEEN 0 AND 1),
  has_multiple_citations BOOLEAN NOT NULL DEFAULT false,
  starts_with_legal_pattern BOOLEAN NOT NULL DEFAULT false,
  quality_score DOUBLE PRECISION CHECK (quality_score BETWEEN 0 AND 1),
  UNIQUE(manifest_id, ingest_batch_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS ix_chunk_features_manifest_batch ON kb.chunk_features(manifest_id, ingest_batch_id);
CREATE INDEX IF NOT EXISTS ix_chunk_features_massima ON kb.chunk_features(massima_id);

-- ============================================================
-- FASE 5: Chunk Labels
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.chunk_labels (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  chunk_feature_id BIGINT NOT NULL REFERENCES kb.chunk_features(id) ON DELETE CASCADE,
  heur_label TEXT NOT NULL,
  heur_confidence DOUBLE PRECISION CHECK (heur_confidence BETWEEN 0 AND 1),
  heur_reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
  llm_label TEXT,
  llm_confidence DOUBLE PRECISION CHECK (llm_confidence BETWEEN 0 AND 1),
  llm_model TEXT,
  llm_response JSONB,
  final_label TEXT NOT NULL,
  final_confidence DOUBLE PRECISION CHECK (final_confidence BETWEEN 0 AND 1),
  label_method TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_chunk_labels_final_label ON kb.chunk_labels(final_label);

-- ============================================================
-- FASE 6: Reference Alignment
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.reference_alignment (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  ref_unit_id BIGINT NOT NULL REFERENCES kb.qa_reference_units(id) ON DELETE CASCADE,
  matched_massima_id UUID REFERENCES kb.massime(id) ON DELETE SET NULL,
  match_type kb.qa_match_type NOT NULL,
  overlap_ratio DOUBLE PRECISION CHECK (overlap_ratio BETWEEN 0 AND 1),
  jaccard_similarity DOUBLE PRECISION CHECK (jaccard_similarity BETWEEN 0 AND 1),
  edit_distance INT CHECK (edit_distance >= 0),
  fragment_count INT NOT NULL DEFAULT 0 CHECK (fragment_count >= 0),
  fusion_count INT NOT NULL DEFAULT 0 CHECK (fusion_count >= 0)
);

CREATE INDEX IF NOT EXISTS ix_ref_align_manifest_batch ON kb.reference_alignment(manifest_id, ingest_batch_id);
CREATE INDEX IF NOT EXISTS ix_ref_align_match_type ON kb.reference_alignment(match_type);

CREATE TABLE IF NOT EXISTS kb.reference_alignment_summary (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  total_ref_units INT NOT NULL CHECK (total_ref_units >= 0),
  matched_count INT NOT NULL CHECK (matched_count >= 0),
  unmatched_count INT NOT NULL CHECK (unmatched_count >= 0),
  coverage_pct DOUBLE PRECISION CHECK (coverage_pct BETWEEN 0 AND 100),
  fragmentation_score DOUBLE PRECISION CHECK (fragmentation_score >= 0),
  fusion_score DOUBLE PRECISION CHECK (fusion_score >= 0),
  avg_overlap DOUBLE PRECISION CHECK (avg_overlap BETWEEN 0 AND 1),
  UNIQUE(manifest_id, ingest_batch_id)
);

-- ============================================================
-- FASE 7: Retrieval Evaluation
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.retrieval_eval_queries (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  query_text TEXT NOT NULL,
  query_type TEXT NOT NULL,
  source_massima_id UUID REFERENCES kb.massime(id) ON DELETE SET NULL,
  ground_truth_ids UUID[],
  keywords TEXT[]
);

CREATE INDEX IF NOT EXISTS ix_retrieval_queries_type ON kb.retrieval_eval_queries(query_type);

CREATE TABLE IF NOT EXISTS kb.retrieval_eval_results (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  query_id BIGINT NOT NULL REFERENCES kb.retrieval_eval_queries(id) ON DELETE CASCADE,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  method TEXT NOT NULL,
  recall_at_5 DOUBLE PRECISION CHECK (recall_at_5 BETWEEN 0 AND 1),
  recall_at_10 DOUBLE PRECISION CHECK (recall_at_10 BETWEEN 0 AND 1),
  mrr DOUBLE PRECISION CHECK (mrr BETWEEN 0 AND 1),
  ndcg_at_10 DOUBLE PRECISION CHECK (ndcg_at_10 BETWEEN 0 AND 1),
  noise_rate_at_10 DOUBLE PRECISION CHECK (noise_rate_at_10 BETWEEN 0 AND 1),
  result_ids UUID[],
  result_scores DOUBLE PRECISION[],
  latency_ms INT CHECK (latency_ms >= 0)
);

CREATE INDEX IF NOT EXISTS ix_retrieval_results_batch_method ON kb.retrieval_eval_results(ingest_batch_id, method);

CREATE TABLE IF NOT EXISTS kb.retrieval_eval_summary (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  method TEXT NOT NULL,
  query_type TEXT NOT NULL,
  query_count INT NOT NULL CHECK (query_count >= 0),
  avg_recall_5 DOUBLE PRECISION CHECK (avg_recall_5 BETWEEN 0 AND 1),
  avg_recall_10 DOUBLE PRECISION CHECK (avg_recall_10 BETWEEN 0 AND 1),
  avg_mrr DOUBLE PRECISION CHECK (avg_mrr BETWEEN 0 AND 1),
  avg_ndcg_10 DOUBLE PRECISION CHECK (avg_ndcg_10 BETWEEN 0 AND 1),
  avg_noise_rate_10 DOUBLE PRECISION CHECK (avg_noise_rate_10 BETWEEN 0 AND 1),
  avg_latency_ms DOUBLE PRECISION CHECK (avg_latency_ms >= 0)
);

CREATE INDEX IF NOT EXISTS ix_retrieval_summary_batch ON kb.retrieval_eval_summary(ingest_batch_id);

-- ============================================================
-- FASE 8: Ingestion Profiles
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.qa_ingestion_profiles (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL UNIQUE REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  profile TEXT NOT NULL,
  confidence DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
  features JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_ingestion_profiles_profile ON kb.qa_ingestion_profiles(profile);

-- ============================================================
-- FASE 9: LLM Decisions
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.llm_decisions (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  trigger_type TEXT NOT NULL,
  input_text TEXT NOT NULL,
  input_page_range INT4RANGE,
  model TEXT NOT NULL,
  prompt_template TEXT,
  raw_response TEXT,
  parsed_output JSONB,
  confidence DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
  tokens_input INT CHECK (tokens_input >= 0),
  tokens_output INT CHECK (tokens_output >= 0),
  cost_usd DOUBLE PRECISION CHECK (cost_usd >= 0),
  latency_ms INT CHECK (latency_ms >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_llm_decisions_trigger ON kb.llm_decisions(trigger_type);

-- ============================================================
-- FASE 10: Reports
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.qa_document_reports (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  manifest_id BIGINT NOT NULL REFERENCES kb.pdf_manifest(id) ON DELETE CASCADE,
  ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id) ON DELETE SET NULL,
  extraction_quality_score DOUBLE PRECISION CHECK (extraction_quality_score BETWEEN 0 AND 1),
  gate_acceptance_rate DOUBLE PRECISION CHECK (gate_acceptance_rate BETWEEN 0 AND 1),
  chunking_quality_score DOUBLE PRECISION CHECK (chunking_quality_score BETWEEN 0 AND 1),
  reference_coverage_pct DOUBLE PRECISION CHECK (reference_coverage_pct BETWEEN 0 AND 100),
  retrieval_self_recall_5 DOUBLE PRECISION CHECK (retrieval_self_recall_5 BETWEEN 0 AND 1),
  composite_risk_score DOUBLE PRECISION CHECK (composite_risk_score BETWEEN 0 AND 1),
  risk_grade TEXT,
  profile TEXT,
  health_flag_count INT NOT NULL DEFAULT 0 CHECK (health_flag_count >= 0),
  recommended_actions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  report_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(manifest_id, ingest_batch_id)
);

CREATE INDEX IF NOT EXISTS ix_doc_reports_risk ON kb.qa_document_reports(composite_risk_score DESC);

CREATE TABLE IF NOT EXISTS kb.qa_global_reports (
  id BIGSERIAL PRIMARY KEY,
  qa_run_id BIGINT REFERENCES kb.qa_runs(id) ON DELETE SET NULL,
  ingest_batch_id BIGINT NOT NULL UNIQUE REFERENCES kb.ingest_batches(id) ON DELETE CASCADE,
  total_documents INT NOT NULL CHECK (total_documents >= 0),
  total_massime INT NOT NULL CHECK (total_massime >= 0),
  avg_extraction_quality DOUBLE PRECISION CHECK (avg_extraction_quality BETWEEN 0 AND 1),
  avg_gate_acceptance_rate DOUBLE PRECISION CHECK (avg_gate_acceptance_rate BETWEEN 0 AND 1),
  avg_reference_coverage DOUBLE PRECISION CHECK (avg_reference_coverage BETWEEN 0 AND 100),
  avg_retrieval_recall_5 DOUBLE PRECISION CHECK (avg_retrieval_recall_5 BETWEEN 0 AND 1),
  grade_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
  profile_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
  top_risk_documents UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
  summary_text TEXT,
  full_report JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- VIEW: PDF Rankings by Risk
-- ============================================================

CREATE OR REPLACE VIEW kb.pdf_rankings_by_risk AS
SELECT
  pm.filename,
  pm.anno,
  pm.tipo,
  qdr.composite_risk_score,
  qdr.risk_grade,
  qdr.profile,
  qdr.health_flag_count,
  qdr.recommended_actions
FROM kb.qa_document_reports qdr
JOIN kb.pdf_manifest pm ON pm.id = qdr.manifest_id
ORDER BY qdr.composite_risk_score DESC;

COMMIT;
