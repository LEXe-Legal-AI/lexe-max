-- ============================================================================
-- Migration 002: Alignment Fix v3.2 (16 Migliorie)
-- ============================================================================
-- Fixes the 0.2% coverage issue by adding proper tracking columns,
-- qa_run_id everywhere, fingerprint support, and performance indexes.
-- ============================================================================

BEGIN;

-- ============================================================================
-- A0. Tabella kb.qa_runs (se non esiste già)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.qa_runs (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    config_json JSONB,
    git_commit TEXT,
    status TEXT DEFAULT 'running'
);

CREATE INDEX IF NOT EXISTS idx_qa_runs_started ON kb.qa_runs(started_at DESC);

-- ============================================================================
-- A0. Aggiungi qa_run_id dove manca (con FK)
-- ============================================================================

-- qa_reference_units
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'qa_reference_units' AND column_name = 'qa_run_id'
    ) THEN
        ALTER TABLE kb.qa_reference_units ADD COLUMN qa_run_id INTEGER REFERENCES kb.qa_runs(id);
    END IF;
END $$;

-- reference_alignment
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'reference_alignment' AND column_name = 'qa_run_id'
    ) THEN
        ALTER TABLE kb.reference_alignment ADD COLUMN qa_run_id INTEGER REFERENCES kb.qa_runs(id);
    END IF;
END $$;

-- reference_alignment_summary
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'reference_alignment_summary' AND column_name = 'qa_run_id'
    ) THEN
        ALTER TABLE kb.reference_alignment_summary ADD COLUMN qa_run_id INTEGER REFERENCES kb.qa_runs(id);
    END IF;
END $$;

-- ============================================================================
-- A1. Colonne nuove per kb.qa_reference_units (NO DEFAULT!)
-- ============================================================================

ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS reference_version TEXT;
ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS extraction_engine TEXT;
ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS normalization_version TEXT;
ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS raw_text TEXT;
ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS spaced_letters_score FLOAT;
ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS text_fingerprint BIGINT;
ALTER TABLE kb.qa_reference_units ADD COLUMN IF NOT EXISTS fingerprint_method TEXT;

-- Backfill record storici (ref_v0 per distinguerli)
UPDATE kb.qa_reference_units
SET reference_version = 'ref_v0',
    extraction_engine = 'unstructured_hi_res',
    normalization_version = 'norm_v0',
    fingerprint_method = 'none'
WHERE reference_version IS NULL;

-- Ora NOT NULL dove ha senso
ALTER TABLE kb.qa_reference_units
    ALTER COLUMN reference_version SET NOT NULL,
    ALTER COLUMN extraction_engine SET NOT NULL;

COMMENT ON COLUMN kb.qa_reference_units.extraction_engine IS 'pymupdf | unstructured_hi_res';
COMMENT ON COLUMN kb.qa_reference_units.spaced_letters_score IS '0-1, alto = testo con spazi tra lettere';
COMMENT ON COLUMN kb.qa_reference_units.fingerprint_method IS 'simhash64_v1 | none';

-- ============================================================================
-- A1. UNIQUE constraint con qa_run_id
-- ============================================================================

-- Drop old constraint if exists
ALTER TABLE kb.qa_reference_units
    DROP CONSTRAINT IF EXISTS qa_reference_units_manifest_unit_key;

-- Add new constraint with qa_run_id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'qa_reference_units_manifest_unit_run_key'
    ) THEN
        ALTER TABLE kb.qa_reference_units ADD CONSTRAINT qa_reference_units_manifest_unit_run_key
            UNIQUE(manifest_id, unit_index, qa_run_id);
    END IF;
END $$;

-- ============================================================================
-- A2. Colonne nuove per kb.reference_alignment
-- ============================================================================

ALTER TABLE kb.reference_alignment ADD COLUMN IF NOT EXISTS match_stage TEXT;
ALTER TABLE kb.reference_alignment ADD COLUMN IF NOT EXISTS match_score FLOAT;
ALTER TABLE kb.reference_alignment ADD COLUMN IF NOT EXISTS normalized_ref_hash TEXT;
ALTER TABLE kb.reference_alignment ADD COLUMN IF NOT EXISTS normalized_chunk_hash TEXT;

COMMENT ON COLUMN kb.reference_alignment.match_stage IS 'exact_hash | token_jaccard | char_ngram | embedding';

-- ============================================================================
-- A2. Colonne in kb.reference_alignment_summary
-- ============================================================================

ALTER TABLE kb.reference_alignment_summary ADD COLUMN IF NOT EXISTS alignment_trust FLOAT;
ALTER TABLE kb.reference_alignment_summary ADD COLUMN IF NOT EXISTS embedding_pct FLOAT;
ALTER TABLE kb.reference_alignment_summary ADD COLUMN IF NOT EXISTS collision_rate FLOAT;

COMMENT ON COLUMN kb.reference_alignment_summary.alignment_trust IS '(exact+jaccard+ngram)/total, >= 0.90';
COMMENT ON COLUMN kb.reference_alignment_summary.collision_rate IS 'pct ref_units che matchano stesso chunk, < 3%';

-- UNIQUE constraint with qa_run_id
ALTER TABLE kb.reference_alignment_summary
    DROP CONSTRAINT IF EXISTS reference_alignment_summary_manifest_batch_key;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'reference_alignment_summary_manifest_batch_run_key'
    ) THEN
        ALTER TABLE kb.reference_alignment_summary ADD CONSTRAINT reference_alignment_summary_manifest_batch_run_key
            UNIQUE(manifest_id, ingest_batch_id, qa_run_id);
    END IF;
END $$;

-- ============================================================================
-- A3. Colonne per kb.massime (content_hash, testo_normalizzato, fingerprint)
-- ============================================================================

ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS testo_normalizzato TEXT;
ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS text_fingerprint BIGINT;

COMMENT ON COLUMN kb.massime.content_hash IS 'sha256(testo_normalizzato)[:40]';
COMMENT ON COLUMN kb.massime.testo_normalizzato IS 'Output di norm_v2';
COMMENT ON COLUMN kb.massime.text_fingerprint IS 'simhash64 per candidate generation';

-- ============================================================================
-- A4. Indici per Performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_qa_ref_units_manifest_run
    ON kb.qa_reference_units(manifest_id, qa_run_id);

CREATE INDEX IF NOT EXISTS idx_massime_document
    ON kb.massime(document_id);

CREATE INDEX IF NOT EXISTS idx_massime_content_hash
    ON kb.massime(content_hash)
    WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_massime_fingerprint
    ON kb.massime(text_fingerprint)
    WHERE text_fingerprint IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_qa_ref_units_fingerprint
    ON kb.qa_reference_units(text_fingerprint)
    WHERE text_fingerprint IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_qa_ref_units_content_hash
    ON kb.qa_reference_units(content_hash)
    WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_reference_alignment_manifest_batch
    ON kb.reference_alignment(manifest_id, ingest_batch_id, qa_run_id);

-- ============================================================================
-- Done
-- ============================================================================

COMMIT;

-- Verification queries
-- SELECT count(*) FROM kb.qa_runs;
-- SELECT column_name FROM information_schema.columns WHERE table_schema='kb' AND table_name='qa_reference_units';
-- SELECT column_name FROM information_schema.columns WHERE table_schema='kb' AND table_name='reference_alignment_summary';
