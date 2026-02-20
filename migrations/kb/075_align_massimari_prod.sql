-- ============================================================================
-- 075_align_massimari_prod.sql
-- Align production massimari tables to match staging schema
-- ============================================================================
-- PURPOSE: Production has the same tables but they are ALL EMPTY.
--          This migration alters prod schema to match staging so that
--          a pg_dump --data-only from staging can be restored cleanly.
-- SAFE: All target tables are empty, so ALTER/DROP is non-destructive.
-- RUN ON: PRODUCTION (49.12.85.92) lexe-max container
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1a. kb.massime — add 4 missing columns + make is_active nullable
-- ============================================================================

ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS text_fingerprint BIGINT;
ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS ingest_batch_id BIGINT;
ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS extraction_mode TEXT;
ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS quality_flags JSONB;
ALTER TABLE kb.massime ALTER COLUMN is_active DROP NOT NULL;

DO $$ BEGIN RAISE NOTICE '1a. kb.massime columns aligned'; END $$;

-- ============================================================================
-- 1b. kb.embeddings — DROP and RECREATE with staging schema
--     Prod has 7 columns, staging has 10. Table is EMPTY so safe to recreate.
-- ============================================================================

DROP TABLE IF EXISTS kb.embeddings CASCADE;

CREATE TABLE kb.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID REFERENCES kb.massime(id),
    model_name TEXT,
    model_version TEXT,
    dimension INTEGER,
    embedding_distance TEXT,
    embedding_batch_id INTEGER,
    embedding vector(1536),
    is_normalized BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_embeddings_massima_id ON kb.embeddings (massima_id);
CREATE INDEX idx_embeddings_hnsw ON kb.embeddings USING hnsw (embedding vector_cosine_ops);

DO $$ BEGIN RAISE NOTICE '1b. kb.embeddings recreated with staging schema'; END $$;

-- ============================================================================
-- 1c. kb.norms — rename full_text → full_ref, add number/year columns
-- ============================================================================

-- Check if full_text exists (it may already be full_ref)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'kb' AND table_name = 'norms' AND column_name = 'full_text'
  ) THEN
    ALTER TABLE kb.norms RENAME COLUMN full_text TO full_ref;
    RAISE NOTICE '1c. Renamed full_text → full_ref';
  ELSE
    RAISE NOTICE '1c. full_ref already exists, skip rename';
  END IF;
END $$;

ALTER TABLE kb.norms ALTER COLUMN article DROP NOT NULL;
ALTER TABLE kb.norms ADD COLUMN IF NOT EXISTS number TEXT;
ALTER TABLE kb.norms ADD COLUMN IF NOT EXISTS year INTEGER;

DO $$ BEGIN RAISE NOTICE '1c. kb.norms columns aligned (article now nullable)'; END $$;

-- ============================================================================
-- 1d. kb.documents — add profile_id
-- ============================================================================

ALTER TABLE kb.documents ADD COLUMN IF NOT EXISTS profile_id UUID;

DO $$ BEGIN RAISE NOTICE '1d. kb.documents profile_id added'; END $$;

-- ============================================================================
-- 1e. kb.sections — add numero, tipo
-- ============================================================================

ALTER TABLE kb.sections ADD COLUMN IF NOT EXISTS numero VARCHAR;
ALTER TABLE kb.sections ADD COLUMN IF NOT EXISTS tipo VARCHAR;

DO $$ BEGIN RAISE NOTICE '1e. kb.sections columns aligned'; END $$;

-- ============================================================================
-- Final report
-- ============================================================================

DO $$
BEGIN
  RAISE NOTICE '';
  RAISE NOTICE '============================================';
  RAISE NOTICE '  075 ALIGN MASSIMARI PROD — COMPLETE';
  RAISE NOTICE '============================================';
  RAISE NOTICE '  kb.massime:     +4 columns, is_active nullable';
  RAISE NOTICE '  kb.embeddings:  recreated (10 columns)';
  RAISE NOTICE '  kb.norms:       full_text→full_ref, +number/year';
  RAISE NOTICE '  kb.documents:   +profile_id';
  RAISE NOTICE '  kb.sections:    +numero, +tipo';
  RAISE NOTICE '============================================';
  RAISE NOTICE '  Ready for pg_restore from staging dump';
  RAISE NOTICE '============================================';
END $$;

COMMIT;
