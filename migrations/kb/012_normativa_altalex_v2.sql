-- Migration: Extend kb.normativa_altalex for Altalex PDF ingestion v2
-- Date: 2026-02-04
-- Purpose: Add columns for articolo normalization, overlap, provenance, and global_key

-- =============================================================================
-- 1. ADD NEW COLUMNS TO kb.normativa_altalex
-- =============================================================================

-- Articolo normalization
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS articolo_num_norm INTEGER,
ADD COLUMN IF NOT EXISTS articolo_suffix TEXT,
ADD COLUMN IF NOT EXISTS articolo_sort_key TEXT;

-- Global key for cross-document lookup (populated by trigger)
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS global_key TEXT;

-- Overlap context for retrieval
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS testo_context TEXT;

-- Provenance (page-based)
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS page_start INTEGER,
ADD COLUMN IF NOT EXISTS page_end INTEGER,
ADD COLUMN IF NOT EXISTS source_span INTEGER[2];

-- Structured commi
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS commi JSONB;

-- References (parsed and raw)
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS riferimenti_parsed JSONB,
ADD COLUMN IF NOT EXISTS riferimenti_raw TEXT[];

-- Marker source tracking
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS marker_block_ids TEXT[],
ADD COLUMN IF NOT EXISTS marker_json_path TEXT;

-- =============================================================================
-- 2. UPDATE articolo_sort_key AS GENERATED COLUMN
-- Note: Can't add GENERATED to existing column, use trigger instead
-- =============================================================================

CREATE OR REPLACE FUNCTION kb.set_altalex_sort_key()
RETURNS TRIGGER AS $$
BEGIN
    -- Sort key: 6-digit padded number + suffix (e.g., '002043.bis' or '002043.00')
    IF NEW.articolo_num_norm IS NOT NULL THEN
        NEW.articolo_sort_key := LPAD(NEW.articolo_num_norm::TEXT, 6, '0') || '.' || COALESCE(NEW.articolo_suffix, '00');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_altalex_sort_key ON kb.normativa_altalex;
CREATE TRIGGER trg_altalex_sort_key
BEFORE INSERT OR UPDATE ON kb.normativa_altalex
FOR EACH ROW EXECUTE FUNCTION kb.set_altalex_sort_key();

-- =============================================================================
-- 3. TRIGGER FOR global_key
-- Format: altalex:{codice}:{articolo_num_norm}:{suffix}
-- =============================================================================

CREATE OR REPLACE FUNCTION kb.set_altalex_global_key()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate required fields
    IF NEW.codice IS NULL THEN
        RAISE EXCEPTION 'codice cannot be NULL';
    END IF;

    IF NEW.articolo_num_norm IS NULL THEN
        -- Try to extract from articolo field
        -- Extended Latin ordinals: bis(2), ter(3), quater(4), quinquies(5), sexies(6),
        -- septies(7), octies(8), novies/nonies(9), decies(10), undecies(11), duodecies(12),
        -- terdecies(13), quaterdecies(14), quinquiesdecies(15), sexiesdecies(16),
        -- septiesdecies(17), octiesdecies(18)
        NEW.articolo_num_norm := (regexp_match(NEW.articolo, '^(\d+)'))[1]::INTEGER;
        NEW.articolo_suffix := (regexp_match(NEW.articolo,
            '^\d+[-\s]*(bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|'
            'decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|'
            'sexiesdecies|septiesdecies|octiesdecies)?', 'i'))[1];
    END IF;

    IF NEW.articolo_num_norm IS NULL THEN
        RAISE EXCEPTION 'Could not extract articolo_num_norm from articolo: %', NEW.articolo;
    END IF;

    -- Build global_key
    NEW.global_key := 'altalex:' || LOWER(NEW.codice) || ':'
                      || NEW.articolo_num_norm || ':'
                      || COALESCE(LOWER(NEW.articolo_suffix), 'null');

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_altalex_global_key ON kb.normativa_altalex;
CREATE TRIGGER trg_altalex_global_key
BEFORE INSERT OR UPDATE ON kb.normativa_altalex
FOR EACH ROW EXECUTE FUNCTION kb.set_altalex_global_key();

-- =============================================================================
-- 4. UNIQUE INDEX ON global_key
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS idx_normativa_altalex_global_key
ON kb.normativa_altalex(global_key);

-- Composite index with COALESCE for NULL suffix
CREATE UNIQUE INDEX IF NOT EXISTS idx_normativa_altalex_composite
ON kb.normativa_altalex(codice, articolo_num_norm, COALESCE(articolo_suffix, 'null'));

-- Index for sorting
CREATE INDEX IF NOT EXISTS idx_normativa_altalex_sort
ON kb.normativa_altalex(codice, articolo_sort_key);

-- =============================================================================
-- 5. INGESTION LOGS TABLE (for quarantine and tracking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS kb.altalex_ingestion_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file TEXT,
    codice TEXT,
    articolo TEXT,
    stage TEXT NOT NULL,        -- 'convert', 'chunk', 'validate', 'embed', 'store'
    status TEXT NOT NULL,       -- 'success', 'warning', 'error', 'quarantine'
    message TEXT,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_altalex_logs_status
ON kb.altalex_ingestion_logs(status);

CREATE INDEX IF NOT EXISTS idx_altalex_logs_source
ON kb.altalex_ingestion_logs(source_file);

CREATE INDEX IF NOT EXISTS idx_altalex_logs_created
ON kb.altalex_ingestion_logs(created_at DESC);

-- =============================================================================
-- 6. EMBEDDING TABLE EXTENSION (multi-dims support)
-- Note: Existing kb.normativa_embeddings uses vector(1536) fixed
-- Create new table for multi-dim support
-- =============================================================================

CREATE TABLE IF NOT EXISTS kb.altalex_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    altalex_id UUID NOT NULL REFERENCES kb.normativa_altalex(id) ON DELETE CASCADE,
    model TEXT NOT NULL CHECK (model <> ''),
    channel TEXT NOT NULL CHECK (channel IN ('testo', 'rubrica', 'contesto')),
    dims INTEGER NOT NULL CHECK (dims IN (384, 768, 1024, 1536)),
    embedding vector NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(altalex_id, model, channel)
);

-- Indexes per dimension (HNSW if pgvector >= 0.5.0)
-- Note: Run separately after verifying pgvector version

-- For 1536 dims (OpenAI)
CREATE INDEX IF NOT EXISTS idx_altalex_emb_1536_hnsw
ON kb.altalex_embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE dims = 1536;

-- For 1024 dims (e5, bge)
CREATE INDEX IF NOT EXISTS idx_altalex_emb_1024_hnsw
ON kb.altalex_embeddings USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE dims = 1024;

-- For 768 dims (legal-bert)
CREATE INDEX IF NOT EXISTS idx_altalex_emb_768_hnsw
ON kb.altalex_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
WHERE dims = 768;

CREATE INDEX IF NOT EXISTS idx_altalex_emb_altalex_id
ON kb.altalex_embeddings(altalex_id);

CREATE INDEX IF NOT EXISTS idx_altalex_emb_model
ON kb.altalex_embeddings(model, channel);

-- =============================================================================
-- 7. EMBEDDING CACHE TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS kb.altalex_embedding_cache (
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    channel TEXT NOT NULL,
    dims INTEGER NOT NULL CHECK (dims IN (384, 768, 1024, 1536)),
    embedding vector NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_hit_at TIMESTAMPTZ,

    UNIQUE(text_hash, model, channel, dims)
);

-- =============================================================================
-- 8. FTS WITH UNACCENT (better Italian search)
-- =============================================================================

-- Check if unaccent extension exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'unaccent') THEN
        CREATE EXTENSION IF NOT EXISTS unaccent;
    END IF;
END $$;

-- Add FTS column with unaccent
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS testo_tsv tsvector;

-- Trigger to update tsvector
CREATE OR REPLACE FUNCTION kb.update_altalex_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.testo_tsv := to_tsvector('italian',
        unaccent(COALESCE(NEW.rubrica, '') || ' ' || COALESCE(NEW.testo, ''))
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_altalex_tsv ON kb.normativa_altalex;
CREATE TRIGGER trg_altalex_tsv
BEFORE INSERT OR UPDATE OF rubrica, testo ON kb.normativa_altalex
FOR EACH ROW EXECUTE FUNCTION kb.update_altalex_tsv();

-- GIN index for FTS
CREATE INDEX IF NOT EXISTS idx_normativa_altalex_tsv
ON kb.normativa_altalex USING GIN(testo_tsv);

-- =============================================================================
-- 9. COMMENTS
-- =============================================================================

COMMENT ON COLUMN kb.normativa_altalex.articolo_num_norm IS 'Numero articolo normalizzato (es. 2043 per "2043-bis")';
COMMENT ON COLUMN kb.normativa_altalex.articolo_suffix IS 'Suffisso articolo: bis, ter, quater, etc.';
COMMENT ON COLUMN kb.normativa_altalex.articolo_sort_key IS 'Chiave per ordinamento: 002043.bis';
COMMENT ON COLUMN kb.normativa_altalex.global_key IS 'Chiave globale: altalex:cc:2043:bis';
COMMENT ON COLUMN kb.normativa_altalex.testo_context IS 'Testo con overlap ±200 chars per retrieval';
COMMENT ON COLUMN kb.normativa_altalex.commi IS 'Array di commi: [{num: 1, testo: "..."}, ...]';
COMMENT ON COLUMN kb.normativa_altalex.riferimenti_parsed IS 'Riferimenti validati: ["CC:1218", "CC:2059"]';
COMMENT ON COLUMN kb.normativa_altalex.riferimenti_raw IS 'Riferimenti raw non parsati';
COMMENT ON COLUMN kb.normativa_altalex.marker_block_ids IS 'IDs dei blocchi marker originali';

COMMENT ON TABLE kb.altalex_ingestion_logs IS 'Log ingestion per tracking errori e quarantina';
COMMENT ON TABLE kb.altalex_embeddings IS 'Embeddings multi-dimensione per articoli Altalex';
COMMENT ON TABLE kb.altalex_embedding_cache IS 'Cache embeddings per evitare ricalcolo';
