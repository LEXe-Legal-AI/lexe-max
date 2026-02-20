-- ============================================================================
-- 060_normativa_search_setup.sql
-- LEXE KB — Normativa Search: Surgical migration for search enablement
-- ============================================================================
-- Purpose: Enable hybrid search (dense + sparse) on kb.normativa articles.
--          Creates ONLY what's needed without V3 full migration.
--          Does NOT rename or drop existing tables/columns.
-- Idempotent: safe to re-run (IF NOT EXISTS, ON CONFLICT DO NOTHING).
-- Rollback: DROP tables + ALTER DROP COLUMN restores original state.
-- ============================================================================

BEGIN;

-- ============================================================================
-- PREREQUISITES
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. kb.work — Minimal work table (standalone, no V3 dependencies)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    title_short TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE kb.work IS 'Legal works (codes/laws) — minimal for normativa search';

-- Seed the 5 fundamental codes
INSERT INTO kb.work (code, title, title_short) VALUES
    ('CC',   'Codice Civile',                 'Cod. Civ.'),
    ('CP',   'Codice Penale',                 'Cod. Pen.'),
    ('CPC',  'Codice di Procedura Civile',    'Cod. Proc. Civ.'),
    ('CPP',  'Codice di Procedura Penale',    'Cod. Proc. Pen.'),
    ('COST', 'Costituzione della Repubblica', 'Cost.')
ON CONFLICT (code) DO NOTHING;

-- ============================================================================
-- 2. ALTER kb.normativa — Add columns for work linkage and article parsing
-- ============================================================================

-- work_id: links to kb.work
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'normativa' AND column_name = 'work_id'
    ) THEN
        ALTER TABLE kb.normativa ADD COLUMN work_id UUID REFERENCES kb.work(id) ON DELETE SET NULL;
    END IF;
END $$;

-- articolo_num: parsed numeric part (e.g. 2043 from "2043-bis")
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'normativa' AND column_name = 'articolo_num'
    ) THEN
        ALTER TABLE kb.normativa ADD COLUMN articolo_num INTEGER;
    END IF;
END $$;

-- articolo_suffix: parsed suffix (e.g. "bis" from "2043-bis")
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'normativa' AND column_name = 'articolo_suffix'
    ) THEN
        ALTER TABLE kb.normativa ADD COLUMN articolo_suffix TEXT;
    END IF;
END $$;

-- articolo_sort_key: zero-padded sort key (e.g. "002043.02" for "2043-bis")
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'normativa' AND column_name = 'articolo_sort_key'
    ) THEN
        ALTER TABLE kb.normativa ADD COLUMN articolo_sort_key TEXT;
    END IF;
END $$;

-- Index on work_id
CREATE INDEX IF NOT EXISTS idx_normativa_work_id ON kb.normativa(work_id);

-- ============================================================================
-- 3. BACKFILL work_id from codice column
-- ============================================================================

UPDATE kb.normativa n
SET work_id = w.id
FROM kb.work w
WHERE UPPER(n.codice) = w.code
  AND n.work_id IS NULL;

-- ============================================================================
-- 4. BACKFILL articolo_num, articolo_suffix, articolo_sort_key
-- ============================================================================

-- Suffix ordinal mapping for sort keys
CREATE OR REPLACE FUNCTION kb.fn_suffix_ordinal(suffix TEXT) RETURNS TEXT AS $$
BEGIN
    RETURN CASE LOWER(TRIM(suffix))
        WHEN 'bis'         THEN '02'
        WHEN 'ter'         THEN '03'
        WHEN 'quater'      THEN '04'
        WHEN 'quinquies'   THEN '05'
        WHEN 'sexies'      THEN '06'
        WHEN 'septies'     THEN '07'
        WHEN 'octies'      THEN '08'
        WHEN 'novies'      THEN '09'
        WHEN 'decies'      THEN '10'
        WHEN 'undecies'    THEN '11'
        WHEN 'duodecies'   THEN '12'
        WHEN 'terdecies'   THEN '13'
        WHEN 'quaterdecies' THEN '14'
        WHEN 'quinquiesdecies' THEN '15'
        WHEN 'sexiesdecies' THEN '16'
        WHEN 'septiesdecies' THEN '17'
        WHEN 'octiesdecies' THEN '18'
        ELSE '00'
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Parse articolo field: "2043" → num=2043, suffix=NULL, sort_key="002043.00"
--                       "360-bis" → num=360, suffix="bis", sort_key="000360.02"
UPDATE kb.normativa
SET
    articolo_num = CASE
        WHEN articolo ~ '^\d+' THEN (regexp_match(articolo, '^(\d+)'))[1]::INTEGER
        ELSE NULL
    END,
    articolo_suffix = CASE
        WHEN articolo ~ '[- ]([a-zA-Z]+)$' THEN LOWER(TRIM((regexp_match(articolo, '[- ]([a-zA-Z]+)$'))[1]))
        ELSE NULL
    END,
    articolo_sort_key = CASE
        WHEN articolo ~ '^\d+' THEN
            LPAD((regexp_match(articolo, '^(\d+)'))[1], 6, '0') || '.' ||
            CASE
                WHEN articolo ~ '[- ]([a-zA-Z]+)$'
                THEN kb.fn_suffix_ordinal((regexp_match(articolo, '[- ]([a-zA-Z]+)$'))[1])
                ELSE '00'
            END
        ELSE articolo  -- keep as-is for special articles
    END
WHERE articolo_sort_key IS NULL;

-- ============================================================================
-- 5. kb.normativa_chunk — Chunk table for RAG retrieval
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_chunk (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    normativa_id UUID NOT NULL REFERENCES kb.normativa(id) ON DELETE CASCADE,
    work_id UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,
    articolo_sort_key TEXT NOT NULL,
    articolo_num INTEGER,
    articolo_suffix TEXT,
    chunk_no INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_est INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(normativa_id, chunk_no),
    CHECK (chunk_no >= 0),
    CHECK (length(trim(text)) >= 30),
    CHECK (token_est > 0),
    CHECK (token_est <= 10000),
    CHECK (char_end > char_start)
);

CREATE INDEX IF NOT EXISTS idx_norm_chunk_work ON kb.normativa_chunk(work_id);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_sort ON kb.normativa_chunk(work_id, articolo_sort_key, chunk_no);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_group ON kb.normativa_chunk(work_id, normativa_id, chunk_no);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_created ON kb.normativa_chunk(created_at);

-- ============================================================================
-- 6. kb.normativa_chunk_fts — Full-text search on chunks (Italian)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_chunk_fts (
    chunk_id UUID PRIMARY KEY REFERENCES kb.normativa_chunk(id) ON DELETE CASCADE,
    tsv_it TSVECTOR
);

CREATE INDEX IF NOT EXISTS idx_norm_chunk_fts ON kb.normativa_chunk_fts USING GIN(tsv_it);

-- Trigger function: auto-populate FTS on chunk INSERT/UPDATE
CREATE OR REPLACE FUNCTION kb.fn_normativa_chunk_fts_update() RETURNS trigger AS $$
BEGIN
    INSERT INTO kb.normativa_chunk_fts(chunk_id, tsv_it)
    VALUES(NEW.id, to_tsvector('italian', unaccent(TRIM(NEW.text))))
    ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Idempotent trigger creation
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'trg_normativa_chunk_fts'
          AND n.nspname = 'kb' AND c.relname = 'normativa_chunk'
    ) THEN
        CREATE TRIGGER trg_normativa_chunk_fts
            AFTER INSERT OR UPDATE OF text ON kb.normativa_chunk
            FOR EACH ROW EXECUTE FUNCTION kb.fn_normativa_chunk_fts_update();
    END IF;
END $$;

-- Backfill FTS for any existing chunks
INSERT INTO kb.normativa_chunk_fts(chunk_id, tsv_it)
SELECT id, to_tsvector('italian', unaccent(trim(text)))
FROM kb.normativa_chunk
ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;

-- ============================================================================
-- 7. kb.normativa_chunk_embeddings — Dense vector embeddings for chunks
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_chunk_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES kb.normativa_chunk(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'testo',
    dims INTEGER NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(chunk_id, model, channel, dims),
    CHECK (channel IN ('testo')),
    CHECK (dims = 1536),
    CHECK (model <> '')
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_norm_chunk_emb_hnsw
    ON kb.normativa_chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_norm_chunk_emb_chunk ON kb.normativa_chunk_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_emb_model ON kb.normativa_chunk_embeddings(model);

-- ============================================================================
-- 8. Helper functions
-- ============================================================================

-- websearch_to_tsquery safe wrapper
CREATE OR REPLACE FUNCTION kb.fn_tsquery_it(q text) RETURNS tsquery AS $$
BEGIN
    RETURN websearch_to_tsquery('italian', unaccent(left(q, 256)));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- 9. Stats view — Coverage per code
-- ============================================================================

CREATE OR REPLACE VIEW kb.v_chunk_stats AS
WITH chunks AS (
    SELECT normativa_id, count(*) AS chunks_per_articolo
    FROM kb.normativa_chunk
    GROUP BY normativa_id
),
embeddings AS (
    SELECT nc.normativa_id, count(*) AS emb_per_articolo
    FROM kb.normativa_chunk_embeddings nce
    JOIN kb.normativa_chunk nc ON nc.id = nce.chunk_id
    GROUP BY nc.normativa_id
)
SELECT
    w.code AS work_code,
    w.title AS work_title,
    count(n.id) AS articoli_tot,
    count(ch.normativa_id) AS articoli_chunkizzati,
    round(100.0 * count(ch.normativa_id) / NULLIF(count(n.id), 0), 2) AS chunk_coverage_pct,
    count(emb.normativa_id) AS articoli_con_embeddings,
    round(100.0 * count(emb.normativa_id) / NULLIF(count(n.id), 0), 2) AS emb_coverage_pct,
    round(avg(coalesce(ch.chunks_per_articolo, 0))::numeric, 2) AS media_chunk_per_art,
    max(coalesce(ch.chunks_per_articolo, 0)) AS max_chunk_per_art
FROM kb.work w
LEFT JOIN kb.normativa n ON n.work_id = w.id
LEFT JOIN chunks ch ON ch.normativa_id = n.id
LEFT JOIN embeddings emb ON emb.normativa_id = n.id
GROUP BY w.code, w.title
ORDER BY w.code;

COMMENT ON VIEW kb.v_chunk_stats IS 'Coverage stats: articles, chunks, embeddings per code';

COMMIT;
