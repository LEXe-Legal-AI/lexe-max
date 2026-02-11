-- migrations/kb/055_chunking_schema.sql
-- V4 Chunking Schema - Production Ready
-- Tabelle chunk per normativa e annotation con FTS
-- HNSW indexes in 056_hnsw_indexes.sql (separato)

BEGIN;

-- ═══════════════════════════════════════════════════════════════
-- PREREQUISITI (guardia idempotente)
-- ═══════════════════════════════════════════════════════════════
-- NOTA: Se fallisce, rilanciare come superuser e riprovare
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ═══════════════════════════════════════════════════════════════
-- CHUNK NORMATIVA
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.normativa_chunk (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  normativa_id UUID NOT NULL REFERENCES kb.normativa(id) ON DELETE CASCADE,
  work_id UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,
  articolo_sort_key TEXT NOT NULL,
  articolo_num INTEGER,           -- join veloci senza normativa
  articolo_suffix TEXT,           -- "bis", "ter"...
  chunk_no INTEGER NOT NULL,
  char_start INTEGER NOT NULL,    -- su testo NORMALIZZATO
  char_end INTEGER NOT NULL,      -- su testo NORMALIZZATO
  text TEXT NOT NULL,
  token_est INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(normativa_id, chunk_no),
  CHECK (chunk_no >= 0),                -- numerazione valida
  CHECK (length(trim(text)) >= 30),     -- anti-vuoto
  CHECK (token_est > 0),                -- anti-micro
  CHECK (token_est <= 10000),           -- anti-bug overflow
  CHECK (char_end > char_start)         -- boundaries coerenti
);

-- Indici per retriever grouping e paginazione batch (tutti idempotenti)
CREATE INDEX IF NOT EXISTS idx_norm_chunk_work ON kb.normativa_chunk(work_id);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_sort ON kb.normativa_chunk(work_id, articolo_sort_key, chunk_no);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_group ON kb.normativa_chunk(work_id, normativa_id, chunk_no);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_created ON kb.normativa_chunk(created_at);

-- ═══════════════════════════════════════════════════════════════
-- FTS CHUNK NORMATIVA
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.normativa_chunk_fts (
  chunk_id UUID PRIMARY KEY REFERENCES kb.normativa_chunk(id) ON DELETE CASCADE,
  tsv_it TSVECTOR
);
CREATE INDEX IF NOT EXISTS idx_norm_chunk_fts ON kb.normativa_chunk_fts USING GIN(tsv_it);

-- Trigger function FTS
CREATE OR REPLACE FUNCTION kb.fn_normativa_chunk_fts_update() RETURNS trigger AS $$
BEGIN
    INSERT INTO kb.normativa_chunk_fts(chunk_id, tsv_it)
    VALUES(NEW.id, to_tsvector('italian', unaccent(TRIM(NEW.text))))
    ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger idempotente con DO block (CREATE TRIGGER non ha IF NOT EXISTS)
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
END
$$;

-- Backfill FTS con repair (ON CONFLICT DO UPDATE)
INSERT INTO kb.normativa_chunk_fts(chunk_id, tsv_it)
SELECT id, to_tsvector('italian', unaccent(trim(text)))
FROM kb.normativa_chunk
ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;

-- ═══════════════════════════════════════════════════════════════
-- EMBEDDINGS CHUNK NORMATIVA
-- ═══════════════════════════════════════════════════════════════

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

-- NOTA: HNSW index in 056_hnsw_indexes.sql

-- ═══════════════════════════════════════════════════════════════
-- CHUNK ANNOTATION
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.annotation_chunk (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  annotation_id UUID NOT NULL REFERENCES kb.annotation(id) ON DELETE CASCADE,
  chunk_no INTEGER NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_est INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(annotation_id, chunk_no),
  CHECK (chunk_no >= 0),
  CHECK (length(trim(text)) >= 30),
  CHECK (token_est > 0),
  CHECK (char_end > char_start)
);

CREATE INDEX IF NOT EXISTS idx_ann_chunk_group ON kb.annotation_chunk(annotation_id, chunk_no);
CREATE INDEX IF NOT EXISTS idx_ann_chunk_created ON kb.annotation_chunk(created_at);

-- ═══════════════════════════════════════════════════════════════
-- FTS CHUNK ANNOTATION
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.annotation_chunk_fts (
  chunk_id UUID PRIMARY KEY REFERENCES kb.annotation_chunk(id) ON DELETE CASCADE,
  tsv_it TSVECTOR
);
CREATE INDEX IF NOT EXISTS idx_ann_chunk_fts ON kb.annotation_chunk_fts USING GIN(tsv_it);

-- Trigger function FTS annotation
CREATE OR REPLACE FUNCTION kb.fn_annotation_chunk_fts_update() RETURNS trigger AS $$
BEGIN
    INSERT INTO kb.annotation_chunk_fts(chunk_id, tsv_it)
    VALUES(NEW.id, to_tsvector('italian', unaccent(TRIM(NEW.text))))
    ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger idempotente con DO block
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger t
    JOIN pg_class c ON c.oid = t.tgrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE t.tgname = 'trg_annotation_chunk_fts'
      AND n.nspname = 'kb' AND c.relname = 'annotation_chunk'
  ) THEN
    CREATE TRIGGER trg_annotation_chunk_fts
      AFTER INSERT OR UPDATE OF text ON kb.annotation_chunk
      FOR EACH ROW EXECUTE FUNCTION kb.fn_annotation_chunk_fts_update();
  END IF;
END
$$;

-- Backfill FTS annotation chunks con repair
INSERT INTO kb.annotation_chunk_fts(chunk_id, tsv_it)
SELECT id, to_tsvector('italian', unaccent(trim(text)))
FROM kb.annotation_chunk
ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;

-- ═══════════════════════════════════════════════════════════════
-- EMBEDDINGS CHUNK ANNOTATION
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.annotation_chunk_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  chunk_id UUID NOT NULL REFERENCES kb.annotation_chunk(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT 'content',
  dims INTEGER NOT NULL,
  embedding vector(1536) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(chunk_id, model, channel, dims),
  CHECK (channel IN ('content')),
  CHECK (dims = 1536),
  CHECK (model <> '')
);

-- NOTA: HNSW index in 056_hnsw_indexes.sql

-- ═══════════════════════════════════════════════════════════════
-- VISTA STATS PER GATE B/C
-- ═══════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW kb.v_chunk_stats AS
WITH chunks AS (
  SELECT normativa_id, count(*) AS chunks_per_articolo
  FROM kb.normativa_chunk
  GROUP BY normativa_id
)
SELECT
  w.code AS work_code,
  count(*) AS articoli_tot,
  count(ch.normativa_id) AS articoli_chunkizzati,
  round(100.0 * count(ch.normativa_id) / nullif(count(*),0), 2) AS articoli_chunkizzati_pct,
  round(avg(coalesce(ch.chunks_per_articolo,0))::numeric, 2) AS media_chunk_per_articolo,
  max(coalesce(ch.chunks_per_articolo,0)) AS max_chunk_per_articolo
FROM kb.work w
JOIN kb.normativa n ON n.work_id = w.id
LEFT JOIN chunks ch ON ch.normativa_id = n.id
GROUP BY w.code;

-- ═══════════════════════════════════════════════════════════════
-- FTS QUERY HELPER (websearch safe)
-- ═══════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION kb.fn_tsquery_it(q text) RETURNS tsquery AS $$
BEGIN
  -- Cap query a 256 chars per evitare parse pesanti
  RETURN websearch_to_tsquery('italian', unaccent(left(q, 256)));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMIT;
