-- Migration 084: Library Chunks for Private Library feature (WS2)
-- Tenant-uploaded document chunks with FTS and embeddings.
-- App-level tenant isolation (no RLS on lexe-max), same pattern as normativa_chunk*.

BEGIN;

-- ═══════════════════════════════════════════════════════════════
-- PREREQUISITI
-- ═══════════════════════════════════════════════════════════════
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ═══════════════════════════════════════════════════════════════
-- LIBRARY CHUNK
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.library_chunk (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id UUID NOT NULL,         -- references core.library_documents(id) via app logic
  tenant_id UUID NOT NULL,           -- app-level tenant isolation (no FK, cross-DB)
  chunk_no INTEGER NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_est INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(document_id, chunk_no),
  CHECK (chunk_no >= 0),
  CHECK (length(trim(text)) >= 10),   -- lower threshold than normativa (user docs may be shorter)
  CHECK (token_est > 0),
  CHECK (token_est <= 10000),
  CHECK (char_end > char_start)
);

CREATE INDEX IF NOT EXISTS idx_lib_chunk_doc ON kb.library_chunk(document_id);
CREATE INDEX IF NOT EXISTS idx_lib_chunk_tenant ON kb.library_chunk(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lib_chunk_sort ON kb.library_chunk(document_id, chunk_no);
CREATE INDEX IF NOT EXISTS idx_lib_chunk_created ON kb.library_chunk(created_at);

-- ═══════════════════════════════════════════════════════════════
-- FTS LIBRARY CHUNK
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.library_chunk_fts (
  chunk_id UUID PRIMARY KEY REFERENCES kb.library_chunk(id) ON DELETE CASCADE,
  tsv_it TSVECTOR
);
CREATE INDEX IF NOT EXISTS idx_lib_chunk_fts ON kb.library_chunk_fts USING GIN(tsv_it);

-- Trigger function FTS
CREATE OR REPLACE FUNCTION kb.fn_library_chunk_fts_update() RETURNS trigger AS $$
BEGIN
    INSERT INTO kb.library_chunk_fts(chunk_id, tsv_it)
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
    WHERE t.tgname = 'trg_library_chunk_fts'
      AND n.nspname = 'kb' AND c.relname = 'library_chunk'
  ) THEN
    CREATE TRIGGER trg_library_chunk_fts
      AFTER INSERT OR UPDATE OF text ON kb.library_chunk
      FOR EACH ROW EXECUTE FUNCTION kb.fn_library_chunk_fts_update();
  END IF;
END
$$;

-- ═══════════════════════════════════════════════════════════════
-- EMBEDDINGS LIBRARY CHUNK
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kb.library_chunk_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  chunk_id UUID NOT NULL REFERENCES kb.library_chunk(id) ON DELETE CASCADE,
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

-- ═══════════════════════════════════════════════════════════════
-- HNSW INDEX (inline, small table expected)
-- ═══════════════════════════════════════════════════════════════

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'kb' AND c.relname = 'idx_lib_chunk_emb_1536'
  ) THEN
    CREATE INDEX idx_lib_chunk_emb_1536
      ON kb.library_chunk_embeddings
      USING hnsw (embedding vector_cosine_ops)
      WHERE dims = 1536;
  END IF;
END
$$;

COMMIT;
