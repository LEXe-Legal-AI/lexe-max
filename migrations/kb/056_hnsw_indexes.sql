-- migrations/kb/056_hnsw_indexes.sql
-- V4 HNSW Indexes - Separato da 055 per essere rilanciabile
-- ESEGUIRE DOPO 055_chunking_schema.sql
--
-- RUNBOOK PRE-INDEX (eseguire in sessione psql PRIMA di questo file):
--   SET maintenance_work_mem = '2GB';
--   SET work_mem = '128MB';
--   SET hnsw.ef_construction = 128;
--   SET hnsw.m = 16;
--   SET lock_timeout = '5s';
--   SET statement_timeout = '0';

BEGIN;

-- ═══════════════════════════════════════════════════════════════
-- HNSW NORMATIVA CHUNK EMBEDDINGS
-- ═══════════════════════════════════════════════════════════════

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'kb' AND c.relname = 'idx_norm_chunk_emb_1536'
  ) THEN
    CREATE INDEX idx_norm_chunk_emb_1536
      ON kb.normativa_chunk_embeddings
      USING hnsw (embedding vector_cosine_ops)
      WHERE dims = 1536 AND model = 'openai/text-embedding-3-small';
  END IF;
END
$$;

-- ═══════════════════════════════════════════════════════════════
-- HNSW ANNOTATION CHUNK EMBEDDINGS
-- ═══════════════════════════════════════════════════════════════

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'kb' AND c.relname = 'idx_ann_chunk_emb_1536'
  ) THEN
    CREATE INDEX idx_ann_chunk_emb_1536
      ON kb.annotation_chunk_embeddings
      USING hnsw (embedding vector_cosine_ops)
      WHERE dims = 1536 AND model = 'openai/text-embedding-3-small';
  END IF;
END
$$;

COMMIT;
