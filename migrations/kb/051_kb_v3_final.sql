-- ============================================================================
-- KB Normativa V3 Final Schema
-- Migration: 051_kb_v3_final.sql
-- Date: 2026-02-06
-- Description: Complete schema for KB Normativa with 3-axis classification
--              (identity_class, quality_class, lifecycle_status)
-- ============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS kb;

-- ============================================================================
-- LEGACY TABLE MIGRATION
-- Rename old normativa tables to preserve data during V3 migration
-- ============================================================================

DO $$
BEGIN
  -- Rename old normativa if exists (preserve data)
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kb' AND table_name = 'normativa') THEN
    -- First drop foreign key constraints that reference normativa
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kb' AND table_name = 'normativa_citations') THEN
      ALTER TABLE kb.normativa_citations DROP CONSTRAINT IF EXISTS normativa_citations_source_id_fkey;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kb' AND table_name = 'normativa_embeddings') THEN
      ALTER TABLE kb.normativa_embeddings DROP CONSTRAINT IF EXISTS normativa_embeddings_normativa_id_fkey;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kb' AND table_name = 'normativa_updates') THEN
      ALTER TABLE kb.normativa_updates DROP CONSTRAINT IF EXISTS normativa_updates_normativa_id_fkey;
    END IF;
    -- Rename old table
    ALTER TABLE kb.normativa RENAME TO normativa_v2_legacy;
    RAISE NOTICE 'Renamed kb.normativa to kb.normativa_v2_legacy';
  END IF;

  -- Rename old normativa_altalex if exists
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kb' AND table_name = 'normativa_altalex') THEN
    ALTER TABLE kb.normativa_altalex RENAME TO normativa_altalex_v2_legacy;
    RAISE NOTICE 'Renamed kb.normativa_altalex to kb.normativa_altalex_v2_legacy';
  END IF;

  -- Rename old normativa_embeddings if exists
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kb' AND table_name = 'normativa_embeddings') THEN
    ALTER TABLE kb.normativa_embeddings RENAME TO normativa_embeddings_v2_legacy;
    RAISE NOTICE 'Renamed kb.normativa_embeddings to kb.normativa_embeddings_v2_legacy';
  END IF;
END$$;

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- AGE opzionale (non blocca migration)
DO $$
BEGIN
  BEGIN
    CREATE EXTENSION IF NOT EXISTS age;
  EXCEPTION WHEN others THEN
    RAISE NOTICE 'AGE extension not available, skipping';
  END;
END$$;

-- ============================================================================
-- ENUMS (idempotent)
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace WHERE n.nspname = 'kb' AND t.typname = 'article_identity_class') THEN
    CREATE TYPE kb.article_identity_class AS ENUM ('BASE','SUFFIX','SPECIAL');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace WHERE n.nspname = 'kb' AND t.typname = 'article_quality_class') THEN
    CREATE TYPE kb.article_quality_class AS ENUM ('VALID_STRONG','VALID_SHORT','WEAK','EMPTY','INVALID');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace WHERE n.nspname = 'kb' AND t.typname = 'lifecycle_status') THEN
    CREATE TYPE kb.lifecycle_status AS ENUM ('CURRENT','HISTORICAL','REPEALED','UNKNOWN');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace WHERE n.nspname = 'kb' AND t.typname = 'annotation_type') THEN
    CREATE TYPE kb.annotation_type AS ENUM ('NOTE','COMMENTO','MASSIMA','BROCARDIO','RIFERIMENTO','ALTRO');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace WHERE n.nspname = 'kb' AND t.typname = 'source_system_code') THEN
    CREATE TYPE kb.source_system_code AS ENUM ('ALTALEX_PDF','BROCARDI_OFFLINE','STUDIO_CATALDI_OFFLINE','BROCARDI_ONLINE','NORMATTIVA_ONLINE','ALTRO');
  END IF;
END$$;

-- ============================================================================
-- SOURCE SYSTEM (fonti normativa)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.source_system (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code kb.source_system_code NOT NULL UNIQUE,
  name TEXT NOT NULL,
  base_url TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO kb.source_system(code,name,base_url) VALUES
('ALTALEX_PDF','Altalex PDF',NULL),
('BROCARDI_OFFLINE','Brocardi Offline',NULL),
('STUDIO_CATALDI_OFFLINE','Studio Cataldi Offline',NULL),
('BROCARDI_ONLINE','Brocardi Online','https://www.brocardi.it'),
('NORMATTIVA_ONLINE','Normattiva','https://www.normattiva.it')
ON CONFLICT (code) DO NOTHING;

-- ============================================================================
-- SOURCE TYPE (gerarchico: CODICE -> CODICE_PROCEDURA)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.source_type (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  parent_id UUID REFERENCES kb.source_type(id) ON DELETE SET NULL
);

-- ============================================================================
-- TOPIC (gerarchico: civile -> civile-procedura)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.topic (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  parent_id UUID REFERENCES kb.topic(id) ON DELETE SET NULL
);

-- ============================================================================
-- NIR MAPPING (code -> URN:NIR base + resolver)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.nir_mapping (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT NOT NULL UNIQUE,
  nir_base TEXT NOT NULL,
  normattiva_resolver TEXT,
  canonical_title TEXT,
  source_system_id UUID REFERENCES kb.source_system(id) ON DELETE SET NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- WORK (atti normativi: CC, CP, CPC, CPP, COST, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  source_type_id UUID REFERENCES kb.source_type(id) ON DELETE SET NULL,
  nir_mapping_id UUID REFERENCES kb.nir_mapping(id) ON DELETE SET NULL,
  publication_date DATE,
  gu_ref TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- WORK ALIAS (CCI -> CCII, c.c. -> CC, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work_alias (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  work_id UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  UNIQUE(work_id, alias)
);

-- ============================================================================
-- WORK TOPIC (M2M work <-> topic)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work_topic (
  work_id UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,
  topic_id UUID NOT NULL REFERENCES kb.topic(id) ON DELETE CASCADE,
  PRIMARY KEY(work_id, topic_id)
);

-- ============================================================================
-- WORK SOURCE LINK (provenance per atto)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work_source_link (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  work_id UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,
  source_system_id UUID NOT NULL REFERENCES kb.source_system(id) ON DELETE CASCADE,
  source_ref TEXT,
  retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(work_id, source_system_id, source_ref)
);

-- ============================================================================
-- NORMATIVA (articoli con 3 assi classificazione)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  work_id UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,

  -- Articolo info
  articolo TEXT NOT NULL,                              -- "2043", "2043-bis"
  articolo_num INTEGER,                                -- 2043
  articolo_suffix TEXT,                                -- "bis", "ter", etc.

  -- 3-axis classification
  identity_class kb.article_identity_class NOT NULL,   -- BASE, SUFFIX, SPECIAL
  quality kb.article_quality_class NOT NULL,           -- VALID_STRONG, VALID_SHORT, WEAK, EMPTY, INVALID
  lifecycle kb.lifecycle_status NOT NULL DEFAULT 'UNKNOWN', -- CURRENT, HISTORICAL, REPEALED, UNKNOWN

  -- Sorting and URN
  articolo_sort_key TEXT NOT NULL,                     -- "002043.00", "002043.02" (for bis)
  urn_nir TEXT,                                        -- Full URN:NIR for article

  -- Content
  rubrica TEXT,                                        -- Article title/heading
  testo TEXT NOT NULL,                                 -- Article text

  -- Versioning
  is_current BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(work_id, articolo_sort_key)
);

CREATE INDEX IF NOT EXISTS idx_normativa_work ON kb.normativa(work_id);
CREATE INDEX IF NOT EXISTS idx_normativa_sort ON kb.normativa(work_id, articolo_sort_key);
CREATE INDEX IF NOT EXISTS idx_normativa_quality ON kb.normativa(work_id, quality);
CREATE INDEX IF NOT EXISTS idx_normativa_lifecycle ON kb.normativa(work_id, lifecycle);

-- ============================================================================
-- NORMATIVA ALTALEX (extension per dati Altalex specifici)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_altalex (
  normativa_id UUID PRIMARY KEY REFERENCES kb.normativa(id) ON DELETE CASCADE,
  global_key TEXT UNIQUE,                              -- "altalex:cc:2043:bis"
  testo_context TEXT                                   -- Overlap +/- 200 chars
);

-- ============================================================================
-- NORMATIVA FIELD SOURCE (provenance per campo: testo, rubrica, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_field_source (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  normativa_id UUID NOT NULL REFERENCES kb.normativa(id) ON DELETE CASCADE,
  field_name TEXT NOT NULL,                            -- "testo", "rubrica"
  source_system_id UUID NOT NULL REFERENCES kb.source_system(id) ON DELETE CASCADE,
  source_ref TEXT,
  similarity REAL,                                     -- Jaccard/trgm similarity
  decided_by TEXT,                                     -- "auto", "manual"
  decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(normativa_id, field_name)
);

-- ============================================================================
-- NORMATIVA FTS (full-text search separato con trigger)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_fts (
  normativa_id UUID PRIMARY KEY REFERENCES kb.normativa(id) ON DELETE CASCADE,
  tsv_it TSVECTOR
);
CREATE INDEX IF NOT EXISTS idx_normativa_fts ON kb.normativa_fts USING GIN(tsv_it);

CREATE OR REPLACE FUNCTION kb.fn_normativa_fts_update() RETURNS trigger AS $$
BEGIN
  INSERT INTO kb.normativa_fts(normativa_id, tsv_it)
  VALUES(NEW.id, to_tsvector('italian', unaccent(coalesce(NEW.rubrica,'') || ' ' || coalesce(NEW.testo,''))))
  ON CONFLICT (normativa_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_normativa_fts ON kb.normativa;
CREATE TRIGGER trg_normativa_fts AFTER INSERT OR UPDATE OF rubrica, testo ON kb.normativa
FOR EACH ROW EXECUTE FUNCTION kb.fn_normativa_fts_update();

-- ============================================================================
-- NORMATIVA EMBEDDINGS (vector indexes per canale)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  normativa_id UUID NOT NULL REFERENCES kb.normativa(id) ON DELETE CASCADE,
  model TEXT NOT NULL,                                 -- "text-embedding-3-small"
  channel TEXT NOT NULL,                               -- "testo", "rubrica"
  dims INTEGER NOT NULL,                               -- 1536, 1024, 768
  embedding vector(1536) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(normativa_id, model, channel, dims)
);

-- HNSW indexes per dimension/channel (partial)
CREATE INDEX IF NOT EXISTS idx_norm_emb_1536_testo
ON kb.normativa_embeddings USING hnsw (embedding vector_cosine_ops)
WHERE dims = 1536 AND channel = 'testo';

CREATE INDEX IF NOT EXISTS idx_norm_emb_1536_rubrica
ON kb.normativa_embeddings USING hnsw (embedding vector_cosine_ops)
WHERE dims = 1536 AND channel = 'rubrica';

-- ============================================================================
-- ANNOTATION (note, commenti, massime, brocardi)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.annotation (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_system_id UUID REFERENCES kb.source_system(id) ON DELETE SET NULL,
  type kb.annotation_type NOT NULL,
  title TEXT,
  content TEXT NOT NULL,
  author TEXT,
  source_ref TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- SENTENZA (giurisprudenza)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.sentenza (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_system_id UUID REFERENCES kb.source_system(id) ON DELETE SET NULL,
  authority TEXT,
  section TEXT,
  decision_date DATE,
  number TEXT,
  title TEXT,
  text TEXT,
  source_ref TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb.sentenza_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  sentenza_id UUID NOT NULL REFERENCES kb.sentenza(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  channel TEXT NOT NULL,
  dims INTEGER NOT NULL,
  embedding vector(1536) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(sentenza_id, model, channel, dims)
);

CREATE INDEX IF NOT EXISTS idx_sent_emb_1536_testo
ON kb.sentenza_embeddings USING hnsw (embedding vector_cosine_ops)
WHERE dims = 1536 AND channel = 'testo';

-- ============================================================================
-- ANNOTATION LINK (M2M annotation <-> normativa/sentenza)
-- Uses surrogate id to avoid PK with nullable columns
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.annotation_link (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  annotation_id UUID NOT NULL REFERENCES kb.annotation(id) ON DELETE CASCADE,
  normativa_id UUID REFERENCES kb.normativa(id) ON DELETE CASCADE,
  sentenza_id UUID REFERENCES kb.sentenza(id) ON DELETE CASCADE,
  relation TEXT NOT NULL DEFAULT 'related',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- XOR: exactly one of normativa_id or sentenza_id must be NOT NULL
  CHECK ((normativa_id IS NOT NULL) <> (sentenza_id IS NOT NULL))
);

-- Partial unique indexes to prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS ux_ann_link_norm
ON kb.annotation_link(annotation_id, normativa_id, relation)
WHERE normativa_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_ann_link_sent
ON kb.annotation_link(annotation_id, sentenza_id, relation)
WHERE sentenza_id IS NOT NULL;

-- ============================================================================
-- INGESTION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.ingestion_run (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  mode TEXT NOT NULL,                                  -- "full", "incremental", "test"
  notes TEXT
);

CREATE TABLE IF NOT EXISTS kb.ingestion_event (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  run_id UUID NOT NULL REFERENCES kb.ingestion_run(id) ON DELETE CASCADE,
  work_code TEXT,
  normativa_id UUID REFERENCES kb.normativa(id) ON DELETE SET NULL,
  level TEXT NOT NULL,                                 -- "info", "warning", "error"
  event_type TEXT NOT NULL,                            -- "extraction", "validation", "conflict"
  message TEXT NOT NULL,
  payload JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ing_event_run ON kb.ingestion_event(run_id);
CREATE INDEX IF NOT EXISTS idx_ing_event_work ON kb.ingestion_event(work_code);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Quality stats per work (for monitoring)
CREATE OR REPLACE VIEW kb.v_quality_stats AS
SELECT
  w.code AS work_code,
  COUNT(n.*) AS total_all,
  MIN(n.articolo_num) FILTER (WHERE n.identity_class IN ('BASE','SUFFIX')) AS range_base_min,
  MAX(n.articolo_num) FILTER (WHERE n.identity_class IN ('BASE','SUFFIX')) AS range_base_max,
  COUNT(*) FILTER (WHERE n.quality IN ('VALID_STRONG','VALID_SHORT')) AS validi,
  COUNT(*) FILTER (WHERE n.quality IN ('WEAK','EMPTY')) AS deboli,
  COUNT(*) FILTER (WHERE n.quality = 'INVALID') AS invalidi,
  COUNT(*) FILTER (WHERE n.lifecycle = 'REPEALED') AS abrogati,
  COUNT(*) FILTER (WHERE n.lifecycle = 'UNKNOWN') AS lifecycle_unknown
FROM kb.work w
LEFT JOIN kb.normativa n ON n.work_id = w.id AND n.is_current = TRUE
GROUP BY w.code;

-- Current normativa (for queries)
CREATE OR REPLACE VIEW kb.v_normativa_current AS
SELECT w.code AS work_code, n.*
FROM kb.normativa n
JOIN kb.work w ON w.id = n.work_id
WHERE n.is_current = TRUE;

COMMIT;
