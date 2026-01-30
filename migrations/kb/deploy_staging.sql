-- ============================================================================
-- KB MASSIMARI - DEPLOY STAGING
-- Script completo per creare il KB verticale su staging
-- Server: 91.99.229.111 (LEXe staging)
-- ============================================================================

-- ============================================================================
-- 1. SCHEMA
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS kb;
SET search_path TO kb, public;

-- ============================================================================
-- 2. EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Load AGE
LOAD 'age';

-- Set search_path per AGE
SET search_path TO ag_catalog, kb, public;

-- Crea grafo (se non esiste)
DO $$
BEGIN
    PERFORM * FROM ag_catalog.ag_graph WHERE name = 'lexe_jurisprudence';
    IF NOT FOUND THEN
        PERFORM ag_catalog.create_graph('lexe_jurisprudence');
        RAISE NOTICE 'Graph lexe_jurisprudence created';
    ELSE
        RAISE NOTICE 'Graph lexe_jurisprudence already exists';
    END IF;
END $$;

SET search_path TO kb, ag_catalog, public;

-- ============================================================================
-- 3. TABELLE CORE
-- ============================================================================

-- Documenti sorgente (PDF massimari)
CREATE TABLE IF NOT EXISTS kb.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL UNIQUE,
    anno INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    tipo VARCHAR(10) NOT NULL CHECK (tipo IN ('civile', 'penale')),
    titolo TEXT,
    pagine INTEGER,
    ocr_quality_score FLOAT,
    metadata JSONB DEFAULT '{}',
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_anno ON kb.documents(anno);
CREATE INDEX IF NOT EXISTS idx_documents_tipo ON kb.documents(tipo);

-- Sezioni gerarchiche
CREATE TABLE IF NOT EXISTS kb.sections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES kb.documents(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES kb.sections(id) ON DELETE CASCADE,
    level INTEGER NOT NULL,
    titolo TEXT NOT NULL,
    pagina_inizio INTEGER,
    pagina_fine INTEGER,
    section_path TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sections_document ON kb.sections(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_parent ON kb.sections(parent_id);

-- Massime (unita atomica di retrieval)
CREATE TABLE IF NOT EXISTS kb.massime (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,

    testo TEXT NOT NULL,
    testo_con_contesto TEXT,
    testo_normalizzato TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,

    sezione VARCHAR(20),
    numero VARCHAR(20),
    anno INTEGER,
    data_decisione DATE,
    rv VARCHAR(30),
    relatore VARCHAR(100),

    pagina_inizio INTEGER,
    pagina_fine INTEGER,
    tipo VARCHAR(10),
    materia VARCHAR(100),
    keywords TEXT[],

    importance_score FLOAT DEFAULT 0.5,
    confirmation_count INTEGER DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_massime_document ON kb.massime(document_id);
CREATE INDEX IF NOT EXISTS idx_massime_section ON kb.massime(section_id);
CREATE INDEX IF NOT EXISTS idx_massime_anno ON kb.massime(anno);
CREATE INDEX IF NOT EXISTS idx_massime_tipo ON kb.massime(tipo);
CREATE INDEX IF NOT EXISTS idx_massime_hash ON kb.massime(content_hash);

-- pg_trgm per fuzzy matching
CREATE INDEX IF NOT EXISTS idx_massime_trgm ON kb.massime
USING gin (testo_normalizzato gin_trgm_ops);

-- ============================================================================
-- 4. FULL-TEXT SEARCH (tsvector columns)
-- ============================================================================

ALTER TABLE kb.massime
ADD COLUMN IF NOT EXISTS tsv_simple TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(testo, ''))) STORED;

ALTER TABLE kb.massime
ADD COLUMN IF NOT EXISTS tsv_italian TSVECTOR
GENERATED ALWAYS AS (to_tsvector('italian', COALESCE(testo, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_massime_tsv_simple ON kb.massime USING gin(tsv_simple);
CREATE INDEX IF NOT EXISTS idx_massime_tsv_italian ON kb.massime USING gin(tsv_italian);

-- ============================================================================
-- 5. EMBEDDING TABLE - MISTRAL (WINNER from benchmark)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.emb_mistral (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    chunk_idx SMALLINT DEFAULT 0,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(massima_id, chunk_idx)
);

-- HNSW index per cosine similarity
CREATE INDEX IF NOT EXISTS idx_emb_mistral_hnsw
ON kb.emb_mistral USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE kb.emb_mistral IS
'Mistral Embed embeddings (1024 dim) via OpenRouter. Winner from benchmark: 0.793 avg similarity score.';

-- ============================================================================
-- 6. TABELLE SUPPLEMENTARI
-- ============================================================================

-- Citazioni estratte
CREATE TABLE IF NOT EXISTS kb.citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    tipo VARCHAR(20) NOT NULL,
    raw_text TEXT NOT NULL,
    articolo VARCHAR(30),
    codice VARCHAR(100),
    anno INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citations_massima ON kb.citations(massima_id);
CREATE INDEX IF NOT EXISTS idx_citations_tipo ON kb.citations(tipo);

-- Edge weights per graph
CREATE TABLE IF NOT EXISTS kb.edge_weights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    edge_type VARCHAR(50) NOT NULL,
    weight FLOAT NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edge_weights_source ON kb.edge_weights(source_id);
CREATE INDEX IF NOT EXISTS idx_edge_weights_target ON kb.edge_weights(target_id);

-- Ingestion jobs tracking
CREATE TABLE IF NOT EXISTS kb.ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES kb.documents(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 7. FUNCTIONS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION kb.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
DROP TRIGGER IF EXISTS tr_documents_updated_at ON kb.documents;
CREATE TRIGGER tr_documents_updated_at
    BEFORE UPDATE ON kb.documents
    FOR EACH ROW EXECUTE FUNCTION kb.update_updated_at();

DROP TRIGGER IF EXISTS tr_massime_updated_at ON kb.massime;
CREATE TRIGGER tr_massime_updated_at
    BEFORE UPDATE ON kb.massime
    FOR EACH ROW EXECUTE FUNCTION kb.update_updated_at();

-- ============================================================================
-- 8. VERIFICATION
-- ============================================================================

DO $$
DECLARE
    ext_record RECORD;
    tbl_record RECORD;
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'KB MASSIMARI - STAGING DEPLOYMENT COMPLETE';
    RAISE NOTICE '============================================================';

    RAISE NOTICE 'Extensions:';
    FOR ext_record IN SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'age', 'pg_trgm') LOOP
        RAISE NOTICE '  - % v%', ext_record.extname, ext_record.extversion;
    END LOOP;

    RAISE NOTICE 'Tables in kb schema:';
    FOR tbl_record IN SELECT tablename FROM pg_tables WHERE schemaname = 'kb' ORDER BY tablename LOOP
        RAISE NOTICE '  - kb.%', tbl_record.tablename;
    END LOOP;

    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Ready for ingestion!';
    RAISE NOTICE '============================================================';
END $$;
