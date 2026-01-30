-- 002_init_schema.sql
-- LEXE Knowledge Base - Schema per massimari giurisprudenziali
-- Multi-embedding flessibile, BM25, Graph pesato

-- ============================================================
-- SCHEMA
-- ============================================================

CREATE SCHEMA IF NOT EXISTS kb;
SET search_path TO kb, ag_catalog, public;

-- ============================================================
-- TABELLE CORE
-- ============================================================

-- Documenti sorgente (PDF massimari)
CREATE TABLE kb.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 del file
    anno INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    tipo VARCHAR(10) NOT NULL CHECK (tipo IN ('civile', 'penale')),
    titolo TEXT,
    pagine INTEGER,

    -- OCR Quality metrics
    ocr_quality_score FLOAT,              -- 0-1, score complessivo
    ocr_valid_chars_ratio FLOAT,          -- % caratteri validi
    ocr_italian_tokens_ratio FLOAT,       -- % token italiani riconosciuti
    ocr_citation_regex_success FLOAT,     -- % citazioni estratte con regex

    metadata JSONB DEFAULT '{}',
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_anno ON kb.documents(anno);
CREATE INDEX idx_documents_tipo ON kb.documents(tipo);
CREATE INDEX idx_documents_hash ON kb.documents(source_hash);

-- ============================================================
-- Nodi gerarchici (struttura editoriale: parte > capitolo > sezione)
-- ============================================================

CREATE TABLE kb.sections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES kb.documents(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES kb.sections(id) ON DELETE CASCADE,
    level INTEGER NOT NULL,                -- 1=parte, 2=capitolo, 3=sezione, 4=sottosezione
    titolo TEXT NOT NULL,
    pagina_inizio INTEGER,
    pagina_fine INTEGER,
    section_path TEXT NOT NULL,            -- "PARTE I > Cap. 1 > Sez. 2"
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sections_document ON kb.sections(document_id);
CREATE INDEX idx_sections_parent ON kb.sections(parent_id);
CREATE INDEX idx_sections_level ON kb.sections(level);

-- ============================================================
-- Massime atomiche (unita primaria di retrieval)
-- ============================================================

CREATE TABLE kb.massime (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,

    -- Contenuto (Upgrade E: Massima + Evidenza)
    testo TEXT NOT NULL,                    -- Chunk A: Massima pulita
    testo_con_contesto TEXT,                -- Chunk B: Con blocchi OCR attigui
    testo_normalizzato TEXT NOT NULL,       -- Per dedup e trgm
    content_hash VARCHAR(64) NOT NULL,      -- SHA256 normalizzato

    -- Citazione normalizzata (Upgrade F: data come prima classe)
    sezione VARCHAR(20),                    -- "Sez. U", "Sez. 1", "Sez. 6-1"
    numero VARCHAR(20),                     -- "12345"
    anno INTEGER,                           -- 2020
    data_decisione DATE,                    -- Data esatta se disponibile
    rv VARCHAR(30),                         -- "Rv. 123456-01"
    relatore VARCHAR(100),

    -- Metadati
    pagina_inizio INTEGER,
    pagina_fine INTEGER,
    tipo VARCHAR(10),                       -- civile/penale
    materia VARCHAR(100),
    keywords TEXT[],

    -- Scoring
    importance_score FLOAT DEFAULT 0.5,
    confirmation_count INTEGER DEFAULT 1,

    -- Citation pinpoint tracking
    citation_extracted BOOLEAN DEFAULT FALSE,
    citation_complete BOOLEAN DEFAULT FALSE,  -- Tutti i campi estratti?

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indici base
CREATE INDEX idx_massime_document ON kb.massime(document_id);
CREATE INDEX idx_massime_section ON kb.massime(section_id);
CREATE INDEX idx_massime_anno ON kb.massime(anno);
CREATE INDEX idx_massime_sezione ON kb.massime(sezione);
CREATE INDEX idx_massime_tipo ON kb.massime(tipo);
CREATE INDEX idx_massime_materia ON kb.massime(materia);
CREATE INDEX idx_massime_hash ON kb.massime(content_hash);
CREATE INDEX idx_massime_data ON kb.massime(data_decisione);

-- pg_trgm per typo catch e fuzzy matching
CREATE INDEX idx_massime_trgm ON kb.massime
USING gin (testo_normalizzato gin_trgm_ops);

-- ============================================================
-- Embeddings FLESSIBILI (Multi-Modello Multi-Canale)
-- Una riga per (massima, model, channel)
-- ============================================================

CREATE TABLE kb.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,

    model VARCHAR(50) NOT NULL,             -- 'qwen3', 'e5-large', 'bge-m3', 'legal-bert-it'
    channel VARCHAR(20) NOT NULL,           -- 'testo', 'tema', 'contesto'
    embedding vector NOT NULL,              -- Dimensioni variabili per modello
    dims INTEGER NOT NULL,                  -- 1536, 1024, 768, etc.

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(massima_id, model, channel)
);

CREATE INDEX idx_embeddings_massima ON kb.embeddings(massima_id);
CREATE INDEX idx_embeddings_model ON kb.embeddings(model);
CREATE INDEX idx_embeddings_channel ON kb.embeddings(channel);

-- ============================================================
-- Partial HNSW indexes per modello (dimensioni diverse)
-- Cast necessario per indicizzare vettori di dimensioni specifiche
-- ============================================================

-- Qwen3 (1536 dims)
CREATE INDEX idx_emb_qwen3_testo ON kb.embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE model = 'qwen3' AND channel = 'testo';

CREATE INDEX idx_emb_qwen3_tema ON kb.embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE model = 'qwen3' AND channel = 'tema';

CREATE INDEX idx_emb_qwen3_contesto ON kb.embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE model = 'qwen3' AND channel = 'contesto';

-- E5-Large (1024 dims)
CREATE INDEX idx_emb_e5_testo ON kb.embeddings
USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE model = 'e5-large' AND channel = 'testo';

CREATE INDEX idx_emb_e5_tema ON kb.embeddings
USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE model = 'e5-large' AND channel = 'tema';

-- BGE-M3 (1024 dims)
CREATE INDEX idx_emb_bge_testo ON kb.embeddings
USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE model = 'bge-m3' AND channel = 'testo';

CREATE INDEX idx_emb_bge_tema ON kb.embeddings
USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE model = 'bge-m3' AND channel = 'tema';

-- LEGAL-BERT-IT (768 dims) - se disponibile
CREATE INDEX idx_emb_legal_testo ON kb.embeddings
USING hnsw ((embedding::vector(768)) vector_cosine_ops)
WHERE model = 'legal-bert-it' AND channel = 'testo';

-- ============================================================
-- Citazioni e norme estratte
-- ============================================================

CREATE TABLE kb.citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,

    tipo VARCHAR(20) NOT NULL CHECK (tipo IN ('pronuncia', 'norma', 'regolamento_ue', 'direttiva_ue')),
    raw_text TEXT NOT NULL,                 -- Testo originale estratto

    -- Pronuncia
    sezione VARCHAR(20),
    numero VARCHAR(20),
    anno INTEGER,
    data_decisione DATE,
    rv VARCHAR(30),

    -- Norma italiana
    articolo VARCHAR(30),
    comma VARCHAR(20),
    codice VARCHAR(100),                    -- "c.c.", "c.p.c.", "c.p.", "cost."

    -- Norme EU
    regolamento VARCHAR(100),
    direttiva VARCHAR(100),
    celex VARCHAR(50),

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_citations_massima ON kb.citations(massima_id);
CREATE INDEX idx_citations_tipo ON kb.citations(tipo);
CREATE INDEX idx_citations_codice ON kb.citations(codice);
CREATE INDEX idx_citations_articolo ON kb.citations(articolo);
CREATE INDEX idx_citations_anno ON kb.citations(anno);
CREATE INDEX idx_citations_sezione ON kb.citations(sezione);

-- trgm per fuzzy search su citazioni raw
CREATE INDEX idx_citations_raw_trgm ON kb.citations
USING gin (raw_text gin_trgm_ops);

-- ============================================================
-- Near-duplicate tracking
-- ============================================================

CREATE TABLE kb.duplicates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    duplicate_of UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    similarity FLOAT NOT NULL,
    strategy VARCHAR(20) NOT NULL CHECK (strategy IN ('skip', 'merge', 'supersede', 'update', 'implicit_confirm')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_duplicates_massima ON kb.duplicates(massima_id);
CREATE INDEX idx_duplicates_of ON kb.duplicates(duplicate_of);

-- ============================================================
-- Graph Edge Weights (metadata per calcolo pesi AGE)
-- ============================================================

CREATE TABLE kb.edge_weights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    edge_type VARCHAR(50) NOT NULL,         -- 'CITES', 'APPLIES', 'SAME_PRINCIPLE', 'CONTRASTS_WITH'
    weight FLOAT NOT NULL DEFAULT 0.5,

    -- Fattori di calcolo peso
    norm_overlap INTEGER DEFAULT 0,         -- Quante norme condividono
    embedding_similarity FLOAT,             -- Cosine similarity embeddings
    temporal_proximity INTEGER,             -- Anni di differenza
    same_section BOOLEAN DEFAULT FALSE,     -- Stessa sezione Cassazione

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX idx_edge_weights_source ON kb.edge_weights(source_id);
CREATE INDEX idx_edge_weights_target ON kb.edge_weights(target_id);
CREATE INDEX idx_edge_weights_type ON kb.edge_weights(edge_type);
CREATE INDEX idx_edge_weights_weight ON kb.edge_weights(weight DESC);

-- ============================================================
-- Ingestion Jobs tracking
-- ============================================================

CREATE TABLE kb.ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES kb.documents(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'retrying')),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ingestion_jobs_status ON kb.ingestion_jobs(status);
CREATE INDEX idx_ingestion_jobs_document ON kb.ingestion_jobs(document_id);

-- ============================================================
-- Benchmark Results tracking
-- ============================================================

CREATE TABLE kb.benchmark_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_name VARCHAR(50) NOT NULL,       -- 'S1_qwen3', 'S2_e5', etc.
    embedding_model VARCHAR(50) NOT NULL,
    system_config VARCHAR(10) NOT NULL,     -- 'S1', 'S2', 'S3', 'S4', 'S5'

    -- Retrieval metrics
    recall_at_20 FLOAT,
    mrr_at_10 FLOAT,
    precision_at_5 FLOAT,

    -- Answer quality
    groundedness FLOAT,
    citation_completeness FLOAT,
    contradiction_rate FLOAT,

    -- System metrics
    latency_p95_ms FLOAT,
    cost_per_query FLOAT,

    -- OCR metrics
    ocr_valid_chars_ratio FLOAT,
    ocr_citation_regex_success FLOAT,
    citation_pinpoint_accuracy FLOAT,

    -- Final score
    final_score FLOAT,

    query_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_benchmark_config ON kb.benchmark_runs(config_name);
CREATE INDEX idx_benchmark_score ON kb.benchmark_runs(final_score DESC);

-- ============================================================
-- FUNCTIONS
-- ============================================================

-- Funzione per calcolare peso edge SAME_PRINCIPLE
CREATE OR REPLACE FUNCTION kb.calculate_edge_weight(
    p_norm_overlap INTEGER,
    p_embedding_similarity FLOAT,
    p_temporal_proximity INTEGER,
    p_same_section BOOLEAN
) RETURNS FLOAT AS $$
DECLARE
    norm_weight FLOAT := 0.3;
    emb_weight FLOAT := 0.4;
    temp_weight FLOAT := 0.2;
    sect_weight FLOAT := 0.1;
    norm_normalized FLOAT;
    temp_decay FLOAT;
    sect_bonus FLOAT;
BEGIN
    -- Normalize norm_overlap (assume max 10 shared norms)
    norm_normalized := LEAST(p_norm_overlap / 10.0, 1.0);

    -- Temporal decay
    temp_decay := 1.0 / (1.0 + p_temporal_proximity);

    -- Section bonus
    sect_bonus := CASE WHEN p_same_section THEN 1.0 ELSE 0.0 END;

    RETURN (
        norm_weight * norm_normalized +
        emb_weight * COALESCE(p_embedding_similarity, 0) +
        temp_weight * temp_decay +
        sect_weight * sect_bonus
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Funzione per update timestamp
CREATE OR REPLACE FUNCTION kb.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers per updated_at
CREATE TRIGGER tr_documents_updated_at
    BEFORE UPDATE ON kb.documents
    FOR EACH ROW EXECUTE FUNCTION kb.update_updated_at();

CREATE TRIGGER tr_massime_updated_at
    BEFORE UPDATE ON kb.massime
    FOR EACH ROW EXECUTE FUNCTION kb.update_updated_at();

CREATE TRIGGER tr_ingestion_jobs_updated_at
    BEFORE UPDATE ON kb.ingestion_jobs
    FOR EACH ROW EXECUTE FUNCTION kb.update_updated_at();

-- ============================================================
-- LOG
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'LEXE Knowledge Base - Schema initialized';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - kb.documents (source PDFs)';
    RAISE NOTICE '  - kb.sections (hierarchical structure)';
    RAISE NOTICE '  - kb.massime (atomic chunks)';
    RAISE NOTICE '  - kb.embeddings (multi-model, multi-channel)';
    RAISE NOTICE '  - kb.citations (norms and precedents)';
    RAISE NOTICE '  - kb.duplicates (near-duplicate tracking)';
    RAISE NOTICE '  - kb.edge_weights (graph weights)';
    RAISE NOTICE '  - kb.ingestion_jobs (job tracking)';
    RAISE NOTICE '  - kb.benchmark_runs (benchmark results)';
    RAISE NOTICE '============================================================';
END $$;
