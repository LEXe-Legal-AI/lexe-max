-- Migration: Multi-Model Embeddings Schema
-- Schema ottimizzato per benchmark comparativo con dimensioni diverse
-- Ogni modello ha la sua tabella con HNSW index dedicato

-- ============================================================================
-- DROP OLD FLEXIBLE TABLE (se vuota)
-- ============================================================================

-- Backup check: solo se vuota
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM kb.embeddings) = 0 THEN
        DROP TABLE IF EXISTS kb.embeddings CASCADE;
        RAISE NOTICE 'Dropped empty kb.embeddings table';
    ELSE
        RAISE NOTICE 'kb.embeddings has data, keeping it';
    END IF;
END $$;

-- ============================================================================
-- TABELLE PER MODELLO (dimensioni fisse per HNSW)
-- ============================================================================

-- BGE-M3: 1024 dim, max 8192 tokens
-- Best for: dense + sparse + colbert-like
CREATE TABLE IF NOT EXISTS kb.emb_bge_m3 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    chunk_idx SMALLINT DEFAULT 0,  -- 0 = full massima, 1+ = sub-chunks
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(massima_id, chunk_idx)
);

-- Multilingual E5-Large: 1024 dim, max 512 tokens
-- Best for: general multilingual, italiano
CREATE TABLE IF NOT EXISTS kb.emb_e5_large (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    chunk_idx SMALLINT DEFAULT 0,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(massima_id, chunk_idx)
);

-- Distil-ITA-Legal-BERT: 768 dim, max 512 tokens
-- Best for: Italian legal domain
CREATE TABLE IF NOT EXISTS kb.emb_ita_legal_bert (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    chunk_idx SMALLINT DEFAULT 0,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(massima_id, chunk_idx)
);

-- Qwen3 (via LiteLLM): 1536 dim (se usi text-embedding-3-large style)
-- Placeholder per eventuale uso futuro
CREATE TABLE IF NOT EXISTS kb.emb_qwen3 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    chunk_idx SMALLINT DEFAULT 0,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(massima_id, chunk_idx)
);

-- ============================================================================
-- HNSW INDEXES (cosine similarity)
-- ============================================================================

-- BGE-M3
CREATE INDEX IF NOT EXISTS idx_emb_bge_m3_hnsw
ON kb.emb_bge_m3 USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- E5-Large
CREATE INDEX IF NOT EXISTS idx_emb_e5_large_hnsw
ON kb.emb_e5_large USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ITA-Legal-BERT
CREATE INDEX IF NOT EXISTS idx_emb_ita_legal_bert_hnsw
ON kb.emb_ita_legal_bert USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Qwen3
CREATE INDEX IF NOT EXISTS idx_emb_qwen3_hnsw
ON kb.emb_qwen3 USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- TABELLA BENCHMARK RUNS
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.embedding_benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name VARCHAR(100) NOT NULL,
    model VARCHAR(50) NOT NULL,  -- bge_m3, e5_large, ita_legal_bert, qwen3
    retrieval_mode VARCHAR(50) NOT NULL,  -- dense, hybrid, hybrid_rerank

    -- Query set info
    query_set VARCHAR(100),
    query_count INTEGER,

    -- Metrics
    mrr FLOAT,           -- Mean Reciprocal Rank
    ndcg_10 FLOAT,       -- NDCG@10
    recall_10 FLOAT,     -- Recall@10
    recall_50 FLOAT,     -- Recall@50
    latency_p50_ms FLOAT,
    latency_p95_ms FLOAT,

    -- Breakdown per category
    metrics_citazione JSONB,   -- {"mrr": 0.8, "recall_10": 0.9}
    metrics_istituto JSONB,
    metrics_avversaria JSONB,

    -- Config
    config JSONB,        -- Parametri usati (reranker, k, etc.)

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- FUNZIONI HELPER PER BENCHMARK
-- ============================================================================

-- Dense search generico (seleziona tabella per modello)
CREATE OR REPLACE FUNCTION kb.dense_search(
    p_model VARCHAR,
    p_query_embedding VECTOR,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(massima_id UUID, distance FLOAT) AS $$
BEGIN
    IF p_model = 'bge_m3' THEN
        RETURN QUERY
        SELECT e.massima_id, e.embedding <=> p_query_embedding as dist
        FROM kb.emb_bge_m3 e
        WHERE e.chunk_idx = 0  -- Solo full massima per ora
        ORDER BY dist
        LIMIT p_limit;
    ELSIF p_model = 'e5_large' THEN
        RETURN QUERY
        SELECT e.massima_id, e.embedding <=> p_query_embedding as dist
        FROM kb.emb_e5_large e
        WHERE e.chunk_idx = 0
        ORDER BY dist
        LIMIT p_limit;
    ELSIF p_model = 'ita_legal_bert' THEN
        RETURN QUERY
        SELECT e.massima_id, e.embedding <=> p_query_embedding as dist
        FROM kb.emb_ita_legal_bert e
        WHERE e.chunk_idx = 0
        ORDER BY dist
        LIMIT p_limit;
    ELSIF p_model = 'qwen3' THEN
        RETURN QUERY
        SELECT e.massima_id, e.embedding <=> p_query_embedding as dist
        FROM kb.emb_qwen3 e
        WHERE e.chunk_idx = 0
        ORDER BY dist
        LIMIT p_limit;
    ELSE
        RAISE EXCEPTION 'Unknown model: %', p_model;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Hybrid search con RRF (BM25 + Dense)
CREATE OR REPLACE FUNCTION kb.hybrid_search_benchmark(
    p_model VARCHAR,
    p_query_text TEXT,
    p_query_embedding VECTOR,
    p_limit INTEGER DEFAULT 10,
    p_rrf_k INTEGER DEFAULT 60
) RETURNS TABLE(
    massima_id UUID,
    rrf_score FLOAT,
    dense_rank INTEGER,
    sparse_rank INTEGER
) AS $$
WITH sparse AS (
    SELECT m.id as massima_id,
           ROW_NUMBER() OVER (ORDER BY m.tsv_italian @@ plainto_tsquery('italian', p_query_text) DESC, m.id) as rank
    FROM kb.massime m
    WHERE m.tsv_italian @@ plainto_tsquery('italian', p_query_text)
    LIMIT p_limit * 3
),
dense AS (
    SELECT ds.massima_id,
           ROW_NUMBER() OVER (ORDER BY ds.distance) as rank
    FROM kb.dense_search(p_model, p_query_embedding, p_limit * 3) ds
),
combined AS (
    SELECT COALESCE(s.massima_id, d.massima_id) as massima_id,
           COALESCE(d.rank, 999999) as dense_rank,
           COALESCE(s.rank, 999999) as sparse_rank,
           (1.0 / (p_rrf_k + COALESCE(d.rank, 999999))) +
           (1.0 / (p_rrf_k + COALESCE(s.rank, 999999))) as rrf
    FROM dense d
    FULL OUTER JOIN sparse s ON d.massima_id = s.massima_id
)
SELECT c.massima_id, c.rrf, c.dense_rank::INTEGER, c.sparse_rank::INTEGER
FROM combined c
ORDER BY c.rrf DESC
LIMIT p_limit;
$$ LANGUAGE sql;

-- ============================================================================
-- VIEW PER STATS EMBEDDINGS
-- ============================================================================

CREATE OR REPLACE VIEW kb.embedding_stats AS
SELECT 'bge_m3' as model, COUNT(*) as total,
       COUNT(DISTINCT massima_id) as unique_massime,
       COUNT(*) FILTER (WHERE chunk_idx > 0) as sub_chunks
FROM kb.emb_bge_m3
UNION ALL
SELECT 'e5_large', COUNT(*), COUNT(DISTINCT massima_id), COUNT(*) FILTER (WHERE chunk_idx > 0)
FROM kb.emb_e5_large
UNION ALL
SELECT 'ita_legal_bert', COUNT(*), COUNT(DISTINCT massima_id), COUNT(*) FILTER (WHERE chunk_idx > 0)
FROM kb.emb_ita_legal_bert
UNION ALL
SELECT 'qwen3', COUNT(*), COUNT(DISTINCT massima_id), COUNT(*) FILTER (WHERE chunk_idx > 0)
FROM kb.emb_qwen3;

-- ============================================================================
-- COMMENTI
-- ============================================================================

COMMENT ON TABLE kb.emb_bge_m3 IS
'BGE-M3 embeddings (1024 dim). Supporta dense+sparse+colbert. Max 8192 tokens.';

COMMENT ON TABLE kb.emb_e5_large IS
'Multilingual E5-Large embeddings (1024 dim). Ottimo per italiano generale. Max 512 tokens.';

COMMENT ON TABLE kb.emb_ita_legal_bert IS
'distil-ita-legal-bert embeddings (768 dim). Dominio legale italiano. Max 512 tokens - richiede chunking per massime lunghe.';

COMMENT ON TABLE kb.emb_qwen3 IS
'Qwen3 embeddings via LiteLLM (1536 dim). Placeholder per uso futuro.';

COMMENT ON COLUMN kb.emb_bge_m3.chunk_idx IS
'0 = embedding full massima, 1+ = sub-chunks per massime > max_seq_length. Usato per modelli con finestra corta.';

COMMENT ON TABLE kb.embedding_benchmarks IS
'Risultati benchmark retrieval. Traccia MRR, NDCG, Recall per modello e modalità.';
