-- 003_init_bm25.sql
-- LEXE Knowledge Base - BM25 indexes via pg_search (ParadeDB)
-- Nota: pg_search ha licenza AGPLv3 - solo uso interno

-- ============================================================
-- BM25 INDEX per massime (se pg_search disponibile)
-- ============================================================

DO $$
BEGIN
    -- Verifica se pg_search e' disponibile
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
        RAISE NOTICE 'pg_search found, creating BM25 indexes...';

        -- Crea BM25 index su testo massime
        -- Usa tokenizer italiano per stemming
        EXECUTE $bm25$
            CALL paradedb.create_bm25_index(
                index_name => 'massime_bm25_idx',
                table_name => 'kb.massime',
                key_field => 'id',
                text_fields => paradedb.field(
                    name => 'testo',
                    tokenizer => paradedb.tokenizer('italian')
                )
            )
        $bm25$;

        RAISE NOTICE 'BM25 index massime_bm25_idx created successfully';

        -- Crea BM25 index su testo_con_contesto (evidenza)
        EXECUTE $bm25$
            CALL paradedb.create_bm25_index(
                index_name => 'massime_context_bm25_idx',
                table_name => 'kb.massime',
                key_field => 'id',
                text_fields => paradedb.field(
                    name => 'testo_con_contesto',
                    tokenizer => paradedb.tokenizer('italian')
                )
            )
        $bm25$;

        RAISE NOTICE 'BM25 index massime_context_bm25_idx created successfully';

        -- Crea BM25 index su citazioni raw
        EXECUTE $bm25$
            CALL paradedb.create_bm25_index(
                index_name => 'citations_bm25_idx',
                table_name => 'kb.citations',
                key_field => 'id',
                text_fields => paradedb.field(
                    name => 'raw_text',
                    tokenizer => paradedb.tokenizer('default')
                )
            )
        $bm25$;

        RAISE NOTICE 'BM25 index citations_bm25_idx created successfully';

    ELSE
        RAISE WARNING 'pg_search extension not found - BM25 indexes NOT created';
        RAISE WARNING 'System will fallback to pg_trgm + tsvector for sparse search';

        -- Fallback: crea tsvector columns per FTS base
        RAISE NOTICE 'Creating tsvector fallback indexes...';

        -- Aggiungi colonne tsvector se non esistono
        ALTER TABLE kb.massime
        ADD COLUMN IF NOT EXISTS tsv_simple TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(testo, ''))) STORED;

        ALTER TABLE kb.massime
        ADD COLUMN IF NOT EXISTS tsv_italian TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('italian', COALESCE(testo, ''))) STORED;

        -- Indici GIN per full-text search
        CREATE INDEX IF NOT EXISTS idx_massime_tsv_simple ON kb.massime USING gin(tsv_simple);
        CREATE INDEX IF NOT EXISTS idx_massime_tsv_italian ON kb.massime USING gin(tsv_italian);

        RAISE NOTICE 'Fallback tsvector indexes created';
    END IF;

EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error creating BM25 indexes: %', SQLERRM;
    RAISE WARNING 'Falling back to tsvector indexes...';

    -- Fallback su errore
    ALTER TABLE kb.massime
    ADD COLUMN IF NOT EXISTS tsv_simple TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(testo, ''))) STORED;

    ALTER TABLE kb.massime
    ADD COLUMN IF NOT EXISTS tsv_italian TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('italian', COALESCE(testo, ''))) STORED;

    CREATE INDEX IF NOT EXISTS idx_massime_tsv_simple ON kb.massime USING gin(tsv_simple);
    CREATE INDEX IF NOT EXISTS idx_massime_tsv_italian ON kb.massime USING gin(tsv_italian);

    RAISE NOTICE 'Fallback tsvector indexes created after error';
END $$;

-- ============================================================
-- FUNZIONI HELPER per BM25 Search
-- ============================================================

-- Funzione wrapper per BM25 search (con fallback a FTS)
CREATE OR REPLACE FUNCTION kb.bm25_search(
    p_query TEXT,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    massima_id UUID,
    score FLOAT
) AS $$
BEGIN
    -- Prova prima BM25
    BEGIN
        RETURN QUERY EXECUTE $q$
            SELECT m.id, paradedb.score(m.id)::FLOAT as score
            FROM kb.massime m
            WHERE m.id @@@ paradedb.search(
                query => paradedb.parse($1),
                index => 'massime_bm25_idx'
            )
            ORDER BY score DESC
            LIMIT $2
        $q$ USING p_query, p_limit;

        -- Se ritorna risultati, esci
        IF FOUND THEN
            RETURN;
        END IF;

    EXCEPTION WHEN OTHERS THEN
        -- BM25 fallito, usa FTS
        RAISE DEBUG 'BM25 search failed, using FTS fallback: %', SQLERRM;
    END;

    -- Fallback a FTS con ts_rank
    RETURN QUERY
    SELECT m.id,
           ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', p_query))::FLOAT as score
    FROM kb.massime m
    WHERE m.tsv_italian @@ plainto_tsquery('italian', p_query)
       OR m.tsv_simple @@ plainto_tsquery('simple', p_query)
    ORDER BY score DESC
    LIMIT p_limit;

END;
$$ LANGUAGE plpgsql;

-- Funzione per hybrid search (dense + sparse via RRF)
CREATE OR REPLACE FUNCTION kb.hybrid_search(
    p_query TEXT,
    p_query_embedding vector,
    p_model VARCHAR DEFAULT 'qwen3',
    p_channel VARCHAR DEFAULT 'testo',
    p_limit INTEGER DEFAULT 20,
    p_rrf_k INTEGER DEFAULT 60
)
RETURNS TABLE (
    massima_id UUID,
    rrf_score FLOAT,
    dense_rank INTEGER,
    sparse_rank INTEGER
) AS $$
WITH dense_results AS (
    SELECT
        e.massima_id,
        1 - (e.embedding <=> p_query_embedding) as similarity,
        ROW_NUMBER() OVER (ORDER BY e.embedding <=> p_query_embedding) as rank
    FROM kb.embeddings e
    WHERE e.model = p_model AND e.channel = p_channel
    ORDER BY e.embedding <=> p_query_embedding
    LIMIT 50
),
sparse_results AS (
    SELECT
        massima_id,
        score,
        ROW_NUMBER() OVER (ORDER BY score DESC) as rank
    FROM kb.bm25_search(p_query, 50)
),
combined AS (
    SELECT
        COALESCE(d.massima_id, s.massima_id) as massima_id,
        COALESCE(1.0 / (p_rrf_k + d.rank), 0) +
        COALESCE(1.0 / (p_rrf_k + s.rank), 0) as rrf_score,
        d.rank as dense_rank,
        s.rank as sparse_rank
    FROM dense_results d
    FULL OUTER JOIN sparse_results s ON d.massima_id = s.massima_id
)
SELECT
    c.massima_id,
    c.rrf_score,
    c.dense_rank::INTEGER,
    c.sparse_rank::INTEGER
FROM combined c
ORDER BY c.rrf_score DESC
LIMIT p_limit;
$$ LANGUAGE SQL;

-- ============================================================
-- LOG
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'LEXE Knowledge Base - BM25/FTS setup complete';
    RAISE NOTICE '============================================================';
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
        RAISE NOTICE 'Mode: BM25 (pg_search)';
    ELSE
        RAISE NOTICE 'Mode: FTS fallback (tsvector)';
    END IF;
    RAISE NOTICE 'Functions created:';
    RAISE NOTICE '  - kb.bm25_search(query, limit)';
    RAISE NOTICE '  - kb.hybrid_search(query, embedding, model, channel, limit, rrf_k)';
    RAISE NOTICE '============================================================';
END $$;
