-- Migration: 008_golden_category_queries.sql
-- Golden set for category-based queries
-- Version: v3.4.0

-- ============================================================
-- GOLDEN CATEGORY QUERIES TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.golden_category_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    query_class VARCHAR(20) NOT NULL CHECK (query_class IN ('topic_only', 'topic_semantic')),
    expected_category_id VARCHAR(50) NOT NULL REFERENCES kb.categories(id),
    expected_level INT NOT NULL CHECK (expected_level IN (1, 2)),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.golden_category_queries IS 'Golden set for category-based query evaluation';
COMMENT ON COLUMN kb.golden_category_queries.query_class IS 'topic_only = pure topic, topic_semantic = topic + semantics';
COMMENT ON COLUMN kb.golden_category_queries.expected_category_id IS 'Expected L1 or L2 category';
COMMENT ON COLUMN kb.golden_category_queries.expected_level IS '1 = L1 expected, 2 = L2 expected';

-- Index
CREATE INDEX IF NOT EXISTS idx_golden_cat_batch ON kb.golden_category_queries(batch_id);
CREATE INDEX IF NOT EXISTS idx_golden_cat_class ON kb.golden_category_queries(query_class);
CREATE INDEX IF NOT EXISTS idx_golden_cat_active ON kb.golden_category_queries(is_active) WHERE is_active = TRUE;

-- ============================================================
-- CATEGORY EVAL RUNS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.category_eval_runs (
    id SERIAL PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    run_at TIMESTAMPTZ DEFAULT NOW(),
    top_k INT NOT NULL,

    -- Topic-only metrics
    topic_only_count INT,
    topic_only_accuracy FLOAT,
    topic_only_mrr FLOAT,

    -- Topic-semantic metrics
    topic_semantic_count INT,
    topic_semantic_accuracy FLOAT,
    topic_semantic_mrr FLOAT,

    -- Overall
    total_queries INT,
    avg_latency_ms FLOAT,

    -- Pass/Fail
    topic_only_pass BOOLEAN,
    topic_semantic_pass BOOLEAN,

    config JSONB
);

COMMENT ON TABLE kb.category_eval_runs IS 'Category evaluation run results';
