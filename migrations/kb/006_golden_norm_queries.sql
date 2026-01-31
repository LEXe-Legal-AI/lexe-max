-- Migration: 006_golden_norm_queries.sql
-- Golden Set for Norm Query Evaluation
-- Sprint: Norm Graph v3.3.0

-- ============================================================
-- GOLDEN NORM QUERIES TABLE
-- ============================================================
-- Test queries for norm lookup and mixed (semantic+norm) evaluation
--
-- query_class:
--   pure_norm: Query is only a norm reference (art. 2043 c.c.)
--   mixed: Query has semantic + norm (danno ingiusto art 2043 c.c.)
--
-- expected_norm_id: Canonical norm ID (CC:2043, DLGS:165:2001)
-- Note: We don't expect a single massima, but any massima citing the norm

CREATE TABLE IF NOT EXISTS kb.golden_norm_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    query_class VARCHAR(20) NOT NULL CHECK (query_class IN ('pure_norm', 'mixed')),
    expected_norm_id VARCHAR(50) NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

COMMENT ON TABLE kb.golden_norm_queries IS 'Golden set for norm lookup evaluation';
COMMENT ON COLUMN kb.golden_norm_queries.query_class IS 'pure_norm = only norm, mixed = semantic + norm';
COMMENT ON COLUMN kb.golden_norm_queries.expected_norm_id IS 'Canonical norm ID: CC:2043, DLGS:165:2001';

-- ============================================================
-- INDEXES
-- ============================================================

-- Active queries by batch
CREATE INDEX IF NOT EXISTS idx_golden_norm_queries_active
    ON kb.golden_norm_queries(is_active, batch_id);

-- Lookup by norm
CREATE INDEX IF NOT EXISTS idx_golden_norm_queries_norm
    ON kb.golden_norm_queries(expected_norm_id);

-- Unique constraint: no duplicate queries per batch
CREATE UNIQUE INDEX IF NOT EXISTS uq_golden_norm_queries_batch_query
    ON kb.golden_norm_queries(batch_id, query_text);

-- ============================================================
-- EVAL RESULTS TABLE (optional, for tracking runs)
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.norm_eval_runs (
    id SERIAL PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    run_at TIMESTAMPTZ DEFAULT NOW(),
    top_k INTEGER NOT NULL DEFAULT 10,

    -- Pure norm metrics
    pure_norm_count INTEGER,
    pure_norm_recall_at_k NUMERIC(5,4),
    pure_norm_mrr NUMERIC(5,4),
    pure_norm_router_accuracy NUMERIC(5,4),

    -- Mixed metrics
    mixed_count INTEGER,
    mixed_norm_hit_rate NUMERIC(5,4),
    mixed_norm_mrr NUMERIC(5,4),

    -- Overall
    total_queries INTEGER,
    avg_latency_ms NUMERIC(10,2),

    notes TEXT
);

COMMENT ON TABLE kb.norm_eval_runs IS 'Evaluation run results for norm golden set';

-- ============================================================
-- HELPER VIEW: Active golden queries with norm info
-- ============================================================

CREATE OR REPLACE VIEW kb.golden_norm_queries_with_info AS
SELECT
    gnq.*,
    n.full_ref,
    n.code,
    n.citation_count
FROM kb.golden_norm_queries gnq
LEFT JOIN kb.norms n ON n.id = gnq.expected_norm_id
WHERE gnq.is_active = TRUE;

COMMENT ON VIEW kb.golden_norm_queries_with_info IS 'Active golden queries with norm metadata';
