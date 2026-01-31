-- Migration: 007_category_graph.sql
-- Category Graph: Topic classification for massime
-- Version: v3.4.0

-- ============================================================
-- CATEGORIES TABLE
-- ============================================================
-- Hierarchical categories for topic classification
-- Level 1: 8 macro areas (always assigned)
-- Level 2: ~40 subcategories (assigned if confidence >= 0.70)
-- Level 3: Future, for specific sub-domains

CREATE TABLE IF NOT EXISTS kb.categories (
    id TEXT PRIMARY KEY,                    -- CIVILE, CIVILE_RESP_CIV, PROC_CIV_IMPUG
    name TEXT NOT NULL,                     -- Human readable name
    description TEXT,                       -- Description for UI
    level INT NOT NULL CHECK (level IN (1, 2, 3)),
    parent_id TEXT REFERENCES kb.categories(id),
    keywords TEXT[] DEFAULT '{}',           -- Keywords for matching
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.categories IS 'Hierarchical topic categories for massime classification';
COMMENT ON COLUMN kb.categories.id IS 'Canonical ID: CIVILE, CIVILE_RESP_CIV';
COMMENT ON COLUMN kb.categories.level IS '1=macro, 2=subcategory, 3=specific';
COMMENT ON COLUMN kb.categories.keywords IS 'Keywords for keyword-based classification';

-- ============================================================
-- CATEGORY ASSIGNMENTS TABLE
-- ============================================================
-- Links massime to categories with confidence scores

CREATE TABLE IF NOT EXISTS kb.category_assignments (
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    category_id TEXT NOT NULL REFERENCES kb.categories(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    method TEXT NOT NULL CHECK (method IN ('keyword', 'embedding', 'hybrid', 'manual')),
    evidence_terms TEXT[] DEFAULT '{}',     -- Terms that triggered the match
    run_id INT,                             -- Optional: track which run assigned this
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (massima_id, category_id)
);

COMMENT ON TABLE kb.category_assignments IS 'Topic assignments for massime with confidence';
COMMENT ON COLUMN kb.category_assignments.confidence IS 'Assignment confidence 0-1';
COMMENT ON COLUMN kb.category_assignments.method IS 'Classification method used';
COMMENT ON COLUMN kb.category_assignments.evidence_terms IS 'Keywords that triggered match';

-- ============================================================
-- INDEXES
-- ============================================================

-- Category lookups
CREATE INDEX IF NOT EXISTS idx_categories_level ON kb.categories(level);
CREATE INDEX IF NOT EXISTS idx_categories_parent ON kb.categories(parent_id);

-- Assignment lookups
CREATE INDEX IF NOT EXISTS idx_cat_assign_category ON kb.category_assignments(category_id);
CREATE INDEX IF NOT EXISTS idx_cat_assign_massima ON kb.category_assignments(massima_id);
CREATE INDEX IF NOT EXISTS idx_cat_assign_confidence ON kb.category_assignments(confidence DESC);

-- Find high-confidence L2 assignments
CREATE INDEX IF NOT EXISTS idx_cat_assign_high_conf ON kb.category_assignments(category_id, confidence)
    WHERE confidence >= 0.70;

-- ============================================================
-- VIEWS
-- ============================================================

-- Categories with assignment counts
CREATE OR REPLACE VIEW kb.category_stats AS
SELECT
    c.id,
    c.name,
    c.level,
    c.parent_id,
    COUNT(ca.massima_id) AS assignment_count,
    AVG(ca.confidence) AS avg_confidence,
    MIN(ca.confidence) AS min_confidence,
    MAX(ca.confidence) AS max_confidence
FROM kb.categories c
LEFT JOIN kb.category_assignments ca ON ca.category_id = c.id
GROUP BY c.id, c.name, c.level, c.parent_id
ORDER BY c.level, assignment_count DESC;

COMMENT ON VIEW kb.category_stats IS 'Category statistics with assignment counts';

-- Massime with their L1 category
CREATE OR REPLACE VIEW kb.massime_with_category AS
SELECT
    m.id AS massima_id,
    m.rv,
    m.sezione,
    m.numero,
    m.anno,
    ca.category_id AS l1_category,
    c.name AS l1_name,
    ca.confidence AS l1_confidence
FROM kb.massime m
LEFT JOIN kb.category_assignments ca ON ca.massima_id = m.id
LEFT JOIN kb.categories c ON c.id = ca.category_id AND c.level = 1
WHERE m.is_active = TRUE;

COMMENT ON VIEW kb.massime_with_category IS 'Active massime with their L1 category';

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Get category hierarchy for a massima
CREATE OR REPLACE FUNCTION kb.get_massima_categories(p_massima_id UUID)
RETURNS TABLE (
    category_id TEXT,
    category_name TEXT,
    level INT,
    parent_id TEXT,
    confidence FLOAT,
    method TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.name,
        c.level,
        c.parent_id,
        ca.confidence,
        ca.method
    FROM kb.category_assignments ca
    JOIN kb.categories c ON c.id = ca.category_id
    WHERE ca.massima_id = p_massima_id
    ORDER BY c.level, ca.confidence DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION kb.get_massima_categories IS 'Get all categories for a massima';

-- Get L1 category for a massima (most confident)
CREATE OR REPLACE FUNCTION kb.get_massima_l1(p_massima_id UUID)
RETURNS TEXT AS $$
    SELECT ca.category_id
    FROM kb.category_assignments ca
    JOIN kb.categories c ON c.id = ca.category_id
    WHERE ca.massima_id = p_massima_id AND c.level = 1
    ORDER BY ca.confidence DESC
    LIMIT 1;
$$ LANGUAGE SQL;

COMMENT ON FUNCTION kb.get_massima_l1 IS 'Get L1 category for a massima';

-- ============================================================
-- CLASSIFICATION RUN TRACKING (optional)
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.category_runs (
    id SERIAL PRIMARY KEY,
    run_type TEXT NOT NULL DEFAULT 'classification',
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_massime INT,
    assigned_l1 INT,
    assigned_l2 INT,
    unknown_count INT,
    config JSONB
);

COMMENT ON TABLE kb.category_runs IS 'Track category classification runs';
