-- Migration: 009_category_v2.sql
-- Category Graph v2.4: Three-Axis Taxonomy (Materia + Natura + Ambito)
-- Version: v2.4
-- Date: 2026-01-31

-- ============================================================
-- FEATURE EXTRACTION VIEW
-- ============================================================
-- Optimized view for classification pipeline
-- NOTE: testo_lower computed in Python to avoid DB cost
-- NOTE: GROUP BY only on m.id, m.sezione, d.tipo (NOT m.testo!)

CREATE OR REPLACE VIEW kb.massime_features_v2 AS
SELECT
    m.id AS massima_id,
    m.sezione,
    d.tipo,
    MIN(LEFT(m.testo, 2000)) AS testo_trunc,
    COALESCE(
        ARRAY_AGG(DISTINCT n.full_ref) FILTER (WHERE n.full_ref IS NOT NULL),
        ARRAY[]::text[]
    ) AS norms_canonical,
    COUNT(DISTINCT n.id) AS norms_count
FROM kb.massime m
LEFT JOIN kb.documents d ON d.id = m.document_id
LEFT JOIN kb.massima_norms mn ON mn.massima_id = m.id
LEFT JOIN kb.norms n ON n.id = mn.norm_id
WHERE m.is_active = TRUE
GROUP BY m.id, m.sezione, d.tipo;

COMMENT ON VIEW kb.massime_features_v2 IS 'Feature view for category v2 classification. testo_lower computed in Python.';

-- ============================================================
-- GOLDEN SET LABELS (API responses)
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.golden_category_labels_v2 (
    id SERIAL PRIMARY KEY,
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    labeler_model VARCHAR(100) NOT NULL,
    materia_l1 VARCHAR(32),
    natura_l1 VARCHAR(16),
    ambito_l1 VARCHAR(32),
    topic_l2 VARCHAR(50),
    confidence FLOAT,
    rationale TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.golden_category_labels_v2 IS 'Raw API labeling responses from models';
COMMENT ON COLUMN kb.golden_category_labels_v2.labeler_model IS 'Model used: qwen/qwen3-235b-a22b-2507, mistralai/mistral-large-2512, openai/gpt-5.2';

CREATE INDEX idx_golden_labels_v2_massima ON kb.golden_category_labels_v2(massima_id);
CREATE INDEX idx_golden_labels_v2_model ON kb.golden_category_labels_v2(labeler_model);

-- ============================================================
-- ADJUDICATED GROUND TRUTH
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.golden_category_adjudicated_v2 (
    massima_id UUID PRIMARY KEY REFERENCES kb.massime(id) ON DELETE CASCADE,
    materia_l1 VARCHAR(32) NOT NULL,
    natura_l1 VARCHAR(16) NOT NULL,
    ambito_l1 VARCHAR(32),
    topic_l2 VARCHAR(50),
    agreement_score FLOAT,
    difficulty_bucket VARCHAR(32),
    split VARCHAR(10) NOT NULL CHECK (split IN ('train', 'test')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.golden_category_adjudicated_v2 IS 'Final ground truth after adjudication';
COMMENT ON COLUMN kb.golden_category_adjudicated_v2.agreement_score IS '1.0=both agree, 0.7=partial, 0.5=judge decided';
COMMENT ON COLUMN kb.golden_category_adjudicated_v2.difficulty_bucket IS 'easy, metadata_ambiguous, procedural_heavy, cross_domain';
COMMENT ON COLUMN kb.golden_category_adjudicated_v2.split IS 'train (420) or test (180)';

CREATE INDEX idx_golden_adjudicated_v2_split ON kb.golden_category_adjudicated_v2(split);
CREATE INDEX idx_golden_adjudicated_v2_bucket ON kb.golden_category_adjudicated_v2(difficulty_bucket);
CREATE INDEX idx_golden_adjudicated_v2_materia ON kb.golden_category_adjudicated_v2(materia_l1);

-- ============================================================
-- CATEGORY PREDICTIONS V2
-- ============================================================
-- Main output table with full audit trail

CREATE TABLE IF NOT EXISTS kb.category_predictions_v2 (
    id SERIAL PRIMARY KEY,
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    run_id INT REFERENCES kb.graph_runs(id) ON DELETE CASCADE,

    -- Axis A: Materia (always assigned)
    materia_l1 VARCHAR(32) NOT NULL
        CHECK (materia_l1 IN ('CIVILE', 'PENALE', 'LAVORO', 'TRIBUTARIO', 'AMMINISTRATIVO', 'CRISI')),
    materia_confidence FLOAT NOT NULL
        CHECK (materia_confidence >= 0 AND materia_confidence <= 1),
    materia_rule VARCHAR(50),
    materia_candidate_set TEXT[],
    materia_reasons TEXT[],

    -- Axis B: Natura (always assigned)
    natura_l1 VARCHAR(16) NOT NULL
        CHECK (natura_l1 IN ('SOSTANZIALE', 'PROCESSUALE')),
    natura_confidence FLOAT NOT NULL
        CHECK (natura_confidence >= 0 AND natura_confidence <= 1),
    natura_rule VARCHAR(50),

    -- Axis C: Ambito (only if natura=PROCESSUALE)
    ambito_l1 VARCHAR(32)
        CHECK (ambito_l1 IS NULL OR ambito_l1 IN ('GIUDIZIO', 'IMPUGNAZIONI', 'ESECUZIONE', 'MISURE', 'UNKNOWN')),
    ambito_confidence FLOAT
        CHECK (ambito_confidence IS NULL OR (ambito_confidence >= 0 AND ambito_confidence <= 1)),
    ambito_rule VARCHAR(50),

    -- Topic L2 (with abstain)
    topic_l2 VARCHAR(50),
    topic_l2_confidence FLOAT,
    topic_l2_flag VARCHAR(16)
        CHECK (topic_l2_flag IS NULL OR topic_l2_flag IN ('auto', 'flagged', 'abstain')),
    abstain_reason TEXT,

    -- Composite confidence
    composite_confidence FLOAT NOT NULL
        CHECK (composite_confidence >= 0 AND composite_confidence <= 1),

    -- Metadata for confidence adjustment
    norms_count INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(massima_id, run_id),

    -- Constraint: ambito must be NULL if natura=SOSTANZIALE
    CONSTRAINT ambito_only_if_processuale
        CHECK (natura_l1 = 'PROCESSUALE' OR ambito_l1 IS NULL)
);

COMMENT ON TABLE kb.category_predictions_v2 IS 'Category predictions with three-axis taxonomy';
COMMENT ON COLUMN kb.category_predictions_v2.materia_l1 IS 'Subject matter: CIVILE, PENALE, LAVORO, TRIBUTARIO, AMMINISTRATIVO, CRISI';
COMMENT ON COLUMN kb.category_predictions_v2.natura_l1 IS 'Legal nature: SOSTANZIALE or PROCESSUALE';
COMMENT ON COLUMN kb.category_predictions_v2.ambito_l1 IS 'Procedural scope (only if PROCESSUALE): GIUDIZIO, IMPUGNAZIONI, ESECUZIONE, MISURE, UNKNOWN';
COMMENT ON COLUMN kb.category_predictions_v2.materia_reasons IS 'Audit trail: all reasons that led to materia decision';
COMMENT ON COLUMN kb.category_predictions_v2.norms_count IS 'Number of norms cited (0 = -0.03 confidence penalty)';

-- Performance indexes
CREATE INDEX idx_category_pred_v2_run ON kb.category_predictions_v2(run_id);
CREATE INDEX idx_category_pred_v2_massima ON kb.category_predictions_v2(massima_id);
CREATE INDEX idx_category_pred_v2_materia ON kb.category_predictions_v2(materia_l1);
CREATE INDEX idx_category_pred_v2_natura ON kb.category_predictions_v2(natura_l1);
CREATE INDEX idx_category_pred_v2_ambito ON kb.category_predictions_v2(ambito_l1)
    WHERE ambito_l1 IS NOT NULL;
CREATE INDEX idx_category_pred_v2_composite ON kb.category_predictions_v2(composite_confidence DESC);

-- ============================================================
-- ENRICHED VIEW
-- ============================================================
-- Joins massime with latest active predictions

CREATE OR REPLACE VIEW kb.massime_enriched AS
SELECT
    m.*,
    p.materia_l1,
    p.materia_confidence,
    p.materia_rule,
    p.materia_candidate_set,
    p.materia_reasons,
    p.natura_l1,
    p.natura_confidence,
    p.natura_rule,
    p.ambito_l1,
    p.ambito_confidence,
    p.ambito_rule,
    p.topic_l2,
    p.topic_l2_confidence,
    p.topic_l2_flag,
    p.composite_confidence
FROM kb.massime m
LEFT JOIN kb.category_predictions_v2 p ON p.massima_id = m.id
    AND p.run_id = (
        SELECT id FROM kb.graph_runs
        WHERE run_type = 'category_v2' AND is_active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
    )
WHERE m.is_active = TRUE;

COMMENT ON VIEW kb.massime_enriched IS 'Active massime with latest category v2 predictions';

-- ============================================================
-- STATISTICS VIEW
-- ============================================================

CREATE OR REPLACE VIEW kb.category_v2_stats AS
SELECT
    materia_l1,
    natura_l1,
    ambito_l1,
    COUNT(*) AS count,
    ROUND(AVG(materia_confidence)::numeric, 3) AS avg_materia_conf,
    ROUND(AVG(natura_confidence)::numeric, 3) AS avg_natura_conf,
    ROUND(AVG(composite_confidence)::numeric, 3) AS avg_composite_conf
FROM kb.category_predictions_v2 p
WHERE p.run_id = (
    SELECT id FROM kb.graph_runs
    WHERE run_type = 'category_v2' AND is_active = TRUE
    ORDER BY created_at DESC
    LIMIT 1
)
GROUP BY materia_l1, natura_l1, ambito_l1
ORDER BY count DESC;

COMMENT ON VIEW kb.category_v2_stats IS 'Statistics for active category v2 run';

-- ============================================================
-- EVAL RUNS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.category_v2_eval_runs (
    id SERIAL PRIMARY KEY,
    run_id INT REFERENCES kb.graph_runs(id),
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),

    -- L1 Gates
    materia_coverage FLOAT,
    materia_accuracy FLOAT,
    natura_coverage FLOAT,
    natura_accuracy FLOAT,
    top2_accuracy FLOAT,
    calibration_error FLOAT,

    -- Ambito Gate
    ambito_coverage FLOAT,
    ambito_unknown_rate FLOAT,

    -- L2 Gates
    l2_abstain_rate FLOAT,
    l2_precision_audit FLOAT,
    l2_precision_auto FLOAT,

    -- Pass/Fail
    all_gates_passed BOOLEAN,
    failed_gates TEXT[],

    notes TEXT
);

COMMENT ON TABLE kb.category_v2_eval_runs IS 'Evaluation results for category v2 runs';
