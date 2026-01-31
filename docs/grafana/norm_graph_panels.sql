-- ============================================================
-- GRAFANA PANELS: Norm Graph Monitoring
-- ============================================================
-- Dashboard: LEXE KB - Norm Graph
-- Created: 2026-01-31

-- ============================================================
-- PANEL 1: Norms Overview (Stat)
-- ============================================================

-- Total norms
SELECT COUNT(*) as "Total Norms" FROM kb.norms;

-- Total edges
SELECT COUNT(*) as "Total Edges" FROM kb.massima_norms;

-- Coverage %
SELECT
    ROUND(
        COUNT(DISTINCT massima_id)::numeric * 100 /
        (SELECT COUNT(*) FROM kb.massime WHERE is_active),
        1
    ) as "Coverage %"
FROM kb.massima_norms;

-- ============================================================
-- PANEL 2: Norms by Code Type (Pie Chart)
-- ============================================================

SELECT
    code as "Code",
    COUNT(*) as "Norms",
    SUM(citation_count) as "Citations"
FROM kb.norms
GROUP BY code
ORDER BY SUM(citation_count) DESC;

-- ============================================================
-- PANEL 3: Top 20 Cited Norms (Table)
-- ============================================================

SELECT
    id as "Norm ID",
    full_ref as "Reference",
    citation_count as "Citations"
FROM kb.norms
ORDER BY citation_count DESC
LIMIT 20;

-- ============================================================
-- PANEL 4: Citation Distribution Histogram
-- ============================================================

SELECT
    width_bucket(citation_count, 0, 500, 10) as bucket,
    COUNT(*) as norms,
    MIN(citation_count) as min_citations,
    MAX(citation_count) as max_citations
FROM kb.norms
GROUP BY bucket
ORDER BY bucket;

-- ============================================================
-- PANEL 5: Recent Graph Runs (Table)
-- ============================================================

SELECT
    id as "Run ID",
    run_type as "Type",
    status as "Status",
    started_at as "Started",
    completed_at as "Completed",
    EXTRACT(EPOCH FROM (completed_at - started_at))::int as "Duration (s)"
FROM kb.graph_runs
WHERE run_type = 'norm_graph'
ORDER BY id DESC
LIMIT 10;

-- ============================================================
-- PANEL 6: Avg Citations per Code (Bar Chart)
-- ============================================================

SELECT
    code as "Code",
    ROUND(AVG(citation_count), 2) as "Avg Citations"
FROM kb.norms
GROUP BY code
ORDER BY AVG(citation_count) DESC;

-- ============================================================
-- PANEL 7: Norms Created Over Time (Time Series)
-- ============================================================
-- Use created_at from kb.norms if available

SELECT
    date_trunc('day', created_at) as time,
    COUNT(*) as new_norms
FROM kb.norms
GROUP BY date_trunc('day', created_at)
ORDER BY time;

-- ============================================================
-- PANEL 8: Top Laws by Jurisdiction (for LEGGE, DLGS, DPR, DL)
-- ============================================================

SELECT
    full_ref as "Law",
    citation_count as "Citations"
FROM kb.norms
WHERE code IN ('LEGGE', 'DLGS', 'DPR', 'DL')
ORDER BY citation_count DESC
LIMIT 15;

-- ============================================================
-- PANEL 9: Constitutional Articles (COST)
-- ============================================================

SELECT
    article as "Article",
    full_ref as "Reference",
    citation_count as "Citations"
FROM kb.norms
WHERE code = 'COST'
ORDER BY citation_count DESC
LIMIT 10;

-- ============================================================
-- PANEL 10: Civil Code Articles (CC) Top 15
-- ============================================================

SELECT
    article as "Article",
    full_ref as "Reference",
    citation_count as "Citations"
FROM kb.norms
WHERE code = 'CC'
ORDER BY citation_count DESC
LIMIT 15;

-- ============================================================
-- ALERT: Low Coverage Warning
-- ============================================================
-- Trigger alert if coverage drops below 50%

SELECT
    CASE
        WHEN (
            SELECT COUNT(DISTINCT massima_id)::float /
                   (SELECT COUNT(*) FROM kb.massime WHERE is_active)
            FROM kb.massima_norms
        ) < 0.50
        THEN 1
        ELSE 0
    END as alert_low_coverage;

-- ============================================================
-- ALERT: Orphan Edges Warning
-- ============================================================
-- Trigger alert if orphan edges found

SELECT
    (SELECT COUNT(*) FROM kb.massima_norms mn
     LEFT JOIN kb.norms n ON mn.norm_id = n.id
     WHERE n.id IS NULL)
    +
    (SELECT COUNT(*) FROM kb.massima_norms mn
     LEFT JOIN kb.massime m ON mn.massima_id = m.id
     WHERE m.id IS NULL)
    as orphan_edges;
