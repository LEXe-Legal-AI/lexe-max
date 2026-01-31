-- Migration: 005_norm_graph.sql
-- Norm Graph: norme estratte dalle massime per lookup e reranking
-- Sprint Fase 3

-- ============================================================
-- NORMS TABLE
-- ============================================================
-- Canonical norm references extracted from massime
-- id format: CC:2043, CPC:360:bis, LEGGE:241:1990, DLGS:50:2016

CREATE TABLE IF NOT EXISTS kb.norms (
    id TEXT PRIMARY KEY,                    -- canonical id
    code TEXT NOT NULL,                     -- CC, CPC, CP, CPP, COST, LEGGE, DLGS, DPR
    article TEXT,                           -- for codes: 2043, 360, 111
    suffix TEXT,                            -- bis, ter, quater, quinquies, sexies, septies, octies, novies, decies
    number TEXT,                            -- for laws: 241, 50, 445
    year INT,                               -- for laws: 1990, 2016, 2000
    full_ref TEXT NOT NULL,                 -- human readable form
    citation_count INT DEFAULT 0,           -- number of massime citing this norm
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Validate code values
ALTER TABLE kb.norms DROP CONSTRAINT IF EXISTS norms_code_check;
ALTER TABLE kb.norms ADD CONSTRAINT norms_code_check
    CHECK (code IN ('CC', 'CPC', 'CP', 'CPP', 'COST', 'LEGGE', 'DLGS', 'DPR', 'DL', 'TULPS', 'TUB', 'TUF', 'CAD'));

COMMENT ON TABLE kb.norms IS 'Canonical norm references extracted from massime';
COMMENT ON COLUMN kb.norms.id IS 'Canonical ID: CC:2043, CPC:360:bis, LEGGE:241:1990';
COMMENT ON COLUMN kb.norms.code IS 'Norm type: CC, CPC, CP, COST, LEGGE, DLGS, DPR, etc.';
COMMENT ON COLUMN kb.norms.article IS 'Article number for codes (CC, CPC, CP, COST)';
COMMENT ON COLUMN kb.norms.suffix IS 'Article suffix: bis, ter, quater, etc.';
COMMENT ON COLUMN kb.norms.number IS 'Law number for LEGGE, DLGS, DPR';
COMMENT ON COLUMN kb.norms.year IS 'Law year for LEGGE, DLGS, DPR';

-- ============================================================
-- MASSIMA-NORM EDGES
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.massima_norms (
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    norm_id TEXT NOT NULL REFERENCES kb.norms(id) ON DELETE CASCADE,
    context_span TEXT,                      -- text snippet around the citation
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (massima_id, norm_id)
);

COMMENT ON TABLE kb.massima_norms IS 'Edges connecting massime to cited norms';
COMMENT ON COLUMN kb.massima_norms.context_span IS 'Text context around the norm citation';

-- ============================================================
-- INDEXES
-- ============================================================

-- Lookup by code + article (for CC, CPC, CP, COST)
CREATE INDEX IF NOT EXISTS idx_norms_code_article
    ON kb.norms(code, article)
    WHERE article IS NOT NULL;

-- Lookup by code + number + year (for LEGGE, DLGS, DPR)
CREATE INDEX IF NOT EXISTS idx_norms_code_number_year
    ON kb.norms(code, number, year)
    WHERE number IS NOT NULL;

-- Top cited norms
CREATE INDEX IF NOT EXISTS idx_norms_citation_count
    ON kb.norms(citation_count DESC);

-- Edges: find massime by norm
CREATE INDEX IF NOT EXISTS idx_massima_norms_norm
    ON kb.massima_norms(norm_id);

-- Edges: find norms by massima
CREATE INDEX IF NOT EXISTS idx_massima_norms_massima
    ON kb.massima_norms(massima_id);

-- ============================================================
-- STATS VIEW
-- ============================================================

CREATE OR REPLACE VIEW kb.norm_stats AS
SELECT
    code,
    COUNT(*) AS norm_count,
    SUM(citation_count) AS total_citations,
    AVG(citation_count)::NUMERIC(10,2) AS avg_citations,
    MAX(citation_count) AS max_citations
FROM kb.norms
GROUP BY code
ORDER BY total_citations DESC;

COMMENT ON VIEW kb.norm_stats IS 'Statistics on norms by code type';

-- ============================================================
-- HELPER FUNCTION: Recompute citation counts
-- ============================================================

CREATE OR REPLACE FUNCTION kb.recompute_norm_citation_counts()
RETURNS INT AS $$
DECLARE
    updated_count INT;
BEGIN
    WITH counts AS (
        SELECT norm_id, COUNT(*) AS cnt
        FROM kb.massima_norms
        GROUP BY norm_id
    )
    UPDATE kb.norms n
    SET citation_count = COALESCE(c.cnt, 0),
        updated_at = NOW()
    FROM counts c
    WHERE n.id = c.norm_id
      AND n.citation_count != c.cnt;

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    -- Reset counts for norms with no edges
    UPDATE kb.norms
    SET citation_count = 0, updated_at = NOW()
    WHERE citation_count > 0
      AND id NOT IN (SELECT DISTINCT norm_id FROM kb.massima_norms);

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION kb.recompute_norm_citation_counts IS 'Recompute citation_count for all norms from massima_norms edges';
