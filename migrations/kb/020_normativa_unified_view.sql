-- Migration: 020_normativa_unified_view.sql
-- Purpose: Create unified view combining Altalex (primary) + Brocardi (enrichment)
-- Date: 2026-02-04

-- ============================================================
-- ADD ENRICHMENT COLUMNS TO ALTALEX TABLE
-- ============================================================

-- Add columns for Brocardi enrichment data
ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS brocardi_is_abrogato BOOLEAN DEFAULT FALSE;

ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS brocardi_abrogation_note TEXT;

ALTER TABLE kb.normativa_altalex
ADD COLUMN IF NOT EXISTS brocardi_cross_refs TEXT[];

COMMENT ON COLUMN kb.normativa_altalex.brocardi_is_abrogato IS 'TRUE if Brocardi marks this article as abrogated';
COMMENT ON COLUMN kb.normativa_altalex.brocardi_abrogation_note IS 'Abrogation note from Brocardi (e.g., "Abrogato da L. 151/1975")';
COMMENT ON COLUMN kb.normativa_altalex.brocardi_cross_refs IS 'Cross-references extracted from Brocardi text [art. X, art. Y]';


-- ============================================================
-- UNIFIED VIEW: ALTALEX PRIMARY + BROCARDI ENRICHMENT
-- ============================================================

CREATE OR REPLACE VIEW kb.normativa_unified AS
WITH altalex_enriched AS (
    -- Altalex articles enriched with Brocardi data
    SELECT
        a.id,
        a.codice,
        a.articolo,
        a.is_preleggi,
        a.is_attuazione,
        COALESCE(a.rubrica, b.rubrica) AS rubrica,  -- Altalex primary, Brocardi fallback
        a.testo,                                      -- ALWAYS use Altalex (full text)
        a.testo_normalizzato,
        a.content_hash,
        a.libro,
        a.titolo,
        a.capo,
        a.sezione,

        -- Brocardi enrichment
        b.id AS brocardi_id,
        COALESCE(a.brocardi_is_abrogato,
            CASE WHEN b.testo ILIKE '%abrogat%' THEN TRUE ELSE FALSE END
        ) AS is_abrogato,
        a.brocardi_abrogation_note,
        a.brocardi_cross_refs,
        a.brocardi_match_status,
        a.brocardi_similarity,

        -- Source tracking
        'altalex' AS primary_source,
        a.source_file,
        a.source_edition,
        a.created_at,
        a.updated_at
    FROM kb.normativa_altalex a
    LEFT JOIN kb.normativa b
        ON b.codice = a.codice
        AND b.articolo = a.articolo
    WHERE NOT a.is_preleggi AND NOT a.is_attuazione
),
brocardi_only AS (
    -- Brocardi articles NOT in Altalex (e.g., CP, CdS)
    SELECT
        b.id,
        b.codice,
        b.articolo,
        FALSE AS is_preleggi,
        FALSE AS is_attuazione,
        b.rubrica,
        b.testo,
        b.testo_normalizzato,
        b.mirror_hash AS content_hash,
        b.libro,
        b.titolo,
        NULL::varchar AS capo,
        NULL::varchar AS sezione,

        -- Brocardi is the only source
        b.id AS brocardi_id,
        CASE WHEN b.testo ILIKE '%abrogat%' THEN TRUE ELSE FALSE END AS is_abrogato,
        NULL::text AS brocardi_abrogation_note,
        NULL::text[] AS brocardi_cross_refs,
        'brocardi_only' AS brocardi_match_status,
        1.0 AS brocardi_similarity,

        -- Source tracking
        'brocardi' AS primary_source,
        NULL AS source_file,
        NULL AS source_edition,
        b.created_at,
        b.updated_at
    FROM kb.normativa b
    WHERE NOT EXISTS (
        SELECT 1 FROM kb.normativa_altalex a
        WHERE a.codice = b.codice AND a.articolo = b.articolo
        AND NOT a.is_preleggi AND NOT a.is_attuazione
    )
),
preleggi_attuazione AS (
    -- Preleggi and Attuazione (Altalex only, Brocardi doesn't have them)
    SELECT
        a.id,
        a.codice,
        a.articolo,
        a.is_preleggi,
        a.is_attuazione,
        a.rubrica,
        a.testo,
        a.testo_normalizzato,
        a.content_hash,
        a.libro,
        a.titolo,
        a.capo,
        a.sezione,

        -- No Brocardi match
        NULL::uuid AS brocardi_id,
        FALSE AS is_abrogato,
        NULL::text AS brocardi_abrogation_note,
        NULL::text[] AS brocardi_cross_refs,
        'altalex_only' AS brocardi_match_status,
        NULL::float AS brocardi_similarity,

        -- Source tracking
        'altalex' AS primary_source,
        a.source_file,
        a.source_edition,
        a.created_at,
        a.updated_at
    FROM kb.normativa_altalex a
    WHERE a.is_preleggi OR a.is_attuazione
)
-- Combine all sources
SELECT * FROM altalex_enriched
UNION ALL
SELECT * FROM brocardi_only
UNION ALL
SELECT * FROM preleggi_attuazione;

COMMENT ON VIEW kb.normativa_unified IS 'Unified normativa view: Altalex (primary, full text) + Brocardi (enrichment, cross-refs)';


-- ============================================================
-- STATS VIEW
-- ============================================================

CREATE OR REPLACE VIEW kb.normativa_stats AS
SELECT
    codice,
    COUNT(*) AS total_articles,
    COUNT(*) FILTER (WHERE primary_source = 'altalex') AS from_altalex,
    COUNT(*) FILTER (WHERE primary_source = 'brocardi') AS from_brocardi,
    COUNT(*) FILTER (WHERE is_preleggi) AS preleggi,
    COUNT(*) FILTER (WHERE is_attuazione) AS attuazione,
    COUNT(*) FILTER (WHERE is_abrogato) AS abrogated,
    ROUND(AVG(LENGTH(testo))) AS avg_text_length,
    MAX(LENGTH(testo)) AS max_text_length
FROM kb.normativa_unified
GROUP BY codice
ORDER BY codice;

COMMENT ON VIEW kb.normativa_stats IS 'Statistics per codice from unified normativa view';


-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_normativa_altalex_brocardi_match
ON kb.normativa_altalex(codice, articolo)
WHERE NOT is_preleggi AND NOT is_attuazione;
