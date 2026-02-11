-- Migration: Create kb.normativa_altalex table
-- Parallel table for Altalex source, with cross-reference to Brocardi
-- Date: 2026-02-04

-- =============================================================================
-- ALTALEX NORMATIVA TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_altalex (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ═══ IDENTIFICAZIONE ═══
    codice VARCHAR(50) NOT NULL,              -- 'CC', 'CP', 'CPC'
    articolo VARCHAR(20) NOT NULL,            -- '2043', '1', '2043-bis'
    is_preleggi BOOLEAN DEFAULT FALSE,        -- Art. 1-31 Disposizioni preliminari

    -- ═══ CONTENUTO ═══
    rubrica TEXT,                             -- 'Risarcimento per fatto illecito'
    testo TEXT NOT NULL,                      -- Testo articolo completo
    testo_normalizzato TEXT,                  -- Per search/comparison
    content_hash VARCHAR(64),                 -- SHA256 del testo normalizzato

    -- ═══ GERARCHIA ═══
    libro VARCHAR(200),                       -- 'Libro IV - Delle obbligazioni'
    titolo VARCHAR(200),                      -- 'Titolo IX - Dei fatti illeciti'
    capo VARCHAR(200),
    sezione VARCHAR(200),

    -- ═══ SOURCE TRACKING ═══
    source_file VARCHAR(500),                 -- Path file MD originale
    source_edition VARCHAR(100),              -- 'Altalex 2025'
    line_start INTEGER,
    line_end INTEGER,

    -- ═══ CROSS-REFERENCE BROCARDI ═══
    brocardi_match_id UUID REFERENCES kb.normativa(id),
    brocardi_similarity FLOAT,                -- 0.0-1.0 Jaccard similarity
    brocardi_match_status VARCHAR(20),        -- 'exact', 'format_diff', 'content_diff', 'no_match'
    brocardi_extras TEXT[],                   -- ['massime', 'relazioni', 'note', 'spiegazione']

    -- ═══ TIMESTAMPS ═══
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- ═══ CONSTRAINTS ═══
    UNIQUE(codice, articolo, is_preleggi)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_normativa_altalex_codice_art
    ON kb.normativa_altalex(codice, articolo);

CREATE INDEX IF NOT EXISTS idx_normativa_altalex_hash
    ON kb.normativa_altalex(content_hash);

CREATE INDEX IF NOT EXISTS idx_normativa_altalex_brocardi
    ON kb.normativa_altalex(brocardi_match_id)
    WHERE brocardi_match_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_normativa_altalex_preleggi
    ON kb.normativa_altalex(codice, is_preleggi)
    WHERE is_preleggi = TRUE;

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_normativa_altalex_fts
    ON kb.normativa_altalex
    USING GIN (to_tsvector('italian', COALESCE(rubrica, '') || ' ' || testo));

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE kb.normativa_altalex IS
    'Articoli di codice da fonte Altalex (PDF export). Parallela a kb.normativa (Brocardi) per confronto.';

COMMENT ON COLUMN kb.normativa_altalex.is_preleggi IS
    'True per Art. 1-31 Disposizioni sulla legge in generale (non presenti in Brocardi)';

COMMENT ON COLUMN kb.normativa_altalex.brocardi_extras IS
    'Array di contenuti extra presenti in Brocardi: massime, relazioni, note, spiegazione';
