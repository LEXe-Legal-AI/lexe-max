-- Migration: kb.ingestion_profiles
-- Sistema di profili per ingestion massimari
-- Trasforma i fix strutturali in configurazione governabile

-- ============================================================================
-- TABELLA PROFILI
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.ingestion_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identificazione
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    version VARCHAR(20) NOT NULL DEFAULT 'v1',

    -- Matching documenti
    doc_type VARCHAR(50),           -- 'penale', 'civile', NULL = tutti
    anno_min INTEGER,               -- Range anni applicabile
    anno_max INTEGER,
    filename_pattern VARCHAR(255),  -- Regex per match filename

    -- Configurazione completa in JSONB
    config JSONB NOT NULL DEFAULT '{}',

    -- Metadata
    is_default BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Solo un profilo default
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_single_default
ON kb.ingestion_profiles (is_default) WHERE is_default = TRUE;

-- Link documento -> profilo usato
ALTER TABLE kb.documents
ADD COLUMN IF NOT EXISTS profile_id UUID REFERENCES kb.ingestion_profiles(id);

-- ============================================================================
-- STRUTTURA CONFIG JSONB
-- ============================================================================
/*
{
  "extraction": {
    "strategy": "fast|hi_res|auto",
    "languages": ["ita"],
    "skip_pages_default": 15,
    "skip_pages_dynamic": true,
    "content_start_buffer": 10
  },

  "gate_policy": {
    "min_length": 150,
    "max_citation_ratio": 0.03,
    "required_keywords_short": true,
    "bad_starts": [",\\s*del\\s+\\d", ",\\s*dep\\.", ",\\s*Rv\\."],
    "max_match_position": 200
  },

  "massima_patterns": [
    {
      "name": "sez_num_date",
      "pattern": "Sez\\s*\\.?\\s*(\\d+)\\s*[ªa°]?\\s*,?\\s*n\\s*\\.?\\s*(\\d+)\\s+del\\s+(\\d{1,2}\\s*/\\s*\\d{1,2}\\s*/\\s*\\d{4})",
      "priority": 1
    }
  ],

  "section_backfill": {
    "strategy": "title_search|toc_only|hybrid",
    "fuzzy_threshold": 0.6,
    "normalize_aggressive": true,
    "page_end_buffer_single": 5
  },

  "qa_thresholds": {
    "max_short_pct": 5.0,
    "max_out_of_range": 10,
    "min_linking_pct": 90.0,
    "max_page_collisions_same_level": 3
  }
}
*/

-- ============================================================================
-- PROFILI BASELINE
-- ============================================================================

-- Profilo default (baseline sicura)
INSERT INTO kb.ingestion_profiles (name, description, version, is_default, config)
VALUES (
    'massimario_default_v1',
    'Profilo baseline per massimari standard. SKIP_PAGES dinamico, gate policy standard.',
    'v1',
    TRUE,
    '{
      "extraction": {
        "strategy": "fast",
        "languages": ["ita"],
        "skip_pages_default": 15,
        "skip_pages_dynamic": true,
        "content_start_buffer": 10
      },
      "gate_policy": {
        "min_length": 150,
        "max_citation_ratio": 0.03,
        "required_keywords_short": true,
        "max_match_position": 200
      },
      "section_backfill": {
        "strategy": "title_search",
        "fuzzy_threshold": 0.6,
        "normalize_aggressive": false,
        "page_end_buffer_single": 5
      },
      "qa_thresholds": {
        "max_short_pct": 5.0,
        "max_out_of_range": 10,
        "min_linking_pct": 90.0
      }
    }'::jsonb
) ON CONFLICT (name) DO NOTHING;

-- Profilo per penali 2021-2023 (formato pulito)
INSERT INTO kb.ingestion_profiles (name, description, version, doc_type, anno_min, anno_max, config)
VALUES (
    'massimario_penale_2021_2023',
    'Penali recenti con TOC ben strutturato. SKIP_PAGES ~22-28.',
    'v1',
    'penale',
    2021,
    2023,
    '{
      "extraction": {
        "strategy": "fast",
        "languages": ["ita"],
        "skip_pages_default": 20,
        "skip_pages_dynamic": true,
        "content_start_buffer": 10
      },
      "gate_policy": {
        "min_length": 150,
        "max_citation_ratio": 0.03,
        "required_keywords_short": true,
        "max_match_position": 200
      },
      "section_backfill": {
        "strategy": "title_search",
        "fuzzy_threshold": 0.6,
        "normalize_aggressive": false,
        "page_end_buffer_single": 5
      },
      "qa_thresholds": {
        "max_short_pct": 2.0,
        "max_out_of_range": 5,
        "min_linking_pct": 95.0
      }
    }'::jsonb
) ON CONFLICT (name) DO NOTHING;

-- Profilo per civili 2018 (TOC con collisioni)
INSERT INTO kb.ingestion_profiles (name, description, version, doc_type, anno_min, anno_max, config)
VALUES (
    'massimario_civile_toc_collision',
    'Civili con TOC che ha collisioni di pagina (es. 2018). Normalizzazione aggressiva.',
    'v1',
    'civile',
    2015,
    2020,
    '{
      "extraction": {
        "strategy": "fast",
        "languages": ["ita"],
        "skip_pages_default": 25,
        "skip_pages_dynamic": true,
        "content_start_buffer": 12
      },
      "gate_policy": {
        "min_length": 150,
        "max_citation_ratio": 0.03,
        "required_keywords_short": true,
        "max_match_position": 200
      },
      "section_backfill": {
        "strategy": "hybrid",
        "fuzzy_threshold": 0.5,
        "normalize_aggressive": true,
        "page_end_buffer_single": 8,
        "handle_page_collisions": true
      },
      "title_normalization": {
        "remove_chapter_prefix": true,
        "unidecode": true,
        "compress_spaces": true,
        "trigram_fallback_threshold": 0.7
      },
      "qa_thresholds": {
        "max_short_pct": 5.0,
        "max_out_of_range": 15,
        "min_linking_pct": 90.0,
        "max_page_collisions_same_level": 5
      }
    }'::jsonb
) ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- FUNZIONI HELPER
-- ============================================================================

-- Trova profilo migliore per un documento
CREATE OR REPLACE FUNCTION kb.find_profile_for_document(
    p_doc_type VARCHAR,
    p_anno INTEGER,
    p_filename VARCHAR DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_profile_id UUID;
BEGIN
    -- Prima cerca match specifico per tipo/anno
    SELECT id INTO v_profile_id
    FROM kb.ingestion_profiles
    WHERE is_active = TRUE
      AND (doc_type IS NULL OR doc_type = p_doc_type)
      AND (anno_min IS NULL OR p_anno >= anno_min)
      AND (anno_max IS NULL OR p_anno <= anno_max)
      AND (filename_pattern IS NULL OR p_filename ~ filename_pattern)
    ORDER BY
        -- Priorita: match specifico > generico
        CASE WHEN doc_type IS NOT NULL THEN 0 ELSE 1 END,
        CASE WHEN anno_min IS NOT NULL THEN 0 ELSE 1 END,
        CASE WHEN filename_pattern IS NOT NULL THEN 0 ELSE 1 END
    LIMIT 1;

    -- Se nessun match, usa default
    IF v_profile_id IS NULL THEN
        SELECT id INTO v_profile_id
        FROM kb.ingestion_profiles
        WHERE is_default = TRUE AND is_active = TRUE
        LIMIT 1;
    END IF;

    RETURN v_profile_id;
END;
$$ LANGUAGE plpgsql;

-- Estrai config specifica da profilo
CREATE OR REPLACE FUNCTION kb.get_profile_config(
    p_profile_id UUID,
    p_path TEXT  -- es. 'gate_policy.min_length'
) RETURNS JSONB AS $$
DECLARE
    v_config JSONB;
    v_path_parts TEXT[];
    v_result JSONB;
BEGIN
    SELECT config INTO v_config
    FROM kb.ingestion_profiles
    WHERE id = p_profile_id;

    IF v_config IS NULL THEN
        RETURN NULL;
    END IF;

    -- Naviga il path
    v_path_parts := string_to_array(p_path, '.');
    v_result := v_config;

    FOR i IN 1..array_length(v_path_parts, 1) LOOP
        v_result := v_result -> v_path_parts[i];
    END LOOP;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- METRICHE QA AVANZATE
-- ============================================================================

-- View per collisioni pagina stesso livello
CREATE OR REPLACE VIEW kb.qa_page_collisions AS
SELECT
    d.id as document_id,
    d.anno,
    d.tipo,
    s.level,
    s.tipo as section_type,
    s.pagina_inizio,
    COUNT(*) as collision_count,
    array_agg(s.titolo ORDER BY s.id) as colliding_titles
FROM kb.sections s
JOIN kb.documents d ON d.id = s.document_id
WHERE s.pagina_inizio IS NOT NULL
GROUP BY d.id, d.anno, d.tipo, s.level, s.tipo, s.pagina_inizio
HAVING COUNT(*) > 1;

-- View per monotonicita gerarchia (figli prima dei padri)
CREATE OR REPLACE VIEW kb.qa_hierarchy_anomalies AS
SELECT
    d.id as document_id,
    d.anno,
    d.tipo,
    child.id as child_id,
    child.tipo as child_type,
    child.titolo as child_title,
    child.pagina_inizio as child_page,
    parent.id as parent_id,
    parent.tipo as parent_type,
    parent.titolo as parent_title,
    parent.pagina_inizio as parent_page
FROM kb.sections child
JOIN kb.sections parent ON parent.id = child.parent_id
JOIN kb.documents d ON d.id = child.document_id
WHERE child.pagina_inizio IS NOT NULL
  AND parent.pagina_inizio IS NOT NULL
  AND child.pagina_inizio < parent.pagina_inizio;

-- View report QA completo per documento
CREATE OR REPLACE VIEW kb.qa_document_report AS
SELECT
    d.id as document_id,
    d.anno,
    d.tipo,
    d.profile_id,
    p.name as profile_name,

    -- Massime stats
    COUNT(DISTINCT m.id) as total_massime,
    COUNT(DISTINCT m.id) FILTER (WHERE LENGTH(m.testo) < 200) as short_massime,
    ROUND(100.0 * COUNT(DISTINCT m.id) FILTER (WHERE LENGTH(m.testo) < 200) / NULLIF(COUNT(DISTINCT m.id), 0), 1) as pct_short,

    -- Linking stats
    COUNT(DISTINCT m.id) FILTER (WHERE m.section_id IS NOT NULL) as linked_massime,
    ROUND(100.0 * COUNT(DISTINCT m.id) FILTER (WHERE m.section_id IS NOT NULL) / NULLIF(COUNT(DISTINCT m.id), 0), 1) as pct_linked,

    -- Section stats
    COUNT(DISTINCT s.id) as total_sections,
    COUNT(DISTINCT s.id) FILTER (WHERE s.pagina_inizio IS NOT NULL) as sections_with_page,
    ROUND(100.0 * COUNT(DISTINCT s.id) FILTER (WHERE s.pagina_inizio IS NOT NULL) / NULLIF(COUNT(DISTINCT s.id), 0), 1) as pct_sections_with_page,

    -- Out of range (subquery necessaria)
    (SELECT COUNT(*) FROM kb.massime m2
     JOIN kb.sections s2 ON s2.id = m2.section_id
     WHERE m2.document_id = d.id
       AND m2.pagina_inizio IS NOT NULL
       AND s2.pagina_inizio IS NOT NULL
       AND s2.pagina_fine IS NOT NULL
       AND (m2.pagina_inizio < s2.pagina_inizio OR m2.pagina_inizio > s2.pagina_fine)
    ) as out_of_range_count

FROM kb.documents d
LEFT JOIN kb.massime m ON m.document_id = d.id
LEFT JOIN kb.sections s ON s.document_id = d.id
LEFT JOIN kb.ingestion_profiles p ON p.id = d.profile_id
GROUP BY d.id, d.anno, d.tipo, d.profile_id, p.name;

-- ============================================================================
-- TRIGGER per updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION kb.update_profile_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_profile_updated ON kb.ingestion_profiles;
CREATE TRIGGER trg_profile_updated
    BEFORE UPDATE ON kb.ingestion_profiles
    FOR EACH ROW
    EXECUTE FUNCTION kb.update_profile_timestamp();

-- ============================================================================
-- COMMENTI
-- ============================================================================

COMMENT ON TABLE kb.ingestion_profiles IS
'Profili di configurazione per ingestion massimari. Ogni profilo contiene parametri per estrazione, gate policy, backfill sezioni e soglie QA.';

COMMENT ON COLUMN kb.ingestion_profiles.config IS
'Configurazione JSONB completa. Struttura: extraction, gate_policy, massima_patterns, section_backfill, qa_thresholds.';

COMMENT ON VIEW kb.qa_page_collisions IS
'Sezioni con stessa pagina_inizio e stesso livello. Indica possibili entry TOC non risolte.';

COMMENT ON VIEW kb.qa_hierarchy_anomalies IS
'Sezioni figlie con pagina prima del padre. Anomalia strutturale da investigare.';

COMMENT ON VIEW kb.qa_document_report IS
'Report QA completo per documento: short massime, linking, sections coverage, out of range.';
