-- ============================================================================
-- KB V3 UNIFIED SCHEMA - LEXE Legal Knowledge Base
-- ============================================================================
-- Version: 3.0.0
-- Date: 2026-02-06
-- Purpose: Unified schema for normativa, massime, sentenze with URN:NIR alignment
--
-- Key Features:
--   1. work → normativa hierarchy (atto → articolo)
--   2. URN:NIR alignment with Normattiva standard
--   3. Multi-source tracking (Altalex, Brocardi, Studio Cataldi, Normattiva)
--   4. Apache AGE graph integration
--   5. Multi-dim vector embeddings with HNSW
--   6. Full-text search (Italian + BM25 via tsvector)
--   7. Annotation system for notes, comments, jurisprudence
-- ============================================================================

BEGIN;

-- ============================================================================
-- SCHEMA & EXTENSIONS
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS kb;

-- Core extensions
CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;         -- Trigram similarity
CREATE EXTENSION IF NOT EXISTS unaccent;        -- Accent-insensitive search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- UUID generation

-- Apache AGE for property graph (optional, may fail if not installed)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS age;
    LOAD 'age';
    SET search_path = ag_catalog, "$user", public, kb;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Apache AGE not available, skipping graph features';
END $$;

-- ============================================================================
-- LOOKUP TABLES - Source Types & Systems
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.source_type (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    priority INT DEFAULT 50,        -- For retrieval ranking
    description TEXT
);

COMMENT ON TABLE kb.source_type IS 'Tipologie di fonti normative: codici, leggi, regolamenti, etc.';

INSERT INTO kb.source_type (id, label, priority, description) VALUES
    ('COSTITUZIONE', 'Costituzione e leggi costituzionali', 100, 'Carta costituzionale e leggi di rango costituzionale'),
    ('UE', 'Normative comunitarie UE', 90, 'Regolamenti, direttive, decisioni UE'),
    ('CODICE', 'Codici', 80, 'Codici civile, penale, procedura, etc.'),
    ('TESTO_UNICO', 'Testi unici', 70, 'TUB, TUF, TUIR, etc.'),
    ('LEGGE_DECRETO', 'Leggi e decreti', 60, 'Leggi ordinarie, DL, DLgs, DPR'),
    ('REGOLAMENTO', 'Regolamenti', 40, 'Regolamenti ministeriali, autorità'),
    ('SOFT_LAW', 'Soft law', 30, 'Linee guida, circolari, prassi'),
    ('ALTRO', 'Altre fonti', 10, 'Fonti non classificate')
ON CONFLICT (id) DO UPDATE SET
    label = EXCLUDED.label,
    priority = EXCLUDED.priority,
    description = EXCLUDED.description;

-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.source_system (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    is_canonical BOOLEAN DEFAULT FALSE,
    is_offline BOOLEAN DEFAULT FALSE,
    rate_limit_per_sec NUMERIC(10,3),
    base_url TEXT,
    notes TEXT
);

COMMENT ON TABLE kb.source_system IS 'Sistemi sorgente per acquisizione: Altalex, Brocardi, Normattiva, etc.';
COMMENT ON COLUMN kb.source_system.is_canonical IS 'TRUE se fonte autorevole/ufficiale';
COMMENT ON COLUMN kb.source_system.is_offline IS 'TRUE se mirror locale disponibile';

INSERT INTO kb.source_system (id, label, is_canonical, is_offline, rate_limit_per_sec, base_url, notes) VALUES
    ('ALTALEX_PDF', 'PDF Altalex', FALSE, TRUE, NULL, NULL, 'Fonte primaria PDF per estrazione batch'),
    ('STUDIO_CATALDI_OFFLINE', 'Studio Cataldi offline', FALSE, TRUE, NULL, 'C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa', 'Mirror locale completo'),
    ('BROCARDI_OFFLINE', 'Brocardi offline', FALSE, TRUE, NULL, 'C:/Mie pagine Web/broc-civ/www.brocardi.it', 'Mirror locale in download'),
    ('BROCARDI_ONLINE', 'Brocardi online', FALSE, FALSE, 0.333, 'https://www.brocardi.it', 'Fallback online, 1 req/3 sec'),
    ('NORMATTIVA_ONLINE', 'Normattiva online', TRUE, FALSE, 1.0, 'https://www.normattiva.it', 'Fonte canonica ufficiale'),
    ('EURLEX_ONLINE', 'EUR-Lex online', TRUE, FALSE, 0.5, 'https://eur-lex.europa.eu', 'Fonte canonica UE'),
    ('GAZZETTA_UFFICIALE', 'Gazzetta Ufficiale', TRUE, FALSE, 0.2, 'https://www.gazzettaufficiale.it', 'Fonte ufficiale pubblicazione')
ON CONFLICT (id) DO UPDATE SET
    label = EXCLUDED.label,
    is_canonical = EXCLUDED.is_canonical,
    is_offline = EXCLUDED.is_offline,
    rate_limit_per_sec = EXCLUDED.rate_limit_per_sec,
    base_url = EXCLUDED.base_url,
    notes = EXCLUDED.notes;

-- ============================================================================
-- NIR MAPPING - URN Alignment Table
-- ============================================================================
-- Enforces consistency between work.code and URN:NIR identifiers
-- Reference: https://www.normattiva.it/staticPage/utilita
-- AgID spec: https://www.agid.gov.it/sites/agid/files/2024-06/Linee_guida_marcatura_documenti_normativi.pdf

CREATE TABLE IF NOT EXISTS kb.nir_mapping (
    code TEXT PRIMARY KEY,              -- Internal code: CC, CP, CPC, etc.
    nir_base TEXT NOT NULL UNIQUE,      -- URN base: urn:nir:stato:regio.decreto:1942-03-16;262
    nome_completo TEXT NOT NULL,        -- Full name in Italian
    nome_breve TEXT,                    -- Short name for display
    abbreviazioni TEXT[],               -- Common abbreviations: ['c.c.', 'cod. civ.']
    source_type_id TEXT REFERENCES kb.source_type(id),
    atto_tipo TEXT NOT NULL,            -- 'regio.decreto', 'decreto.legislativo', 'legge', etc.
    atto_data DATE NOT NULL,            -- Publication date
    atto_numero TEXT NOT NULL,          -- Act number
    normattiva_url TEXT,                -- Direct Normattiva URL
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE kb.nir_mapping IS 'Mapping tra code interno e URN:NIR Normattiva - enforces consistency';
COMMENT ON COLUMN kb.nir_mapping.code IS 'Codice interno coerente: CC, CP, CPC, CCII, TUB, etc.';
COMMENT ON COLUMN kb.nir_mapping.nir_base IS 'URN:NIR base atto secondo standard AgID';

-- Gold mappings per i principali codici e testi unici
INSERT INTO kb.nir_mapping (code, nir_base, nome_completo, nome_breve, abbreviazioni, source_type_id, atto_tipo, atto_data, atto_numero, normattiva_url) VALUES
    -- COSTITUZIONE
    ('COST', 'urn:nir:stato:costituzione:1947-12-27', 'Costituzione della Repubblica Italiana', 'Costituzione', ARRAY['cost.', 'costituzione'], 'COSTITUZIONE', 'costituzione', '1947-12-27', '0', 'https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:costituzione:1947-12-27'),

    -- CODICI FONDAMENTALI
    ('CC', 'urn:nir:stato:regio.decreto:1942-03-16;262', 'Codice Civile', 'Cod. Civ.', ARRAY['c.c.', 'cod. civ.', 'codice civile'], 'CODICE', 'regio.decreto', '1942-03-16', '262', 'https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262'),
    ('CP', 'urn:nir:stato:regio.decreto:1930-10-19;1398', 'Codice Penale', 'Cod. Pen.', ARRAY['c.p.', 'cod. pen.', 'codice penale'], 'CODICE', 'regio.decreto', '1930-10-19', '1398', 'https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1930-10-19;1398'),
    ('CPC', 'urn:nir:stato:regio.decreto:1940-10-28;1443', 'Codice di Procedura Civile', 'Cod. Proc. Civ.', ARRAY['c.p.c.', 'cod. proc. civ.'], 'CODICE', 'regio.decreto', '1940-10-28', '1443', 'https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1940-10-28;1443'),
    ('CPP', 'urn:nir:stato:decreto.presidente.repubblica:1988-09-22;447', 'Codice di Procedura Penale', 'Cod. Proc. Pen.', ARRAY['c.p.p.', 'cod. proc. pen.'], 'CODICE', 'decreto.presidente.repubblica', '1988-09-22', '447', 'https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.presidente.repubblica:1988-09-22;447'),

    -- CODICE CRISI
    ('CCII', 'urn:nir:stato:decreto.legislativo:2019-01-12;14', 'Codice della Crisi d''Impresa e dell''Insolvenza', 'Cod. Crisi', ARRAY['c.c.i.i.', 'ccii', 'cod. crisi'], 'CODICE', 'decreto.legislativo', '2019-01-12', '14', 'https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2019-01-12;14'),

    -- CODICI SETTORIALI
    ('CDS', 'urn:nir:stato:decreto.legislativo:1992-04-30;285', 'Codice della Strada', 'Cod. Strada', ARRAY['c.d.s.', 'cod. strada'], 'CODICE', 'decreto.legislativo', '1992-04-30', '285', NULL),
    ('CCONS', 'urn:nir:stato:decreto.legislativo:2005-09-06;206', 'Codice del Consumo', 'Cod. Consumo', ARRAY['cod. cons.', 'cod. consumo'], 'CODICE', 'decreto.legislativo', '2005-09-06', '206', NULL),
    ('CAMB', 'urn:nir:stato:decreto.legislativo:2006-04-03;152', 'Codice dell''Ambiente', 'Cod. Ambiente', ARRAY['cod. amb.', 'cod. ambiente'], 'CODICE', 'decreto.legislativo', '2006-04-03', '152', NULL),
    ('CAD', 'urn:nir:stato:decreto.legislativo:2005-03-07;82', 'Codice dell''Amministrazione Digitale', 'CAD', ARRAY['cad', 'cod. amm. dig.'], 'CODICE', 'decreto.legislativo', '2005-03-07', '82', NULL),
    ('CNAV', 'urn:nir:stato:regio.decreto:1942-03-30;327', 'Codice della Navigazione', 'Cod. Nav.', ARRAY['c.nav.', 'cod. nav.'], 'CODICE', 'regio.decreto', '1942-03-30', '327', NULL),
    ('CASS', 'urn:nir:stato:decreto.legislativo:2005-09-07;209', 'Codice delle Assicurazioni Private', 'Cod. Ass.', ARRAY['cod. ass.', 'c.ass.'], 'CODICE', 'decreto.legislativo', '2005-09-07', '209', NULL),
    ('CAPPALTI', 'urn:nir:stato:decreto.legislativo:2023-03-31;36', 'Codice degli Appalti', 'Cod. Appalti', ARRAY['cod. app.', 'd.lgs. 36/2023'], 'CODICE', 'decreto.legislativo', '2023-03-31', '36', NULL),

    -- TESTI UNICI
    ('TUB', 'urn:nir:stato:decreto.legislativo:1993-09-01;385', 'Testo Unico Bancario', 'TUB', ARRAY['tub', 't.u.b.', 'd.lgs. 385/1993'], 'TESTO_UNICO', 'decreto.legislativo', '1993-09-01', '385', NULL),
    ('TUF', 'urn:nir:stato:decreto.legislativo:1998-02-24;58', 'Testo Unico della Finanza', 'TUF', ARRAY['tuf', 't.u.f.', 'd.lgs. 58/1998'], 'TESTO_UNICO', 'decreto.legislativo', '1998-02-24', '58', NULL),
    ('TUIR', 'urn:nir:stato:decreto.presidente.repubblica:1986-12-22;917', 'Testo Unico Imposte sui Redditi', 'TUIR', ARRAY['tuir', 't.u.i.r.', 'dpr 917/1986'], 'TESTO_UNICO', 'decreto.presidente.repubblica', '1986-12-22', '917', NULL),
    ('TUEL', 'urn:nir:stato:decreto.legislativo:2000-08-18;267', 'Testo Unico Enti Locali', 'TUEL', ARRAY['tuel', 't.u.e.l.', 'd.lgs. 267/2000'], 'TESTO_UNICO', 'decreto.legislativo', '2000-08-18', '267', NULL),
    ('TUE', 'urn:nir:stato:decreto.presidente.repubblica:1998-04-28;275', 'Testo Unico Edilizia', 'TUE', ARRAY['tue', 't.u.e.', 'dpr 275/1998'], 'TESTO_UNICO', 'decreto.presidente.repubblica', '1998-04-28', '275', NULL),
    ('TUSL', 'urn:nir:stato:decreto.legislativo:2008-04-09;81', 'Testo Unico Sicurezza Lavoro', 'TUSL', ARRAY['tusl', 't.u.s.l.', 'd.lgs. 81/2008'], 'TESTO_UNICO', 'decreto.legislativo', '2008-04-09', '81', NULL),
    ('TUI', 'urn:nir:stato:decreto.legislativo:1998-07-25;286', 'Testo Unico Immigrazione', 'TUI', ARRAY['tui', 't.u.i.', 'd.lgs. 286/1998'], 'TESTO_UNICO', 'decreto.legislativo', '1998-07-25', '286', NULL),

    -- PRIVACY / GDPR
    ('GDPR', 'urn:nir:unione.europea:regolamento:2016-04-27;2016-679', 'Regolamento Generale Protezione Dati', 'GDPR', ARRAY['gdpr', 'reg. ue 2016/679'], 'UE', 'regolamento', '2016-04-27', '2016/679', NULL),
    ('CPRIVACY', 'urn:nir:stato:decreto.legislativo:2003-06-30;196', 'Codice Privacy', 'Cod. Privacy', ARRAY['cod. privacy', 'd.lgs. 196/2003'], 'CODICE', 'decreto.legislativo', '2003-06-30', '196', NULL)

ON CONFLICT (code) DO UPDATE SET
    nir_base = EXCLUDED.nir_base,
    nome_completo = EXCLUDED.nome_completo,
    nome_breve = EXCLUDED.nome_breve,
    abbreviazioni = EXCLUDED.abbreviazioni,
    source_type_id = EXCLUDED.source_type_id,
    atto_tipo = EXCLUDED.atto_tipo,
    atto_data = EXCLUDED.atto_data,
    atto_numero = EXCLUDED.atto_numero,
    normattiva_url = EXCLUDED.normattiva_url;

-- Validation function
CREATE OR REPLACE FUNCTION kb.validate_nir_code(p_code TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (SELECT 1 FROM kb.nir_mapping WHERE code = UPPER(p_code));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Get NIR base from code
CREATE OR REPLACE FUNCTION kb.get_nir_base(p_code TEXT)
RETURNS TEXT AS $$
    SELECT nir_base FROM kb.nir_mapping WHERE code = UPPER(p_code);
$$ LANGUAGE sql STABLE;

-- Build article URN
CREATE OR REPLACE FUNCTION kb.build_article_urn(
    p_code TEXT,
    p_articolo TEXT,
    p_comma TEXT DEFAULT NULL
)
RETURNS TEXT AS $$
DECLARE
    v_nir_base TEXT;
    v_urn TEXT;
BEGIN
    SELECT nir_base INTO v_nir_base FROM kb.nir_mapping WHERE code = UPPER(p_code);
    IF v_nir_base IS NULL THEN
        RETURN NULL;
    END IF;

    v_urn := v_nir_base || '~art' || p_articolo;
    IF p_comma IS NOT NULL THEN
        v_urn := v_urn || '-com' || p_comma;
    END IF;

    RETURN v_urn;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- TOPIC HIERARCHY (for classification)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.topic (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    level INT NOT NULL CHECK (level IN (1, 2, 3)),
    parent_id TEXT REFERENCES kb.topic(id),
    sort_order INT DEFAULT 0,
    CONSTRAINT topic_hierarchy CHECK (
        (level = 1 AND parent_id IS NULL) OR
        (level > 1 AND parent_id IS NOT NULL)
    )
);

COMMENT ON TABLE kb.topic IS 'Gerarchia temi giuridici (3 livelli): macro > meso > micro';

-- Example topics (can be extended)
INSERT INTO kb.topic (id, label, level, parent_id, sort_order) VALUES
    ('DIR_CIVILE', 'Diritto Civile', 1, NULL, 10),
    ('DIR_PENALE', 'Diritto Penale', 1, NULL, 20),
    ('DIR_AMM', 'Diritto Amministrativo', 1, NULL, 30),
    ('DIR_COMM', 'Diritto Commerciale', 1, NULL, 40),
    ('DIR_LAVORO', 'Diritto del Lavoro', 1, NULL, 50),
    ('DIR_TRIBUTARIO', 'Diritto Tributario', 1, NULL, 60),
    -- Level 2 examples
    ('OBBLIGAZIONI', 'Obbligazioni', 2, 'DIR_CIVILE', 10),
    ('CONTRATTI', 'Contratti', 2, 'DIR_CIVILE', 20),
    ('RESP_CIVILE', 'Responsabilità civile', 2, 'DIR_CIVILE', 30),
    ('FAMIGLIA', 'Famiglia e successioni', 2, 'DIR_CIVILE', 40),
    ('REATI_PERSONA', 'Reati contro la persona', 2, 'DIR_PENALE', 10),
    ('REATI_PATRIMONIO', 'Reati contro il patrimonio', 2, 'DIR_PENALE', 20),
    ('SOCIETA', 'Società', 2, 'DIR_COMM', 10),
    ('FALLIMENTO', 'Crisi d''impresa', 2, 'DIR_COMM', 20)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- WORK (Documents/Acts)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identity (linked to nir_mapping)
    code TEXT NOT NULL REFERENCES kb.nir_mapping(code),
    title TEXT NOT NULL,
    title_short TEXT,

    -- Type and status
    source_type_id TEXT REFERENCES kb.source_type(id),
    is_abrogated BOOLEAN DEFAULT FALSE,
    is_consolidated BOOLEAN DEFAULT TRUE,  -- Testo vigente vs originale

    -- Dates
    edition_date DATE,                      -- Edition/consolidation date
    publication_date DATE,                  -- GU publication date

    -- URN (denormalized from nir_mapping for convenience)
    nir_base TEXT,

    -- Metadata
    meta JSONB DEFAULT '{}'::jsonb,
    article_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(code, edition_date)
);

COMMENT ON TABLE kb.work IS 'Atti normativi: codici, leggi, decreti, regolamenti';
COMMENT ON COLUMN kb.work.code IS 'Codice identificativo allineato a kb.nir_mapping';
COMMENT ON COLUMN kb.work.nir_base IS 'URN:NIR base (denormalized da nir_mapping)';

-- Trigger to auto-fill nir_base from nir_mapping
CREATE OR REPLACE FUNCTION kb.set_work_nir_base()
RETURNS TRIGGER AS $$
BEGIN
    SELECT nir_base INTO NEW.nir_base
    FROM kb.nir_mapping
    WHERE code = NEW.code;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_work_nir_base ON kb.work;
CREATE TRIGGER trg_work_nir_base
    BEFORE INSERT OR UPDATE OF code ON kb.work
    FOR EACH ROW EXECUTE FUNCTION kb.set_work_nir_base();

CREATE INDEX IF NOT EXISTS idx_work_code ON kb.work(code);
CREATE INDEX IF NOT EXISTS idx_work_type ON kb.work(source_type_id);

-- ============================================================================
-- WORK SOURCE LINK (Provenance tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work_source_link (
    work_id UUID REFERENCES kb.work(id) ON DELETE CASCADE,
    source_system_id TEXT REFERENCES kb.source_system(id),
    source_locator TEXT NOT NULL,           -- File path, URL, or identifier
    is_primary BOOLEAN DEFAULT FALSE,
    extracted_at TIMESTAMPTZ,
    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (work_id, source_system_id, source_locator)
);

COMMENT ON TABLE kb.work_source_link IS 'Link tra work e sistemi sorgente (tracking provenance)';

-- ============================================================================
-- WORK TOPIC (Many-to-many)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.work_topic (
    work_id UUID REFERENCES kb.work(id) ON DELETE CASCADE,
    topic_id TEXT REFERENCES kb.topic(id),
    PRIMARY KEY (work_id, topic_id)
);

-- ============================================================================
-- ARTICLE QUALITY ENUM
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'article_quality') THEN
        CREATE TYPE kb.article_quality AS ENUM (
            'VALID_STRONG',   -- Testo presente, struttura coerente, >150 chars
            'VALID_SHORT',    -- Corto ma semanticamente pieno
            'WEAK',           -- Incompleto, segnali taglio, rumore
            'EMPTY',          -- Vuoto, "abrogato", "omissis"
            'INVALID'         -- Estrazione rotta, heading catturato
        );
    END IF;
END $$;

-- ============================================================================
-- NORMATIVA (Articles)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Parent work
    work_id UUID REFERENCES kb.work(id) ON DELETE CASCADE,

    -- URN:NIR (computed from work + article)
    urn_nir VARCHAR(250) UNIQUE,

    -- Article identification
    codice TEXT NOT NULL,                   -- CC, CP, CCII (denormalized)
    articolo TEXT NOT NULL,                 -- '2043', '360-bis'
    comma TEXT,                             -- '1', '2', 'bis'

    -- Parsed article number (for sorting)
    articolo_num INTEGER,                   -- 2043
    articolo_suffix TEXT,                   -- 'bis', 'ter'
    articolo_sort_key TEXT,                 -- '002043.02'

    -- Hierarchy
    libro TEXT,
    titolo TEXT,
    capo TEXT,
    sezione TEXT,

    -- Content
    rubrica TEXT,
    testo TEXT NOT NULL,
    testo_normalizzato TEXT,                -- Lowercase, normalized for dedup

    -- Cross-validation
    canonical_source TEXT,                  -- 'NORMATTIVA_ONLINE'
    canonical_hash VARCHAR(64),
    mirror_source TEXT,                     -- 'BROCARDI_OFFLINE'
    mirror_hash VARCHAR(64),
    validation_status TEXT DEFAULT 'pending'
        CHECK (validation_status IN ('pending', 'verified', 'format_diff', 'content_diff', 'review_needed')),

    -- Versioning (multivigenza)
    data_vigenza_da DATE,
    data_vigenza_a DATE,
    is_current BOOLEAN DEFAULT TRUE,
    nota_modifica TEXT,
    previous_version_id UUID REFERENCES kb.normativa(id),

    -- Quality
    quality kb.article_quality DEFAULT 'VALID_STRONG',
    warnings JSONB DEFAULT '[]'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    -- Constraints
    UNIQUE(work_id, articolo, comma, data_vigenza_da)
);

COMMENT ON TABLE kb.normativa IS 'Articoli di normativa con cross-validation e multivigenza';
COMMENT ON COLUMN kb.normativa.articolo_sort_key IS 'NNNNNN.SS format: 002043.02 for art. 2043-ter';

-- Trigger for sort_key and URN
CREATE OR REPLACE FUNCTION kb.set_normativa_computed()
RETURNS TRIGGER AS $$
DECLARE
    v_suffix_order INT;
    v_nir_base TEXT;
BEGIN
    -- Parse article number and suffix if not provided
    IF NEW.articolo_num IS NULL THEN
        NEW.articolo_num := (regexp_match(NEW.articolo, '^(\d+)'))[1]::INTEGER;
    END IF;

    IF NEW.articolo_suffix IS NULL AND NEW.articolo ~ '(bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|octiesdecies)' THEN
        NEW.articolo_suffix := (regexp_match(NEW.articolo, '(bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|octiesdecies)', 'i'))[1];
    END IF;

    -- Compute sort_key: NNNNNN.SS
    v_suffix_order := CASE LOWER(NEW.articolo_suffix)
        WHEN 'bis' THEN 2 WHEN 'ter' THEN 3 WHEN 'quater' THEN 4
        WHEN 'quinquies' THEN 5 WHEN 'sexies' THEN 6 WHEN 'septies' THEN 7
        WHEN 'octies' THEN 8 WHEN 'novies' THEN 9 WHEN 'nonies' THEN 9
        WHEN 'decies' THEN 10 WHEN 'undecies' THEN 11 WHEN 'duodecies' THEN 12
        WHEN 'terdecies' THEN 13 WHEN 'quaterdecies' THEN 14 WHEN 'quinquiesdecies' THEN 15
        WHEN 'sexiesdecies' THEN 16 WHEN 'septiesdecies' THEN 17 WHEN 'octiesdecies' THEN 18
        ELSE 0
    END;

    IF NEW.articolo_num IS NOT NULL THEN
        NEW.articolo_sort_key := LPAD(NEW.articolo_num::TEXT, 6, '0') || '.' || LPAD(v_suffix_order::TEXT, 2, '0');
    ELSE
        NEW.articolo_sort_key := '999999.99';  -- Special articles at end
    END IF;

    -- Build URN if work_id provided
    IF NEW.work_id IS NOT NULL THEN
        SELECT nir_base INTO v_nir_base FROM kb.work WHERE id = NEW.work_id;
        IF v_nir_base IS NOT NULL THEN
            NEW.urn_nir := v_nir_base || '~art' || NEW.articolo;
            IF NEW.comma IS NOT NULL THEN
                NEW.urn_nir := NEW.urn_nir || '-com' || NEW.comma;
            END IF;
        END IF;
    END IF;

    -- Normalize text
    IF NEW.testo IS NOT NULL AND NEW.testo_normalizzato IS NULL THEN
        NEW.testo_normalizzato := LOWER(regexp_replace(NEW.testo, '\s+', ' ', 'g'));
    END IF;

    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_normativa_computed ON kb.normativa;
CREATE TRIGGER trg_normativa_computed
    BEFORE INSERT OR UPDATE ON kb.normativa
    FOR EACH ROW EXECUTE FUNCTION kb.set_normativa_computed();

-- Indexes
CREATE INDEX IF NOT EXISTS idx_normativa_lookup_current
    ON kb.normativa (codice, articolo, comma) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_normativa_work ON kb.normativa (work_id);
CREATE INDEX IF NOT EXISTS idx_normativa_quality ON kb.normativa (quality);
CREATE INDEX IF NOT EXISTS idx_normativa_sort ON kb.normativa (codice, articolo_sort_key);
CREATE INDEX IF NOT EXISTS idx_normativa_validation ON kb.normativa (validation_status);

-- Trigram index for fuzzy search
CREATE INDEX IF NOT EXISTS idx_normativa_trgm
    ON kb.normativa USING gin (testo_normalizzato gin_trgm_ops);

-- Full-text search Italian
CREATE INDEX IF NOT EXISTS idx_normativa_fts
    ON kb.normativa USING gin (to_tsvector('italian', COALESCE(rubrica, '') || ' ' || testo));

-- ============================================================================
-- NORMATIVA ALTALEX (Altalex-specific extension)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_altalex (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    normativa_id UUID UNIQUE REFERENCES kb.normativa(id) ON DELETE CASCADE,

    -- Altalex-specific parsing
    global_key TEXT UNIQUE,                 -- altalex:cc:2043:bis
    testo_context TEXT,                     -- Overlap ±200 chars

    -- Structured content
    commi JSONB,                            -- [{num: 1, testo: "..."}, ...]
    riferimenti_parsed JSONB,               -- ["CC:1218", "CC:2059"]
    riferimenti_raw TEXT[],

    -- Page provenance
    page_start INTEGER,
    page_end INTEGER,

    -- Full-text search (Italian + unaccent)
    testo_tsv TSVECTOR,

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Trigger for global_key and tsvector
CREATE OR REPLACE FUNCTION kb.set_altalex_computed()
RETURNS TRIGGER AS $$
DECLARE
    v_codice TEXT;
    v_articolo_num INT;
    v_articolo_suffix TEXT;
    v_testo TEXT;
    v_rubrica TEXT;
BEGIN
    -- Get data from parent normativa
    SELECT codice, articolo_num, articolo_suffix, testo, rubrica
    INTO v_codice, v_articolo_num, v_articolo_suffix, v_testo, v_rubrica
    FROM kb.normativa WHERE id = NEW.normativa_id;

    -- Build global_key
    IF v_articolo_num IS NOT NULL THEN
        NEW.global_key := 'altalex:' || LOWER(v_codice) || ':' || v_articolo_num || ':' || COALESCE(LOWER(v_articolo_suffix), 'null');
    END IF;

    -- Update tsvector
    NEW.testo_tsv := to_tsvector('italian',
        unaccent(COALESCE(v_rubrica, '') || ' ' || COALESCE(v_testo, ''))
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_altalex_computed ON kb.normativa_altalex;
CREATE TRIGGER trg_altalex_computed
    BEFORE INSERT OR UPDATE ON kb.normativa_altalex
    FOR EACH ROW EXECUTE FUNCTION kb.set_altalex_computed();

CREATE INDEX IF NOT EXISTS idx_altalex_global_key ON kb.normativa_altalex(global_key);
CREATE INDEX IF NOT EXISTS idx_altalex_tsv ON kb.normativa_altalex USING gin (testo_tsv);

-- ============================================================================
-- NORMATIVA EMBEDDINGS (Multi-dimensional)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.normativa_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    normativa_id UUID REFERENCES kb.normativa(id) ON DELETE CASCADE,

    model TEXT NOT NULL,                    -- 'text-embedding-3-small', 'e5-large'
    channel TEXT NOT NULL,                  -- 'testo', 'rubrica', 'combined'
    dims INT NOT NULL,
    embedding vector NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(normativa_id, model, channel)
);

COMMENT ON TABLE kb.normativa_embeddings IS 'Embeddings multi-dim per articoli normativa';

-- HNSW indexes per dimension (partial indexes)
CREATE INDEX IF NOT EXISTS idx_norm_emb_1536_testo
    ON kb.normativa_embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
    WHERE dims = 1536 AND channel = 'testo';

CREATE INDEX IF NOT EXISTS idx_norm_emb_1024_testo
    ON kb.normativa_embeddings USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
    WHERE dims = 1024 AND channel = 'testo';

CREATE INDEX IF NOT EXISTS idx_norm_emb_768_testo
    ON kb.normativa_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WHERE dims = 768 AND channel = 'testo';

CREATE INDEX IF NOT EXISTS idx_norm_emb_1536_rubrica
    ON kb.normativa_embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
    WHERE dims = 1536 AND channel = 'rubrica';

-- ============================================================================
-- ANNOTATION (Notes, comments, doctrine)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.annotation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    annotation_type TEXT NOT NULL,          -- 'nota', 'commento', 'massima', 'brocardo', 'ratio'
    source_system_id TEXT REFERENCES kb.source_system(id),

    author TEXT,
    title TEXT,
    text TEXT NOT NULL,
    text_normalized TEXT,

    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE kb.annotation IS 'Annotazioni: note, commenti, massime, brocardi, ratio decidendi';

CREATE INDEX IF NOT EXISTS idx_annotation_type ON kb.annotation(annotation_type);
CREATE INDEX IF NOT EXISTS idx_annotation_trgm ON kb.annotation USING gin (text_normalized gin_trgm_ops);

-- ============================================================================
-- ANNOTATION EMBEDDINGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.annotation_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    annotation_id UUID REFERENCES kb.annotation(id) ON DELETE CASCADE,

    model TEXT NOT NULL,
    channel TEXT NOT NULL,
    dims INT NOT NULL,
    embedding vector NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(annotation_id, model, channel)
);

CREATE INDEX IF NOT EXISTS idx_ann_emb_1536_text
    ON kb.annotation_embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
    WHERE dims = 1536 AND channel = 'text';

-- ============================================================================
-- SENTENZA (Court decisions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.sentenza (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    court TEXT NOT NULL,                    -- 'CASS', 'CEDU', 'CGUE', 'TAR', 'CDS'
    sezione TEXT,                           -- 'III', 'SS.UU.'
    numero TEXT NOT NULL,
    anno INT NOT NULL,
    data_decisione DATE,

    title TEXT,
    text TEXT NOT NULL,
    text_normalized TEXT,

    -- Source tracking
    source_system_id TEXT REFERENCES kb.source_system(id),
    source_locator TEXT,

    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(court, sezione, numero, anno)
);

COMMENT ON TABLE kb.sentenza IS 'Sentenze e decisioni giurisprudenziali';

CREATE INDEX IF NOT EXISTS idx_sentenza_court ON kb.sentenza(court, anno);
CREATE INDEX IF NOT EXISTS idx_sentenza_trgm ON kb.sentenza USING gin (text_normalized gin_trgm_ops);

-- ============================================================================
-- SENTENZA EMBEDDINGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.sentenza_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sentenza_id UUID REFERENCES kb.sentenza(id) ON DELETE CASCADE,

    model TEXT NOT NULL,
    channel TEXT NOT NULL,
    dims INT NOT NULL,
    embedding vector NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(sentenza_id, model, channel)
);

CREATE INDEX IF NOT EXISTS idx_sent_emb_1536_text
    ON kb.sentenza_embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
    WHERE dims = 1536 AND channel = 'text';

-- ============================================================================
-- ANNOTATION LINK (Many-to-many: annotation ↔ normativa/sentenza)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.annotation_link (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    annotation_id UUID REFERENCES kb.annotation(id) ON DELETE CASCADE,

    normativa_id UUID REFERENCES kb.normativa(id) ON DELETE CASCADE,
    sentenza_id UUID REFERENCES kb.sentenza(id) ON DELETE CASCADE,

    relevance DOUBLE PRECISION DEFAULT 1.0,
    span JSONB DEFAULT '{}'::jsonb,         -- {start: 100, end: 200}

    created_at TIMESTAMPTZ DEFAULT now(),

    CHECK (
        (normativa_id IS NOT NULL)::int +
        (sentenza_id IS NOT NULL)::int >= 1
    )
);

CREATE INDEX IF NOT EXISTS idx_ann_link_normativa ON kb.annotation_link(normativa_id);
CREATE INDEX IF NOT EXISTS idx_ann_link_sentenza ON kb.annotation_link(sentenza_id);
CREATE INDEX IF NOT EXISTS idx_ann_link_annotation ON kb.annotation_link(annotation_id);

-- ============================================================================
-- NORM REFERENCES (Cross-references between normativa)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.norms (
    id TEXT PRIMARY KEY,                    -- 'CC:2043', 'L:241:1990'
    code TEXT NOT NULL,
    article TEXT,
    suffix TEXT,
    number TEXT,
    year INT,
    full_ref TEXT NOT NULL,
    citation_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb.normativa_norms (
    normativa_id UUID REFERENCES kb.normativa(id) ON DELETE CASCADE,
    norm_id TEXT REFERENCES kb.norms(id) ON DELETE CASCADE,
    context_span TEXT,                      -- Surrounding text
    run_id UUID,
    PRIMARY KEY (normativa_id, norm_id)
);

-- ============================================================================
-- INGESTION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb.ingestion_run (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ,
    source_system_id TEXT REFERENCES kb.source_system(id),
    work_code TEXT,                         -- Target work code (e.g., 'CC')
    status TEXT DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    stats JSONB DEFAULT '{}'::jsonb,        -- {total: 100, inserted: 95, errors: 5}
    meta JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS kb.ingestion_event (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES kb.ingestion_run(id) ON DELETE CASCADE,

    entity_type TEXT NOT NULL,              -- 'normativa', 'annotation', 'embedding'
    entity_id UUID,
    action TEXT NOT NULL,                   -- 'insert', 'update', 'skip', 'error'

    severity TEXT DEFAULT 'info'
        CHECK (severity IN ('debug', 'info', 'warning', 'error')),
    message TEXT,
    meta JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ing_event_run ON kb.ingestion_event(run_id);
CREATE INDEX IF NOT EXISTS idx_ing_event_severity ON kb.ingestion_event(severity);

-- ============================================================================
-- APACHE AGE GRAPH (if available)
-- ============================================================================

DO $$
BEGIN
    -- Create graph if AGE is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'age') THEN
        IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'kb_graph') THEN
            PERFORM ag_catalog.create_graph('kb_graph');
        END IF;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping AGE graph creation: %', SQLERRM;
END $$;

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- Current articles view
CREATE OR REPLACE VIEW kb.v_normativa_current AS
SELECT
    n.id,
    n.codice,
    n.articolo,
    n.comma,
    n.articolo_sort_key,
    n.rubrica,
    n.testo,
    n.quality,
    n.validation_status,
    w.title as work_title,
    w.edition_date,
    nm.nome_breve
FROM kb.normativa n
LEFT JOIN kb.work w ON n.work_id = w.id
LEFT JOIN kb.nir_mapping nm ON n.codice = nm.code
WHERE n.is_current = TRUE;

-- Quality stats per work
CREATE OR REPLACE VIEW kb.v_quality_stats AS
SELECT
    codice,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE quality IN ('VALID_STRONG', 'VALID_SHORT')) as validi,
    COUNT(*) FILTER (WHERE quality IN ('WEAK', 'EMPTY')) as deboli,
    COUNT(*) FILTER (WHERE quality = 'INVALID') as invalidi,
    MIN(articolo_sort_key) as dal,
    MAX(articolo_sort_key) as al,
    ROUND(100.0 * COUNT(*) FILTER (WHERE quality IN ('VALID_STRONG', 'VALID_SHORT')) / NULLIF(COUNT(*), 0), 1) as coverage_pct
FROM kb.normativa
WHERE is_current = TRUE
GROUP BY codice
ORDER BY codice;

-- Annotation counts per article
CREATE OR REPLACE VIEW kb.v_normativa_annotations AS
SELECT
    n.id,
    n.codice,
    n.articolo,
    COUNT(DISTINCT al.annotation_id) FILTER (WHERE a.annotation_type = 'nota') as note_count,
    COUNT(DISTINCT al.annotation_id) FILTER (WHERE a.annotation_type = 'commento') as commenti_count,
    COUNT(DISTINCT al.annotation_id) FILTER (WHERE a.annotation_type = 'massima') as massime_count
FROM kb.normativa n
LEFT JOIN kb.annotation_link al ON al.normativa_id = n.id
LEFT JOIN kb.annotation a ON al.annotation_id = a.id
WHERE n.is_current = TRUE
GROUP BY n.id, n.codice, n.articolo;

-- ============================================================================
-- GRANTS (customize as needed)
-- ============================================================================

-- GRANT USAGE ON SCHEMA kb TO lexe_api;
-- GRANT SELECT ON ALL TABLES IN SCHEMA kb TO lexe_api;
-- GRANT INSERT, UPDATE ON kb.normativa, kb.annotation, kb.ingestion_run, kb.ingestion_event TO lexe_api;

COMMIT;
