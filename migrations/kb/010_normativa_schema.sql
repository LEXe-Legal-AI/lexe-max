-- 010_normativa_schema.sql
-- LEXE Knowledge Base - Schema per normativa italiana (codici, leggi)
-- Con URN:NIR standard, cross-validation, e number-anchored graph

-- ============================================================
-- TABELLA NORMATIVA (Articoli di codice)
-- ============================================================

CREATE TABLE kb.normativa (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ═══ IDENTIFICAZIONE URN:NIR ═══
    -- Standard Normattiva: urn:nir:stato:legge:1942-03-16;262:art2043
    urn_nir VARCHAR(200) UNIQUE,
    codice VARCHAR(50) NOT NULL,            -- 'CC', 'CP', 'CPC', 'CPP', 'COST', 'CDS'
    articolo VARCHAR(20) NOT NULL,          -- '2043', '1', '360-bis'
    comma VARCHAR(10),                      -- '1', '2', 'bis', 'ter'

    -- ═══ GERARCHIA ═══
    libro VARCHAR(150),                     -- 'Libro IV - Delle obbligazioni'
    titolo VARCHAR(150),                    -- 'Titolo IX - Dei fatti illeciti'
    capo VARCHAR(150),
    sezione VARCHAR(150),

    -- ═══ CONTENUTO ═══
    rubrica TEXT,                           -- Titolo articolo: "Risarcimento per fatto illecito"
    testo TEXT NOT NULL,                    -- Testo completo articolo
    testo_normalizzato TEXT,                -- Per search/dedup (lowercase, no extra spaces)

    -- ═══ CROSS-VALIDATION (Cintura e Bretelle) ═══
    canonical_source VARCHAR(50),           -- 'normattiva' | 'gazzetta'
    canonical_url VARCHAR(500),             -- URL fonte canonica
    canonical_retrieved_at TIMESTAMPTZ,
    canonical_hash VARCHAR(64),             -- SHA256 testo normalizzato

    -- Mirror validation
    mirror_source VARCHAR(50),              -- 'studiocataldi' | 'brocardi'
    mirror_url VARCHAR(500),
    mirror_hash VARCHAR(64),
    validation_status VARCHAR(20) DEFAULT 'pending'
        CHECK (validation_status IN ('pending', 'verified', 'format_diff', 'content_diff', 'review_needed')),
    validation_diff TEXT,                   -- Summary delle differenze se presenti
    validated_at TIMESTAMPTZ,

    -- ═══ VERSIONING (Multivigenza) ═══
    data_vigenza_da DATE,                   -- Da quando è in vigore
    data_vigenza_a DATE,                    -- Fino a quando (NULL = vigente)
    is_current BOOLEAN DEFAULT TRUE,        -- Versione corrente
    nota_modifica TEXT,                     -- "Modificato da L. 123/2020"
    previous_version_id UUID REFERENCES kb.normativa(id),

    -- ═══ SOURCE TRACKING ═══
    source_file VARCHAR(200),               -- File mirror originale
    ingestion_version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- ═══ CONSTRAINTS ═══
    UNIQUE(codice, articolo, comma, data_vigenza_da)
);

-- Commenti per documentazione
COMMENT ON TABLE kb.normativa IS 'Articoli di codice italiano con cross-validation e multivigenza';
COMMENT ON COLUMN kb.normativa.urn_nir IS 'Identificativo URN:NIR standard Normattiva';
COMMENT ON COLUMN kb.normativa.canonical_hash IS 'SHA256 del testo_normalizzato dalla fonte canonica';
COMMENT ON COLUMN kb.normativa.validation_status IS 'Stato cross-validation: pending, verified, format_diff, content_diff, review_needed';

-- Indexes per performance
CREATE INDEX idx_normativa_urn ON kb.normativa(urn_nir);
CREATE INDEX idx_normativa_codice_art ON kb.normativa(codice, articolo);
CREATE INDEX idx_normativa_current ON kb.normativa(codice, articolo) WHERE is_current = TRUE;
CREATE INDEX idx_normativa_validation ON kb.normativa(validation_status);
CREATE INDEX idx_normativa_codice ON kb.normativa(codice);

-- Trigram index per ricerca fuzzy
CREATE INDEX idx_normativa_trgm ON kb.normativa USING GIN (testo_normalizzato gin_trgm_ops);

-- Full-text search italiano
CREATE INDEX idx_normativa_fts ON kb.normativa
USING GIN (to_tsvector('italian', testo));

-- ============================================================
-- TABELLA NORMATIVA EMBEDDINGS
-- ============================================================

CREATE TABLE kb.normativa_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    normativa_id UUID NOT NULL REFERENCES kb.normativa(id) ON DELETE CASCADE,
    model VARCHAR(50) NOT NULL,             -- 'text-embedding-3-small', 'e5-large'
    channel VARCHAR(20) NOT NULL DEFAULT 'testo',  -- 'testo', 'rubrica', 'combined'
    embedding vector(1536),
    dims INTEGER DEFAULT 1536,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(normativa_id, model, channel)
);

-- HNSW index per vector search
CREATE INDEX idx_normativa_emb_hnsw ON kb.normativa_embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_normativa_emb_normativa ON kb.normativa_embeddings(normativa_id);
CREATE INDEX idx_normativa_emb_model ON kb.normativa_embeddings(model);

-- ============================================================
-- TABELLA NORMATIVA CITATIONS (Cross-references tra articoli)
-- ============================================================

CREATE TABLE kb.normativa_citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES kb.normativa(id) ON DELETE CASCADE,
    target_codice VARCHAR(50) NOT NULL,
    target_articolo VARCHAR(20) NOT NULL,
    target_comma VARCHAR(10),
    citation_type VARCHAR(30) DEFAULT 'reference'
        CHECK (citation_type IN ('reference', 'see_also', 'replaces', 'amended_by', 'repealed_by')),
    raw_citation TEXT,                      -- Testo originale: "art. 2044 c.c."
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_id, target_codice, target_articolo, target_comma)
);

CREATE INDEX idx_normativa_citations_source ON kb.normativa_citations(source_id);
CREATE INDEX idx_normativa_citations_target ON kb.normativa_citations(target_codice, target_articolo);

-- ============================================================
-- TABELLA NORMATIVA UPDATES (Tracking aggiornamenti settimanali)
-- ============================================================

CREATE TABLE kb.normativa_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    normativa_id UUID REFERENCES kb.normativa(id) ON DELETE CASCADE,
    check_date DATE NOT NULL,
    previous_hash VARCHAR(64),
    new_hash VARCHAR(64),
    change_detected BOOLEAN NOT NULL DEFAULT FALSE,
    change_type VARCHAR(30)
        CHECK (change_type IN ('new_version', 'correction', 'abrogation', 'formatting')),
    change_summary TEXT,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_normativa_updates_normativa ON kb.normativa_updates(normativa_id);
CREATE INDEX idx_normativa_updates_date ON kb.normativa_updates(check_date);
CREATE INDEX idx_normativa_updates_pending ON kb.normativa_updates(processed) WHERE processed = FALSE;

-- ============================================================
-- TABELLA LEGAL NUMBERS (Master index numeri legali)
-- Number-Anchored Knowledge Graph
-- ============================================================

CREATE TABLE kb.legal_numbers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identificativo canonico univoco
    canonical_id VARCHAR(100) UNIQUE NOT NULL,  -- "CC:2043", "L:241:1990", "CASS:12345:2020"

    -- Parsing strutturato
    number_type VARCHAR(20) NOT NULL
        CHECK (number_type IN ('article', 'law', 'decree', 'sentence', 'tu', 'regulation')),
    codice VARCHAR(30),                     -- 'CC', 'CP', 'L', 'DLGS', 'CASS', 'DPR'
    numero VARCHAR(50),                     -- '2043', '241', '12345'
    anno INTEGER,                           -- 1990, 2020
    comma VARCHAR(20),                      -- 'bis', 'ter', '1', '2'

    -- Metadata
    description TEXT,                       -- "Risarcimento per fatto illecito"
    full_reference TEXT,                    -- "Art. 2043 Codice Civile"
    is_vigente BOOLEAN DEFAULT TRUE,

    -- Stats (aggiornate periodicamente)
    citation_count INTEGER DEFAULT 0,       -- Quante volte è citato
    citing_massime_count INTEGER DEFAULT 0,
    citing_normativa_count INTEGER DEFAULT 0,
    citing_sentenze_count INTEGER DEFAULT 0,

    -- Timestamps
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    stats_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.legal_numbers IS 'Master index di tutti i numeri legali per Number-Anchored Knowledge Graph';
COMMENT ON COLUMN kb.legal_numbers.canonical_id IS 'Formato: TYPE:NUMBER[:YEAR] - es. CC:2043, L:241:1990, CASS:12345:2020';

-- Indexes
CREATE INDEX idx_legal_numbers_canonical ON kb.legal_numbers(canonical_id);
CREATE INDEX idx_legal_numbers_type ON kb.legal_numbers(number_type, codice);
CREATE INDEX idx_legal_numbers_codice ON kb.legal_numbers(codice);
CREATE INDEX idx_legal_numbers_citation_count ON kb.legal_numbers(citation_count DESC);
CREATE INDEX idx_legal_numbers_vigente ON kb.legal_numbers(is_vigente) WHERE is_vigente = TRUE;

-- ============================================================
-- TABELLA NUMBER CITATIONS (Edges: chi cita chi)
-- ============================================================

CREATE TABLE kb.number_citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source: chi cita
    source_type VARCHAR(20) NOT NULL
        CHECK (source_type IN ('massima', 'normativa', 'sentenza', 'commento', 'brocardi')),
    source_id UUID NOT NULL,

    -- Target: cosa è citato (numero)
    target_number_id UUID REFERENCES kb.legal_numbers(id) ON DELETE CASCADE,
    target_canonical VARCHAR(100) NOT NULL, -- Denormalizzato per performance

    -- Context
    raw_citation TEXT,                      -- "art. 2043 c.c."
    citation_context TEXT,                  -- Frase che contiene la citazione
    confidence FLOAT DEFAULT 1.0,           -- Confidence dell'estrazione

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_type, source_id, target_canonical)
);

COMMENT ON TABLE kb.number_citations IS 'Edges del Number-Anchored Knowledge Graph: chi cita quali numeri';

-- Indexes per traversal grafo
CREATE INDEX idx_number_citations_source ON kb.number_citations(source_type, source_id);
CREATE INDEX idx_number_citations_target ON kb.number_citations(target_canonical);
CREATE INDEX idx_number_citations_number_id ON kb.number_citations(target_number_id);

-- ============================================================
-- VIEW MATERIALIZZATA: Numeri più citati
-- ============================================================

CREATE MATERIALIZED VIEW kb.top_cited_numbers AS
SELECT
    ln.id,
    ln.canonical_id,
    ln.number_type,
    ln.codice,
    ln.numero,
    ln.anno,
    ln.description,
    ln.full_reference,
    COUNT(nc.id) as total_citations,
    COUNT(CASE WHEN nc.source_type = 'massima' THEN 1 END) as massima_citations,
    COUNT(CASE WHEN nc.source_type = 'normativa' THEN 1 END) as normativa_citations,
    COUNT(CASE WHEN nc.source_type = 'sentenza' THEN 1 END) as sentenza_citations,
    array_agg(DISTINCT nc.source_type) as cited_by_types
FROM kb.legal_numbers ln
LEFT JOIN kb.number_citations nc ON nc.target_number_id = ln.id
GROUP BY ln.id
ORDER BY total_citations DESC;

CREATE UNIQUE INDEX idx_top_cited_numbers_id ON kb.top_cited_numbers(id);
CREATE INDEX idx_top_cited_numbers_citations ON kb.top_cited_numbers(total_citations DESC);

-- ============================================================
-- FUNZIONI HELPER
-- ============================================================

-- Funzione per refresh stats dei legal_numbers
CREATE OR REPLACE FUNCTION kb.refresh_legal_number_stats()
RETURNS void AS $$
BEGIN
    -- Update citation counts
    UPDATE kb.legal_numbers ln
    SET
        citation_count = COALESCE(stats.total, 0),
        citing_massime_count = COALESCE(stats.massime, 0),
        citing_normativa_count = COALESCE(stats.normativa, 0),
        citing_sentenze_count = COALESCE(stats.sentenze, 0),
        stats_updated_at = NOW()
    FROM (
        SELECT
            target_number_id,
            COUNT(*) as total,
            COUNT(CASE WHEN source_type = 'massima' THEN 1 END) as massime,
            COUNT(CASE WHEN source_type = 'normativa' THEN 1 END) as normativa,
            COUNT(CASE WHEN source_type = 'sentenza' THEN 1 END) as sentenze
        FROM kb.number_citations
        GROUP BY target_number_id
    ) stats
    WHERE ln.id = stats.target_number_id;

    -- Refresh materialized view
    REFRESH MATERIALIZED VIEW CONCURRENTLY kb.top_cited_numbers;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION kb.refresh_legal_number_stats IS 'Aggiorna le statistiche delle citazioni per tutti i legal_numbers';

-- Funzione per normalizzare canonical_id
CREATE OR REPLACE FUNCTION kb.normalize_canonical_id(
    p_type VARCHAR,
    p_codice VARCHAR,
    p_numero VARCHAR,
    p_anno INTEGER DEFAULT NULL
)
RETURNS VARCHAR AS $$
BEGIN
    IF p_anno IS NOT NULL THEN
        RETURN UPPER(p_codice) || ':' || p_numero || ':' || p_anno;
    ELSE
        RETURN UPPER(p_codice) || ':' || p_numero;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger per aggiornare updated_at
CREATE OR REPLACE FUNCTION kb.update_normativa_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_normativa_updated
    BEFORE UPDATE ON kb.normativa
    FOR EACH ROW
    EXECUTE FUNCTION kb.update_normativa_timestamp();

-- ============================================================
-- LOOKUP TABLES (Mapping codici)
-- ============================================================

CREATE TABLE kb.codice_lookup (
    codice VARCHAR(30) PRIMARY KEY,
    nome_completo VARCHAR(200) NOT NULL,
    abbreviazioni TEXT[],                   -- ['c.c.', 'cod. civ.', 'codice civile']
    urn_prefix VARCHAR(100),                -- 'urn:nir:stato:legge:1942-03-16;262'
    data_emanazione DATE,
    note TEXT
);

INSERT INTO kb.codice_lookup (codice, nome_completo, abbreviazioni, urn_prefix, data_emanazione) VALUES
('CC', 'Codice Civile', ARRAY['c.c.', 'cod. civ.', 'codice civile'], 'urn:nir:stato:legge:1942-03-16;262', '1942-03-16'),
('CP', 'Codice Penale', ARRAY['c.p.', 'cod. pen.', 'codice penale'], 'urn:nir:stato:decreto:1930-10-19;1398', '1930-10-19'),
('CPC', 'Codice di Procedura Civile', ARRAY['c.p.c.', 'cod. proc. civ.'], 'urn:nir:stato:decreto:1940-10-28;1443', '1940-10-28'),
('CPP', 'Codice di Procedura Penale', ARRAY['c.p.p.', 'cod. proc. pen.'], 'urn:nir:stato:decreto:1988-09-22;447', '1988-09-22'),
('COST', 'Costituzione della Repubblica Italiana', ARRAY['cost.', 'costituzione'], 'urn:nir:stato:costituzione:1947-12-27', '1947-12-27'),
('CDS', 'Codice della Strada', ARRAY['c.d.s.', 'cod. strada'], 'urn:nir:stato:decreto:1992-04-30;285', '1992-04-30'),
('CDPRIV', 'Codice della Privacy', ARRAY['cod. privacy', 'd.lgs. 196/2003'], 'urn:nir:stato:decreto.legislativo:2003-06-30;196', '2003-06-30'),
('CCONS', 'Codice del Consumo', ARRAY['cod. consumo', 'd.lgs. 206/2005'], 'urn:nir:stato:decreto.legislativo:2005-09-06;206', '2005-09-06'),
('CAPPALTI', 'Codice degli Appalti', ARRAY['cod. appalti', 'd.lgs. 50/2016'], 'urn:nir:stato:decreto.legislativo:2016-04-18;50', '2016-04-18'),
('CAMB', 'Codice dell''Ambiente', ARRAY['cod. ambiente', 'd.lgs. 152/2006'], 'urn:nir:stato:decreto.legislativo:2006-04-03;152', '2006-04-03'),
('CNAV', 'Codice della Navigazione', ARRAY['cod. nav.', 'c.nav.'], 'urn:nir:stato:decreto:1942-03-30;327', '1942-03-30'),
('CNAUT', 'Codice della Nautica', ARRAY['cod. nautica', 'd.lgs. 171/2005'], 'urn:nir:stato:decreto.legislativo:2005-07-18;171', '2005-07-18'),
('CASS', 'Codice delle Assicurazioni Private', ARRAY['cod. ass.', 'd.lgs. 209/2005'], 'urn:nir:stato:decreto.legislativo:2005-09-07;209', '2005-09-07'),
('CBENI', 'Codice dei Beni Culturali', ARRAY['cod. beni cult.', 'd.lgs. 42/2004'], 'urn:nir:stato:decreto.legislativo:2004-01-22;42', '2004-01-22'),
('CPIND', 'Codice della Proprietà Industriale', ARRAY['cod. prop. ind.', 'd.lgs. 30/2005'], 'urn:nir:stato:decreto.legislativo:2005-02-10;30', '2005-02-10');

COMMENT ON TABLE kb.codice_lookup IS 'Mapping tra sigle codici e riferimenti completi per URN:NIR';

-- ============================================================
-- GRANT PERMISSIONS
-- ============================================================

-- Grant select to read-only users (if needed)
-- GRANT SELECT ON ALL TABLES IN SCHEMA kb TO lexe_readonly;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA kb TO lexe_readonly;
