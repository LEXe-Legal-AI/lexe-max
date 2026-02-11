-- 011_brocardi_dizionario_schema.sql
-- LEXE Knowledge Base - Schema per brocardi latini e dizionario giuridico

-- ============================================================
-- TABELLA BROCARDI (Massime latine)
-- ============================================================

CREATE TABLE kb.brocardi (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ═══ CONTENUTO ═══
    latino TEXT NOT NULL,                   -- "Ad impossibilia nemo tenetur"
    italiano TEXT,                          -- "Nessuno è tenuto all'impossibile"
    significato TEXT,                       -- Spiegazione operativa
    esempi_uso TEXT[],                      -- Esempi pratici di applicazione

    -- ═══ CLASSIFICAZIONE ═══
    tags TEXT[],                            -- ['contratti', 'obbligazioni', 'impossibilità']
    categoria VARCHAR(50)                   -- 'principio', 'massima', 'locuzione', 'adagio'
        CHECK (categoria IN ('principio', 'massima', 'locuzione', 'adagio', 'brocardo')),
    area_diritto VARCHAR(50),               -- 'civile', 'penale', 'processuale', 'generale'

    -- ═══ FONTI ═══
    source VARCHAR(50) NOT NULL,            -- 'brocardi.it', 'avvocatoandreani', 'manuale'
    source_url VARCHAR(500),
    attribution TEXT,                       -- Autore/fonte storica se nota

    -- ═══ TIMESTAMPS ═══
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.brocardi IS 'Brocardi latini con traduzione e significato operativo';

-- Indexes
CREATE INDEX idx_brocardi_latino_trgm ON kb.brocardi USING GIN (latino gin_trgm_ops);
CREATE INDEX idx_brocardi_italiano_trgm ON kb.brocardi USING GIN (italiano gin_trgm_ops);
CREATE INDEX idx_brocardi_tags ON kb.brocardi USING GIN (tags);
CREATE INDEX idx_brocardi_categoria ON kb.brocardi(categoria);
CREATE INDEX idx_brocardi_area ON kb.brocardi(area_diritto);

-- Full-text search
CREATE INDEX idx_brocardi_fts ON kb.brocardi
USING GIN (to_tsvector('italian', COALESCE(italiano, '') || ' ' || COALESCE(significato, '')));

-- ============================================================
-- TABELLA BROCARDI EMBEDDINGS
-- ============================================================

CREATE TABLE kb.brocardi_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brocardi_id UUID NOT NULL REFERENCES kb.brocardi(id) ON DELETE CASCADE,
    model VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL DEFAULT 'combined'
        CHECK (channel IN ('latino', 'italiano', 'significato', 'combined')),
    embedding vector(1536),
    dims INTEGER DEFAULT 1536,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(brocardi_id, model, channel)
);

-- HNSW index per vector search
CREATE INDEX idx_brocardi_emb_hnsw ON kb.brocardi_embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_brocardi_emb_brocardi ON kb.brocardi_embeddings(brocardi_id);

-- ============================================================
-- TABELLA BROCARDI LINKS (Collegamenti a normativa)
-- ============================================================

CREATE TABLE kb.brocardi_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brocardi_id UUID NOT NULL REFERENCES kb.brocardi(id) ON DELETE CASCADE,

    -- Target: può essere normativa, massima, o legal_number
    link_type VARCHAR(30) NOT NULL
        CHECK (link_type IN ('normativa', 'massima', 'legal_number')),
    target_id UUID,                         -- ID del target (normativa/massima/legal_number)
    target_canonical VARCHAR(100),          -- Per legal_number: "CC:2043"

    -- Metadata
    link_reason VARCHAR(100),               -- 'principio_applicato', 'massima_correlata', 'articolo_tipico'
    relevance_score FLOAT DEFAULT 0.5,      -- 0-1
    note TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(brocardi_id, link_type, target_id)
);

COMMENT ON TABLE kb.brocardi_links IS 'Collegamenti tra brocardi e articoli/massime/norme';

CREATE INDEX idx_brocardi_links_brocardi ON kb.brocardi_links(brocardi_id);
CREATE INDEX idx_brocardi_links_target ON kb.brocardi_links(link_type, target_id);
CREATE INDEX idx_brocardi_links_canonical ON kb.brocardi_links(target_canonical);

-- ============================================================
-- TABELLA DIZIONARIO GIURIDICO
-- ============================================================

CREATE TABLE kb.dizionario (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ═══ CONTENUTO ═══
    voce VARCHAR(200) NOT NULL,             -- "Capacità giuridica"
    voce_normalizzata VARCHAR(200),         -- Lowercase, no accents per search
    definizione TEXT NOT NULL,              -- Definizione completa
    definizione_breve TEXT,                 -- One-liner per quick lookup

    -- ═══ COLLEGAMENTI ═══
    sinonimi TEXT[],                        -- ['capacità di diritto']
    contrari TEXT[],                        -- Termini opposti
    vedi_anche TEXT[],                      -- ['Capacità di agire', 'Persona giuridica']

    -- ═══ CLASSIFICAZIONE ═══
    area VARCHAR(50),                       -- 'civile', 'penale', 'processuale', 'amministrativo'
    sotto_area VARCHAR(100),                -- 'obbligazioni', 'famiglia', 'successioni'
    tags TEXT[],
    livello VARCHAR(20) DEFAULT 'base'      -- 'base', 'intermedio', 'avanzato', 'tecnico'
        CHECK (livello IN ('base', 'intermedio', 'avanzato', 'tecnico')),

    -- ═══ FONTI ═══
    source VARCHAR(50) NOT NULL,            -- 'brocardi.it', 'treccani', 'manuale'
    source_url VARCHAR(500),

    -- ═══ TIMESTAMPS ═══
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE kb.dizionario IS 'Dizionario giuridico italiano con definizioni e collegamenti';

-- Unique constraint sulla voce normalizzata
CREATE UNIQUE INDEX idx_dizionario_voce_unique ON kb.dizionario(voce_normalizzata);

-- Indexes per search
CREATE INDEX idx_dizionario_voce_trgm ON kb.dizionario USING GIN (voce gin_trgm_ops);
CREATE INDEX idx_dizionario_definizione_trgm ON kb.dizionario USING GIN (definizione gin_trgm_ops);
CREATE INDEX idx_dizionario_sinonimi ON kb.dizionario USING GIN (sinonimi);
CREATE INDEX idx_dizionario_tags ON kb.dizionario USING GIN (tags);
CREATE INDEX idx_dizionario_area ON kb.dizionario(area);

-- Full-text search
CREATE INDEX idx_dizionario_fts ON kb.dizionario
USING GIN (to_tsvector('italian', voce || ' ' || definizione));

-- ============================================================
-- TABELLA DIZIONARIO EMBEDDINGS
-- ============================================================

CREATE TABLE kb.dizionario_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dizionario_id UUID NOT NULL REFERENCES kb.dizionario(id) ON DELETE CASCADE,
    model VARCHAR(50) NOT NULL,
    embedding vector(1536),
    dims INTEGER DEFAULT 1536,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(dizionario_id, model)
);

-- HNSW index per vector search
CREATE INDEX idx_dizionario_emb_hnsw ON kb.dizionario_embeddings
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_dizionario_emb_dizionario ON kb.dizionario_embeddings(dizionario_id);

-- ============================================================
-- TABELLA DIZIONARIO LINKS (Collegamenti a normativa/brocardi)
-- ============================================================

CREATE TABLE kb.dizionario_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dizionario_id UUID NOT NULL REFERENCES kb.dizionario(id) ON DELETE CASCADE,

    -- Target
    link_type VARCHAR(30) NOT NULL
        CHECK (link_type IN ('normativa', 'brocardi', 'legal_number', 'dizionario')),
    target_id UUID,
    target_canonical VARCHAR(100),          -- Per legal_number

    -- Metadata
    link_reason VARCHAR(100),               -- 'definizione_di', 'articolo_correlato', 'brocardo_correlato'
    relevance_score FLOAT DEFAULT 0.5,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(dizionario_id, link_type, target_id)
);

CREATE INDEX idx_dizionario_links_dizionario ON kb.dizionario_links(dizionario_id);
CREATE INDEX idx_dizionario_links_target ON kb.dizionario_links(link_type, target_id);

-- ============================================================
-- FUNZIONI HELPER
-- ============================================================

-- Funzione per normalizzare voce dizionario
CREATE OR REPLACE FUNCTION kb.normalize_voce(p_voce TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN LOWER(
        TRANSLATE(
            p_voce,
            'àèéìòùÀÈÉÌÒÙ',
            'aeeiouAEEIOU'
        )
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger per normalizzare automaticamente la voce
CREATE OR REPLACE FUNCTION kb.normalize_dizionario_voce()
RETURNS TRIGGER AS $$
BEGIN
    NEW.voce_normalizzata = kb.normalize_voce(NEW.voce);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_dizionario_normalize
    BEFORE INSERT OR UPDATE ON kb.dizionario
    FOR EACH ROW
    EXECUTE FUNCTION kb.normalize_dizionario_voce();

-- Trigger per updated_at su brocardi
CREATE TRIGGER trg_brocardi_updated
    BEFORE UPDATE ON kb.brocardi
    FOR EACH ROW
    EXECUTE FUNCTION kb.update_normativa_timestamp();

-- Trigger per updated_at su dizionario
CREATE TRIGGER trg_dizionario_updated
    BEFORE UPDATE ON kb.dizionario
    FOR EACH ROW
    EXECUTE FUNCTION kb.update_normativa_timestamp();

-- ============================================================
-- VIEW: Ricerca unificata brocardi + dizionario
-- ============================================================

CREATE VIEW kb.legal_glossary AS
SELECT
    'brocardi' as tipo,
    id,
    latino as termine,
    COALESCE(italiano, latino) as termine_italiano,
    significato as definizione,
    tags,
    categoria as categoria,
    area_diritto as area
FROM kb.brocardi
UNION ALL
SELECT
    'dizionario' as tipo,
    id,
    voce as termine,
    voce as termine_italiano,
    definizione,
    tags,
    livello as categoria,
    area
FROM kb.dizionario;

COMMENT ON VIEW kb.legal_glossary IS 'Vista unificata per ricerca su brocardi e dizionario';

-- ============================================================
-- SAMPLE DATA: Brocardi più comuni
-- ============================================================

-- Inserisco alcuni brocardi fondamentali come esempio
INSERT INTO kb.brocardi (latino, italiano, significato, tags, categoria, area_diritto, source) VALUES
('Ad impossibilia nemo tenetur', 'Nessuno è tenuto all''impossibile',
 'Principio secondo cui non si può pretendere da qualcuno l''adempimento di una prestazione oggettivamente impossibile. Fondamento dell''impossibilità sopravvenuta della prestazione (art. 1256 c.c.).',
 ARRAY['impossibilità', 'obbligazioni', 'prestazione'], 'principio', 'civile', 'manuale'),

('Ignorantia legis non excusat', 'L''ignoranza della legge non scusa',
 'Principio fondamentale secondo cui nessuno può invocare a propria discolpa l''ignoranza della legge. Codificato nell''art. 5 c.p. (con le eccezioni introdotte dalla Corte Costituzionale).',
 ARRAY['ignoranza', 'legge', 'responsabilità'], 'principio', 'penale', 'manuale'),

('In dubio pro reo', 'Nel dubbio, a favore dell''imputato',
 'Principio cardine del processo penale: se il giudice non raggiunge la certezza della colpevolezza, deve assolvere l''imputato. Espressione della presunzione di innocenza (art. 27 Cost., art. 533 c.p.p.).',
 ARRAY['dubbio', 'imputato', 'assoluzione', 'presunzione innocenza'], 'principio', 'penale', 'manuale'),

('Nemo plus iuris transferre potest quam ipse habet', 'Nessuno può trasferire più diritti di quanti ne abbia',
 'Principio che regola i trasferimenti di diritti: il dante causa non può trasmettere un diritto più ampio di quello che possiede. Eccezioni: acquisto a titolo originario, acquisti a non domino in buona fede.',
 ARRAY['trasferimento', 'diritti', 'proprietà'], 'principio', 'civile', 'manuale'),

('Pacta sunt servanda', 'I patti devono essere rispettati',
 'Principio fondamentale del diritto dei contratti: le parti sono vincolate agli accordi liberamente assunti. Base dell''art. 1372 c.c. (il contratto ha forza di legge tra le parti).',
 ARRAY['contratto', 'obbligazioni', 'adempimento'], 'principio', 'civile', 'manuale'),

('Nullum crimen, nulla poena sine lege', 'Nessun crimine, nessuna pena senza legge',
 'Principio di legalità penale: un fatto può essere punito solo se previsto come reato da una legge vigente al momento della sua commissione. Art. 25 Cost., art. 1 c.p.',
 ARRAY['legalità', 'tipicità', 'riserva di legge'], 'principio', 'penale', 'manuale'),

('Res iudicata pro veritate habetur', 'La cosa giudicata è tenuta per verità',
 'Principio per cui la sentenza passata in giudicato fa stato tra le parti e non può più essere messa in discussione. Art. 2909 c.c., art. 324 c.p.c.',
 ARRAY['giudicato', 'sentenza', 'cosa giudicata'], 'principio', 'processuale', 'manuale'),

('Nemo auditur propriam turpitudinem allegans', 'Nessuno può essere ascoltato se allega la propria turpitudine',
 'Principio secondo cui non si può invocare a proprio vantaggio una situazione di fatto derivante dalla propria condotta illecita o immorale.',
 ARRAY['buona fede', 'turpitudine', 'condotta'], 'adagio', 'civile', 'manuale');

-- ============================================================
-- SAMPLE DATA: Voci dizionario fondamentali
-- ============================================================

INSERT INTO kb.dizionario (voce, definizione, definizione_breve, area, sotto_area, tags, source) VALUES
('Capacità giuridica',
 'Idoneità di un soggetto ad essere titolare di diritti e di obblighi. Si acquista alla nascita (art. 1 c.c.) e si perde con la morte. È attributo di ogni persona fisica.',
 'Idoneità ad essere titolare di diritti e obblighi',
 'civile', 'persone', ARRAY['soggetti', 'capacità', 'diritti'], 'brocardi.it'),

('Capacità di agire',
 'Idoneità di un soggetto a compiere validamente atti giuridici. Si acquista con la maggiore età (art. 2 c.c.) e può essere limitata o esclusa da incapacità naturale o legale.',
 'Idoneità a compiere validamente atti giuridici',
 'civile', 'persone', ARRAY['soggetti', 'capacità', 'atti'], 'brocardi.it'),

('Dolo',
 'Nel diritto civile: artifici o raggiri usati per indurre altri in errore (vizio del consenso, art. 1439 c.c.). Nel diritto penale: volontà di realizzare un fatto previsto dalla legge come reato.',
 'Inganno deliberato o volontà di commettere reato',
 'civile', 'contratti', ARRAY['vizio', 'consenso', 'inganno'], 'brocardi.it'),

('Colpa',
 'Negligenza, imprudenza o imperizia nell''agire, ovvero inosservanza di leggi, regolamenti, ordini o discipline. Nel penale: art. 43 c.p. Nel civile: fondamento della responsabilità extracontrattuale.',
 'Negligenza, imprudenza o imperizia',
 'civile', 'responsabilità', ARRAY['responsabilità', 'danno', 'negligenza'], 'brocardi.it'),

('Giudicato',
 'Qualità della sentenza non più impugnabile con i mezzi ordinari. Il giudicato formale riguarda l''immutabilità della sentenza; il giudicato sostanziale (art. 2909 c.c.) fa stato tra le parti.',
 'Sentenza non più impugnabile',
 'processuale', 'sentenza', ARRAY['sentenza', 'impugnazione', 'definitività'], 'brocardi.it');
