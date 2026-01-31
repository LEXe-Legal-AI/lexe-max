-- 004_graph_schema.sql
-- LEXE Knowledge Base - Graph Schema Extension v3.2.1
-- Citation Graph, Topic Classification, Turning Points, GraphRAG

-- ============================================================
-- NOTA: Richiede 001_init_extensions.sql e 002_init_schema.sql
-- ============================================================

SET search_path TO kb, ag_catalog, public;

-- ============================================================
-- GRAPH RUNS VERSIONING (Miglioria #2 orig + #7 v3.2.1)
-- Track each graph build run for idempotency and cache invalidation
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.graph_runs (
    id SERIAL PRIMARY KEY,
    run_type VARCHAR(50) NOT NULL,      -- citation_extraction, topic_classification, norm_extraction, turning_points
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'rolled_back')),
    is_active BOOLEAN DEFAULT FALSE,    -- Solo 1 run attivo per tipo (cache invalidation v3.2.1)
    config JSONB DEFAULT '{}',          -- Parametri usati per il run
    metrics JSONB DEFAULT '{}',         -- Metriche finali (resolution_rate, edge_count, etc.)
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vincolo: solo 1 run attivo per tipo
CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_runs_active_unique
    ON kb.graph_runs(run_type) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_graph_runs_type ON kb.graph_runs(run_type);
CREATE INDEX IF NOT EXISTS idx_graph_runs_status ON kb.graph_runs(status);

-- ============================================================
-- GRAPH EDGES SQL (Migliorie #0, #1, #2 v3.2.1)
-- Tabella SQL per reranking low-latency (dual-write con AGE)
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.graph_edges (
    id SERIAL PRIMARY KEY,
    source_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    edge_type VARCHAR(30) NOT NULL,     -- CITES (primary), CITES_NORM (to norms)
    relation_subtype VARCHAR(30),       -- CONFIRMS, DISTINGUISHES, OVERRULES, NULL=generic cite

    -- Scoring (Miglioria #1 v3.2.1)
    confidence FLOAT NOT NULL DEFAULT 1.0,  -- Quanto siamo sicuri del match
    weight FLOAT NOT NULL DEFAULT 1.0,      -- Per pruning (keep if >= 0.6)

    -- Evidence (Miglioria #1 v3.2.1) - Debug trail
    evidence JSONB DEFAULT '{}',        -- {pattern, indicator, resolver, raw_span}

    context_span TEXT,                  -- Frase dove appare la citazione
    run_id INTEGER REFERENCES kb.graph_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Dedup semantico (Miglioria #2 v3.2.1): 1 edge per (src, tgt, subtype) per run
    CONSTRAINT uq_graph_edges_dedup UNIQUE (source_id, target_id, edge_type, relation_subtype, run_id)
);

-- Indici per retrieval veloce
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON kb.graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON kb.graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON kb.graph_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_subtype ON kb.graph_edges(relation_subtype) WHERE relation_subtype IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_graph_edges_run ON kb.graph_edges(run_id);

-- Partial index per pruning (solo edges validi per retrieval)
CREATE INDEX IF NOT EXISTS idx_graph_edges_weight_valid
    ON kb.graph_edges(weight DESC) WHERE weight >= 0.6;

-- GIN index per evidence search
CREATE INDEX IF NOT EXISTS idx_graph_edges_evidence ON kb.graph_edges USING gin(evidence);

-- ============================================================
-- CATEGORIES (6-8 Macro L1, predisposte L2/L3)
-- Miglioria #6 v3.2.1: centroid con ricetta esplicita
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.categories (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    level INTEGER NOT NULL DEFAULT 1 CHECK (level IN (1, 2, 3)),  -- L1=macro, L2=sub, L3=micro
    parent_id VARCHAR(50) REFERENCES kb.categories(id) ON DELETE SET NULL,
    keywords TEXT[] DEFAULT '{}',       -- Per keyword matching

    -- Embedding centroid (Miglioria #6 v3.2.1)
    centroid vector(1536),              -- Media embeddings massime seed
    seed_count INTEGER,                 -- Quante massime usate per il centroid
    seed_query TEXT,                    -- Query SQL usata per selezionare seeds (audit)

    is_active BOOLEAN DEFAULT TRUE,     -- L2/L3 disattivate inizialmente
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_categories_parent ON kb.categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_categories_level ON kb.categories(level);
CREATE INDEX IF NOT EXISTS idx_categories_active ON kb.categories(is_active) WHERE is_active = TRUE;

-- HNSW index per centroid similarity (se usiamo embedding classification)
CREATE INDEX IF NOT EXISTS idx_categories_centroid ON kb.categories
    USING hnsw (centroid vector_cosine_ops)
    WHERE centroid IS NOT NULL;

-- ============================================================
-- CATEGORY ASSIGNMENTS
-- Massima -> Categories (multi-label)
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.category_assignments (
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    category_id VARCHAR(50) NOT NULL REFERENCES kb.categories(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    method VARCHAR(20) CHECK (method IN ('keyword', 'embedding', 'hybrid', 'manual')),
    evidence_terms TEXT[] DEFAULT '{}',  -- Keyword che hanno matchato
    run_id INTEGER REFERENCES kb.graph_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (massima_id, category_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_category_assignments_massima ON kb.category_assignments(massima_id);
CREATE INDEX IF NOT EXISTS idx_category_assignments_category ON kb.category_assignments(category_id);
CREATE INDEX IF NOT EXISTS idx_category_assignments_confidence ON kb.category_assignments(confidence DESC);

-- ============================================================
-- NORMS (Canonicalizzati)
-- Miglioria #9 orig: nodi norma per citation graph
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.norms (
    id VARCHAR(50) PRIMARY KEY,         -- Canonical: "CC:2043", "CPC:183:bis"
    code VARCHAR(20) NOT NULL,          -- CC, CPC, CP, CPP, COST, LEGGE
    article VARCHAR(20) NOT NULL,       -- "2043", "183", "111"
    suffix VARCHAR(10),                 -- bis, ter, quater, quinquies, sexies
    full_text TEXT,                     -- "art. 2043 c.c."
    citation_count INTEGER DEFAULT 0,   -- Quante massime citano questa norma
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_norms_code ON kb.norms(code);
CREATE INDEX IF NOT EXISTS idx_norms_count ON kb.norms(citation_count DESC);

-- ============================================================
-- MASSIMA -> NORM EDGES
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.massima_norms (
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    norm_id VARCHAR(50) NOT NULL REFERENCES kb.norms(id) ON DELETE CASCADE,
    context_span TEXT,                  -- Frase dove appare la norma
    run_id INTEGER REFERENCES kb.graph_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (massima_id, norm_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_massima_norms_massima ON kb.massima_norms(massima_id);
CREATE INDEX IF NOT EXISTS idx_massima_norms_norm ON kb.massima_norms(norm_id);

-- ============================================================
-- TURNING POINTS
-- Miglioria #5 v3.2.1: subtype + flag, NON nodi evento
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.turning_points (
    id SERIAL PRIMARY KEY,
    massima_id UUID NOT NULL REFERENCES kb.massime(id) ON DELETE CASCADE,
    overruled_massima_id UUID REFERENCES kb.massime(id) ON DELETE SET NULL,

    turning_point_type VARCHAR(30) CHECK (turning_point_type IN (
        'SEZ_UNITE',            -- Sezioni Unite che risolvono contrasto
        'CONTRASTO_RISOLTO',    -- Risoluzione contrasto esplicita
        'MUTAMENTO',            -- Mutamento orientamento
        'ABBANDONO_INDIRIZZO'   -- Abbandono indirizzo precedente
    )),

    is_turning_point BOOLEAN DEFAULT TRUE,  -- Flag esplicito (v3.2.1)
    rationale_span TEXT,                    -- Frase che indica il turning point
    signal_pattern TEXT,                    -- Regex pattern che ha matchato
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    is_verified BOOLEAN DEFAULT FALSE,      -- Verificato manualmente

    run_id INTEGER REFERENCES kb.graph_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Un turning point per coppia massima->overruled per run
    CONSTRAINT uq_turning_points_dedup UNIQUE (massima_id, overruled_massima_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_turning_points_massima ON kb.turning_points(massima_id);
CREATE INDEX IF NOT EXISTS idx_turning_points_overruled ON kb.turning_points(overruled_massima_id);
CREATE INDEX IF NOT EXISTS idx_turning_points_type ON kb.turning_points(turning_point_type);
CREATE INDEX IF NOT EXISTS idx_turning_points_verified ON kb.turning_points(is_verified) WHERE is_verified = TRUE;

-- ============================================================
-- GOLDEN QUERIES (per QA evaluation)
-- Estensione per graph evaluation
-- ============================================================

-- Add column for expected graph neighbors if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'golden_queries'
        AND column_name = 'expected_neighbors'
    ) THEN
        ALTER TABLE kb.golden_queries ADD COLUMN expected_neighbors UUID[] DEFAULT '{}';
    END IF;
END $$;

-- ============================================================
-- RETRIEVAL LOGS EXTENSION
-- Add graph-related fields
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'retrieval_logs'
        AND column_name = 'graph_expanded_ids'
    ) THEN
        ALTER TABLE kb.retrieval_logs ADD COLUMN graph_expanded_ids UUID[] DEFAULT '{}';
        ALTER TABLE kb.retrieval_logs ADD COLUMN graph_hit_count INTEGER DEFAULT 0;
        ALTER TABLE kb.retrieval_logs ADD COLUMN graph_boost_applied BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- ============================================================
-- SEED CATEGORIES (6 Macro L1)
-- ============================================================

INSERT INTO kb.categories (id, name, level, is_active, keywords, description) VALUES
    ('CIVILE', 'Diritto Civile', 1, TRUE,
     ARRAY['civile', 'obbligazione', 'contratto', 'proprietà', 'famiglia', 'successione', 'c.c.'],
     'Diritto civile sostanziale'),
    ('PENALE', 'Diritto Penale', 1, TRUE,
     ARRAY['penale', 'reato', 'pena', 'imputato', 'condanna', 'c.p.'],
     'Diritto penale sostanziale'),
    ('PROCESSUALE_CIVILE', 'Procedura Civile', 1, TRUE,
     ARRAY['procedura civile', 'c.p.c.', 'giudizio', 'udienza', 'sentenza', 'appello', 'ricorso'],
     'Diritto processuale civile'),
    ('PROCESSUALE_PENALE', 'Procedura Penale', 1, TRUE,
     ARRAY['procedura penale', 'c.p.p.', 'processo penale', 'pm', 'gip', 'dibattimento'],
     'Diritto processuale penale'),
    ('AMMINISTRATIVO', 'Diritto Amministrativo', 1, TRUE,
     ARRAY['amministrativo', 'tar', 'consiglio di stato', 'pa', 'pubblica amministrazione'],
     'Diritto amministrativo'),
    ('TRIBUTARIO', 'Diritto Tributario', 1, TRUE,
     ARRAY['tributo', 'imposta', 'evasione', 'accertamento', 'fiscale', 'agenzia entrate'],
     'Diritto tributario e fiscale'),
    ('UNKNOWN', 'Non Classificato', 1, TRUE,
     ARRAY[]::TEXT[],
     'Massime non ancora classificate')
ON CONFLICT (id) DO UPDATE SET
    keywords = EXCLUDED.keywords,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Funzione per invalidare cache quando cambia run attivo
CREATE OR REPLACE FUNCTION kb.set_active_run(p_run_id INTEGER, p_run_type VARCHAR)
RETURNS VOID AS $$
BEGIN
    -- Disattiva tutti i run dello stesso tipo
    UPDATE kb.graph_runs
    SET is_active = FALSE
    WHERE run_type = p_run_type AND is_active = TRUE;

    -- Attiva il nuovo run
    UPDATE kb.graph_runs
    SET is_active = TRUE
    WHERE id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- Funzione per contare citation per norma (da chiamare dopo build)
CREATE OR REPLACE FUNCTION kb.update_norm_citation_counts()
RETURNS VOID AS $$
BEGIN
    UPDATE kb.norms n
    SET citation_count = (
        SELECT COUNT(DISTINCT massima_id)
        FROM kb.massima_norms mn
        WHERE mn.norm_id = n.id
    );
END;
$$ LANGUAGE plpgsql;

-- Funzione per ottenere neighbors (per cache)
CREATE OR REPLACE FUNCTION kb.get_neighbors(
    p_massima_id UUID,
    p_depth INTEGER DEFAULT 1,
    p_min_weight FLOAT DEFAULT 0.6
)
RETURNS TABLE (
    neighbor_id UUID,
    edge_type VARCHAR(30),
    relation_subtype VARCHAR(30),
    weight FLOAT,
    hop INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE neighbors AS (
        -- Base: direct neighbors
        SELECT
            e.target_id as neighbor_id,
            e.edge_type,
            e.relation_subtype,
            e.weight,
            1 as hop
        FROM kb.graph_edges e
        WHERE e.source_id = p_massima_id
        AND e.weight >= p_min_weight

        UNION

        -- Recursive: neighbors of neighbors (up to depth)
        SELECT
            e.target_id,
            e.edge_type,
            e.relation_subtype,
            e.weight,
            n.hop + 1
        FROM kb.graph_edges e
        JOIN neighbors n ON e.source_id = n.neighbor_id
        WHERE n.hop < p_depth
        AND e.weight >= p_min_weight
        AND e.target_id != p_massima_id  -- No cycles back to start
    )
    SELECT DISTINCT ON (n.neighbor_id)
        n.neighbor_id,
        n.edge_type,
        n.relation_subtype,
        n.weight,
        n.hop
    FROM neighbors n
    ORDER BY n.neighbor_id, n.weight DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- LOG
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'LEXE Knowledge Base - Graph Schema v3.2.1 initialized';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Tables created/updated:';
    RAISE NOTICE '  - kb.graph_runs (versioning + cache invalidation)';
    RAISE NOTICE '  - kb.graph_edges (with weight, evidence, dedup)';
    RAISE NOTICE '  - kb.categories (L1 macro, centroid support)';
    RAISE NOTICE '  - kb.category_assignments (multi-label)';
    RAISE NOTICE '  - kb.norms (canonicalized norm nodes)';
    RAISE NOTICE '  - kb.massima_norms (massima->norm edges)';
    RAISE NOTICE '  - kb.turning_points (subtype + flag)';
    RAISE NOTICE '';
    RAISE NOTICE 'Functions:';
    RAISE NOTICE '  - kb.set_active_run(run_id, run_type)';
    RAISE NOTICE '  - kb.update_norm_citation_counts()';
    RAISE NOTICE '  - kb.get_neighbors(massima_id, depth, min_weight)';
    RAISE NOTICE '';
    RAISE NOTICE 'Categories seeded: 6 L1 + UNKNOWN';
    RAISE NOTICE '============================================================';
END $$;
