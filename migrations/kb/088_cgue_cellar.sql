-- Migration 088: CGUE sentences + citation graph (Sprint 30 P1.1)
--
-- Source: EUR-Lex CELLAR SPARQL + REST (https://publications.europa.eu)
-- CELEX filter: 6*CJ* (Court of Justice judgments)
-- Scope: CGUE judgments 2020-2026, GDPR / biometria / profilazione /
--        trasferimenti extra-UE (~3-5K sentences estimated).
--
-- Design notes:
-- 1. `celex` is the natural primary key (globally unique, stable identifier
--    assigned by Publications Office). Upserts are idempotent on this key.
-- 2. `kb.cgue_citations` is a PARALLEL graph-edge table intentionally kept
--    separate from `kb.graph_edges` (which has FK CASCADE to kb.massime(id))
--    and from `kb.sentenze_cc_edges` (which is rooted in kb.sentenze_cc(id)).
--    Same column shape as the other edge tables so Sprint 31+ can unify.
-- 3. Schema only for `kb.cgue_citations`: Sprint 30 P1.1 does NOT populate
--    edges; extraction is P1.5 (citation graph builder).
-- 4. `data` is the judgment date (date of delivery). Separate `ingested_at`
--    tracks freshness for nightly sync SLA reporting.

BEGIN;

CREATE TABLE IF NOT EXISTS kb.cgue_sentences (
    celex           TEXT PRIMARY KEY,
    ecli            TEXT,
    data            DATE,
    parti           TEXT,
    testo_integrale TEXT,
    massime         TEXT,
    lingua          TEXT NOT NULL DEFAULT 'it',
    source_url      TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv_italian     tsvector,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Populate tsvector from testo_integrale + massime + parti
UPDATE kb.cgue_sentences
SET tsv_italian = to_tsvector('italian',
    COALESCE(testo_integrale, '') || ' ' ||
    COALESCE(massime, '') || ' ' ||
    COALESCE(parti, '')
)
WHERE tsv_italian IS NULL;

CREATE INDEX IF NOT EXISTS idx_cgue_sentences_tsv
    ON kb.cgue_sentences USING GIN (tsv_italian);
CREATE INDEX IF NOT EXISTS idx_cgue_sentences_data
    ON kb.cgue_sentences(data DESC) WHERE data IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cgue_sentences_ecli
    ON kb.cgue_sentences(ecli) WHERE ecli IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cgue_sentences_ingested
    ON kb.cgue_sentences(ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_cgue_sentences_metadata
    ON kb.cgue_sentences USING gin (metadata);

-- ---------------------------------------------------------------------------
-- Parallel citation-edge table (schema only — populated by P1.5 graph builder)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb.cgue_citations (
    id                SERIAL PRIMARY KEY,
    source_celex      TEXT NOT NULL REFERENCES kb.cgue_sentences(celex) ON DELETE CASCADE,
    target_kind       TEXT NOT NULL CHECK (target_kind IN ('cgue','norm','ecli','url','celex_ext')),
    target_celex      TEXT,
    target_ecli       TEXT,
    target_raw        TEXT NOT NULL,
    target_url        TEXT,
    edge_type         VARCHAR(40) NOT NULL,
    relation_subtype  VARCHAR(40) NOT NULL DEFAULT 'unspecified',
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    weight            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    evidence          JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_span      TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_cgue_citations_dedup
    ON kb.cgue_citations (source_celex, target_kind, target_raw, edge_type, relation_subtype);

CREATE INDEX IF NOT EXISTS idx_cgue_citations_source
    ON kb.cgue_citations(source_celex);
CREATE INDEX IF NOT EXISTS idx_cgue_citations_target_celex
    ON kb.cgue_citations(target_celex) WHERE target_celex IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cgue_citations_target_ecli
    ON kb.cgue_citations(target_ecli) WHERE target_ecli IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cgue_citations_edge_type
    ON kb.cgue_citations(edge_type);

COMMIT;
