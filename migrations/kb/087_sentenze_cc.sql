-- Migration 087: Corte Costituzionale decisions (giurcost.org crawler)
-- Sprint 27 P6 Day 2 — deposits sentenze CC + embeddings + parallel graph edges.
--
-- Design notes:
-- 1. vector(1536) to match kb.normativa_chunk_embeddings / kb.embeddings
--    convention (text-embedding-3-small) — cross-domain retrieval requires
--    same embedding space. The T7 prompt requested 1024 but production KB
--    uses 1536, so we align with prod for consistency.
-- 2. kb.sentenze_cc_edges is a PARALLEL graph-edge table instead of reusing
--    kb.graph_edges because the latter has hard FK CASCADE to kb.massime(id)
--    on both source_id and target_id. Reusing it would require dropping that
--    FK — out of scope for Sprint 27. Same column shape for future merge.
-- 3. UNIQUE(tipo, numero, anno) drives idempotent UPSERT from NDJSON ingest.

BEGIN;

CREATE TABLE IF NOT EXISTS kb.sentenze_cc (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url      TEXT NOT NULL,
    filename_slug   TEXT NOT NULL,
    tipo            TEXT NOT NULL CHECK (tipo IN ('sentenza','ordinanza','decreto')),
    numero          INTEGER NOT NULL,
    anno            INTEGER NOT NULL,
    presidente      TEXT,
    relatore        TEXT,
    data_udienza    DATE,
    data_deposito   DATE,
    fascicolo       TEXT,
    dispositivo     TEXT,
    testo_integrale TEXT,
    external_refs   JSONB NOT NULL DEFAULT '[]'::jsonb,
    parse_warnings  JSONB NOT NULL DEFAULT '[]'::jsonb,
    crawled_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_sentenze_cc_tipo_num_anno UNIQUE (tipo, numero, anno),
    CONSTRAINT uq_sentenze_cc_slug UNIQUE (filename_slug)
);

CREATE INDEX IF NOT EXISTS idx_sentenze_cc_anno_numero
    ON kb.sentenze_cc(anno DESC, numero DESC);
CREATE INDEX IF NOT EXISTS idx_sentenze_cc_tipo
    ON kb.sentenze_cc(tipo);
CREATE INDEX IF NOT EXISTS idx_sentenze_cc_deposito
    ON kb.sentenze_cc(data_deposito DESC) WHERE data_deposito IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sentenze_cc_external_refs
    ON kb.sentenze_cc USING gin (external_refs);

CREATE TABLE IF NOT EXISTS kb.sentenze_cc_embedding (
    sentenza_id  UUID NOT NULL REFERENCES kb.sentenze_cc(id) ON DELETE CASCADE,
    chunk_idx    INTEGER NOT NULL,
    testo_chunk  TEXT NOT NULL,
    embedding    vector(1536) NOT NULL,
    model_name   TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (sentenza_id, chunk_idx, model_name)
);

CREATE INDEX IF NOT EXISTS idx_sentenze_cc_embedding_hnsw
    ON kb.sentenze_cc_embedding USING hnsw (embedding vector_cosine_ops);

-- Parallel graph table: sentenza_cc -> normativa OR sentenza_cc -> sentenza_cc.
-- Same column shape as kb.graph_edges so we can unify later (Sprint 28+).
CREATE TABLE IF NOT EXISTS kb.sentenze_cc_edges (
    id                SERIAL PRIMARY KEY,
    source_id         UUID NOT NULL REFERENCES kb.sentenze_cc(id) ON DELETE CASCADE,
    target_kind       TEXT NOT NULL CHECK (target_kind IN ('sentenza_cc','norm','url')),
    target_id         UUID,
    target_raw        TEXT NOT NULL,
    target_url        TEXT,
    edge_type         VARCHAR(30) NOT NULL,
    relation_subtype  VARCHAR(30) NOT NULL DEFAULT 'unspecified',
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    weight            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    evidence          JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_span      TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sentenze_cc_edges_dedup
    ON kb.sentenze_cc_edges (source_id, target_kind, target_raw, edge_type, relation_subtype);

CREATE INDEX IF NOT EXISTS idx_sentenze_cc_edges_source
    ON kb.sentenze_cc_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_sentenze_cc_edges_target_raw
    ON kb.sentenze_cc_edges(target_raw);
CREATE INDEX IF NOT EXISTS idx_sentenze_cc_edges_target_id
    ON kb.sentenze_cc_edges(target_id) WHERE target_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sentenze_cc_edges_type
    ON kb.sentenze_cc_edges(edge_type);

COMMIT;
