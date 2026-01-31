-- ============================================================
-- Migration 003: Cut Decisions Audit Table
-- ============================================================
-- Traccia tutte le decisioni di taglio chunk per audit e debug.
-- Supporta sia tagli deterministici che LLM-validated.
-- ============================================================

CREATE TABLE IF NOT EXISTS kb.cut_decisions (
    id BIGSERIAL PRIMARY KEY,
    qa_run_id INTEGER REFERENCES kb.qa_runs(id),
    ingest_batch_id INTEGER,
    manifest_id INTEGER REFERENCES kb.pdf_manifest(id),
    chunk_temp_id TEXT,               -- id provvisorio: "page:anchor_idx"
    page_number INTEGER,

    -- Metodo e trigger
    method TEXT NOT NULL,             -- 'deterministic' | 'llm_validated' | 'llm_skipped_low_conf'
    trigger_type TEXT,                -- 'forced_cut' | 'suspicious_end' | 'ambiguous_candidates' | NULL

    -- Parametri usati
    soft_cap INTEGER NOT NULL,
    hard_cap INTEGER NOT NULL,

    -- Risultato taglio
    original_len INTEGER NOT NULL,
    chosen_cut_offset INTEGER NOT NULL,
    chosen_candidate_index INTEGER,
    forced_cut BOOLEAN DEFAULT FALSE,

    -- Candidati e contesto
    candidates_json JSONB,            -- lista candidati con offset + reason
    snippet_json JSONB,               -- before/after window per debug

    -- LLM (se usato)
    llm_model TEXT,
    llm_confidence FLOAT,
    llm_response JSONB,
    latency_ms INTEGER,
    cost_usd FLOAT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indici per query comuni
CREATE INDEX IF NOT EXISTS idx_cut_decisions_manifest_batch
    ON kb.cut_decisions(manifest_id, ingest_batch_id);

CREATE INDEX IF NOT EXISTS idx_cut_decisions_trigger
    ON kb.cut_decisions(trigger_type);

CREATE INDEX IF NOT EXISTS idx_cut_decisions_method
    ON kb.cut_decisions(method);

CREATE INDEX IF NOT EXISTS idx_cut_decisions_forced
    ON kb.cut_decisions(forced_cut)
    WHERE forced_cut = TRUE;

-- Commenti
COMMENT ON TABLE kb.cut_decisions IS 'Audit log per decisioni di taglio chunk citation-anchored';
COMMENT ON COLUMN kb.cut_decisions.method IS 'deterministic | llm_validated | llm_skipped_low_conf';
COMMENT ON COLUMN kb.cut_decisions.trigger_type IS 'Cosa ha triggerato LLM: forced_cut | suspicious_end | ambiguous_candidates';
COMMENT ON COLUMN kb.cut_decisions.candidates_json IS 'Array di {offset, kind, reason, preview}';
COMMENT ON COLUMN kb.cut_decisions.snippet_json IS '{before: str, after: str, cut_preview: str}';
