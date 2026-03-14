-- Migration 080: Concept hierarchy for Italian legal code structure
-- Inspired by H-CMR (hierarchical concept reasoning) and SAT-Graph RAG
--
-- Adds hierarchical structure (Codice→Libro→Titolo→Capo→Sezione→Articolo)
-- for better retrieval via "hierarchical expansion".

-- Add concept_path to normativa (e.g., ["CC", "Libro IV", "Titolo IX", "Capo I"])
ALTER TABLE kb.normativa ADD COLUMN IF NOT EXISTS concept_path TEXT[];
ALTER TABLE kb.normativa ADD COLUMN IF NOT EXISTS parent_abrogated BOOLEAN DEFAULT FALSE;

-- Hierarchical structure table
CREATE TABLE IF NOT EXISTS kb.normativa_structure (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_id UUID NOT NULL,
    level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 6),
    label TEXT NOT NULL,
    parent_id UUID REFERENCES kb.normativa_structure(id),
    sort_key TEXT NOT NULL,
    articolo_range_start TEXT,
    articolo_range_end TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(work_id, level, sort_key)
);

-- Link articles to structure nodes
CREATE TABLE IF NOT EXISTS kb.normativa_structure_link (
    normativa_id UUID NOT NULL,
    structure_id UUID NOT NULL REFERENCES kb.normativa_structure(id) ON DELETE CASCADE,
    PRIMARY KEY(normativa_id, structure_id)
);

-- Indexes for efficient hierarchy traversal
CREATE INDEX IF NOT EXISTS idx_structure_work_level
    ON kb.normativa_structure(work_id, level);
CREATE INDEX IF NOT EXISTS idx_structure_parent
    ON kb.normativa_structure(parent_id);
CREATE INDEX IF NOT EXISTS idx_normativa_concept_path
    ON kb.normativa USING gin(concept_path);
