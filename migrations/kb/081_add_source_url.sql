-- Migration 081: Add source_url to kb.massime for persisting validated URLs
--
-- When SearXNG resolves a massima URL and it passes HTTP HEAD + LLM validation,
-- the URL is written back here. Next time the massima is returned by kb_search,
-- source_url is included — skipping SearXNG entirely.
--
-- 2026-03-15

ALTER TABLE kb.massime ADD COLUMN IF NOT EXISTS source_url TEXT;

-- Index for quick lookup of massime without source_url (for batch re-resolve)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_massime_source_url_null
    ON kb.massime (id)
    WHERE source_url IS NULL AND is_active = true;
