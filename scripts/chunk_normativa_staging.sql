-- ============================================================================
-- chunk_normativa_staging.sql
-- Chunk all kb.normativa articles into kb.normativa_chunk
-- Run inside lexe-max container: psql -U lexe_kb -d lexe_kb -f /tmp/chunk.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. Create chunking function
-- ============================================================================
CREATE OR REPLACE FUNCTION kb.fn_chunk_text(
    p_text TEXT,
    p_target_chars INT DEFAULT 1000,
    p_overlap_chars INT DEFAULT 150,
    p_min_chunk_len INT DEFAULT 30
) RETURNS TABLE(
    chunk_no INT,
    char_start INT,
    char_end INT,
    chunk_text TEXT,
    token_est INT
) AS $$
DECLARE
    v_text TEXT;
    v_len INT;
    v_pos INT := 0;
    v_end_pos INT;
    v_chunk_no INT := 0;
    v_chunk TEXT;
    v_split_pos INT;
    v_search_start INT;
    v_search_end INT;
    v_search_text TEXT;
    v_found_pos INT;
BEGIN
    -- Normalize whitespace
    v_text := regexp_replace(p_text, E'\\xC2\\xA0', ' ', 'g');  -- NBSP
    v_text := regexp_replace(v_text, E'[ \\t]+', ' ', 'g');
    v_text := regexp_replace(v_text, E'\\n{3,}', E'\\n\\n', 'g');
    v_text := btrim(v_text);
    v_len := length(v_text);

    IF v_len < p_min_chunk_len THEN
        RETURN;
    END IF;

    WHILE v_pos < v_len LOOP
        v_end_pos := LEAST(v_pos + p_target_chars, v_len);

        -- Find smart split point if not at end
        IF v_end_pos < v_len THEN
            v_search_start := GREATEST(0, v_end_pos - 100);
            v_search_end := LEAST(v_len, v_end_pos + 100);
            v_search_text := substring(v_text FROM v_search_start + 1 FOR v_search_end - v_search_start);
            v_split_pos := NULL;

            -- Try paragraph break (double newline)
            SELECT max(s) INTO v_found_pos
            FROM regexp_matches(v_search_text, E'\\n\\n', 'g') AS m,
                 LATERAL (SELECT position(m[1] IN v_search_text)) AS s(s)
            WHERE FALSE;  -- skip complex regex, use simple approach

            -- Simple approach: scan backwards from end for split points
            -- Try ". " (sentence end)
            v_found_pos := NULL;
            FOR i IN REVERSE length(v_search_text)..1 LOOP
                IF substring(v_search_text FROM i FOR 2) = '. ' THEN
                    v_found_pos := v_search_start + i + 1;  -- after ". "
                    EXIT;
                END IF;
            END LOOP;

            IF v_found_pos IS NOT NULL AND v_found_pos > v_pos THEN
                v_split_pos := v_found_pos;
            END IF;

            -- Try ", " (comma)
            IF v_split_pos IS NULL THEN
                FOR i IN REVERSE length(v_search_text)..1 LOOP
                    IF substring(v_search_text FROM i FOR 2) = ', ' THEN
                        v_found_pos := v_search_start + i + 1;
                        EXIT;
                    END IF;
                END LOOP;
                IF v_found_pos IS NOT NULL AND v_found_pos > v_pos THEN
                    v_split_pos := v_found_pos;
                END IF;
            END IF;

            -- Try space
            IF v_split_pos IS NULL THEN
                FOR i IN REVERSE length(v_search_text)..1 LOOP
                    IF substring(v_search_text FROM i FOR 1) = ' ' THEN
                        v_found_pos := v_search_start + i;
                        EXIT;
                    END IF;
                END LOOP;
                IF v_found_pos IS NOT NULL AND v_found_pos > v_pos THEN
                    v_split_pos := v_found_pos;
                END IF;
            END IF;

            IF v_split_pos IS NOT NULL THEN
                v_end_pos := v_split_pos;
            END IF;
        END IF;

        v_chunk := btrim(substring(v_text FROM v_pos + 1 FOR v_end_pos - v_pos));

        IF length(v_chunk) >= p_min_chunk_len THEN
            chunk_no := v_chunk_no;
            char_start := v_pos;
            char_end := v_end_pos;
            chunk_text := v_chunk;
            token_est := GREATEST(1, length(v_chunk) / 4);
            RETURN NEXT;
            v_chunk_no := v_chunk_no + 1;
        END IF;

        IF v_end_pos >= v_len THEN
            EXIT;
        END IF;
        v_pos := GREATEST(v_pos + 1, v_end_pos - p_overlap_chars);
    END LOOP;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- 2. Clear existing chunks (idempotent re-run)
-- ============================================================================
TRUNCATE kb.normativa_chunk CASCADE;

-- ============================================================================
-- 3. Insert chunks for all normativa articles with work_id
-- ============================================================================
INSERT INTO kb.normativa_chunk (
    normativa_id, work_id, articolo_sort_key,
    articolo_num, articolo_suffix,
    chunk_no, char_start, char_end, text, token_est
)
SELECT
    n.id,
    n.work_id,
    COALESCE(n.articolo_sort_key, ''),
    n.articolo_num,
    n.articolo_suffix,
    c.chunk_no,
    c.char_start,
    c.char_end,
    c.chunk_text,
    c.token_est
FROM kb.normativa n
CROSS JOIN LATERAL kb.fn_chunk_text(n.testo) c
WHERE n.work_id IS NOT NULL
  AND n.testo IS NOT NULL
  AND length(btrim(n.testo)) >= 30;

-- ============================================================================
-- 4. Verify FTS trigger populated (backfill if needed)
-- ============================================================================
INSERT INTO kb.normativa_chunk_fts(chunk_id, tsv_it)
SELECT id, to_tsvector('italian', unaccent(btrim(text)))
FROM kb.normativa_chunk
ON CONFLICT (chunk_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;

-- ============================================================================
-- 5. Report
-- ============================================================================
SELECT '=== CHUNK RESULTS ===' as info;

SELECT
    w.code,
    count(DISTINCT n.id) as articles,
    count(nc.id) as chunks,
    round(avg(nc.token_est)) as avg_tokens,
    round(100.0 * count(DISTINCT n.id) / NULLIF((SELECT count(*) FROM kb.normativa WHERE work_id = w.id), 0), 1) as coverage_pct
FROM kb.work w
LEFT JOIN kb.normativa n ON n.work_id = w.id AND n.testo IS NOT NULL AND length(btrim(n.testo)) >= 30
LEFT JOIN kb.normativa_chunk nc ON nc.normativa_id = n.id
GROUP BY w.code, w.id
ORDER BY w.code;

SELECT '=== TOTALS ===' as info;
SELECT count(*) as total_chunks FROM kb.normativa_chunk;
SELECT count(*) as total_fts FROM kb.normativa_chunk_fts;

COMMIT;
