"""
QA Protocol - Phase 4: Chunk Analysis

For each massima in the current batch, computes deterministic features:
- TOC infiltration score (0-1)
- Citation list score (0-1)
- Short/long flags, sentence count, quality score
- starts_with_legal_pattern flag

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    uv run python scripts/qa/s4_chunk_analysis.py
"""

import asyncio
import re

import asyncpg

from qa_config import DB_URL

# Citation patterns (from massima_extractor.py)
CITATION_PATTERN = re.compile(
    r"(?:Sez\.?|Sezione|Sezioni)\s*(?:U(?:nite)?|L(?:av)?|[0-9]+)",
    re.IGNORECASE,
)
CITATION_SIMPLE = re.compile(
    r"(?:Sez\.?|Sezione)\s*[UL0-9]+[,\s]+n\.?\s*[0-9]+",
    re.IGNORECASE,
)

# Legal pattern starts
LEGAL_STARTS = (
    "in tema di", "in materia di", "la sentenza", "l'ordinanza",
    "il principio", "ai fini", "nel caso", "qualora", "ove",
    "allorché", "allorche'",
)


def compute_toc_score(text: str) -> float:
    """TOC infiltration score (0-1)."""
    lines = text.split("\n") if "\n" in text else [text]
    total_lines = max(len(lines), 1)

    # Dotted lines
    dotted = len(re.findall(r"\.{3,}", text))
    dotted_ratio = min(dotted / total_lines, 1.0)

    # Lines ending with page number
    page_num_end = sum(
        1 for line in lines
        if re.search(r"\d{1,4}\s*$", line.strip()) and len(line.strip()) < 80
    )
    page_num_ratio = min(page_num_end / total_lines, 1.0)

    # Short lines ending with number
    short_num = sum(
        1 for line in lines
        if len(line.strip()) < 40 and re.search(r"\d{1,4}\s*$", line.strip())
    )
    short_num_ratio = min(short_num / total_lines, 1.0)

    # TOC keywords
    toc_kw = sum(
        1 for kw in ("indice", "sommario", "capitolo", "sezione")
        if kw in text.lower()
    )
    toc_kw_ratio = min(toc_kw / 4, 1.0)

    score = (
        dotted_ratio * 0.3
        + page_num_ratio * 0.3
        + short_num_ratio * 0.1
        + toc_kw_ratio * 0.3
    )
    return min(max(score, 0.0), 1.0)


def compute_citation_list_score(text: str) -> float:
    """Citation list score (0-1)."""
    if not text:
        return 0.0

    # Citation character ratio
    citations = CITATION_PATTERN.findall(text)
    citation_chars = sum(len(c) for c in citations)
    citation_ratio = citation_chars / len(text) if len(text) > 0 else 0

    # Sequential citations (comma-separated)
    seq_pattern = re.compile(
        r"(?:Sez\.?|Sezione)\s*[UL0-9]+[,\s]+n\.?\s*[0-9]+(?:\s*[,;]\s*(?:Sez\.?|n\.?)\s*[0-9]+){2,}",
        re.IGNORECASE,
    )
    sequential = len(seq_pattern.findall(text))
    seq_score = min(sequential / 3, 1.0)

    score = citation_ratio * 0.6 + seq_score * 0.4
    return min(max(score, 0.0), 1.0)


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r"[.!?]+\s", text)
    return len([s for s in sentences if len(s.strip()) > 10])


def compute_quality_score(text: str) -> float:
    """Quality score for valid legal text (0-1)."""
    if not text:
        return 0.0
    valid = sum(
        1 for c in text
        if c.isalnum() or c.isspace() or c in '.,;:!?()-"\''
    )
    return valid / len(text) if len(text) > 0 else 0.0


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 4: CHUNK ANALYSIS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )
    print(f"[OK] qa_run_id={qa_run_id}, batch_id={batch_id}")

    # Get all manifest → massime
    manifests = await conn.fetch(
        "SELECT id, doc_id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    total_chunks = 0
    total_short = 0
    total_long = 0
    total_toc = 0
    total_citation = 0

    for m in manifests:
        manifest_id = m["id"]
        doc_id = m["doc_id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT count(*) FROM kb.chunk_features WHERE manifest_id = $1 AND qa_run_id = $2",
            manifest_id, qa_run_id,
        )
        if existing > 0:
            continue

        # Get massime for this document
        massime = await conn.fetch(
            "SELECT id, testo FROM kb.massime WHERE document_id = $1 ORDER BY id",
            doc_id,
        )
        if not massime:
            continue

        doc_toc = 0
        doc_citation = 0

        for idx, row in enumerate(massime):
            testo = row["testo"] or ""
            massima_id = row["id"]

            char_count = len(testo)
            word_count = len(testo.split())
            sentence_count = count_sentences(testo)
            is_short = char_count < 150
            is_very_long = char_count > 2500

            toc_score = compute_toc_score(testo)
            cit_score = compute_citation_list_score(testo)
            quality = compute_quality_score(testo)

            has_mult_citations = len(CITATION_SIMPLE.findall(testo)) > 1
            starts_legal = any(
                testo.lower().startswith(p) for p in LEGAL_STARTS
            )

            await conn.execute(
                """
                INSERT INTO kb.chunk_features
                  (qa_run_id, manifest_id, ingest_batch_id, massima_id,
                   chunk_index, char_count, word_count, sentence_count,
                   is_short, is_very_long,
                   toc_infiltration_score, citation_list_score,
                   has_multiple_citations, starts_with_legal_pattern,
                   quality_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (manifest_id, ingest_batch_id, chunk_index) DO NOTHING
                """,
                qa_run_id, manifest_id, batch_id, massima_id,
                idx, char_count, word_count, sentence_count,
                is_short, is_very_long,
                round(toc_score, 4), round(cit_score, 4),
                has_mult_citations, starts_legal,
                round(quality, 4),
            )

            total_chunks += 1
            if is_short:
                total_short += 1
            if is_very_long:
                total_long += 1
            if toc_score > 0.6:
                total_toc += 1
                doc_toc += 1
            if cit_score > 0.7:
                total_citation += 1
                doc_citation += 1

        if doc_toc > 0 or doc_citation > 0:
            print(f"  [ALERT] {filename}: toc_like={doc_toc}, citation_like={doc_citation}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"CHUNK ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total chunks: {total_chunks}")
    print(f"Short (<150): {total_short} ({total_short/total_chunks*100:.1f}%)" if total_chunks > 0 else "Short: 0")
    print(f"Very long (>2500): {total_long}")
    print(f"TOC-like (score>0.6): {total_toc}")
    print(f"Citation-list-like (score>0.7): {total_citation}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
