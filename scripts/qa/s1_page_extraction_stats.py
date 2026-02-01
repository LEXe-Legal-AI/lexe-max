"""
QA Protocol - Phase 1: Page Extraction Stats

Per-page statistics: char_count, word_count, line_count, element_count,
valid_chars_ratio, italian_tokens_ratio, non_alnum_ratio.
Flags: is_empty, is_ocr_candidate, is_toc_candidate.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s1_page_extraction_stats.py
"""

import asyncio
import re
from pathlib import Path

import asyncpg
import httpx

# ── Config ────────────────────────────────────────────────────────
from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL

# Italian stopwords for ratio calculation
ITALIAN_STOPWORDS = {
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "e", "che", "non", "è", "sono", "del", "della", "dei", "delle",
    "al", "alla", "ai", "alle", "dal", "dalla", "dai", "dalle",
    "nel", "nella", "nei", "nelle", "sul", "sulla", "sui", "sulle",
    "come", "quando", "dove", "se", "anche", "più", "ma", "però",
    "quindi", "così", "sia", "o", "ed", "cui", "quale", "quali",
}

# TOC detection keywords
TOC_KEYWORDS = {"indice", "sommario", "capitolo", "sezione", "parte"}


def compute_page_stats(elements: list[dict], page: int) -> dict:
    """Compute stats for a single page from its elements."""
    page_elems = [
        e for e in elements
        if e.get("metadata", {}).get("page_number") == page
    ]

    all_text = " ".join(e.get("text", "") for e in page_elems if e.get("text"))
    lines = all_text.split("\n") if all_text else []

    char_count = len(all_text)
    words = re.findall(r"\b[a-zA-ZàèéìòùÀÈÉÌÒÙ]+\b", all_text.lower())
    word_count = len(words)
    line_count = len(lines)
    element_count = len(page_elems)

    # Valid chars ratio
    if char_count > 0:
        valid = sum(
            1 for c in all_text
            if c.isalnum() or c.isspace() or c in '.,;:!?()-"\''
        )
        valid_chars_ratio = valid / char_count
    else:
        valid_chars_ratio = 0.0

    # Italian tokens ratio
    if words:
        italian = sum(1 for w in words if w in ITALIAN_STOPWORDS or len(w) > 2)
        italian_tokens_ratio = italian / len(words)
    else:
        italian_tokens_ratio = 0.0

    # Non-alphanumeric ratio
    if char_count > 0:
        non_alnum = sum(1 for c in all_text if not c.isalnum() and not c.isspace())
        non_alnum_ratio = non_alnum / char_count
    else:
        non_alnum_ratio = 0.0

    # Flags
    is_empty = char_count < 10
    is_ocr_candidate = valid_chars_ratio < 0.7 and char_count > 20

    # TOC candidate heuristic
    is_toc_candidate = False
    if char_count > 20:
        dotted_lines = len(re.findall(r"\.{3,}", all_text))
        short_num_lines = sum(
            1 for line in lines
            if len(line.strip()) < 60 and re.search(r"\d{1,4}\s*$", line.strip())
        )
        toc_kw = sum(1 for kw in TOC_KEYWORDS if kw in all_text.lower())
        if (dotted_lines >= 3 or short_num_lines >= 5 or
                (toc_kw >= 2 and short_num_lines >= 2)):
            is_toc_candidate = True

    # Category flags
    categories = {e.get("type", "") for e in page_elems}
    has_narrative = "NarrativeText" in categories
    has_title = "Title" in categories
    has_table = "Table" in categories

    return {
        "page_number": page,
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "element_count": element_count,
        "has_narrative_text": has_narrative,
        "has_title": has_title,
        "has_table": has_table,
        "is_empty": is_empty,
        "is_ocr_candidate": is_ocr_candidate,
        "is_toc_candidate": is_toc_candidate,
        "valid_chars_ratio": round(valid_chars_ratio, 4),
        "italian_tokens_ratio": round(italian_tokens_ratio, 4),
        "non_alnum_ratio": round(non_alnum_ratio, 4),
    }


async def extract_fast(client: httpx.AsyncClient, pdf_path: Path) -> list[dict]:
    """Extract with Unstructured fast strategy."""
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = await client.post(
            UNSTRUCTURED_URL,
            files=files,
            data={"strategy": "fast", "output_format": "application/json"},
            timeout=300.0,
        )
    if response.status_code != 200:
        return []
    return response.json()


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 1: PAGE EXTRACTION STATS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] Using qa_run_id={qa_run_id}")

    manifests = await conn.fetch(
        "SELECT id, filename, pages FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    total_pages = 0
    total_empty = 0
    total_ocr = 0
    total_toc = 0

    async with httpx.AsyncClient() as client:
        for m in manifests:
            manifest_id = m["id"]
            filename = m["filename"]
            expected_pages = m["pages"]

            # Check if already done
            existing = await conn.fetchval(
                "SELECT count(*) FROM kb.page_extraction_stats WHERE manifest_id = $1",
                manifest_id,
            )
            if existing > 0:
                print(f"  [SKIP] {filename}: {existing} page stats exist")
                continue

            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                pdf_path = PDF_DIR / "new" / filename
            if not pdf_path.exists():
                print(f"  [WARN] PDF not found: {filename}")
                continue

            print(f"\n  [{filename}] extracting...")
            elements = await extract_fast(client, pdf_path)
            if not elements:
                print(f"  [WARN] No elements: {filename}")
                continue

            # Get all page numbers
            pages = set()
            for e in elements:
                p = e.get("metadata", {}).get("page_number")
                if p:
                    pages.add(p)

            # Add missing pages (empty)
            for p in range(1, expected_pages + 1):
                pages.add(p)

            print(f"  {len(elements)} elements, {len(pages)} pages")

            for page in sorted(pages):
                stats = compute_page_stats(elements, page)

                await conn.execute(
                    """
                    INSERT INTO kb.page_extraction_stats
                      (qa_run_id, manifest_id, page_number,
                       char_count, word_count, line_count, element_count,
                       has_narrative_text, has_title, has_table,
                       is_empty, is_ocr_candidate, is_toc_candidate,
                       valid_chars_ratio, italian_tokens_ratio, non_alnum_ratio)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (manifest_id, page_number) DO NOTHING
                    """,
                    qa_run_id, manifest_id, stats["page_number"],
                    stats["char_count"], stats["word_count"],
                    stats["line_count"], stats["element_count"],
                    stats["has_narrative_text"], stats["has_title"], stats["has_table"],
                    stats["is_empty"], stats["is_ocr_candidate"], stats["is_toc_candidate"],
                    stats["valid_chars_ratio"], stats["italian_tokens_ratio"],
                    stats["non_alnum_ratio"],
                )

                total_pages += 1
                if stats["is_empty"]:
                    total_empty += 1
                if stats["is_ocr_candidate"]:
                    total_ocr += 1
                if stats["is_toc_candidate"]:
                    total_toc += 1

            print(f"  [OK] {filename}: {len(pages)} page stats inserted")

    # Verify
    stats_count = await conn.fetchval("SELECT count(*) FROM kb.page_extraction_stats")
    docs_covered = await conn.fetchval(
        "SELECT count(DISTINCT manifest_id) FROM kb.page_extraction_stats"
    )

    print(f"\n{'=' * 70}")
    print(f"PAGE STATS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total page stats: {stats_count}")
    print(f"Documents covered: {docs_covered}")
    print(f"Empty pages: {total_empty}")
    print(f"OCR candidates: {total_ocr}")
    print(f"TOC candidates: {total_toc}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
