#!/usr/bin/env python
"""
Import ingested JSON to kb.normativa on STAGING.

Uses SSH tunnel on port 5437 -> staging:5436.

Usage:
    uv run python scripts/import_to_staging.py --all
    uv run python scripts/import_to_staging.py --doc CC --dry-run
"""

import asyncio
import json
import argparse
import platform
from pathlib import Path

# Windows event loop fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncpg

# Database connection - STAGING via SSH tunnel
DB_HOST = "localhost"
DB_PORT = 5437
DB_USER = "lexe_max"
DB_PASS = "lexe_max_dev_password"
DB_NAME = "lexe_max"

INGESTED_ROOT = Path(__file__).parent.parent / "data" / "ingested"

# Latin suffix order for sort_key
SUFFIX_ORDER = {
    None: 0, "": 0,
    "bis": 1, "ter": 2, "quater": 3, "quinquies": 4,
    "sexies": 5, "septies": 6, "octies": 7, "novies": 8,
    "decies": 9, "undecies": 10, "duodecies": 11, "terdecies": 12,
    "quaterdecies": 13, "quinquiesdecies": 14, "sexiesdecies": 15,
    "septiesdecies": 16, "octiesdecies": 17,
}


def compute_sort_key(articolo_num: int | None, suffix: str | None) -> str:
    """Compute sort key for article ordering."""
    num_part = articolo_num or 0
    suffix_order = SUFFIX_ORDER.get(suffix, 99)
    return f"{num_part:06d}.{suffix_order:02d}"


def determine_identity_class(articolo: str, suffix: str | None) -> str:
    """Determine identity class for article."""
    if suffix:
        return "SUFFIX"
    if any(x in articolo.lower() for x in ["preleggi", "disp.", "transitorie", "allegato"]):
        return "SPECIAL"
    return "BASE"


async def get_connection():
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        ssl=False
    )


async def get_work_id(conn, code: str):
    """Get work_id from kb.work by code."""
    row = await conn.fetchrow("SELECT id FROM kb.work WHERE code = $1", code)
    return row["id"] if row else None


async def import_document(conn, doc_code: str, dry_run: bool = False) -> dict:
    """Import single document from ingested JSON."""
    json_path = INGESTED_ROOT / doc_code / "ingested.json"

    if not json_path.exists():
        print(f"  SKIP: {json_path} not found")
        return {"code": doc_code, "status": "not_found", "imported": 0}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    work_id = await get_work_id(conn, doc_code)
    if not work_id:
        print(f"  ERROR: work '{doc_code}' not found in kb.work")
        return {"code": doc_code, "status": "work_not_found", "imported": 0}

    articles = data.get("articles", [])
    imported = 0
    skipped = 0
    errors = 0

    for art in articles:
        # Get text
        text_final = art.get("text_final", "") or art.get("text", "")
        if not text_final or len(text_final.strip()) < 10:
            skipped += 1
            continue

        # Build article identifier
        base_num = art.get("base_num")
        suffix = art.get("suffix")

        if base_num:
            articolo = str(base_num)
            if suffix:
                articolo = f"{articolo}-{suffix}"
        else:
            articolo = art.get("articolo", str(art.get("article_num", "?")))

        articolo_num = base_num or art.get("article_num")

        # Quality mapping
        quality_raw = art.get("quality_class", "VALID_STRONG")
        quality_map = {
            "VALID": "VALID_STRONG",
            "VALID_STRONG": "VALID_STRONG",
            "VALID_SHORT": "VALID_SHORT",
            "WEAK": "WEAK",
            "EMPTY": "EMPTY",
            "INVALID": "INVALID",
        }
        quality = quality_map.get(quality_raw, "VALID_STRONG")

        identity_class = determine_identity_class(articolo, suffix)
        sort_key = compute_sort_key(articolo_num, suffix)
        rubrica = art.get("rubrica_final") or art.get("rubrica")

        if dry_run:
            imported += 1
            continue

        try:
            await conn.execute("""
                INSERT INTO kb.normativa (
                    work_id, articolo, articolo_num, articolo_suffix,
                    identity_class, quality, lifecycle, articolo_sort_key,
                    rubrica, testo, is_current
                ) VALUES (
                    $1, $2, $3, $4,
                    $5::kb.article_identity_class, $6::kb.article_quality_class, 'UNKNOWN'::kb.lifecycle_status, $7,
                    $8, $9, TRUE
                )
                ON CONFLICT (work_id, articolo_sort_key) DO UPDATE SET
                    testo = EXCLUDED.testo,
                    rubrica = EXCLUDED.rubrica,
                    quality = EXCLUDED.quality,
                    articolo = EXCLUDED.articolo,
                    updated_at = now()
            """, work_id, articolo, articolo_num, suffix,
                identity_class, quality, sort_key,
                rubrica, text_final)
            imported += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    ERROR {doc_code}:{articolo}: {e}")

    status = "dry_run" if dry_run else "imported"
    print(f"  {doc_code}: {imported} imported, {skipped} skipped, {errors} errors")
    return {"code": doc_code, "status": status, "imported": imported, "skipped": skipped, "errors": errors}


async def main():
    parser = argparse.ArgumentParser(description="Import ingested JSON to kb.normativa (STAGING)")
    parser.add_argument("--doc", help="Document code (e.g., CC, COST)")
    parser.add_argument("--all", action="store_true", help="Import all available documents")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually insert")
    args = parser.parse_args()

    if not args.doc and not args.all:
        parser.error("Either --doc or --all is required")

    print("Connecting to STAGING via SSH tunnel (localhost:5437)...")
    conn = await get_connection()
    print("Connected!\n")

    try:
        if args.all:
            docs = sorted([d.name for d in INGESTED_ROOT.iterdir()
                          if d.is_dir() and (d / "ingested.json").exists()])
        else:
            docs = [args.doc]

        print(f"=== IMPORT TO STAGING ===")
        print(f"Documents: {len(docs)}")
        print(f"Dry run: {args.dry_run}\n")

        results = []
        for doc in docs:
            result = await import_document(conn, doc, args.dry_run)
            results.append(result)

        # Summary
        total_imported = sum(r.get("imported", 0) for r in results)
        total_skipped = sum(r.get("skipped", 0) for r in results)
        total_errors = sum(r.get("errors", 0) for r in results)

        print(f"\n=== SUMMARY ===")
        print(f"Total imported: {total_imported}")
        print(f"Total skipped: {total_skipped}")
        print(f"Total errors: {total_errors}")

        # Verify in DB
        if not args.dry_run:
            count = await conn.fetchval("SELECT COUNT(*) FROM kb.normativa")
            print(f"Total in DB: {count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
