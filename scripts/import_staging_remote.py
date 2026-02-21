#!/usr/bin/env python3
"""
Import ingested JSON to kb.normativa on STAGING — runs directly on server.
Uses psycopg2 (no asyncpg needed). Connects to localhost:5436.

Usage (on staging server):
    python3 /tmp/import_staging_remote.py --all
    python3 /tmp/import_staging_remote.py --doc CC --dry-run
"""

import json
import argparse
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

# Database connection - STAGING localhost
DB_HOST = "localhost"
DB_PORT = 5436
DB_USER = "lexe_kb"
DB_PASS = "lexe_kb_secret"
DB_NAME = "lexe_kb"

INGESTED_ROOT = Path("/tmp/ingested")

# Latin suffix order for sort_key
SUFFIX_ORDER = {
    None: 0, "": 0,
    "bis": 1, "ter": 2, "quater": 3, "quinquies": 4,
    "sexies": 5, "septies": 6, "octies": 7, "novies": 8,
    "decies": 9, "undecies": 10, "duodecies": 11, "terdecies": 12,
    "quaterdecies": 13, "quinquiesdecies": 14, "sexiesdecies": 15,
    "septiesdecies": 16, "octiesdecies": 17,
}


def compute_sort_key(articolo_num, suffix):
    num_part = articolo_num or 0
    suffix_order = SUFFIX_ORDER.get(suffix, 99)
    return f"{num_part:06d}.{suffix_order:02d}"


def determine_identity_class(articolo, suffix):
    if suffix:
        return "SUFFIX"
    if any(x in articolo.lower() for x in ["preleggi", "disp.", "transitorie", "allegato"]):
        return "SPECIAL"
    return "BASE"


def get_work_id(cur, code):
    cur.execute("SELECT id FROM kb.work WHERE code = %s", (code,))
    row = cur.fetchone()
    return row[0] if row else None


def import_document(cur, doc_code, dry_run=False):
    json_path = INGESTED_ROOT / doc_code / "ingested.json"

    if not json_path.exists():
        print(f"  SKIP: {json_path} not found")
        return {"code": doc_code, "status": "not_found", "imported": 0, "skipped": 0, "errors": 0}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    work_id = get_work_id(cur, doc_code)
    if not work_id:
        print(f"  SKIP: work '{doc_code}' not found in kb.work")
        return {"code": doc_code, "status": "work_not_found", "imported": 0, "skipped": 0, "errors": 0}

    articles = data.get("articles", [])
    imported = 0
    skipped = 0
    errors = 0

    for art in articles:
        text_final = art.get("text_final", "") or art.get("text", "")
        if not text_final or len(text_final.strip()) < 10:
            skipped += 1
            continue

        base_num = art.get("base_num")
        suffix = art.get("suffix")

        if base_num:
            articolo = str(base_num)
            if suffix:
                articolo = f"{articolo}-{suffix}"
        else:
            articolo = art.get("articolo", str(art.get("article_num", "?")))

        articolo_num = base_num or art.get("article_num")

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
            cur.execute("""
                INSERT INTO kb.normativa (
                    codice, work_id, articolo, articolo_num, articolo_suffix,
                    identity_class, quality, lifecycle, articolo_sort_key,
                    rubrica, testo, is_current
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s::kb.article_identity_class, %s::kb.article_quality_class,
                    'UNKNOWN'::kb.lifecycle_status, %s,
                    %s, %s, TRUE
                )
                ON CONFLICT (work_id, articolo_sort_key) DO UPDATE SET
                    testo = EXCLUDED.testo,
                    rubrica = EXCLUDED.rubrica,
                    quality = EXCLUDED.quality,
                    articolo = EXCLUDED.articolo,
                    codice = EXCLUDED.codice,
                    updated_at = now()
            """, (doc_code, str(work_id), articolo, articolo_num, suffix,
                  identity_class, quality, sort_key,
                  rubrica, text_final))
            imported += 1
        except Exception as e:
            conn = cur.connection
            conn.rollback()
            errors += 1
            if errors <= 5:
                print(f"    ERROR {doc_code}:{articolo}: {e}")

    print(f"  {doc_code}: {imported} imported, {skipped} skipped, {errors} errors")
    return {"code": doc_code, "status": "ok", "imported": imported, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Import ingested JSON to kb.normativa (STAGING REMOTE)")
    parser.add_argument("--doc", help="Document code (e.g., CC, COST)")
    parser.add_argument("--all", action="store_true", help="Import all available documents")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually insert")
    args = parser.parse_args()

    if not args.doc and not args.all:
        parser.error("Either --doc or --all is required")

    print(f"Connecting to STAGING DB (localhost:{DB_PORT})...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        dbname=DB_NAME
    )
    conn.autocommit = False
    cur = conn.cursor()
    print("Connected!\n")

    try:
        if args.all:
            docs = sorted([d.name for d in INGESTED_ROOT.iterdir()
                          if d.is_dir() and (d / "ingested.json").exists()])
        else:
            docs = [args.doc]

        print(f"=== IMPORT TO STAGING (REMOTE) ===")
        print(f"Documents: {len(docs)}")
        print(f"Dry run: {args.dry_run}\n")

        results = []
        for doc in docs:
            result = import_document(cur, doc, args.dry_run)
            results.append(result)
            # Commit after each document
            if not args.dry_run:
                conn.commit()

        total_imported = sum(r.get("imported", 0) for r in results)
        total_skipped = sum(r.get("skipped", 0) for r in results)
        total_errors = sum(r.get("errors", 0) for r in results)
        work_found = sum(1 for r in results if r["status"] == "ok")
        work_missing = sum(1 for r in results if r["status"] == "work_not_found")

        print(f"\n=== SUMMARY ===")
        print(f"Works found: {work_found}, missing: {work_missing}")
        print(f"Total imported: {total_imported}")
        print(f"Total skipped: {total_skipped}")
        print(f"Total errors: {total_errors}")

        if not args.dry_run:
            cur.execute("SELECT COUNT(*) FROM kb.normativa")
            count = cur.fetchone()[0]
            print(f"Total in DB: {count}")

            # Per-code stats
            cur.execute("""
                SELECT w.code, COUNT(n.id)
                FROM kb.work w JOIN kb.normativa n ON n.work_id = w.id
                GROUP BY w.code ORDER BY COUNT(n.id) DESC LIMIT 25
            """)
            print(f"\nTop 25 codes by article count:")
            for code, cnt in cur.fetchall():
                print(f"  {code}: {cnt}")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
