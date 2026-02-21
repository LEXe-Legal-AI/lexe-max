#!/usr/bin/env python3
"""
Export Brocardi articles to JSON for ingest_normativa_to_db.py

Parses local HTML mirrors for COST, CPC, CPP using BrocardiParser
and exports to JSON files compatible with the ingest script.

Usage:
    cd lexe-max
    uv run python scripts/export_brocardi_json.py --codes COST CPC CPP
    uv run python scripts/export_brocardi_json.py --codes ALL
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add scripts dir to path for brocardi_parser import
sys.path.insert(0, str(Path(__file__).parent))
from brocardi_parser import BrocardiParser, CODE_TO_PATH


def export_code(parser: BrocardiParser, code: str, output_dir: Path) -> dict:
    """Parse and export a single code to JSON."""
    print(f"\n{'='*60}")
    print(f"  Parsing {code}...")
    print(f"{'='*60}")

    folder = parser.get_folder(code)
    if not folder:
        print(f"  ERROR: No folder found for {code}")
        return {"code": code, "count": 0, "error": "folder_not_found"}

    articles = parser.get_articles(code)
    print(f"  Found {len(articles)} articles")

    if not articles:
        return {"code": code, "count": 0, "error": "no_articles"}

    # Convert to ingest format
    json_articles = []
    empty_count = 0
    for art in articles:
        if not art.testo or len(art.testo.strip()) < 10:
            empty_count += 1
            continue

        json_articles.append({
            "codice": art.code,
            "articolo": art.article_num,
            "rubrica": art.rubrica,
            "testo": art.testo,
            "source_url": f"file://{art.source_file}",
            "citations": [],
        })

    if empty_count > 0:
        print(f"  Skipped {empty_count} empty/short articles")

    # Write JSON
    output_file = output_dir / f"{code.lower()}_brocardi_export.json"
    output = {
        "codice": code,
        "source": "brocardi",
        "exported_at": datetime.now().isoformat(),
        "total_parsed": len(articles),
        "total_exported": len(json_articles),
        "articles": json_articles,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Exported {len(json_articles)} articles to {output_file.name}")

    # Show sample
    if json_articles:
        first = json_articles[0]
        last = json_articles[-1]
        print(f"  First: Art. {first['articolo']} - {first.get('rubrica', 'N/A')}")
        print(f"  Last:  Art. {last['articolo']} - {last.get('rubrica', 'N/A')}")

    return {"code": code, "count": len(json_articles), "file": str(output_file)}


def main():
    ap = argparse.ArgumentParser(description="Export Brocardi articles to JSON")
    ap.add_argument(
        "--codes",
        nargs="+",
        default=["COST", "CPC", "CPP"],
        help="Codes to export (default: COST CPC CPP). Use ALL for all 5."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory (default: scripts/)"
    )
    args = ap.parse_args()

    codes = list(CODE_TO_PATH.keys()) if "ALL" in [c.upper() for c in args.codes] else [c.upper() for c in args.codes]

    print(f"\n{'='*60}")
    print(f"  BROCARDI EXPORT TO JSON")
    print(f"  {datetime.now().isoformat()}")
    print(f"{'='*60}")
    print(f"  Codes: {', '.join(codes)}")
    print(f"  Output: {args.output_dir}")

    parser = BrocardiParser()

    # Check availability
    available = parser.get_available_codes()
    print(f"  Available mirrors: {', '.join(available)}")

    missing = [c for c in codes if c not in available]
    if missing:
        print(f"  WARNING: Missing mirrors for: {', '.join(missing)}")
        codes = [c for c in codes if c in available]

    if not codes:
        print("  ERROR: No codes to process!")
        return

    # Export each code
    results = []
    for code in codes:
        result = export_code(parser, code, args.output_dir)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total = 0
    for r in results:
        status = f"{r['count']} articles" if r['count'] > 0 else f"ERROR: {r.get('error', 'unknown')}"
        print(f"  {r['code']}: {status}")
        total += r['count']
    print(f"  TOTAL: {total} articles exported")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
