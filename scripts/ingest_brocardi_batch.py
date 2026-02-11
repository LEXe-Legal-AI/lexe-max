#!/usr/bin/env python
"""
Batch ingestion of Brocardi HTML documents.

Usage:
    uv run python scripts/ingest_brocardi_batch.py --all
    uv run python scripts/ingest_brocardi_batch.py --doc DISP_CC
"""

import json
import re
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

BROC_ROOT = Path("C:/Mie pagine Web/broc-resto/www.brocardi.it")
OUTPUT_ROOT = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/ingested")

# Document mapping: sigla -> folder name
DOCUMENTS = {
    "DISP_CC": "disposizioni-per-attuazione-del-codice-civile",
    "DISP_CPC": "disposizioni-per-attuazione-codice-procedura-civile",
    "DISP_CPP": "disposizioni-per-attuazione-codice-procedura-penale",
    "DISP_CP": "disposizioni-transitorie-codice-penale",
    "CPROT": "codice-protezione-civile",
}


def parse_article(html_path: Path, doc_id: str) -> dict:
    """Parse a Brocardi article HTML file."""
    try:
        with open(html_path, encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except Exception as e:
        return None

    # Extract article number from filename
    filename = html_path.stem
    match = re.match(r'art(\d+)(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?', filename)
    if not match:
        return None

    base_num = int(match.group(1))
    suffix = match.group(2)

    # Extract title/rubrica
    rubrica = None
    rubrica_el = soup.find('div', class_='rubrica') or soup.find('h2', class_='rubrica')
    if rubrica_el:
        rubrica = rubrica_el.get_text(strip=True)

    # Extract text
    text = ''
    content_div = soup.find('div', class_='testo-articolo') or \
                  soup.find('div', class_='corpo') or \
                  soup.find('div', id='corpo')

    if content_div:
        for unwanted in content_div.find_all(['script', 'style', 'ins', 'nav']):
            unwanted.decompose()
        text = content_div.get_text(separator=' ', strip=True)

    if not text or len(text) < 50:
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    text = re.sub(r'\s+', ' ', text).strip()

    return {
        'article_key': f'{doc_id}:{base_num}' + (f'-{suffix}' if suffix else ''),
        'doc_id': doc_id,
        'base_num': base_num,
        'suffix': suffix,
        'text_primary': text,
        'rubrica_primary': rubrica,
        'text_final': text,
        'rubrica_final': rubrica,
        'final_source': 'brocardi_local',
        'status': 'ready' if len(text) > 50 else 'needs_review',
        'quality_class': 'VALID' if len(text) > 50 else 'WEAK',
        'warnings': [] if len(text) > 50 else ['short_text'],
        'provenance': {'source': 'brocardi_resto', 'path': str(html_path.relative_to(BROC_ROOT))}
    }


def ingest_document(doc_id: str, folder_name: str) -> dict:
    """Ingest a single document."""
    doc_path = BROC_ROOT / folder_name
    if not doc_path.exists():
        print(f"ERROR: Path not found: {doc_path}")
        return None

    # Find all article files
    article_files = list(doc_path.rglob('art*.html'))
    article_files = [f for f in article_files if not f.name.endswith('.tmp')]

    print(f"\n=== {doc_id} ===")
    print(f"Path: {doc_path}")
    print(f"Found {len(article_files)} article files")

    # Parse articles
    articles = []
    for f in article_files:
        article = parse_article(f, doc_id)
        if article:
            articles.append(article)

    # Sort by article number
    articles.sort(key=lambda a: (a['base_num'], a['suffix'] or ''))

    # Stats
    ready = sum(1 for a in articles if a['status'] == 'ready')
    review = sum(1 for a in articles if a['status'] == 'needs_review')

    stats = {
        'doc_id': doc_id,
        'total': len(articles),
        'ready': ready,
        'needs_review': review,
        'skipped': 0,
        'error': 0,
        'used_primary': 0,
        'used_cataldi': 0,
        'used_brocardi': len(articles),
        'used_online': 0,
        'found_cataldi': 0,
        'found_brocardi': len(articles),
        'confirmed': 0,
        'partial': 0,
        'conflicts': 0
    }

    # Save
    output_dir = OUTPUT_ROOT / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'ingested.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'doc_id': doc_id,
            'stats': stats,
            'articles': articles
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_file}")
    print(f"Total: {len(articles)} | Ready: {ready} | Review: {review}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch ingest Brocardi documents")
    parser.add_argument("--doc", help="Document sigla (e.g., DISP_CC)")
    parser.add_argument("--all", action="store_true", help="Ingest all documents")

    args = parser.parse_args()

    if args.doc:
        if args.doc not in DOCUMENTS:
            print(f"Unknown document: {args.doc}")
            print(f"Available: {', '.join(DOCUMENTS.keys())}")
            return
        ingest_document(args.doc, DOCUMENTS[args.doc])

    elif args.all:
        total_articles = 0
        total_ready = 0

        for doc_id, folder_name in DOCUMENTS.items():
            stats = ingest_document(doc_id, folder_name)
            if stats:
                total_articles += stats['total']
                total_ready += stats['ready']

        print("\n" + "=" * 60)
        print("BATCH COMPLETE")
        print("=" * 60)
        print(f"Total documents: {len(DOCUMENTS)}")
        print(f"Total articles: {total_articles}")
        print(f"Total ready: {total_ready}")

    else:
        print("Usage: --doc SIGLA or --all")
        print(f"Available documents: {', '.join(DOCUMENTS.keys())}")


if __name__ == "__main__":
    main()
