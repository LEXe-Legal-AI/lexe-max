#!/usr/bin/env python
"""
Ingest LLM-extracted JSON files from altalex pdf/ to kb.normativa + chunks.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-xxx
    uv run python scripts/ingest_llm_extracted.py
    uv run python scripts/ingest_llm_extracted.py --dry-run
    uv run python scripts/ingest_llm_extracted.py --skip-existing
"""

import asyncio
import json
import argparse
import hashlib
import os
import re
from pathlib import Path
from uuid import uuid4
from datetime import datetime

import asyncpg
import httpx

# Database connection - STAGING (lexe-max container on port 5436)
DB_URL = os.environ.get(
    "LEXE_KB_DATABASE_URL",
    "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"
)

# Path to extracted JSONs
ALTALEX_PDF_ROOT = Path(__file__).parent.parent.parent / "altalex pdf"

# Code mapping from filename patterns
CODICE_MAP = {
    "gdpr": "GDPR",
    "codice-civile": "CC",
    "codice-penale": "CP",
    "codice-procedura-civile": "CPC",
    "codice-procedura-penale": "CPP",
    "costituzione": "COST",
    "codice-ambiente": "CAMB",
    "codice-amministrazione-digitale": "CAD",
    "codice-antimafia": "CAMAFIA",
    "codice-appalti": "CAPP",
    "codice-beni-culturali": "CBC",
    "codice-crisi-impresa": "CCI",
    "codice-consumo": "CCONS",
    "codice-deontologico-forense": "CDF",
    "codice-del-turismo": "CTUR",
    "codice-giustizia-contabile": "CGC",
    "codice-giustizia-sportiva": "CGS",
    "codice-medicinali": "CMED",
    "codice-nautica-diporto": "CND",
    "codice-pari-opportunita": "CPO",
    "codice-privacy": "CPRIV",
    "codice-processo-amministrativo": "CPA",
    "codice-processo-penale-minorile": "CPPM",
    "codice-processo-tributario": "CPT",
    "codice-proprieta-industriale": "CPI",
    "codice-terzo-settore": "CTS",
    "codice-assicurazioni": "CASS",
    "codice-della-strada": "CDS",
    "codice-comunicazioni": "CCE",
    "dichiarazione-universale": "DUDU",
    "legge-diritto-autore": "LDA",
    "legge-diritto-internazionale": "LDIP",
    "legge-divorzio": "LDIV",
    "legge-fallimentare": "LFAL",
    "legge-locazioni": "LLOC",
    "legge-professionale-forense": "LPF",
    "legge-reati-tributari": "LRT",
    "legge-sciopero": "LSCI",
    "legge-depenalizzazione": "LDEP",
    "mediazione-civile": "LMED",
    "ordinamento-penitenziario": "OP",
    "responsabilita-amministrativa": "DLGS231",
    "riforma-fornero": "RFORN",
    "riforma-biagi": "RBIAGI",
    "sicurezza-urbana": "LSU",
    "statuto-contribuente": "SCON",
    "statuto-lavoratori": "SLAV",
    "testo-unico-bancario": "TUB",
    "testo-unico-finanza": "TUF",
    "testo-unico-edilizia": "TUE",
    "testo-unico-enti-locali": "TUEL",
    "testo-unico-espropriazioni": "TUESP",
    "testo-unico-immigrazione": "TUIMM",
    "testo-unico-imposte-redditi": "TUIR",
    "testo-unico-istruzione": "TUIST",
    "testo-unico-iva": "TUIVA",
    "testo-unico-maternita": "TUMAT",
    "testo-unico-previdenza": "TUPREV",
    "testo-unico-pubblico-impiego": "TUPI",
    "testo-unico-sicurezza": "TUSL",
    "testo-unico-societa-partecipate": "TUSP",
    "testo-unico-stupefacenti": "TUSTUP",
    "testo-unico-casellario": "TUCAS",
    "testo-unico-foreste": "TUFOR",
    "testo-unico-spese-giustizia": "TUSG",
    "testo-unico-radiotelevisione": "TURTV",
    "testo-unico-documentazione": "TUDA",
    "regolamento-codice-strada": "RCDS",
}

# Latin suffix order for sort_key
SUFFIX_ORDER = {
    None: 0, "": 0,
    "bis": 1, "ter": 2, "quater": 3, "quinquies": 4,
    "sexies": 5, "septies": 6, "octies": 7, "novies": 8,
    "decies": 9, "undecies": 10, "duodecies": 11, "terdecies": 12,
    "quaterdecies": 13, "quinquiesdecies": 14, "sexiesdecies": 15,
}


def extract_code_from_filename(filename: str) -> str:
    """Extract code from filename using pattern matching."""
    name_lower = filename.lower()
    for pattern, code in CODICE_MAP.items():
        if pattern in name_lower:
            return code
    # Fallback: use first word capitalized
    return name_lower.split("-")[0].upper()[:10]


def compute_hash(text: str) -> str:
    """SHA256 hash."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_article_number(articolo: str) -> tuple[int | None, str | None]:
    """Parse article number and suffix from string like '2043-bis'."""
    if not articolo:
        return None, None

    # Pattern: number + optional suffix
    match = re.match(r'^(\d+)(?:[-\s]*(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies))?',
                     articolo, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        suffix = match.group(2).lower() if match.group(2) else None
        return num, suffix

    # Try just number
    try:
        return int(articolo), None
    except ValueError:
        return None, None


def compute_sort_key(articolo_num: int | None, suffix: str | None) -> str:
    """Compute sort key for article ordering."""
    num_part = articolo_num or 0
    suffix_order = SUFFIX_ORDER.get(suffix, 99)
    return f"{num_part:06d}.{suffix_order:02d}"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end in last 100 chars
            search_start = max(start + chunk_size - 100, start)
            search_text = text[search_start:end]
            for sep in ['. ', '.\n', '; ', ';\n']:
                last_sep = search_text.rfind(sep)
                if last_sep > 0:
                    end = search_start + last_sep + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


async def get_embedding(text: str, api_key: str) -> list[float]:
    """Get embedding from OpenRouter."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "openai/text-embedding-3-small",
                "input": text[:8000]  # Truncate if needed
            }
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


async def ensure_work_exists(conn: asyncpg.Connection, code: str, title: str) -> str:
    """Ensure work exists in kb.work and return id."""
    row = await conn.fetchrow("SELECT id FROM kb.work WHERE code = $1", code)
    if row:
        return str(row["id"])

    work_id = str(uuid4())
    await conn.execute("""
        INSERT INTO kb.work (id, code, title, source, created_at)
        VALUES ($1, $2, $3, 'altalex', NOW())
    """, work_id, code, title)
    return work_id


async def process_json_file(
    conn: asyncpg.Connection,
    json_path: Path,
    api_key: str,
    dry_run: bool = False
) -> dict:
    """Process a single JSON file."""
    print(f"\n{'='*60}")
    print(f"Processing: {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    if not articles:
        print(f"  SKIP: No articles")
        return {"file": json_path.name, "status": "empty", "articles": 0}

    code = data.get("codice") or extract_code_from_filename(json_path.name)
    title = json_path.stem.replace(".llm_extracted", "").replace("-", " ").title()

    print(f"  Code: {code}")
    print(f"  Articles: {len(articles)}")

    if dry_run:
        print(f"  DRY RUN - skipping DB writes")
        return {"file": json_path.name, "status": "dry_run", "code": code, "articles": len(articles)}

    # Ensure work exists
    work_id = await ensure_work_exists(conn, code, title)

    # Process articles
    inserted_articles = 0
    inserted_chunks = 0
    inserted_embeddings = 0

    for art in articles:
        articolo = art.get("articolo", "")
        if not articolo:
            continue

        testo = art.get("testo", "")
        rubrica = art.get("rubrica", "")

        articolo_num, suffix = parse_article_number(articolo)
        sort_key = compute_sort_key(articolo_num, suffix)
        content_hash = compute_hash(testo) if testo else None

        # Determine identity and quality class
        identity_class = "SUFFIX" if suffix else "BASE"
        if any(x in articolo.lower() for x in ["preleggi", "disp.", "transitorie"]):
            identity_class = "SPECIAL"

        quality_class = "VALID_STRONG"
        if not testo or len(testo) < 50:
            quality_class = "EMPTY" if not testo else "WEAK"
        elif len(testo) < 150:
            quality_class = "VALID_SHORT"

        # Insert article
        try:
            art_id = str(uuid4())
            await conn.execute("""
                INSERT INTO kb.normativa (
                    id, work_id, articolo, rubrica, testo, content_hash,
                    articolo_num_norm, articolo_suffix, articolo_sort_key,
                    identity_class, quality_class, canonical_source, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'altalex', NOW())
                ON CONFLICT (work_id, articolo) DO UPDATE SET
                    testo = EXCLUDED.testo,
                    rubrica = EXCLUDED.rubrica,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = NOW()
                RETURNING id
            """, art_id, work_id, articolo, rubrica, testo, content_hash,
                articolo_num, suffix, sort_key, identity_class, quality_class)
            inserted_articles += 1
        except Exception as e:
            print(f"  ERROR inserting article {articolo}: {e}")
            continue

        # Chunk the text
        if testo and len(testo) > 50:
            chunks = chunk_text(testo)
            for i, chunk_text_content in enumerate(chunks):
                chunk_id = str(uuid4())
                try:
                    await conn.execute("""
                        INSERT INTO kb.normativa_chunk (
                            id, normativa_id, chunk_no, testo, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                        ON CONFLICT (normativa_id, chunk_no) DO UPDATE SET
                            testo = EXCLUDED.testo
                    """, chunk_id, art_id, i, chunk_text_content)
                    inserted_chunks += 1

                    # Generate embedding
                    if api_key:
                        try:
                            embedding = await get_embedding(chunk_text_content, api_key)
                            await conn.execute("""
                                INSERT INTO kb.normativa_chunk_embeddings (
                                    chunk_id, embedding, model_name, created_at
                                ) VALUES ($1, $2, 'text-embedding-3-small', NOW())
                                ON CONFLICT (chunk_id) DO UPDATE SET
                                    embedding = EXCLUDED.embedding
                            """, chunk_id, embedding)
                            inserted_embeddings += 1
                        except Exception as e:
                            print(f"  WARN: Embedding failed for chunk {i}: {e}")
                except Exception as e:
                    print(f"  ERROR inserting chunk {i}: {e}")

    print(f"  Inserted: {inserted_articles} articles, {inserted_chunks} chunks, {inserted_embeddings} embeddings")

    return {
        "file": json_path.name,
        "status": "ok",
        "code": code,
        "articles": inserted_articles,
        "chunks": inserted_chunks,
        "embeddings": inserted_embeddings
    }


async def get_existing_codes(conn: asyncpg.Connection) -> set[str]:
    """Get codes already in DB."""
    rows = await conn.fetch("SELECT DISTINCT code FROM kb.work WHERE source = 'altalex'")
    return {r["code"] for r in rows}


async def run_ingestion(skip_existing: bool = False, dry_run: bool = False):
    """Run ingestion on all LLM-extracted JSON files."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not dry_run:
        print("WARNING: No OPENROUTER_API_KEY - embeddings will be skipped")

    # Find all JSON files
    json_files = list(ALTALEX_PDF_ROOT.rglob("*.llm_extracted.json"))
    print(f"Found {len(json_files)} JSON files")

    if not json_files:
        print("No files to process")
        return

    # Connect to database
    print(f"\nConnecting to: {DB_URL.split('@')[1] if '@' in DB_URL else DB_URL}")
    conn = await asyncpg.connect(DB_URL)

    try:
        existing_codes = await get_existing_codes(conn) if skip_existing else set()
        if existing_codes:
            print(f"Existing codes in DB: {existing_codes}")

        results = []
        for i, json_path in enumerate(sorted(json_files), 1):
            code = extract_code_from_filename(json_path.name)

            if skip_existing and code in existing_codes:
                print(f"[{i}/{len(json_files)}] SKIP {code} (already exists)")
                continue

            result = await process_json_file(conn, json_path, api_key, dry_run)
            results.append(result)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        total_articles = sum(r.get("articles", 0) for r in results)
        total_chunks = sum(r.get("chunks", 0) for r in results)
        total_embeddings = sum(r.get("embeddings", 0) for r in results)
        print(f"Files processed: {len(results)}")
        print(f"Articles: {total_articles}")
        print(f"Chunks: {total_chunks}")
        print(f"Embeddings: {total_embeddings}")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest LLM-extracted JSON to DB")
    parser.add_argument("--skip-existing", action="store_true", help="Skip codes already in DB")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    asyncio.run(run_ingestion(skip_existing=args.skip_existing, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
