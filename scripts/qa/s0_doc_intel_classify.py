#!/usr/bin/env python3
"""
Document Intelligence - Phase 0.2: Classificazione con LLM.

1. Estrae le 5 finestre selezionate con Unstructured Cloud
2. Chiama LLM UNA volta per documento con tutti i campioni
3. Salva classificazione e parametri suggeriti

Usage:
    uv run python scripts/qa/s0_doc_intel_classify.py
    uv run python scripts/qa/s0_doc_intel_classify.py --dry-run
    uv run python scripts/qa/s0_doc_intel_classify.py --pdf "Volume I_2020_Massimario_Civile.pdf"
"""

import argparse
import asyncio
import json
import re
import tempfile
import time
from pathlib import Path

import asyncpg
import fitz
import httpx

from qa_config import DB_URL, PDF_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, LLM_MODEL

# ── Config ──────────────────────────────────────────────────────────
CLOUD_API_URL = "https://api.unstructured.io/general/v0/general"
CLOUD_API_KEY = "h7nQP3E52xtFGxLJwuh3guHk4ehtNL"

MAX_CHARS_PER_SAMPLE = 4000  # Comprimi campioni per LLM
MAX_CONCURRENT_CLOUD = 3


# ── LLM Prompt ──────────────────────────────────────────────────────
DOC_INTEL_PROMPT = """Sei un esperto di documenti legali italiani, in particolare massimari della Corte di Cassazione.

Analizza questi 5 campioni estratti da diverse sezioni del documento e classifica il tipo di documento.

CAMPIONI:
{samples}

Rispondi SOLO con un JSON valido, senza altro testo:

{{
  "doc_type": "list_only | massima_plus_commentary | mixed | toc_heavy | ocr_needed",
  "confidence": 0.0-1.0,
  "doc_type_reasoning": "breve spiegazione",

  "massima_anchor_patterns": ["pattern1", "pattern2", ...],
  "commentary_link_likelihood": 0.0-1.0,

  "suggested_chunking": "by_title | by_similarity | pattern_based | parent_child",
  "chunking_reasoning": "perché questa strategia",

  "suggested_gate": {{
    "min_length": 100-300,
    "max_citation_ratio": 0.01-0.10,
    "skip_pages_start": 0-20,
    "skip_pages_end": 0-10
  }},

  "suggested_profile": "clean_standard | legacy_layout | toc_heavy | citation_dense | ocr_needed",

  "sample_observations": [
    {{"band": 0, "observation": "cosa contiene questo campione"}},
    {{"band": 1, "observation": "..."}},
    ...
  ],

  "warnings": ["eventuali problemi rilevati"]
}}

DOC_TYPE DEFINITIONS:
- list_only: Elenco di massime brevi senza spiegazioni dettagliate
- massima_plus_commentary: Ogni massima ha spiegazione/motivazione estesa (pagine)
- mixed: Mix di sezioni con e senza commento
- toc_heavy: Documento con molto indice/TOC infiltrato nel contenuto
- ocr_needed: Testo illeggibile, scanning quality bassa

CHUNKING STRATEGIES:
- by_title: Usa struttura documento (headers), buono per list_only
- by_similarity: Raggruppa per similarità semantica, buono per massima_plus_commentary
- pattern_based: Usa regex per trovare inizio massime (Sez., N., etc.)
- parent_child: Chunk piccoli per search + parent per context LLM"""


def extract_pdf_window(pdf_path: Path, page_start: int, page_end: int) -> Path:
    """Estrae una finestra di pagine dal PDF in un file temporaneo."""
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()

    # page_start e page_end sono 1-based
    new_doc.insert_pdf(doc, from_page=page_start - 1, to_page=page_end - 1)

    # Save to temp file
    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    new_doc.save(temp_path)
    new_doc.close()
    doc.close()

    return temp_path


async def extract_with_cloud(
    pdf_path: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> str:
    """Estrae testo da PDF con Unstructured Cloud."""
    async with semaphore:
        with open(pdf_path, "rb") as f:
            response = await client.post(
                CLOUD_API_URL,
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={
                    "strategy": "hi_res",
                    "output_format": "application/json",
                },
                headers={"unstructured-api-key": CLOUD_API_KEY},
                timeout=300.0,
            )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")

        elements = response.json()

        # Concatena testo
        texts = []
        for elem in elements:
            text = elem.get("text", "").strip()
            if text and len(text) > 20:
                texts.append(text)

        return "\n\n".join(texts)


def compress_sample(text: str, max_chars: int = MAX_CHARS_PER_SAMPLE) -> str:
    """Comprimi campione per LLM, mantieni parti significative."""
    if len(text) <= max_chars:
        return text

    # Prendi inizio e fine
    half = max_chars // 2
    return text[:half] + "\n\n[...]\n\n" + text[-half:]


async def call_llm_classify(samples: list[dict]) -> dict:
    """Chiama LLM per classificare il documento."""
    # Format samples for prompt
    samples_text = ""
    for i, s in enumerate(samples):
        samples_text += f"\n=== CAMPIONE {i} (pagine {s['page_start']}-{s['page_end']}) ===\n"
        samples_text += s['text'][:MAX_CHARS_PER_SAMPLE]
        samples_text += "\n"

    prompt = DOC_INTEL_PROMPT.format(samples=samples_text)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=60.0,
        )

        if response.status_code != 200:
            raise Exception(f"LLM HTTP {response.status_code}: {response.text[:200]}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON from response
        # Try to extract JSON from markdown code block if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        return json.loads(content)


async def process_document(
    conn: asyncpg.Connection,
    manifest_id: int,
    filename: str,
    windows: list[dict],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> dict | None:
    """Processa un singolo documento."""
    # Find PDF
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        pdf_path = PDF_DIR / "new" / filename
    if not pdf_path.exists():
        return None

    samples = []
    temp_files = []

    try:
        # Extract each window with Cloud
        for w in windows:
            # Extract window to temp PDF
            temp_pdf = extract_pdf_window(pdf_path, w['page_start'], w['page_end'])
            temp_files.append(temp_pdf)

            # Extract text with Cloud
            text = await extract_with_cloud(temp_pdf, client, semaphore)

            samples.append({
                'band_index': w['band_index'],
                'page_start': w['page_start'],
                'page_end': w['page_end'],
                'text': compress_sample(text),
                'char_count': len(text),
            })

        if dry_run:
            print(f"    [DRY RUN] Estratti {len(samples)} campioni, tot {sum(s['char_count'] for s in samples)} chars")
            return None

        # Call LLM
        classification = await call_llm_classify(samples)

        # Save to database
        await save_classification(conn, manifest_id, classification, samples)

        return classification

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except:
                pass


async def save_classification(
    conn: asyncpg.Connection,
    manifest_id: int,
    classification: dict,
    samples: list[dict],
):
    """Salva classificazione nel database."""
    # Save to llm_decisions
    await conn.execute(
        """
        INSERT INTO kb.llm_decisions
          (manifest_id, trigger_type, input_summary, output_json, confidence, model_used)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (manifest_id, trigger_type) DO UPDATE SET
          output_json = EXCLUDED.output_json,
          confidence = EXCLUDED.confidence,
          decided_at = now()
        """,
        manifest_id,
        'doc_intel_5x15',
        f"{len(samples)} samples, {sum(s['char_count'] for s in samples)} chars",
        json.dumps(classification),
        classification.get('confidence', 0.5),
        LLM_MODEL,
    )

    # Update ingestion_profiles with suggested profile
    suggested_profile = classification.get('suggested_profile', 'clean_standard')
    suggested_chunking = classification.get('suggested_chunking', 'by_title')
    suggested_gate = classification.get('suggested_gate', {})

    await conn.execute(
        """
        INSERT INTO kb.ingestion_profiles
          (manifest_id, profile_name, chunking_strategy, gate_params, confidence, assigned_by)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (manifest_id) DO UPDATE SET
          profile_name = EXCLUDED.profile_name,
          chunking_strategy = EXCLUDED.chunking_strategy,
          gate_params = EXCLUDED.gate_params,
          confidence = EXCLUDED.confidence,
          assigned_at = now()
        """,
        manifest_id,
        suggested_profile,
        suggested_chunking,
        json.dumps(suggested_gate),
        classification.get('confidence', 0.5),
        'doc_intel_llm',
    )


async def ensure_tables(conn: asyncpg.Connection):
    """Assicura che le tabelle necessarie esistano."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.llm_decisions (
            id SERIAL PRIMARY KEY,
            manifest_id INTEGER REFERENCES kb.pdf_manifest(id),
            trigger_type TEXT NOT NULL,
            input_summary TEXT,
            output_json JSONB,
            confidence REAL,
            model_used TEXT,
            tokens_used INTEGER,
            cost_usd REAL,
            decided_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE(manifest_id, trigger_type)
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.ingestion_profiles (
            id SERIAL PRIMARY KEY,
            manifest_id INTEGER REFERENCES kb.pdf_manifest(id) UNIQUE,
            profile_name TEXT NOT NULL,
            chunking_strategy TEXT,
            gate_params JSONB,
            confidence REAL,
            assigned_by TEXT DEFAULT 'manual',
            assigned_at TIMESTAMPTZ DEFAULT now()
        )
    """)


async def main(dry_run: bool = False, single_pdf: str | None = None):
    """Main entry point."""
    print("=" * 70)
    print("DOCUMENT INTELLIGENCE - Classificazione LLM")
    print("=" * 70)
    print()

    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY non configurata!")
        print("Esporta: export OPENROUTER_API_KEY=sk-or-...")
        return

    conn = await asyncpg.connect(DB_URL)
    await ensure_tables(conn)

    # Get documents with sample windows
    if single_pdf:
        query = """
            SELECT m.id, m.filename,
                   array_agg(json_build_object(
                       'band_index', w.band_index,
                       'page_start', w.page_start,
                       'page_end', w.page_end
                   ) ORDER BY w.band_index) as windows
            FROM kb.pdf_manifest m
            JOIN kb.qa_sample_windows w ON w.manifest_id = m.id
            WHERE m.filename = $1
            GROUP BY m.id, m.filename
        """
        rows = await conn.fetch(query, single_pdf)
    else:
        query = """
            SELECT m.id, m.filename,
                   array_agg(json_build_object(
                       'band_index', w.band_index,
                       'page_start', w.page_start,
                       'page_end', w.page_end
                   ) ORDER BY w.band_index) as windows
            FROM kb.pdf_manifest m
            JOIN kb.qa_sample_windows w ON w.manifest_id = m.id
            LEFT JOIN kb.llm_decisions d ON d.manifest_id = m.id AND d.trigger_type = 'doc_intel_5x15'
            WHERE d.id IS NULL  -- Solo documenti non ancora classificati
            GROUP BY m.id, m.filename
            ORDER BY m.filename
        """
        rows = await conn.fetch(query)

    print(f"Documenti da classificare: {len(rows)}")
    print(f"Dry run: {dry_run}")
    print()

    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLOUD)

    async with httpx.AsyncClient() as client:
        for i, row in enumerate(rows):
            manifest_id = row['id']
            filename = row['filename']
            windows = [json.loads(w) if isinstance(w, str) else w for w in row['windows']]

            print(f"[{i+1}/{len(rows)}] {filename[:55]}")
            print(f"         Windows: {len(windows)}")

            start = time.time()

            try:
                classification = await process_document(
                    conn, manifest_id, filename, windows,
                    client, semaphore, dry_run
                )

                elapsed = time.time() - start

                if classification:
                    doc_type = classification.get('doc_type', '?')
                    confidence = classification.get('confidence', 0)
                    chunking = classification.get('suggested_chunking', '?')
                    profile = classification.get('suggested_profile', '?')

                    print(f"         -> {doc_type} (conf={confidence:.2f})")
                    print(f"         -> chunking={chunking}, profile={profile}")
                    print(f"         ({elapsed:.1f}s)")

                    results.append({
                        'filename': filename,
                        'doc_type': doc_type,
                        'confidence': confidence,
                        'chunking': chunking,
                        'profile': profile,
                    })

            except Exception as e:
                print(f"         [ERROR] {e}")

    await conn.close()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        # Group by doc_type
        by_type = {}
        for r in results:
            by_type.setdefault(r['doc_type'], []).append(r)

        for doc_type, docs in sorted(by_type.items()):
            print(f"\n{doc_type}: {len(docs)} documenti")
            for d in docs[:3]:  # Show first 3
                print(f"  - {d['filename'][:50]} (conf={d['confidence']:.2f})")
            if len(docs) > 3:
                print(f"  ... e altri {len(docs) - 3}")

        # Chunking strategies
        print("\nChunking strategies:")
        by_chunk = {}
        for r in results:
            by_chunk.setdefault(r['chunking'], []).append(r)
        for chunk, docs in sorted(by_chunk.items()):
            print(f"  {chunk}: {len(docs)} documenti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Intelligence Classify")
    parser.add_argument("--dry-run", action="store_true", help="Solo estrazione, no LLM")
    parser.add_argument("--pdf", type=str, help="Singolo PDF da classificare")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, single_pdf=args.pdf))
