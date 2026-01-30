#!/usr/bin/env python3
"""
Document Intelligence - Classificazione con LLM (estrazione locale).

Usa PyMuPDF per estrarre testo (no Cloud), poi chiama LLM per classificare.

Usage:
    uv run python scripts/qa/s0_doc_intel_classify_local.py
    uv run python scripts/qa/s0_doc_intel_classify_local.py --pdf "Volume I_2020.pdf"
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import asyncpg
import fitz  # PyMuPDF
import httpx

from qa_config import DB_URL, PDF_DIR, OPENROUTER_API_KEY, OPENROUTER_URL

# Model
LLM_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"

MAX_CHARS_PER_SAMPLE = 4000


# ── LLM Prompt ──────────────────────────────────────────────────────
DOC_INTEL_PROMPT = """Sei un esperto di documenti legali italiani, in particolare massimari della Corte di Cassazione.

Analizza questi 5 campioni estratti da diverse sezioni del documento e classifica il tipo di documento.

CAMPIONI:
{samples}

Rispondi SOLO con un JSON valido (senza markdown, senza ```), strutturato così:

{{
  "doc_type": "list_only | massima_plus_commentary | mixed | toc_heavy | ocr_needed",
  "confidence": 0.0-1.0,
  "doc_type_reasoning": "breve spiegazione",

  "massima_anchor_patterns": ["pattern regex 1", "pattern regex 2"],
  "commentary_link_likelihood": 0.0-1.0,

  "suggested_chunking": "by_title | by_similarity | pattern_based | parent_child",
  "chunking_reasoning": "perché questa strategia",

  "suggested_gate": {{
    "min_length": 100-300,
    "max_citation_ratio": 0.01-0.10,
    "skip_pages_start": 0-20,
    "skip_pages_end": 0-10
  }},

  "suggested_profile": "structured_by_title | structured_parent_child | baseline_toc_filter | legacy_layout | list_pure | mixed_hybrid",

  "sample_observations": [
    {{"band": 0, "observation": "cosa contiene questo campione"}},
    {{"band": 1, "observation": "..."}},
    {{"band": 2, "observation": "..."}},
    {{"band": 3, "observation": "..."}},
    {{"band": 4, "observation": "..."}}
  ],

  "warnings": ["eventuali problemi rilevati"]
}}

DOC_TYPE:
- list_only: Elenco di massime brevi senza spiegazioni
- massima_plus_commentary: Ogni massima ha spiegazione/motivazione estesa
- mixed: Mix di sezioni con e senza commento
- toc_heavy: Documento con molto indice/TOC infiltrato
- ocr_needed: Testo illeggibile, scanning quality bassa

CHUNKING:
- by_title: Usa struttura documento (headers)
- by_similarity: Raggruppa per similarità semantica
- pattern_based: Usa regex per trovare inizio massime
- parent_child: Chunk piccoli per search + parent per context"""


def extract_window_text(pdf_path: Path, page_start: int, page_end: int) -> str:
    """Estrae testo da una finestra di pagine con PyMuPDF."""
    doc = fitz.open(pdf_path)
    texts = []

    for i in range(page_start - 1, min(page_end, len(doc))):
        page = doc[i]
        text = page.get_text()
        if text.strip():
            texts.append(f"--- Pagina {i+1} ---\n{text}")

    doc.close()
    return "\n\n".join(texts)


def compress_sample(text: str, max_chars: int = MAX_CHARS_PER_SAMPLE) -> str:
    """Comprimi campione per LLM."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...]\n\n" + text[-half:]


async def call_llm_classify(samples: list[dict], api_key: str) -> dict:
    """Chiama LLM per classificare il documento."""
    samples_text = ""
    for i, s in enumerate(samples):
        samples_text += f"\n=== CAMPIONE {i} (pagine {s['page_start']}-{s['page_end']}) ===\n"
        samples_text += compress_sample(s['text'])
        samples_text += "\n"

    prompt = DOC_INTEL_PROMPT.format(samples=samples_text)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=120.0,
        )

        if response.status_code != 200:
            raise Exception(f"LLM HTTP {response.status_code}: {response.text[:500]}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON
        # Remove markdown code blocks if present
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        content = content.strip()

        return json.loads(content)


async def process_document(
    conn: asyncpg.Connection,
    manifest_id: int,
    filename: str,
    windows: list[dict],
    api_key: str,
) -> dict | None:
    """Processa un singolo documento."""
    # Find PDF
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        pdf_path = PDF_DIR / "new" / filename
    if not pdf_path.exists():
        print(f"    [ERROR] PDF non trovato: {filename}")
        return None

    samples = []

    # Extract each window locally
    for w in windows:
        text = extract_window_text(pdf_path, w['page_start'], w['page_end'])
        samples.append({
            'band_index': w['band_index'],
            'page_start': w['page_start'],
            'page_end': w['page_end'],
            'text': text,
            'char_count': len(text),
        })

    total_chars = sum(s['char_count'] for s in samples)
    print(f"    Estratti {len(samples)} campioni, {total_chars} chars totali")

    # Call LLM
    print(f"    Chiamando LLM ({LLM_MODEL})...")
    classification = await call_llm_classify(samples, api_key)

    # Save to database
    await save_classification(conn, manifest_id, classification, samples)

    return classification


async def save_classification(
    conn: asyncpg.Connection,
    manifest_id: int,
    classification: dict,
    samples: list[dict],
):
    """Salva classificazione nel database."""
    # Ensure tables exist
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

    # Update ingestion_profiles
    suggested_profile = classification.get('suggested_profile', 'structured_by_title')
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


async def main(single_pdf: str | None = None, api_key: str | None = None):
    """Main entry point."""
    print("=" * 70)
    print("DOCUMENT INTELLIGENCE - Classificazione LLM (Local Extraction)")
    print("=" * 70)
    print()

    if not api_key:
        api_key = OPENROUTER_API_KEY

    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY non configurata!")
        return

    print(f"Model: {LLM_MODEL}")
    print()

    conn = await asyncpg.connect(DB_URL)

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
            WHERE d.id IS NULL
            GROUP BY m.id, m.filename
            ORDER BY m.filename
        """
        rows = await conn.fetch(query)

    print(f"Documenti da classificare: {len(rows)}")
    print()

    results = []

    for i, row in enumerate(rows):
        manifest_id = row['id']
        filename = row['filename']
        windows = [json.loads(w) if isinstance(w, str) else w for w in row['windows']]

        print(f"[{i+1}/{len(rows)}] {filename}")
        print(f"    Windows: {len(windows)}")

        start = time.time()

        try:
            classification = await process_document(
                conn, manifest_id, filename, windows, api_key
            )

            elapsed = time.time() - start

            if classification:
                doc_type = classification.get('doc_type', '?')
                confidence = classification.get('confidence', 0)
                chunking = classification.get('suggested_chunking', '?')
                profile = classification.get('suggested_profile', '?')

                print(f"    -> doc_type: {doc_type} (conf={confidence:.2f})")
                print(f"    -> chunking: {chunking}")
                print(f"    -> profile: {profile}")
                print(f"    ({elapsed:.1f}s)")

                # Show observations
                for obs in classification.get('sample_observations', [])[:2]:
                    print(f"       B{obs.get('band', '?')}: {obs.get('observation', '')[:60]}...")

                results.append({
                    'filename': filename,
                    'doc_type': doc_type,
                    'confidence': confidence,
                    'chunking': chunking,
                    'profile': profile,
                    'classification': classification,
                })

        except Exception as e:
            print(f"    [ERROR] {e}")

        print()

    await conn.close()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        for r in results:
            print(f"\n{r['filename']}")
            print(f"  doc_type: {r['doc_type']} (conf={r['confidence']:.2f})")
            print(f"  chunking: {r['chunking']}")
            print(f"  profile: {r['profile']}")

            warnings = r['classification'].get('warnings', [])
            if warnings:
                print(f"  warnings: {warnings}")

            anchors = r['classification'].get('massima_anchor_patterns', [])
            if anchors:
                print(f"  anchors: {anchors[:3]}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Document Intelligence Classify (Local)")
    parser.add_argument("--pdf", type=str, help="Singolo PDF da classificare")
    parser.add_argument("--api-key", type=str, help="OpenRouter API key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")

    asyncio.run(main(single_pdf=args.pdf, api_key=api_key))
