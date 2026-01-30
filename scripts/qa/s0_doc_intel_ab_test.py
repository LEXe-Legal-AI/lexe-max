#!/usr/bin/env python3
"""
Document Intelligence - A/B Test con temperature 0.0 vs 0.1

Run A: temperature=0.0, top_p=1.0 (deterministico)
Run B: temperature=0.1, top_p=0.5 (quasi-deterministico)

Salva entrambi i run nel DB per confronto.

Usage:
    uv run python scripts/qa/s0_doc_intel_ab_test.py --api-key "sk-or-..."
    uv run python scripts/qa/s0_doc_intel_ab_test.py --api-key "sk-or-..." --run-only A
    uv run python scripts/qa/s0_doc_intel_ab_test.py --compare-only
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import asyncpg
import fitz
import httpx

from qa_config import DB_URL, PDF_DIR, OPENROUTER_API_KEY, OPENROUTER_URL

LLM_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
MAX_CHARS_PER_SAMPLE = 4000


# ═══════════════════════════════════════════════════════════════════
# PROMPT - Vincoli forti per output deterministico
# ═══════════════════════════════════════════════════════════════════

DOC_INTEL_PROMPT = """Analizza questi 5 campioni da un documento legale italiano (massimario Corte di Cassazione).

CAMPIONI:
{samples}

OUTPUT: Solo JSON valido. No markdown. No chiavi extra. Scegli UNA sola label per ogni campo.

{{
  "doc_type": "<list_only|massima_plus_commentary|mixed|toc_heavy|ocr_needed>",
  "profile": "<structured_by_title|structured_parent_child|baseline_toc_filter|legacy_layout|list_pure|mixed_hybrid>",
  "chunking_strategy": "<by_title|by_similarity|pattern_based|parent_child>",
  "confidence": <0.0-1.0>,
  "suggested_gate": {{
    "min_length": <80-300>,
    "citation_ratio": <0.01-0.10>,
    "skip_pages_start": <0-20>,
    "skip_pages_end": <0-15>
  }},
  "toc_heavy_score": <0.0-1.0>,
  "commentary_link_likelihood": <0.0-1.0>,
  "anchor_patterns": ["<regex1>", "<regex2>", "<regex3>"],
  "evidence": [
    {{"band": 0, "page_range": "<start>-<end>", "quote": "<30 chars>", "observation": "<cosa contiene>"}},
    {{"band": 1, "page_range": "<start>-<end>", "quote": "<30 chars>", "observation": "<cosa contiene>"}}
  ],
  "warnings": ["<problema1>", "<problema2>"]
}}

DEFINIZIONI:
- list_only: Elenco massime brevi senza spiegazioni
- massima_plus_commentary: Ogni massima ha commento esteso (pagine)
- mixed: Sezioni con e senza commento
- toc_heavy: Molto indice/TOC infiltrato
- ocr_needed: Testo illeggibile, OCR scarso"""


@dataclass
class RunConfig:
    """Configurazione per un run A/B."""
    name: str
    temperature: float
    top_p: float


RUN_A = RunConfig(name="A", temperature=0.0, top_p=1.0)
RUN_B = RunConfig(name="B", temperature=0.1, top_p=0.5)


def extract_window_text(pdf_path: Path, page_start: int, page_end: int) -> str:
    """Estrae testo da una finestra di pagine."""
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
    """Comprimi campione."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[...]\n" + text[-half:]


async def call_llm(samples: list[dict], api_key: str, config: RunConfig) -> dict:
    """Chiama LLM con configurazione specifica."""
    samples_text = ""
    for i, s in enumerate(samples):
        samples_text += f"\n=== CAMPIONE {i} (pag {s['page_start']}-{s['page_end']}) ===\n"
        samples_text += compress_sample(s['text'])

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
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": 1500,
            },
            timeout=120.0,
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text[:300]}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        content = content.strip()

        return json.loads(content)


async def ensure_ab_tables(conn: asyncpg.Connection):
    """Crea tabelle per A/B test."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.doc_intel_ab_runs (
            id SERIAL PRIMARY KEY,
            run_name TEXT NOT NULL,  -- 'A' or 'B'
            temperature REAL,
            top_p REAL,
            model TEXT,
            started_at TIMESTAMPTZ DEFAULT now(),
            completed_at TIMESTAMPTZ,
            doc_count INTEGER,
            UNIQUE(run_name)
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.doc_intel_ab_results (
            id SERIAL PRIMARY KEY,
            run_name TEXT NOT NULL,
            manifest_id INTEGER REFERENCES kb.pdf_manifest(id),
            filename TEXT,
            doc_type TEXT,
            profile TEXT,
            chunking_strategy TEXT,
            confidence REAL,
            min_length INTEGER,
            citation_ratio REAL,
            skip_start INTEGER,
            skip_end INTEGER,
            toc_heavy_score REAL,
            commentary_likelihood REAL,
            anchor_patterns JSONB,
            evidence JSONB,
            warnings JSONB,
            raw_json JSONB,
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE(run_name, manifest_id)
        )
    """)


async def save_result(conn: asyncpg.Connection, run_name: str, manifest_id: int,
                      filename: str, result: dict):
    """Salva risultato di un documento."""
    gate = result.get('suggested_gate', {})

    await conn.execute("""
        INSERT INTO kb.doc_intel_ab_results
          (run_name, manifest_id, filename, doc_type, profile, chunking_strategy,
           confidence, min_length, citation_ratio, skip_start, skip_end,
           toc_heavy_score, commentary_likelihood, anchor_patterns, evidence,
           warnings, raw_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        ON CONFLICT (run_name, manifest_id) DO UPDATE SET
          doc_type = EXCLUDED.doc_type,
          profile = EXCLUDED.profile,
          chunking_strategy = EXCLUDED.chunking_strategy,
          confidence = EXCLUDED.confidence,
          min_length = EXCLUDED.min_length,
          citation_ratio = EXCLUDED.citation_ratio,
          skip_start = EXCLUDED.skip_start,
          skip_end = EXCLUDED.skip_end,
          toc_heavy_score = EXCLUDED.toc_heavy_score,
          commentary_likelihood = EXCLUDED.commentary_likelihood,
          anchor_patterns = EXCLUDED.anchor_patterns,
          evidence = EXCLUDED.evidence,
          warnings = EXCLUDED.warnings,
          raw_json = EXCLUDED.raw_json,
          created_at = now()
    """,
        run_name,
        manifest_id,
        filename,
        result.get('doc_type'),
        result.get('profile'),
        result.get('chunking_strategy'),
        result.get('confidence'),
        gate.get('min_length'),
        gate.get('citation_ratio'),
        gate.get('skip_pages_start'),
        gate.get('skip_pages_end'),
        result.get('toc_heavy_score'),
        result.get('commentary_link_likelihood'),
        json.dumps(result.get('anchor_patterns', [])),
        json.dumps(result.get('evidence', [])),
        json.dumps(result.get('warnings', [])),
        json.dumps(result),
    )


async def run_classification(conn: asyncpg.Connection, api_key: str, config: RunConfig):
    """Esegue classificazione per tutti i documenti."""
    print(f"\n{'='*70}")
    print(f"RUN {config.name}: temperature={config.temperature}, top_p={config.top_p}")
    print(f"{'='*70}\n")

    # Get documents with windows
    rows = await conn.fetch("""
        SELECT m.id, m.filename,
               array_agg(json_build_object(
                   'band_index', w.band_index,
                   'page_start', w.page_start,
                   'page_end', w.page_end
               ) ORDER BY w.band_index) as windows
        FROM kb.pdf_manifest m
        JOIN kb.qa_sample_windows w ON w.manifest_id = m.id
        GROUP BY m.id, m.filename
        ORDER BY m.filename
    """)

    print(f"Documenti: {len(rows)}")

    # Register run
    await conn.execute("""
        INSERT INTO kb.doc_intel_ab_runs (run_name, temperature, top_p, model, doc_count)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (run_name) DO UPDATE SET
          temperature = EXCLUDED.temperature,
          top_p = EXCLUDED.top_p,
          started_at = now(),
          completed_at = NULL,
          doc_count = EXCLUDED.doc_count
    """, config.name, config.temperature, config.top_p, LLM_MODEL, len(rows))

    success = 0
    errors = 0

    for i, row in enumerate(rows):
        manifest_id = row['id']
        filename = row['filename']
        windows = [json.loads(w) if isinstance(w, str) else w for w in row['windows']]

        # Find PDF
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"[{i+1}/{len(rows)}] {filename[:45]:45} SKIP (not found)")
            continue

        try:
            # Extract samples
            samples = []
            for w in windows:
                text = extract_window_text(pdf_path, w['page_start'], w['page_end'])
                samples.append({
                    'band_index': w['band_index'],
                    'page_start': w['page_start'],
                    'page_end': w['page_end'],
                    'text': text,
                })

            # Call LLM
            start = time.time()
            result = await call_llm(samples, api_key, config)
            elapsed = time.time() - start

            # Save
            await save_result(conn, config.name, manifest_id, filename, result)

            doc_type = result.get('doc_type', '?')
            profile = result.get('profile', '?')
            conf = result.get('confidence', 0)

            print(f"[{i+1}/{len(rows)}] {filename[:40]:40} {doc_type:25} {profile:25} {conf:.2f} ({elapsed:.1f}s)")
            success += 1

        except Exception as e:
            print(f"[{i+1}/{len(rows)}] {filename[:40]:40} ERROR: {str(e)[:50]}")
            errors += 1

        # Rate limiting
        await asyncio.sleep(0.5)

    # Mark complete
    await conn.execute("""
        UPDATE kb.doc_intel_ab_runs SET completed_at = now() WHERE run_name = $1
    """, config.name)

    print(f"\nRun {config.name} completato: {success} OK, {errors} errori")
    return success, errors


async def compare_runs(conn: asyncpg.Connection):
    """Confronta Run A vs Run B e genera report."""
    print("\n" + "="*70)
    print("CONFRONTO RUN A vs RUN B")
    print("="*70 + "\n")

    # Get results from both runs
    rows = await conn.fetch("""
        SELECT
            a.filename,
            a.doc_type as a_doc_type, b.doc_type as b_doc_type,
            a.profile as a_profile, b.profile as b_profile,
            a.chunking_strategy as a_chunking, b.chunking_strategy as b_chunking,
            a.confidence as a_conf, b.confidence as b_conf,
            a.min_length as a_min_len, b.min_length as b_min_len,
            a.citation_ratio as a_cit_ratio, b.citation_ratio as b_cit_ratio,
            a.skip_start as a_skip_start, b.skip_start as b_skip_start,
            a.skip_end as a_skip_end, b.skip_end as b_skip_end,
            a.toc_heavy_score as a_toc, b.toc_heavy_score as b_toc
        FROM kb.doc_intel_ab_results a
        JOIN kb.doc_intel_ab_results b ON a.manifest_id = b.manifest_id
        WHERE a.run_name = 'A' AND b.run_name = 'B'
        ORDER BY a.filename
    """)

    if not rows:
        print("Nessun risultato da confrontare. Esegui prima entrambi i run.")
        return

    total = len(rows)
    print(f"Documenti confrontati: {total}\n")

    # 1. Label flip rate
    doc_type_flips = sum(1 for r in rows if r['a_doc_type'] != r['b_doc_type'])
    profile_flips = sum(1 for r in rows if r['a_profile'] != r['b_profile'])
    chunking_flips = sum(1 for r in rows if r['a_chunking'] != r['b_chunking'])

    print("1. LABEL FLIP RATE")
    print("-" * 40)
    print(f"   doc_type flip:  {doc_type_flips:3}/{total} ({100*doc_type_flips/total:.1f}%)")
    print(f"   profile flip:   {profile_flips:3}/{total} ({100*profile_flips/total:.1f}%)")
    print(f"   chunking flip:  {chunking_flips:3}/{total} ({100*chunking_flips/total:.1f}%)")

    # 2. Param drift score
    print("\n2. PARAM DRIFT SCORE")
    print("-" * 40)

    drift_scores = []
    for r in rows:
        score = 0
        details = []

        # min_length diff >= 20
        if r['a_min_len'] and r['b_min_len']:
            if abs(r['a_min_len'] - r['b_min_len']) >= 20:
                score += 1
                details.append(f"min_len:{r['a_min_len']}→{r['b_min_len']}")

        # citation_ratio diff >= 0.01
        if r['a_cit_ratio'] and r['b_cit_ratio']:
            if abs(r['a_cit_ratio'] - r['b_cit_ratio']) >= 0.01:
                score += 1
                details.append(f"cit_ratio:{r['a_cit_ratio']:.2f}→{r['b_cit_ratio']:.2f}")

        # chunking change
        if r['a_chunking'] != r['b_chunking']:
            score += 1
            details.append(f"chunking:{r['a_chunking']}→{r['b_chunking']}")

        # profile change
        if r['a_profile'] != r['b_profile']:
            score += 1
            details.append(f"profile:{r['a_profile']}→{r['b_profile']}")

        drift_scores.append((r['filename'], score, details))

    # Top 10 drift
    drift_scores.sort(key=lambda x: -x[1])
    print("   Top 10 documenti con maggior drift:")
    for fname, score, details in drift_scores[:10]:
        if score > 0:
            print(f"   [{score}] {fname[:45]:45} {', '.join(details)}")

    # 3. Confidence stability
    print("\n3. CONFIDENCE STABILITY")
    print("-" * 40)

    conf_diffs = [abs(r['a_conf'] - r['b_conf']) for r in rows if r['a_conf'] and r['b_conf']]
    if conf_diffs:
        avg_diff = sum(conf_diffs) / len(conf_diffs)
        max_diff = max(conf_diffs)
        print(f"   Media diff confidence: {avg_diff:.3f}")
        print(f"   Max diff confidence:   {max_diff:.3f}")

    # 4. High confidence disagreement
    print("\n4. HIGH CONFIDENCE DISAGREEMENT (conf > 0.8)")
    print("-" * 40)

    high_conf_disagree = [
        r for r in rows
        if r['a_conf'] and r['b_conf']
        and r['a_conf'] > 0.8 and r['b_conf'] > 0.8
        and r['a_profile'] != r['b_profile']
    ]
    print(f"   Documenti con conf>0.8 in entrambi ma profilo diverso: {len(high_conf_disagree)}")
    for r in high_conf_disagree[:5]:
        print(f"   - {r['filename'][:40]:40} A:{r['a_profile']} B:{r['b_profile']}")

    # 5. Distribution summary
    print("\n5. DISTRIBUZIONE doc_type")
    print("-" * 40)

    from collections import Counter
    a_types = Counter(r['a_doc_type'] for r in rows)
    b_types = Counter(r['b_doc_type'] for r in rows)

    all_types = set(a_types.keys()) | set(b_types.keys())
    print(f"   {'doc_type':<30} {'Run A':>8} {'Run B':>8}")
    for t in sorted(all_types):
        print(f"   {t:<30} {a_types.get(t, 0):>8} {b_types.get(t, 0):>8}")

    print("\n6. DISTRIBUZIONE profile")
    print("-" * 40)

    a_profiles = Counter(r['a_profile'] for r in rows)
    b_profiles = Counter(r['b_profile'] for r in rows)

    all_profiles = set(a_profiles.keys()) | set(b_profiles.keys())
    print(f"   {'profile':<30} {'Run A':>8} {'Run B':>8}")
    for p in sorted(all_profiles):
        print(f"   {p:<30} {a_profiles.get(p, 0):>8} {b_profiles.get(p, 0):>8}")

    # 7. Recommendation
    print("\n" + "="*70)
    print("RACCOMANDAZIONE")
    print("="*70)

    total_flips = doc_type_flips + profile_flips
    if total_flips == 0:
        print("✓ Run A e B producono risultati IDENTICI. Usa temperature=0.0 (deterministico).")
    elif total_flips <= total * 0.05:
        print(f"~ {total_flips} flip su {total} ({100*total_flips/total:.1f}%). Differenza minima.")
        print("  Usa temperature=0.0 per riproducibilità, 0.1 se vuoi più variabilità.")
    else:
        print(f"⚠ {total_flips} flip su {total} ({100*total_flips/total:.1f}%). Differenza significativa!")
        if len(high_conf_disagree) > 0:
            print("  ⚠ Ci sono disagreement anche con alta confidence. temperature=0.1 aggiunge rumore.")
        print("  Raccomandazione: usa temperature=0.0")


async def main(api_key: str | None = None, run_only: str | None = None, compare_only: bool = False):
    """Main entry point."""
    conn = await asyncpg.connect(DB_URL)
    await ensure_ab_tables(conn)

    if compare_only:
        await compare_runs(conn)
        await conn.close()
        return

    if not api_key:
        print("ERROR: --api-key richiesto")
        await conn.close()
        return

    # First ensure all docs have windows (run prescan if needed)
    window_count = await conn.fetchval("SELECT count(*) FROM kb.qa_sample_windows")
    manifest_count = await conn.fetchval("SELECT count(*) FROM kb.pdf_manifest")

    if window_count < manifest_count:
        print(f"ATTENZIONE: Solo {window_count}/{manifest_count} documenti hanno finestre.")
        print("Esegui prima: uv run python scripts/qa/s0_doc_intel_prescan.py")
        await conn.close()
        return

    # Run A and/or B
    if run_only is None or run_only == 'A':
        await run_classification(conn, api_key, RUN_A)

    if run_only is None or run_only == 'B':
        await run_classification(conn, api_key, RUN_B)

    # Compare if both runs done
    if run_only is None:
        await compare_runs(conn)

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Intelligence A/B Test")
    parser.add_argument("--api-key", type=str, help="OpenRouter API key")
    parser.add_argument("--run-only", type=str, choices=['A', 'B'], help="Esegui solo un run")
    parser.add_argument("--compare-only", action="store_true", help="Solo confronto (no LLM)")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")

    asyncio.run(main(api_key=api_key, run_only=args.run_only, compare_only=args.compare_only))
