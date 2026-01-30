#!/usr/bin/env python3
"""
Compare Cloud Chunking vs Local Extraction.

Confronta i risultati di:
1. Unstructured Cloud chunking (by_title)
2. Local extraction (reference units da Phase 0)

Su 5 PDF eterogenei selezionati per massima diversità.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import asyncpg
import httpx

from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL

# Cloud API (uses env var or default)
UNSTRUCTURED_CLOUD_URL = os.getenv(
    "UNSTRUCTURED_CLOUD_URL",
    "https://api.unstructuredapp.io/general/v0/general",
)
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")

# 5 PDF selezionati per eterogeneità
SELECTED_PDFS = [
    "2014 Mass civile Vol 1 pagg 408.pdf",           # 1 ref unit (problematico)
    "2015 pricipi di diritto processuale Volume 2 massimario Civile_401_650.pdf",  # 165 ref units
    "Rassegna Penale 2011.pdf",                      # 39 ref units
    "Rassegna Civile 2012 - II volume.pdf",          # 17 ref units
    "Rassegna Penale 2012.pdf",                      # 117 ref units
]

# Pattern per identificare massime
MASSIMA_PATTERNS = [
    r"Sez\.\s*[IVX\d]+",           # Sez. I, Sez. II, Sez. Un.
    r"Cass\.",                      # Cassazione
    r"sent\.\s*n\.\s*\d+",          # sent. n. 1234
    r"ord\.\s*n\.\s*\d+",           # ord. n. 1234
    r"Rv\.\s*\d+",                  # Rv. 123456
    r"n\.\s*\d+/\d{2,4}",           # n. 1234/2020
    r"art\.\s*\d+",                 # art. 123
    r"c\.p\.c\.|c\.p\.p\.|c\.c\.",  # codici
]


@dataclass
class ChunkStats:
    """Statistiche per un set di chunks."""
    pdf_name: str
    source: str  # 'cloud_by_title' o 'local'
    chunk_count: int
    avg_chars: float
    min_chars: int
    max_chars: int
    pct_with_massima_pattern: float
    pct_short: float  # < 150 chars
    pct_long: float   # > 2500 chars
    extraction_time: float


def has_massima_pattern(text: str) -> bool:
    """Verifica se il testo contiene pattern tipici di massima."""
    for pattern in MASSIMA_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def analyze_chunks(chunks: list[str], pdf_name: str, source: str, time_taken: float) -> ChunkStats:
    """Analizza una lista di chunks e calcola statistiche."""
    if not chunks:
        return ChunkStats(
            pdf_name=pdf_name,
            source=source,
            chunk_count=0,
            avg_chars=0,
            min_chars=0,
            max_chars=0,
            pct_with_massima_pattern=0,
            pct_short=0,
            pct_long=0,
            extraction_time=time_taken,
        )

    lengths = [len(c) for c in chunks]
    with_pattern = sum(1 for c in chunks if has_massima_pattern(c))
    short = sum(1 for l in lengths if l < 150)
    long_chunks = sum(1 for l in lengths if l > 2500)

    return ChunkStats(
        pdf_name=pdf_name,
        source=source,
        chunk_count=len(chunks),
        avg_chars=sum(lengths) / len(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        pct_with_massima_pattern=100 * with_pattern / len(chunks),
        pct_short=100 * short / len(chunks),
        pct_long=100 * long_chunks / len(chunks),
        extraction_time=time_taken,
    )


async def extract_cloud_by_title(pdf_path: Path) -> tuple[list[str], float]:
    """Estrazione cloud con chunking by_title ottimizzato."""
    print(f"  Cloud by_title: {pdf_path.name}...")

    if not UNSTRUCTURED_API_KEY:
        print("    [SKIP] No UNSTRUCTURED_API_KEY set")
        return [], 0.0

    start = time.time()

    async with httpx.AsyncClient(timeout=1800.0) as client:
        with open(pdf_path, "rb") as f:
            response = await client.post(
                UNSTRUCTURED_CLOUD_URL,
                headers={"unstructured-api-key": UNSTRUCTURED_API_KEY},
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={
                    "strategy": "hi_res",
                    "chunking_strategy": "by_title",
                    "max_characters": "3000",
                    "new_after_n_chars": "2500",
                    "combine_text_under_n_chars": "200",
                    "output_format": "application/json",
                },
            )

    elapsed = time.time() - start

    if response.status_code != 200:
        print(f"    ERROR: {response.status_code} - {response.text[:200]}")
        return [], elapsed

    elements = response.json()
    chunks = [e.get("text", "") for e in elements if e.get("text")]

    print(f"    {len(chunks)} chunks in {elapsed:.1f}s")
    return chunks, elapsed


async def get_local_reference_units(pdf_name: str) -> tuple[list[str], float]:
    """Recupera le reference units locali dal database."""
    print(f"  Local reference: {pdf_name}...")

    start = time.time()

    conn = await asyncpg.connect(DB_URL)
    try:
        # Recupera reference units
        rows = await conn.fetch("""
            SELECT r.testo
            FROM kb.qa_reference_units r
            JOIN kb.pdf_manifest m ON r.manifest_id = m.id
            WHERE m.filename = $1
            ORDER BY r.unit_index
        """, pdf_name)

        chunks = [row["testo"] for row in rows]
    finally:
        await conn.close()

    elapsed = time.time() - start
    print(f"    {len(chunks)} units in {elapsed:.1f}s")
    return chunks, elapsed


def print_comparison_table(stats_list: list[ChunkStats]):
    """Stampa tabella comparativa."""
    print("\n" + "=" * 110)
    print("CONFRONTO CLOUD BY_TITLE vs LOCAL REFERENCE UNITS")
    print("=" * 110)

    # Raggruppa per PDF
    by_pdf = {}
    for s in stats_list:
        if s.pdf_name not in by_pdf:
            by_pdf[s.pdf_name] = {}
        by_pdf[s.pdf_name][s.source] = s

    print(f"\n{'PDF':<45} {'Source':<18} {'Chunks':>8} {'Avg':>8} {'%Mass':>8} {'%Short':>8} {'Time':>8}")
    print("-" * 110)

    for pdf_name, sources in by_pdf.items():
        short_name = pdf_name[:43] if len(pdf_name) > 43 else pdf_name
        for source, stats in sources.items():
            print(f"{short_name:<45} {source:<18} {stats.chunk_count:>8} {stats.avg_chars:>8.0f} {stats.pct_with_massima_pattern:>7.1f}% {stats.pct_short:>7.1f}% {stats.extraction_time:>7.1f}s")
        print()

    # Summary
    cloud_stats = [s for s in stats_list if s.source == 'cloud_by_title']
    local_stats = [s for s in stats_list if s.source == 'local']

    if cloud_stats and local_stats:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        avg_cloud_massima = sum(s.pct_with_massima_pattern for s in cloud_stats) / len(cloud_stats) if cloud_stats else 0
        avg_local_massima = sum(s.pct_with_massima_pattern for s in local_stats) / len(local_stats) if local_stats else 0

        avg_cloud_short = sum(s.pct_short for s in cloud_stats) / len(cloud_stats) if cloud_stats else 0
        avg_local_short = sum(s.pct_short for s in local_stats) / len(local_stats) if local_stats else 0

        avg_cloud_chars = sum(s.avg_chars for s in cloud_stats) / len(cloud_stats) if cloud_stats else 0
        avg_local_chars = sum(s.avg_chars for s in local_stats) / len(local_stats) if local_stats else 0

        total_cloud_chunks = sum(s.chunk_count for s in cloud_stats)
        total_local_chunks = sum(s.chunk_count for s in local_stats)

        total_cloud_time = sum(s.extraction_time for s in cloud_stats)
        total_local_time = sum(s.extraction_time for s in local_stats)

        print(f"\nMetric                    Cloud by_title    Local Reference")
        print(f"-" * 70)
        print(f"Total chunks              {total_cloud_chunks:>14}    {total_local_chunks:>14}")
        print(f"Avg chars per chunk       {avg_cloud_chars:>14.0f}    {avg_local_chars:>14.0f}")
        print(f"Avg % with massima        {avg_cloud_massima:>13.1f}%   {avg_local_massima:>13.1f}%")
        print(f"Avg % short (<150)        {avg_cloud_short:>13.1f}%   {avg_local_short:>13.1f}%")
        print(f"Total time                {total_cloud_time:>13.1f}s   {total_local_time:>13.1f}s")

        # Analisi
        print("\n" + "-" * 70)
        print("ANALISI:")

        if avg_local_chars > 10000:
            print("  [!] LOCAL ha chunk molto grandi - possibile under-segmentation")

        if avg_cloud_short > 20:
            print("  [!] CLOUD ha molti chunk corti - possibile over-segmentation")

        if total_local_chunks < total_cloud_chunks * 0.5:
            print("  [!] LOCAL produce molti meno chunk - segmentazione conservativa")

        if avg_local_massima < avg_cloud_massima * 0.8:
            print("  [!] LOCAL ha meno pattern massima - possibile perdita informazione")

        # Winner (euristico)
        cloud_score = (
            (100 - avg_cloud_short) * 0.3 +  # Meno short è meglio
            avg_cloud_massima * 0.4 +         # Più pattern è meglio
            (1 if 2000 < avg_cloud_chars < 5000 else 0) * 30  # Char range ideale
        )
        local_score = (
            (100 - avg_local_short) * 0.3 +
            avg_local_massima * 0.4 +
            (1 if 2000 < avg_local_chars < 5000 else 0) * 30
        )

        print(f"\n  Score euristico: Cloud={cloud_score:.1f}, Local={local_score:.1f}")


async def run_comparison():
    """Esegue il confronto completo."""
    all_stats = []

    for pdf_name in SELECTED_PDFS:
        pdf_path = PDF_DIR / pdf_name

        if not pdf_path.exists():
            print(f"\n[SKIP] {pdf_name} - file non trovato")
            continue

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")
        print(f"{'='*60}")

        # Cloud extraction (se API key disponibile)
        cloud_chunks, cloud_time = await extract_cloud_by_title(pdf_path)
        if cloud_chunks:
            cloud_stats = analyze_chunks(cloud_chunks, pdf_name, "cloud_by_title", cloud_time)
            all_stats.append(cloud_stats)

        # Local reference units
        local_chunks, local_time = await get_local_reference_units(pdf_name)
        if local_chunks:
            local_stats = analyze_chunks(local_chunks, pdf_name, "local", local_time)
            all_stats.append(local_stats)

    # Print comparison
    print_comparison_table(all_stats)

    return all_stats


if __name__ == "__main__":
    print("Confronto Cloud Chunking vs Local Extraction")
    print("5 PDF eterogenei selezionati")
    print(f"Cloud URL: {UNSTRUCTURED_CLOUD_URL}")
    print(f"API Key: {'SET' if UNSTRUCTURED_API_KEY else 'NOT SET'}")
    print()

    asyncio.run(run_comparison())
