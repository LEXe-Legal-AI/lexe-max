#!/usr/bin/env python3
"""
Document Intelligence - Phase 0.1: Pre-scan locale per page stats.

Usa PyMuPDF (veloce, gratis) per calcolare statistiche per pagina.
Output: page_stats per ogni PDF, usate per scegliere le 5 finestre migliori.

Usage:
    uv run python scripts/qa/s0_doc_intel_prescan.py
    uv run python scripts/qa/s0_doc_intel_prescan.py --pdf "Volume I_2020_Massimario_Civile.pdf"
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import asyncpg
import fitz  # PyMuPDF

from qa_config import DB_URL, PDF_DIR


@dataclass
class PageStats:
    """Statistiche per singola pagina."""
    page_num: int  # 1-based
    char_count: int
    word_count: int
    line_count: int
    valid_chars_ratio: float  # % caratteri alfanumerici
    non_alnum_ratio: float
    is_empty: bool  # < 50 chars
    is_ocr_candidate: bool  # valid_chars_ratio < 0.7
    is_toc_candidate: bool  # puntini, numeri a fine riga
    toc_signals: int  # count of TOC-like patterns
    info_score: float  # score complessivo (0-1)


def analyze_page(page: fitz.Page, page_num: int) -> PageStats:
    """Analizza una singola pagina e ritorna statistiche."""
    text = page.get_text()

    char_count = len(text)
    word_count = len(text.split())
    lines = text.strip().split('\n')
    line_count = len(lines)

    # Valid chars ratio (alfanumerici + spazi)
    if char_count > 0:
        alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
        valid_chars_ratio = alnum_count / char_count
        non_alnum_ratio = 1 - valid_chars_ratio
    else:
        valid_chars_ratio = 0.0
        non_alnum_ratio = 1.0

    # Flags
    is_empty = char_count < 50
    is_ocr_candidate = valid_chars_ratio < 0.7 and char_count > 100

    # TOC signals
    toc_signals = 0
    toc_patterns = [
        r"\.{3,}",  # puntini ...
        r"\.\s*\d+\s*$",  # numero a fine riga
        r"^[IVXLCDM]+\s*[\.\-]",  # numeri romani
        r"^\d+\.\s+[A-Z]",  # 1. TITOLO
        r"(?:indice|sommario|capitolo|sezione)",  # keywords
    ]

    for line in lines:
        for pattern in toc_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                toc_signals += 1
                break

    is_toc_candidate = toc_signals > line_count * 0.3 if line_count > 5 else False

    # Info score (0-1): quanto è "informativa" la pagina
    # Penalizza: vuoto, OCR bad, TOC
    # Premia: testo abbondante, buon ratio
    info_score = 0.0
    if not is_empty:
        # Base: normalizza char_count (assume 3000 chars = pagina piena)
        info_score = min(char_count / 3000, 1.0) * 0.4
        # Bonus: valid chars
        info_score += valid_chars_ratio * 0.3
        # Penalità: TOC
        if is_toc_candidate:
            info_score *= 0.3
        # Penalità: OCR candidate
        if is_ocr_candidate:
            info_score *= 0.5
        # Bonus: word density (testo vero, non garbage)
        word_density = word_count / max(line_count, 1)
        if word_density > 5:  # ~5+ parole per riga = testo normale
            info_score += 0.3

    info_score = min(max(info_score, 0.0), 1.0)

    return PageStats(
        page_num=page_num,
        char_count=char_count,
        word_count=word_count,
        line_count=line_count,
        valid_chars_ratio=round(valid_chars_ratio, 3),
        non_alnum_ratio=round(non_alnum_ratio, 3),
        is_empty=is_empty,
        is_ocr_candidate=is_ocr_candidate,
        is_toc_candidate=is_toc_candidate,
        toc_signals=toc_signals,
        info_score=round(info_score, 3),
    )


def prescan_pdf(pdf_path: Path) -> list[PageStats]:
    """Pre-scan completo di un PDF, ritorna stats per ogni pagina."""
    doc = fitz.open(pdf_path)
    stats = []

    for i in range(len(doc)):
        page = doc[i]
        page_stats = analyze_page(page, i + 1)  # 1-based
        stats.append(page_stats)

    doc.close()
    return stats


def select_sample_windows(
    page_stats: list[PageStats],
    num_windows: int = 5,
    window_size: int = 15,
    max_empty_pct: float = 0.3,
) -> list[dict]:
    """
    Seleziona le migliori finestre di campionamento.

    Divide il documento in N bande, trova la pagina più informativa
    in ogni banda, centra una finestra attorno ad essa.

    Returns:
        Lista di dict con: band_index, center_page, page_start, page_end,
                          avg_info_score, empty_pct
    """
    total_pages = len(page_stats)

    # Adjust for small documents
    if total_pages < 75:
        num_windows = 3
    if total_pages < 45:
        num_windows = 2
        window_size = 10
    if total_pages < 20:
        num_windows = 1
        window_size = total_pages

    windows = []
    band_size = total_pages / num_windows

    for band_idx in range(num_windows):
        # Range della banda
        band_start = int(band_idx * band_size)
        band_end = int((band_idx + 1) * band_size)
        band_pages = page_stats[band_start:band_end]

        if not band_pages:
            continue

        # Trova pagina con max info_score nella banda
        best_page = max(band_pages, key=lambda p: p.info_score)
        center = best_page.page_num  # 1-based

        # Calcola finestra centrata
        half_window = window_size // 2
        page_start = max(1, center - half_window)
        page_end = min(total_pages, page_start + window_size - 1)

        # Aggiusta se finestra va oltre
        if page_end - page_start + 1 < window_size:
            page_start = max(1, page_end - window_size + 1)

        # Stats della finestra
        window_pages = page_stats[page_start - 1:page_end]
        empty_count = sum(1 for p in window_pages if p.is_empty)
        empty_pct = empty_count / len(window_pages) if window_pages else 1.0
        avg_info = sum(p.info_score for p in window_pages) / len(window_pages) if window_pages else 0

        # Se troppo vuota, shift verso zona più densa
        if empty_pct > max_empty_pct:
            # Trova la sotto-finestra più densa nella banda
            best_start = page_start
            best_score = avg_info

            for shift_start in range(band_start + 1, band_end - window_size + 2):
                shift_pages = page_stats[shift_start - 1:shift_start - 1 + window_size]
                shift_empty = sum(1 for p in shift_pages if p.is_empty) / len(shift_pages)
                shift_score = sum(p.info_score for p in shift_pages) / len(shift_pages)

                if shift_empty < empty_pct and shift_score > best_score:
                    best_start = shift_start
                    best_score = shift_score

            page_start = best_start
            page_end = min(total_pages, page_start + window_size - 1)
            window_pages = page_stats[page_start - 1:page_end]
            empty_pct = sum(1 for p in window_pages if p.is_empty) / len(window_pages)
            avg_info = sum(p.info_score for p in window_pages) / len(window_pages)

        windows.append({
            'band_index': band_idx,
            'center_page': center,
            'page_start': page_start,
            'page_end': page_end,
            'window_size': page_end - page_start + 1,
            'avg_info_score': round(avg_info, 3),
            'empty_pct': round(empty_pct, 3),
        })

    return windows


async def save_prescan_results(
    conn: asyncpg.Connection,
    manifest_id: int,
    page_stats: list[PageStats],
    windows: list[dict],
):
    """Salva risultati pre-scan nel database."""
    # Salva page stats in page_extraction_stats (se tabella esiste)
    # Per ora salviamo solo le windows selezionate

    # Delete existing windows for this manifest
    await conn.execute(
        "DELETE FROM kb.qa_sample_windows WHERE manifest_id = $1",
        manifest_id
    )

    # Insert new windows
    for w in windows:
        await conn.execute(
            """
            INSERT INTO kb.qa_sample_windows
              (manifest_id, band_index, page_start, page_end,
               avg_info_score, empty_pct, selection_method)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            manifest_id,
            w['band_index'],
            w['page_start'],
            w['page_end'],
            w['avg_info_score'],
            w['empty_pct'],
            'info_score_v1',
        )


async def main(single_pdf: str | None = None):
    """Main entry point."""
    print("=" * 70)
    print("DOCUMENT INTELLIGENCE - Pre-scan Locale")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Ensure table exists
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.qa_sample_windows (
            id SERIAL PRIMARY KEY,
            manifest_id INTEGER REFERENCES kb.pdf_manifest(id),
            band_index INTEGER NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            avg_info_score REAL,
            empty_pct REAL,
            selection_method TEXT DEFAULT 'info_score_v1',
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE(manifest_id, band_index)
        )
    """)

    # Get PDFs to process
    if single_pdf:
        rows = await conn.fetch(
            "SELECT id, filename FROM kb.pdf_manifest WHERE filename = $1",
            single_pdf
        )
    else:
        rows = await conn.fetch(
            "SELECT id, filename FROM kb.pdf_manifest ORDER BY filename"
        )

    print(f"PDF da analizzare: {len(rows)}")
    print()

    results = []

    for i, row in enumerate(rows):
        manifest_id = row['id']
        filename = row['filename']

        # Find PDF
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"[{i+1}/{len(rows)}] [SKIP] {filename}: non trovato")
            continue

        start = time.time()

        # Pre-scan
        page_stats = prescan_pdf(pdf_path)

        # Select windows
        windows = select_sample_windows(page_stats)

        elapsed = time.time() - start

        # Summary
        total_pages = len(page_stats)
        empty_pages = sum(1 for p in page_stats if p.is_empty)
        toc_pages = sum(1 for p in page_stats if p.is_toc_candidate)
        ocr_pages = sum(1 for p in page_stats if p.is_ocr_candidate)
        avg_info = sum(p.info_score for p in page_stats) / total_pages if total_pages else 0

        print(f"[{i+1}/{len(rows)}] {filename[:50]:50}")
        print(f"         {total_pages} pag, {empty_pages} vuote, {toc_pages} TOC, {ocr_pages} OCR")
        print(f"         avg_info={avg_info:.2f}, windows={len(windows)}, {elapsed:.1f}s")

        for w in windows:
            print(f"           B{w['band_index']}: pag {w['page_start']}-{w['page_end']} "
                  f"(info={w['avg_info_score']:.2f}, empty={w['empty_pct']:.0%})")

        # Save to DB
        await save_prescan_results(conn, manifest_id, page_stats, windows)

        results.append({
            'filename': filename,
            'total_pages': total_pages,
            'empty_pages': empty_pages,
            'toc_pages': toc_pages,
            'ocr_pages': ocr_pages,
            'avg_info_score': round(avg_info, 3),
            'windows': windows,
        })

    await conn.close()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_windows = sum(len(r['windows']) for r in results)
    total_pages_sampled = sum(
        sum(w['page_end'] - w['page_start'] + 1 for w in r['windows'])
        for r in results
    )

    print(f"PDF analizzati: {len(results)}")
    print(f"Finestre selezionate: {total_windows}")
    print(f"Pagine da campionare: {total_pages_sampled}")
    print()

    # Docs with issues
    high_empty = [r for r in results if r['empty_pages'] / r['total_pages'] > 0.1]
    high_ocr = [r for r in results if r['ocr_pages'] / r['total_pages'] > 0.2]
    high_toc = [r for r in results if r['toc_pages'] / r['total_pages'] > 0.1]

    if high_empty:
        print(f"[!] {len(high_empty)} PDF con >10% pagine vuote")
    if high_ocr:
        print(f"[!] {len(high_ocr)} PDF con >20% pagine OCR candidate")
    if high_toc:
        print(f"[!] {len(high_toc)} PDF con >10% pagine TOC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Intelligence Pre-scan")
    parser.add_argument("--pdf", type=str, help="Singolo PDF da analizzare")
    args = parser.parse_args()

    asyncio.run(main(single_pdf=args.pdf))
