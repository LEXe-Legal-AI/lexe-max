#!/usr/bin/env python3
"""
Dry-Run Statistico Citation-Anchored con Guardrail Stretti

Calcola metriche dettagliate per ogni documento PRIMA di scrivere sul DB.
Include soglie PASS/FAIL per evitare "massime-spam".

Guardrail:
- min_char = 180 (non 80)
- max_char = 1400
- max_massime_per_page = 25
- dedupe per content_hash
- anchor_quality: almeno 2 tra Sez., n., Rv., anno

Soglie FAIL:
- pct_short(<180) > 8% → FAIL
- massime/page p95 > 25 → FAIL
- duplicates_by_hash_pct > 3% → FAIL
- pct_with_complete_citation < 50% → WARNING

Usage:
    uv run python scripts/qa/dryrun_civile_statistical.py
    uv run python scripts/qa/dryrun_civile_statistical.py --doc "Volume I_2016"
"""

import argparse
import asyncio
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, quantiles

import asyncpg
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lexe_api.kb.ingestion.massima_extractor import (
    CITATION_ANCHOR_PATTERNS,
    CitationAnchor,
    ExtractedMassima,
    extract_citation,
    find_citation_anchors,
    split_into_sentences,
)
from lexe_api.kb.ingestion.cleaner import clean_legal_text, compute_content_hash, normalize_for_hash
from qa_config import DB_URL, PDF_DIR


# ============================================================
# GUARDRAIL STRETTI
# ============================================================

STRICT_GATES = {
    "min_char": 180,           # Più alto per evitare micro-chunk
    "max_char": 1400,          # Split se supera
    "max_massime_per_page": 25,  # Flag se supera
    "window_before": 2,
    "window_after": 1,
    "toc_skip_pages": 25,
}

# Soglie FAIL/WARNING
THRESHOLDS = {
    "pct_short_fail": 8.0,      # % massime < 180 char
    "massime_per_page_p95_fail": 25,
    "duplicates_pct_fail": 3.0,
    "pct_complete_citation_warn": 50.0,
    "pct_toc_like_fail": 5.0,
    "pct_citation_list_fail": 7.0,
}


@dataclass
class AnchorQuality:
    """Valuta qualità di un'ancora."""
    has_sez: bool = False
    has_numero: bool = False
    has_rv: bool = False
    has_anno: bool = False

    @property
    def score(self) -> int:
        return sum([self.has_sez, self.has_numero, self.has_rv, self.has_anno])

    @property
    def is_valid(self) -> bool:
        """Almeno 2 tra: Sez., n., Rv., anno."""
        return self.score >= 2


def assess_anchor_quality(text: str) -> AnchorQuality:
    """Valuta qualità di un'ancora basandosi sul testo circostante."""
    q = AnchorQuality()
    q.has_sez = bool(re.search(r"Sez\.?\s*(?:U|L|[IVX0-9]+)", text, re.IGNORECASE))
    q.has_numero = bool(re.search(r"n\.?\s*\d+", text, re.IGNORECASE))
    q.has_rv = bool(re.search(r"Rv\.?\s*\d{5,6}", text, re.IGNORECASE))
    q.has_anno = bool(re.search(r"(?:19|20)\d{2}", text))
    return q


def is_toc_like(text: str) -> bool:
    """Rileva se il testo sembra un indice/TOC."""
    # Pattern TOC: molti numeri di pagina, punti di sospensione
    dotted_lines = len(re.findall(r"\.{3,}\s*\d+", text))
    if dotted_lines >= 3:
        return True

    # Troppi titoli di sezione
    section_headers = len(re.findall(r"^(?:Capitolo|Sezione|Parte|INDICE)", text, re.MULTILINE | re.IGNORECASE))
    if section_headers >= 2:
        return True

    return False


def is_citation_list_like(text: str) -> bool:
    """Rileva se il testo è solo una lista di citazioni senza contenuto."""
    # Conta citazioni vs parole normali
    citations = len(re.findall(r"(?:Sez\.|Cass\.|Rv\.|n\.)\s*\d+", text, re.IGNORECASE))
    words = len(text.split())

    if words < 20:
        return False

    # Se >40% del testo sono citazioni, è una lista
    citation_ratio = (citations * 5) / words  # ~5 char per citazione
    return citation_ratio > 0.4


def extract_with_strict_gates(
    text: str,
    page_num: int,
) -> tuple[list[ExtractedMassima], dict]:
    """
    Estrai massime con guardrail stretti.

    Returns:
        (massime_accettate, stats)
    """
    stats = {
        "anchors_found": 0,
        "anchors_valid": 0,
        "anchors_rejected_quality": 0,
        "chunks_total": 0,
        "chunks_accepted": 0,
        "chunks_rejected_short": 0,
        "chunks_rejected_long_split": 0,
        "chunks_toc_like": 0,
        "chunks_citation_list": 0,
        "char_lengths": [],
    }

    # Trova ancore
    anchors = find_citation_anchors(text)
    stats["anchors_found"] = len(anchors)

    if not anchors:
        return [], stats

    # Filtra ancore per qualità
    valid_anchors = []
    for anchor in anchors:
        # Estrai contesto attorno all'ancora
        start = max(0, anchor.start_pos - 200)
        end = min(len(text), anchor.end_pos + 200)
        context = text[start:end]

        quality = assess_anchor_quality(context)
        if quality.is_valid:
            valid_anchors.append(anchor)
        else:
            stats["anchors_rejected_quality"] += 1

    stats["anchors_valid"] = len(valid_anchors)

    if not valid_anchors:
        return [], stats

    # Pre-calcola frasi
    sentences = split_into_sentences(text)

    massime = []
    seen_hashes = set()

    for anchor in valid_anchors:
        # Trova la frase che contiene la citazione
        citation_sentence_idx = -1
        for i, (sent, start, end) in enumerate(sentences):
            if start <= anchor.start_pos < end:
                citation_sentence_idx = i
                break

        if citation_sentence_idx == -1:
            continue

        # Calcola range frasi
        start_idx = max(0, citation_sentence_idx - STRICT_GATES["window_before"])
        end_idx = min(len(sentences), citation_sentence_idx + STRICT_GATES["window_after"] + 1)

        # Estrai window
        window_start = sentences[start_idx][1]
        window_end = sentences[end_idx - 1][2]
        window_text = text[window_start:window_end].strip()

        stats["chunks_total"] += 1

        # Check lunghezza
        if len(window_text) < STRICT_GATES["min_char"]:
            stats["chunks_rejected_short"] += 1
            continue

        # Se troppo lungo, tronca (non split multiplo per semplicità)
        if len(window_text) > STRICT_GATES["max_char"]:
            window_text = window_text[:STRICT_GATES["max_char"]]
            stats["chunks_rejected_long_split"] += 1

        # Check TOC-like
        if is_toc_like(window_text):
            stats["chunks_toc_like"] += 1
            continue

        # Check citation-list-like
        if is_citation_list_like(window_text):
            stats["chunks_citation_list"] += 1
            continue

        # Dedupe per hash
        testo = clean_legal_text(window_text)
        testo_norm = normalize_for_hash(testo)
        content_hash = compute_content_hash(testo)

        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Estrai citazione
        citation = extract_citation(window_text)

        massima = ExtractedMassima(
            testo=testo,
            testo_normalizzato=testo_norm,
            content_hash=content_hash,
            testo_con_contesto=window_text,
            citation=citation,
            section_context="",
            section_path=None,
            page_start=page_num,
            page_end=page_num,
            element_index=len(massime),
            citation_complete=citation.is_complete,
            text_quality_score=0.0,
        )

        massime.append(massima)
        stats["chunks_accepted"] += 1
        stats["char_lengths"].append(len(testo))

    return massime, stats


def analyze_document(pdf_path: Path, toc_skip: int = 25) -> dict:
    """Analizza un documento con statistiche complete."""
    doc = fitz.open(pdf_path)

    all_massime = []
    all_stats = {
        "anchors_found": 0,
        "anchors_valid": 0,
        "anchors_rejected_quality": 0,
        "chunks_total": 0,
        "chunks_accepted": 0,
        "chunks_rejected_short": 0,
        "chunks_rejected_long_split": 0,
        "chunks_toc_like": 0,
        "chunks_citation_list": 0,
        "char_lengths": [],
        "massime_per_page": [],
    }

    for i in range(len(doc)):
        page_num = i + 1

        # Skip TOC
        if page_num <= toc_skip:
            continue

        text = doc[i].get_text()
        if len(text.strip()) < 100:
            continue

        massime, stats = extract_with_strict_gates(text, page_num)

        # Accumula stats
        all_stats["anchors_found"] += stats["anchors_found"]
        all_stats["anchors_valid"] += stats["anchors_valid"]
        all_stats["anchors_rejected_quality"] += stats["anchors_rejected_quality"]
        all_stats["chunks_total"] += stats["chunks_total"]
        all_stats["chunks_accepted"] += stats["chunks_accepted"]
        all_stats["chunks_rejected_short"] += stats["chunks_rejected_short"]
        all_stats["chunks_rejected_long_split"] += stats["chunks_rejected_long_split"]
        all_stats["chunks_toc_like"] += stats["chunks_toc_like"]
        all_stats["chunks_citation_list"] += stats["chunks_citation_list"]
        all_stats["char_lengths"].extend(stats["char_lengths"])
        all_stats["massime_per_page"].append(len(massime))

        all_massime.extend(massime)

    doc.close()

    # Dedupe globale per content_hash
    seen_hashes = set()
    unique_massime = []
    duplicates = 0
    for m in all_massime:
        if m.content_hash not in seen_hashes:
            seen_hashes.add(m.content_hash)
            unique_massime.append(m)
        else:
            duplicates += 1

    # Calcola metriche finali
    n_massime = len(unique_massime)
    char_lengths = all_stats["char_lengths"]
    massime_per_page = all_stats["massime_per_page"]

    result = {
        "filename": pdf_path.name,
        "total_pages": len(fitz.open(pdf_path)),
        "pages_processed": len(massime_per_page),
        "n_massime": n_massime,
        "duplicates": duplicates,
        "duplicates_pct": (duplicates / (n_massime + duplicates) * 100) if (n_massime + duplicates) > 0 else 0,

        # Anchors
        "anchors_found": all_stats["anchors_found"],
        "anchors_valid": all_stats["anchors_valid"],
        "anchors_rejected_quality": all_stats["anchors_rejected_quality"],

        # Chunks
        "chunks_total": all_stats["chunks_total"],
        "chunks_accepted": all_stats["chunks_accepted"],
        "chunks_rejected_short": all_stats["chunks_rejected_short"],
        "chunks_rejected_long_split": all_stats["chunks_rejected_long_split"],
        "chunks_toc_like": all_stats["chunks_toc_like"],
        "chunks_citation_list": all_stats["chunks_citation_list"],

        # Reject rates
        "reject_short_pct": (all_stats["chunks_rejected_short"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,
        "reject_toc_pct": (all_stats["chunks_toc_like"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,
        "reject_citation_list_pct": (all_stats["chunks_citation_list"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,

        # Char length distribution
        "p50_char": int(median(char_lengths)) if char_lengths else 0,
        "p90_char": int(quantiles(char_lengths, n=10)[8]) if len(char_lengths) >= 10 else (max(char_lengths) if char_lengths else 0),
        "pct_short": (sum(1 for c in char_lengths if c < 180) / len(char_lengths) * 100) if char_lengths else 0,
        "pct_long": (sum(1 for c in char_lengths if c > 1400) / len(char_lengths) * 100) if char_lengths else 0,

        # Massime per page distribution
        "massime_per_page_avg": mean(massime_per_page) if massime_per_page else 0,
        "massime_per_page_p95": quantiles(massime_per_page, n=20)[18] if len(massime_per_page) >= 20 else (max(massime_per_page) if massime_per_page else 0),

        # Citation quality
        "pct_complete_citation": (sum(1 for m in unique_massime if m.citation_complete) / n_massime * 100) if n_massime > 0 else 0,

        # Massime list (for further analysis)
        "_massime": unique_massime,
    }

    return result


def evaluate_pass_fail(result: dict) -> tuple[str, list[str]]:
    """Valuta PASS/FAIL/WARNING basandosi sulle soglie."""
    failures = []
    warnings = []

    # FAIL conditions
    if result["pct_short"] > THRESHOLDS["pct_short_fail"]:
        failures.append(f"pct_short={result['pct_short']:.1f}% > {THRESHOLDS['pct_short_fail']}%")

    if result["massime_per_page_p95"] > THRESHOLDS["massime_per_page_p95_fail"]:
        failures.append(f"massime/page p95={result['massime_per_page_p95']:.1f} > {THRESHOLDS['massime_per_page_p95_fail']}")

    if result["duplicates_pct"] > THRESHOLDS["duplicates_pct_fail"]:
        failures.append(f"duplicates={result['duplicates_pct']:.1f}% > {THRESHOLDS['duplicates_pct_fail']}%")

    if result["reject_toc_pct"] > THRESHOLDS["pct_toc_like_fail"]:
        failures.append(f"toc_like={result['reject_toc_pct']:.1f}% > {THRESHOLDS['pct_toc_like_fail']}%")

    if result["reject_citation_list_pct"] > THRESHOLDS["pct_citation_list_fail"]:
        failures.append(f"citation_list={result['reject_citation_list_pct']:.1f}% > {THRESHOLDS['pct_citation_list_fail']}%")

    # WARNING conditions
    if result["pct_complete_citation"] < THRESHOLDS["pct_complete_citation_warn"]:
        warnings.append(f"pct_complete_citation={result['pct_complete_citation']:.1f}% < {THRESHOLDS['pct_complete_citation_warn']}%")

    if failures:
        return "FAIL", failures + warnings
    elif warnings:
        return "WARNING", warnings
    else:
        return "PASS", []


async def get_current_massime(conn, filename: str) -> int:
    """Get current massime count for a document."""
    row = await conn.fetchrow("""
        SELECT count(ma.id) as n_massime
        FROM kb.pdf_manifest m
        LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
        WHERE m.filename = $1
        GROUP BY m.id
    """, filename)
    return row["n_massime"] if row else 0


async def main(doc_filter: str = None):
    print("=" * 80)
    print("DRY-RUN STATISTICO - CITATION-ANCHORED CON GUARDRAIL STRETTI")
    print("=" * 80)
    print()

    print("GUARDRAIL:")
    for k, v in STRICT_GATES.items():
        print(f"  {k}: {v}")
    print()

    print("SOGLIE FAIL:")
    for k, v in THRESHOLDS.items():
        print(f"  {k}: {v}")
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get documents
    if doc_filter:
        query = """
            SELECT m.filename, s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.filename ILIKE $1
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query, f"%{doc_filter}%")
    else:
        # Canary: solo i 2 documenti target
        query = """
            SELECT m.filename, s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.filename ILIKE '%Volume I_2016_Massimario_Civile%'
               OR m.filename ILIKE '%Volume II_2024_Massimario_Civile%'
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query)

    print(f"Documenti da analizzare: {len(docs)}")
    print()

    results = []
    total_pass = 0
    total_fail = 0
    total_warn = 0

    for i, doc in enumerate(docs, 1):
        filename = doc["filename"]
        coverage = float(doc["coverage_pct"] or 0)

        print(f"[{i}/{len(docs)}] {filename[:55]}...")

        # Find PDF
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  [SKIP] PDF non trovato")
            continue

        # Get current massime
        current = await get_current_massime(conn, filename)

        # Analyze
        result = analyze_document(pdf_path, toc_skip=STRICT_GATES["toc_skip_pages"])
        result["current_massime"] = current
        result["current_coverage"] = coverage

        # Evaluate
        status, issues = evaluate_pass_fail(result)
        result["status"] = status
        result["issues"] = issues

        if status == "PASS":
            total_pass += 1
        elif status == "FAIL":
            total_fail += 1
        else:
            total_warn += 1

        # Print summary
        improvement = result["n_massime"] / current if current > 0 else float('inf')
        imp_str = f"{improvement:.1f}x" if improvement != float('inf') else "NEW"

        print(f"  Status: {status}")
        print(f"  Current: {current}, New: {result['n_massime']}, Improvement: {imp_str}")
        print(f"  p50_char: {result['p50_char']}, p90_char: {result['p90_char']}")
        print(f"  pct_short: {result['pct_short']:.1f}%, pct_long: {result['pct_long']:.1f}%")
        print(f"  duplicates: {result['duplicates_pct']:.1f}%")
        print(f"  massime/page avg: {result['massime_per_page_avg']:.1f}, p95: {result['massime_per_page_p95']:.1f}")
        print(f"  pct_complete_citation: {result['pct_complete_citation']:.1f}%")
        print(f"  reject_short: {result['reject_short_pct']:.1f}%, toc: {result['reject_toc_pct']:.1f}%, citation_list: {result['reject_citation_list_pct']:.1f}%")

        if issues:
            print(f"  Issues:")
            for issue in issues:
                print(f"    - {issue}")

        print()

        # Remove massime list for summary
        result_copy = {k: v for k, v in result.items() if not k.startswith("_")}
        results.append(result_copy)

    await conn.close()

    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  PASS: {total_pass}")
    print(f"  WARNING: {total_warn}")
    print(f"  FAIL: {total_fail}")
    print()

    if total_fail == 0:
        print("  +------------------------------------------+")
        print("  |  ALL DOCUMENTS PASS - READY FOR CANARY  |")
        print("  +------------------------------------------+")
    else:
        print("  +------------------------------------------+")
        print("  |  SOME DOCUMENTS FAIL - FIX REQUIRED     |")
        print("  +------------------------------------------+")

    # Save detailed report
    report_file = Path(__file__).parent / "DRYRUN_STATISTICAL_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Dry-Run Statistico - Citation-Anchored\n\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().isoformat()}\n\n")

        f.write("## Guardrail\n\n")
        f.write("| Parameter | Value |\n|-----------|-------|\n")
        for k, v in STRICT_GATES.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n## Soglie\n\n")
        f.write("| Threshold | Value |\n|-----------|-------|\n")
        for k, v in THRESHOLDS.items():
            f.write(f"| {k} | {v} |\n")

        f.write(f"\n## Summary\n\n")
        f.write(f"| Status | Count |\n|--------|-------|\n")
        f.write(f"| PASS | {total_pass} |\n")
        f.write(f"| WARNING | {total_warn} |\n")
        f.write(f"| FAIL | {total_fail} |\n")

        f.write(f"\n## Per-Document Results\n\n")
        f.write("| Document | Status | Current | New | p50 | p90 | pct_short | dup% | mpp_p95 | cit% |\n")
        f.write("|----------|--------|---------|-----|-----|-----|-----------|------|---------|------|\n")
        for r in results:
            f.write(f"| {r['filename'][:35]} | {r['status']} | {r['current_massime']} | {r['n_massime']} | ")
            f.write(f"{r['p50_char']} | {r['p90_char']} | {r['pct_short']:.1f}% | {r['duplicates_pct']:.1f}% | ")
            f.write(f"{r['massime_per_page_p95']:.1f} | {r['pct_complete_citation']:.1f}% |\n")

    print(f"\nReport salvato: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry-run statistico citation-anchored")
    parser.add_argument("--doc", type=str, help="Filter by document name (default: canary docs)")
    args = parser.parse_args()

    asyncio.run(main(doc_filter=args.doc))
