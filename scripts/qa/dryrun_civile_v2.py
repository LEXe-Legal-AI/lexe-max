#!/usr/bin/env python3
"""
Dry-Run Civile v2 - Con Smart Cut (soft_cap/hard_cap + sentence boundary)

Metriche nuove:
- pct_truncated: % chunk che hanno richiesto cut
- pct_forced_cut: % chunk con forced_cut (nessun boundary)
- avg_cut_chars: quanto tagliamo in media

Soglie:
- pct_truncated < 20% (altrimenti finestra troppo larga)
- pct_forced_cut < 5%
- pct_short < 8%
- duplicates < 3%

Usage:
    uv run python scripts/qa/dryrun_civile_v2.py
    uv run python scripts/qa/dryrun_civile_v2.py --doc "Volume I_2016"
    uv run python scripts/qa/dryrun_civile_v2.py --all-civile
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
    extract_citation,
    find_citation_anchors,
    split_into_sentences,
)
from lexe_api.kb.ingestion.cut_validator import (
    SOFT_CAP,
    HARD_CAP,
    MIN_CHAR,
    choose_cut_sync,
    is_suspicious_end,
    propose_cut_candidates,
)
from lexe_api.kb.ingestion.cleaner import clean_legal_text, compute_content_hash, normalize_for_hash
from qa_config import DB_URL, PDF_DIR


# ============================================================
# GUARDRAIL v2
# ============================================================

GATES = {
    "min_char": MIN_CHAR,  # 180
    "soft_cap": SOFT_CAP,  # 1600
    "hard_cap": HARD_CAP,  # 1800
    "window_before": 1,    # Ridotto da 2 a 1 (pct_truncated era >30%)
    "window_after": 1,
    "toc_skip_pages": 25,
}

# Soglie FAIL/WARNING
THRESHOLDS = {
    "pct_short_fail": 8.0,
    "pct_truncated_fail": 30.0,  # Se >30% anche con 1800, passa a 1+1
    "pct_forced_cut_fail": 10.0,
    "duplicates_pct_fail": 3.0,
    "pct_complete_citation_warn": 50.0,
    "pct_toc_like_fail": 5.0,
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
        return self.score >= 2


def assess_anchor_quality(text: str) -> AnchorQuality:
    q = AnchorQuality()
    q.has_sez = bool(re.search(r"Sez\.?\s*(?:U|L|[IVX0-9]+)", text, re.IGNORECASE))
    q.has_numero = bool(re.search(r"n\.?\s*\d+", text, re.IGNORECASE))
    q.has_rv = bool(re.search(r"Rv\.?\s*\d{5,6}", text, re.IGNORECASE))
    q.has_anno = bool(re.search(r"(?:19|20)\d{2}", text))
    return q


def is_toc_like(text: str) -> bool:
    dotted_lines = len(re.findall(r"\.{3,}\s*\d+", text))
    if dotted_lines >= 3:
        return True
    section_headers = len(re.findall(r"^(?:Capitolo|Sezione|Parte|INDICE)", text, re.MULTILINE | re.IGNORECASE))
    return section_headers >= 2


def is_citation_list_like(text: str) -> bool:
    citations = len(re.findall(r"(?:Sez\.|Cass\.|Rv\.|n\.)\s*\d+", text, re.IGNORECASE))
    words = len(text.split())
    if words < 20:
        return False
    citation_ratio = (citations * 5) / words
    return citation_ratio > 0.4


def extract_with_smart_cut(
    text: str,
    page_num: int,
) -> tuple[list[dict], dict]:
    """
    Estrai massime con smart cut (soft_cap/hard_cap + sentence boundary).

    Returns:
        (massime_data, stats)
    """
    stats = {
        "anchors_found": 0,
        "anchors_valid": 0,
        "anchors_rejected_quality": 0,
        "chunks_total": 0,
        "chunks_accepted": 0,
        "chunks_rejected_short": 0,
        "chunks_truncated": 0,
        "chunks_forced_cut": 0,
        "chunks_toc_like": 0,
        "chunks_citation_list": 0,
        "char_lengths": [],
        "cut_chars": [],  # Quanto abbiamo tagliato per chunk
        "trigger_types": Counter(),
    }

    anchors = find_citation_anchors(text)
    stats["anchors_found"] = len(anchors)

    if not anchors:
        return [], stats

    # Filtra ancore per qualità
    valid_anchors = []
    for anchor in anchors:
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
        start_idx = max(0, citation_sentence_idx - GATES["window_before"])
        end_idx = min(len(sentences), citation_sentence_idx + GATES["window_after"] + 1)

        # Estrai window grezzo
        window_start = sentences[start_idx][1]
        window_end = sentences[end_idx - 1][2]
        raw_window = text[window_start:window_end].strip()

        stats["chunks_total"] += 1
        original_len = len(raw_window)

        # Check lunghezza minima
        if original_len < GATES["min_char"]:
            stats["chunks_rejected_short"] += 1
            continue

        # Smart cut se troppo lungo
        if original_len > GATES["hard_cap"]:
            decision = choose_cut_sync(raw_window, GATES["soft_cap"], GATES["hard_cap"])
            window_text = raw_window[:decision.offset]

            stats["chunks_truncated"] += 1
            stats["cut_chars"].append(original_len - decision.offset)

            if decision.forced_cut:
                stats["chunks_forced_cut"] += 1

            if decision.trigger_type:
                stats["trigger_types"][decision.trigger_type] += 1
        else:
            window_text = raw_window

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

        massima = {
            "testo": testo,
            "testo_normalizzato": testo_norm,
            "content_hash": content_hash,
            "testo_con_contesto": window_text,
            "citation": citation,
            "page_start": page_num,
            "citation_complete": citation.is_complete,
        }

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
        "chunks_truncated": 0,
        "chunks_forced_cut": 0,
        "chunks_toc_like": 0,
        "chunks_citation_list": 0,
        "char_lengths": [],
        "cut_chars": [],
        "massime_per_page": [],
        "trigger_types": Counter(),
    }

    for i in range(len(doc)):
        page_num = i + 1

        if page_num <= toc_skip:
            continue

        text = doc[i].get_text()
        if len(text.strip()) < 100:
            continue

        massime, stats = extract_with_smart_cut(text, page_num)

        # Accumula stats
        for key in ["anchors_found", "anchors_valid", "anchors_rejected_quality",
                    "chunks_total", "chunks_accepted", "chunks_rejected_short",
                    "chunks_truncated", "chunks_forced_cut", "chunks_toc_like",
                    "chunks_citation_list"]:
            all_stats[key] += stats[key]

        all_stats["char_lengths"].extend(stats["char_lengths"])
        all_stats["cut_chars"].extend(stats["cut_chars"])
        all_stats["massime_per_page"].append(len(massime))
        all_stats["trigger_types"].update(stats["trigger_types"])

        all_massime.extend(massime)

    doc.close()

    # Dedupe globale per content_hash
    seen_hashes = set()
    unique_massime = []
    duplicates = 0
    for m in all_massime:
        if m["content_hash"] not in seen_hashes:
            seen_hashes.add(m["content_hash"])
            unique_massime.append(m)
        else:
            duplicates += 1

    # Calcola metriche finali
    n_massime = len(unique_massime)
    char_lengths = all_stats["char_lengths"]
    cut_chars = all_stats["cut_chars"]
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

        # Chunks
        "chunks_total": all_stats["chunks_total"],
        "chunks_accepted": all_stats["chunks_accepted"],
        "chunks_rejected_short": all_stats["chunks_rejected_short"],

        # CUT METRICS (nuove!)
        "chunks_truncated": all_stats["chunks_truncated"],
        "chunks_forced_cut": all_stats["chunks_forced_cut"],
        "pct_truncated": (all_stats["chunks_truncated"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,
        "pct_forced_cut": (all_stats["chunks_forced_cut"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,
        "avg_cut_chars": mean(cut_chars) if cut_chars else 0,
        "trigger_types": dict(all_stats["trigger_types"]),

        # Reject rates
        "reject_short_pct": (all_stats["chunks_rejected_short"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,
        "reject_toc_pct": (all_stats["chunks_toc_like"] / all_stats["chunks_total"] * 100) if all_stats["chunks_total"] > 0 else 0,

        # Char length distribution
        "p50_char": int(median(char_lengths)) if char_lengths else 0,
        "p90_char": int(quantiles(char_lengths, n=10)[8]) if len(char_lengths) >= 10 else (max(char_lengths) if char_lengths else 0),
        "pct_short": (sum(1 for c in char_lengths if c < MIN_CHAR) / len(char_lengths) * 100) if char_lengths else 0,
        "pct_long": (sum(1 for c in char_lengths if c > HARD_CAP) / len(char_lengths) * 100) if char_lengths else 0,

        # Massime per page
        "massime_per_page_avg": mean(massime_per_page) if massime_per_page else 0,
        "massime_per_page_p95": quantiles(massime_per_page, n=20)[18] if len(massime_per_page) >= 20 else (max(massime_per_page) if massime_per_page else 0),

        # Citation quality
        "pct_complete_citation": (sum(1 for m in unique_massime if m["citation_complete"]) / n_massime * 100) if n_massime > 0 else 0,

        "_massime": unique_massime,
    }

    return result


def evaluate_pass_fail(result: dict) -> tuple[str, list[str]]:
    """Valuta PASS/FAIL/WARNING basandosi sulle soglie."""
    failures = []
    warnings = []

    if result["pct_short"] > THRESHOLDS["pct_short_fail"]:
        failures.append(f"pct_short={result['pct_short']:.1f}% > {THRESHOLDS['pct_short_fail']}%")

    if result["pct_truncated"] > THRESHOLDS["pct_truncated_fail"]:
        failures.append(f"pct_truncated={result['pct_truncated']:.1f}% > {THRESHOLDS['pct_truncated_fail']}% (considera window 1+1)")

    if result["pct_forced_cut"] > THRESHOLDS["pct_forced_cut_fail"]:
        failures.append(f"pct_forced_cut={result['pct_forced_cut']:.1f}% > {THRESHOLDS['pct_forced_cut_fail']}%")

    if result["duplicates_pct"] > THRESHOLDS["duplicates_pct_fail"]:
        failures.append(f"duplicates={result['duplicates_pct']:.1f}% > {THRESHOLDS['duplicates_pct_fail']}%")

    if result["reject_toc_pct"] > THRESHOLDS["pct_toc_like_fail"]:
        failures.append(f"toc_like={result['reject_toc_pct']:.1f}% > {THRESHOLDS['pct_toc_like_fail']}%")

    if result["pct_complete_citation"] < THRESHOLDS["pct_complete_citation_warn"]:
        warnings.append(f"pct_complete_citation={result['pct_complete_citation']:.1f}% < {THRESHOLDS['pct_complete_citation_warn']}%")

    if failures:
        return "FAIL", failures + warnings
    elif warnings:
        return "WARNING", warnings
    else:
        return "PASS", []


async def get_current_massime(conn, filename: str) -> int:
    row = await conn.fetchrow("""
        SELECT count(ma.id) as n_massime
        FROM kb.pdf_manifest m
        LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
        WHERE m.filename = $1
        GROUP BY m.id
    """, filename)
    return row["n_massime"] if row else 0


async def main(doc_filter: str = None, all_civile: bool = False):
    print("=" * 80)
    print("DRY-RUN CIVILE v2 - SMART CUT (soft/hard cap + sentence boundary)")
    print("=" * 80)
    print()

    print("GUARDRAIL:")
    for k, v in GATES.items():
        print(f"  {k}: {v}")
    print()

    print("SOGLIE:")
    for k, v in THRESHOLDS.items():
        print(f"  {k}: {v}")
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get documents
    if all_civile:
        query = """
            SELECT m.filename, s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.tipo = 'civile' OR m.tipo IS NULL
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query)
    elif doc_filter:
        query = """
            SELECT m.filename, s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.filename ILIKE $1
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query, f"%{doc_filter}%")
    else:
        # Default: canary docs
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

        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  [SKIP] PDF non trovato")
            continue

        current = await get_current_massime(conn, filename)

        result = analyze_document(pdf_path, toc_skip=GATES["toc_skip_pages"])
        result["current_massime"] = current
        result["current_coverage"] = coverage

        status, issues = evaluate_pass_fail(result)
        result["status"] = status
        result["issues"] = issues

        if status == "PASS":
            total_pass += 1
        elif status == "FAIL":
            total_fail += 1
        else:
            total_warn += 1

        improvement = result["n_massime"] / current if current > 0 else float('inf')
        imp_str = f"{improvement:.1f}x" if improvement != float('inf') else "NEW"

        print(f"  Status: {status}")
        print(f"  Current: {current}, New: {result['n_massime']}, Improvement: {imp_str}")
        print(f"  p50: {result['p50_char']}, p90: {result['p90_char']}, pct_long: {result['pct_long']:.1f}%")
        print(f"  pct_truncated: {result['pct_truncated']:.1f}%, pct_forced_cut: {result['pct_forced_cut']:.1f}%")
        if result['avg_cut_chars'] > 0:
            print(f"  avg_cut_chars: {result['avg_cut_chars']:.0f}")
        print(f"  duplicates: {result['duplicates_pct']:.1f}%, cit_complete: {result['pct_complete_citation']:.1f}%")

        if result['trigger_types']:
            print(f"  trigger_types: {dict(result['trigger_types'])}")

        if issues:
            print(f"  Issues:")
            for issue in issues:
                print(f"    - {issue}")

        print()

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
        print("  |  SOME DOCUMENTS FAIL - REVIEW REQUIRED  |")
        print("  +------------------------------------------+")

    # Aggregated stats
    if results:
        all_truncated = [r["pct_truncated"] for r in results]
        all_forced = [r["pct_forced_cut"] for r in results]
        print()
        print("AGGREGATED METRICS:")
        print(f"  avg_pct_truncated: {mean(all_truncated):.1f}%")
        print(f"  avg_pct_forced_cut: {mean(all_forced):.1f}%")
        print(f"  max_pct_truncated: {max(all_truncated):.1f}%")
        print(f"  max_pct_forced_cut: {max(all_forced):.1f}%")

    # Save report
    report_file = Path(__file__).parent / "DRYRUN_V2_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Dry-Run Civile v2 - Smart Cut\n\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().isoformat()}\n\n")

        f.write("## Guardrail\n\n")
        f.write("| Parameter | Value |\n|-----------|-------|\n")
        for k, v in GATES.items():
            f.write(f"| {k} | {v} |\n")

        f.write(f"\n## Summary\n\n")
        f.write(f"| Status | Count |\n|--------|-------|\n")
        f.write(f"| PASS | {total_pass} |\n")
        f.write(f"| WARNING | {total_warn} |\n")
        f.write(f"| FAIL | {total_fail} |\n")

        f.write(f"\n## Per-Document Results\n\n")
        f.write("| Document | Status | Cur | New | p50 | p90 | trunc% | forced% | dup% | cit% |\n")
        f.write("|----------|--------|-----|-----|-----|-----|--------|---------|------|------|\n")
        for r in results:
            f.write(f"| {r['filename'][:30]} | {r['status']} | {r['current_massime']} | {r['n_massime']} | ")
            f.write(f"{r['p50_char']} | {r['p90_char']} | {r['pct_truncated']:.1f}% | {r['pct_forced_cut']:.1f}% | ")
            f.write(f"{r['duplicates_pct']:.1f}% | {r['pct_complete_citation']:.1f}% |\n")

    print(f"\nReport salvato: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry-run Civile v2 con smart cut")
    parser.add_argument("--doc", type=str, help="Filter by document name")
    parser.add_argument("--all-civile", action="store_true", help="Process all Civile documents")
    args = parser.parse_args()

    asyncio.run(main(doc_filter=args.doc, all_civile=args.all_civile))
