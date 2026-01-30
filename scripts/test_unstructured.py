"""
Test Unstructured API - KB Massimari
Estrae PDF usando Unstructured API e confronta con PyMuPDF.
"""
import asyncio
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from uuid import uuid4

import asyncpg
import fitz  # PyMuPDF
import httpx

# Unstructured API
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"

# Database
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# PDF directory
PDF_DIR = Path("C:/Users/Fra/Documents/lexe/collezione zecca/New folder (2)/Massimario_PDF/")

# Test files
TEST_FILES = [
    "2021_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2023_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2018_MASSIMARIO CIVILE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTIDELLE SEZIONI CIVILI.pdf",
]

# Mesi italiani
MESI_IT = {
    'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4,
    'maggio': 5, 'giugno': 6, 'luglio': 7, 'agosto': 8,
    'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12
}

# Pattern massime - SPACE TOLERANT per Unstructured
MASSIMA_PATTERNS = [
    # Sez. 4, n. 6513 del 27/01/2021 (space tolerant)
    re.compile(
        r"Sez\s*\.?\s*(\d+)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.IGNORECASE
    ),
    # Sez. Un., n. 12345 del 27/01/2021 (space tolerant)
    re.compile(
        r"Sez\s*\.?\s*(Un\.?|Unite)\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.IGNORECASE
    ),
    # Simple: Sez X n Y (senza data - per match più ampi)
    re.compile(
        r"Sez\s*\.?\s*(\d+)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d{3,})",
        re.IGNORECASE
    ),
    # Cass., sez. X, DD mese YYYY, n. XXXXX (date italiane, space tolerant)
    re.compile(
        r"Cass\s*\.?\s*,?\s*sez\s*\.?\s*(un\.?|unite|\d+)\s*[ªa°]?\s*,?\s*"
        r"(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})\s*,?\s*n\s*\.?\s*(\d+)",
        re.IGNORECASE
    ),
]


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


async def extract_with_unstructured(pdf_path: Path, strategy: str = "auto") -> dict:
    """Extract PDF using Unstructured API."""
    print(f"  Extracting with Unstructured (strategy={strategy})...")

    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            data = {
                "strategy": strategy,
                "languages": ["ita"],
                "pdf_infer_table_structure": "false",
            }

            start_time = time.time()
            response = await client.post(UNSTRUCTURED_URL, files=files, data=data)
            elapsed = time.time() - start_time

    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} - {response.text[:200]}")
        return {"elements": [], "elapsed": elapsed, "error": response.text}

    elements = response.json()
    return {
        "elements": elements,
        "elapsed": elapsed,
        "element_count": len(elements),
    }


def extract_with_pymupdf(pdf_path: Path, start_page: int = 20, max_pages: int = 80) -> dict:
    """Extract PDF using PyMuPDF."""
    print(f"  Extracting with PyMuPDF...")

    start_time = time.time()
    doc = fitz.open(pdf_path)

    elements = []
    end_page = min(doc.page_count, start_page + max_pages)

    for i in range(start_page, end_page):
        page = doc[i]
        text = page.get_text()
        if text.strip() and len(text) > 100:
            elements.append({
                "type": "NarrativeText",
                "text": text,
                "metadata": {"page_number": i + 1}
            })

    doc.close()
    elapsed = time.time() - start_time

    return {
        "elements": elements,
        "elapsed": elapsed,
        "element_count": len(elements),
    }


def extract_massime_from_elements(elements: list, source_type: str) -> list:
    """Extract massime from elements."""
    massime = []
    seen_hashes = set()

    all_text = " ".join([e.get("text", "") for e in elements])

    for pattern in MASSIMA_PATTERNS:
        for match in pattern.finditer(all_text):
            start = max(0, match.start() - 200)
            end = min(len(all_text), match.end() + 500)
            context = re.sub(r'\s+', ' ', all_text[start:end].strip())

            if len(context) < 50:
                continue

            content_hash = compute_hash(context)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            groups = match.groups()
            sezione = str(groups[0]) if groups[0] else "unknown"
            numero = str(groups[1]) if len(groups) > 1 and groups[1] else "0"

            massime.append({
                "sezione": sezione,
                "numero": numero,
                "testo": context[:500],
                "hash": content_hash,
            })

    return massime


def analyze_elements(elements: list) -> dict:
    """Analyze element types and statistics."""
    stats = {
        "total_elements": len(elements),
        "total_chars": sum(len(e.get("text", "")) for e in elements),
        "element_types": {},
        "avg_element_length": 0,
    }

    for e in elements:
        etype = e.get("type", "Unknown")
        stats["element_types"][etype] = stats["element_types"].get(etype, 0) + 1

    if elements:
        stats["avg_element_length"] = stats["total_chars"] // len(elements)

    return stats


async def compare_extractions(pdf_path: Path, source_type: str):
    """Compare PyMuPDF vs Unstructured extraction."""
    print(f"\n{'='*70}")
    print(f"Comparing: {pdf_path.name[:60]}...")
    print("="*70)

    # PyMuPDF extraction
    pymupdf_result = extract_with_pymupdf(pdf_path)
    pymupdf_stats = analyze_elements(pymupdf_result["elements"])
    pymupdf_massime = extract_massime_from_elements(pymupdf_result["elements"], source_type)

    print(f"\n  PyMuPDF:")
    print(f"    Time: {pymupdf_result['elapsed']:.2f}s")
    print(f"    Elements: {pymupdf_stats['total_elements']}")
    print(f"    Total chars: {pymupdf_stats['total_chars']:,}")
    print(f"    Massime found: {len(pymupdf_massime)}")

    # Unstructured extraction (fast strategy first)
    unstructured_result = await extract_with_unstructured(pdf_path, strategy="fast")
    unstructured_stats = analyze_elements(unstructured_result.get("elements", []))
    unstructured_massime = extract_massime_from_elements(
        unstructured_result.get("elements", []), source_type
    )

    print(f"\n  Unstructured (fast):")
    print(f"    Time: {unstructured_result['elapsed']:.2f}s")
    print(f"    Elements: {unstructured_stats['total_elements']}")
    print(f"    Total chars: {unstructured_stats['total_chars']:,}")
    print(f"    Element types: {unstructured_stats['element_types']}")
    print(f"    Massime found: {len(unstructured_massime)}")

    # Try hi_res for OCR if available
    if unstructured_stats['total_elements'] < 50:
        print("\n  Trying hi_res strategy (OCR)...")
        hires_result = await extract_with_unstructured(pdf_path, strategy="hi_res")
        hires_stats = analyze_elements(hires_result.get("elements", []))
        hires_massime = extract_massime_from_elements(
            hires_result.get("elements", []), source_type
        )

        print(f"\n  Unstructured (hi_res):")
        print(f"    Time: {hires_result['elapsed']:.2f}s")
        print(f"    Elements: {hires_stats['total_elements']}")
        print(f"    Total chars: {hires_stats['total_chars']:,}")
        print(f"    Massime found: {len(hires_massime)}")

    return {
        "filename": pdf_path.name,
        "pymupdf": {
            "time": pymupdf_result["elapsed"],
            "elements": pymupdf_stats["total_elements"],
            "chars": pymupdf_stats["total_chars"],
            "massime": len(pymupdf_massime),
        },
        "unstructured": {
            "time": unstructured_result["elapsed"],
            "elements": unstructured_stats["total_elements"],
            "chars": unstructured_stats["total_chars"],
            "massime": len(unstructured_massime),
            "types": unstructured_stats["element_types"],
        },
    }


async def main():
    print("="*70)
    print("KB Massimari - PyMuPDF vs Unstructured Comparison")
    print("="*70)

    # Check Unstructured API
    print("\nChecking Unstructured API...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:8500/healthcheck")
            print(f"  API Status: {resp.json()}")
        except Exception as e:
            print(f"  ERROR: Unstructured API not available: {e}")
            return

    results = []

    for filename in TEST_FILES:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            print(f"SKIP: {filename} not found")
            continue

        # Determine type
        source_type = "penale" if "PENALE" in filename else "civile"

        result = await compare_extractions(pdf_path, source_type)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'File':<35} {'PyMuPDF':>12} {'Unstructured':>12} {'Winner':>10}")
    print(f"{'':35} {'Massime':>12} {'Massime':>12} {'':>10}")
    print("-"*70)

    for r in results:
        pm = r["pymupdf"]["massime"]
        un = r["unstructured"]["massime"]
        winner = "PyMuPDF" if pm > un else ("Unstruct" if un > pm else "Tie")
        print(f"{r['filename'][:33]:<35} {pm:>12} {un:>12} {winner:>10}")

    print("-"*70)
    total_pm = sum(r["pymupdf"]["massime"] for r in results)
    total_un = sum(r["unstructured"]["massime"] for r in results)
    winner = "PyMuPDF" if total_pm > total_un else ("Unstruct" if total_un > total_pm else "Tie")
    print(f"{'TOTAL':<35} {total_pm:>12} {total_un:>12} {winner:>10}")

    print("\n" + "="*70)
    print("TIMING COMPARISON")
    print("="*70)
    print(f"{'File':<35} {'PyMuPDF':>12} {'Unstructured':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['filename'][:33]:<35} {r['pymupdf']['time']:>10.2f}s {r['unstructured']['time']:>10.2f}s")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
