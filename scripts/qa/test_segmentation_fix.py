#!/usr/bin/env python3
"""Test segmentation fix for 2014 Mass civile."""

import re
import httpx
from pathlib import Path

MIN_LENGTH_REF = 80

def has_citation(text: str) -> bool:
    return bool(re.search(r"(?:Sez\.?|Sezione)\s*[UuLl0-9]", text, re.IGNORECASE))


def segment_reference_units_OLD(elements: list[dict]) -> list[dict]:
    """Old segmentation - only Sez. pattern."""
    units = []
    current_texts = []
    current_page_start = None
    current_page_end = None

    for elem in elements:
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")
        page = elem.get("metadata", {}).get("page_number")

        if elem_type in ("Header", "Footer", "PageNumber"):
            continue

        # OLD pattern
        if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+)", text, re.IGNORECASE):
            if current_texts:
                full = " ".join(current_texts)
                if len(full) >= MIN_LENGTH_REF:
                    units.append({"testo": full, "page_start": current_page_start, "page_end": current_page_end})
            current_texts = [text]
            current_page_start = page
            current_page_end = page
        elif text:
            if not current_texts:
                current_page_start = page
            current_texts.append(text)
            current_page_end = page

    if current_texts:
        full = " ".join(current_texts)
        if len(full) >= MIN_LENGTH_REF:
            units.append({"testo": full, "page_start": current_page_start, "page_end": current_page_end})

    return units


def segment_reference_units_NEW(elements: list[dict]) -> list[dict]:
    """New segmentation - includes La Corte, In tema."""
    units = []
    current_texts = []
    current_page_start = None
    current_page_end = None

    for elem in elements:
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")
        page = elem.get("metadata", {}).get("page_number")

        if elem_type in ("Header", "Footer", "PageNumber"):
            continue

        # NEW pattern with La Corte, In tema
        if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+|La Corte|In tema)", text, re.IGNORECASE):
            if current_texts:
                full = " ".join(current_texts)
                if len(full) >= MIN_LENGTH_REF:
                    units.append({"testo": full, "page_start": current_page_start, "page_end": current_page_end})
            current_texts = [text]
            current_page_start = page
            current_page_end = page
        elif text:
            if not current_texts:
                current_page_start = page
            current_texts.append(text)
            current_page_end = page

    if current_texts:
        full = " ".join(current_texts)
        if len(full) >= MIN_LENGTH_REF:
            units.append({"testo": full, "page_start": current_page_start, "page_end": current_page_end})

    return units


pdf_path = Path(r"C:\PROJECTS\lexe-genesis\data\raccolta\2014 Mass civile Vol 1 pagg 408.pdf")

print(f"Testing on: {pdf_path.name}")
print("Extracting with local Unstructured (fast)...")

with open(pdf_path, "rb") as f:
    response = httpx.post(
        "http://localhost:8500/general/v0/general",
        files={"files": (pdf_path.name, f, "application/pdf")},
        data={"strategy": "fast", "output_format": "application/json"},
        timeout=300.0,
    )

elements = response.json()
print(f"Elements: {len(elements)}")

# Compare old vs new
old_units = segment_reference_units_OLD(elements)
new_units = segment_reference_units_NEW(elements)

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"OLD segmentation: {len(old_units)} units")
print(f"NEW segmentation: {len(new_units)} units")

if old_units:
    old_avg = sum(len(u["testo"]) for u in old_units) / len(old_units)
    print(f"OLD avg chars: {old_avg:.0f}")

if new_units:
    new_avg = sum(len(u["testo"]) for u in new_units) / len(new_units)
    print(f"NEW avg chars: {new_avg:.0f}")

print()
print("NEW units sample (first 5):")
for i, u in enumerate(new_units[:5]):
    print(f"  {i+1}. [{u['page_start']}-{u['page_end']}] {u['testo'][:80]}...")
