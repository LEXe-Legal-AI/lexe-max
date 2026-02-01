#!/usr/bin/env python3
"""Quick test cloud extraction on a single PDF."""

import os
import time
from pathlib import Path
import httpx

UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")
UNSTRUCTURED_CLOUD_URL = "https://api.unstructuredapp.io/general/v0/general"

PDF_PATH = Path(r"C:\PROJECTS\lexe-genesis\data\raccolta\Rassegna Penale 2012.pdf")

print(f"API Key: {'SET' if UNSTRUCTURED_API_KEY else 'NOT SET'}")
print(f"API Key value: {UNSTRUCTURED_API_KEY[:10]}..." if UNSTRUCTURED_API_KEY else "NO KEY")
print(f"PDF: {PDF_PATH.name}")
print(f"PDF exists: {PDF_PATH.exists()}")
print()

if not UNSTRUCTURED_API_KEY:
    print("ERROR: No API key!")
    exit(1)

print("Starting cloud extraction (by_title)...")
start = time.time()

with open(PDF_PATH, "rb") as f:
    response = httpx.post(
        UNSTRUCTURED_CLOUD_URL,
        headers={"unstructured-api-key": UNSTRUCTURED_API_KEY},
        files={"files": (PDF_PATH.name, f, "application/pdf")},
        data={
            "strategy": "hi_res",
            "chunking_strategy": "by_title",
            "max_characters": "3000",
            "new_after_n_chars": "2500",
            "combine_text_under_n_chars": "200",
            "output_format": "application/json",
        },
        timeout=1800.0,
    )

elapsed = time.time() - start

print(f"Status: {response.status_code}")
print(f"Time: {elapsed:.1f}s")

if response.status_code == 200:
    elements = response.json()
    chunks = [e.get("text", "") for e in elements if e.get("text")]
    print(f"Chunks: {len(chunks)}")
    if chunks:
        lengths = [len(c) for c in chunks]
        print(f"Avg chars: {sum(lengths)/len(lengths):.0f}")
        print(f"Min: {min(lengths)}, Max: {max(lengths)}")
else:
    print(f"ERROR: {response.text[:500]}")
