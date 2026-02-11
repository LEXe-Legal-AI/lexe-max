#!/usr/bin/env python
"""
Test 3 approcci di chunking/embedding:

1. CHUNKING FISSO - Chunk 500 chars con overlap 100
2. REGEX - Pattern matching per articoli
3. LLM - Estrazione con modello economico (Llama 3.2 3B)

Usage:
    uv run --no-sync python scripts/test_3_approaches.py <pdf_path> --codice DUDU

    # Con OpenRouter per approccio 3
    OPENROUTER_API_KEY=sk-or-... uv run --no-sync python scripts/test_3_approaches.py ...
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import torch


@dataclass
class Chunk:
    """Chunk generico."""
    chunk_id: int
    text: str
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


def convert_pdf_docling(pdf_path: str) -> str:
    """Converte PDF in markdown con Docling."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice
    )

    accel = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    )
    pipeline_opts = PdfPipelineOptions(
        accelerator_options=accel,
        do_ocr=False,
        do_table_structure=False,
    )
    converter = DocumentConverter(
        format_options={'pdf': PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


# ============================================================
# APPROCCIO 1: CHUNKING FISSO
# ============================================================

def approach_1_fixed_chunking(text: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Chunking fisso con overlap."""
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                metadata={"approach": "fixed_chunking"}
            ))
            chunk_id += 1

        start += chunk_size - overlap

    return chunks


# ============================================================
# APPROCCIO 2: REGEX
# ============================================================

ARTICOLO_PATTERNS = [
    re.compile(r'[-•]\s*Art\.?\s*(\d+(?:-\w+)?)\.\s*(.+)', re.IGNORECASE),
    re.compile(r'(?:Articolo|Art\.?)\s+(\d+(?:-\w+)?)', re.IGNORECASE),
]


def approach_2_regex(text: str) -> list[Chunk]:
    """Estrazione articoli con regex."""
    chunks = []
    lines = text.split('\n')

    current_article = None
    current_text = []
    chunk_id = 0

    for line in lines:
        article_match = None
        for pattern in ARTICOLO_PATTERNS:
            m = pattern.search(line)
            if m:
                article_match = m.group(1)
                break

        if article_match:
            # Save previous
            if current_article and current_text:
                full_text = '\n'.join(current_text).strip()
                if len(full_text) > 20:
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=full_text,
                        metadata={"approach": "regex", "articolo": current_article}
                    ))
                    chunk_id += 1

            current_article = article_match
            current_text = [line]
        elif current_article:
            current_text.append(line)

    # Last article
    if current_article and current_text:
        full_text = '\n'.join(current_text).strip()
        if len(full_text) > 20:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=full_text,
                metadata={"approach": "regex", "articolo": current_article}
            ))

    return chunks


# ============================================================
# APPROCCIO 3: LLM EXTRACTION
# ============================================================

def approach_3_llm(text: str, model: str = "meta-llama/llama-3.2-3b-instruct") -> list[Chunk]:
    """Estrazione articoli con LLM economico."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  SKIP: OPENROUTER_API_KEY not set")
        return []

    prompt = f"""Estrai TUTTI gli articoli dal seguente testo legale.

Per ogni articolo restituisci:
- numero: il numero dell'articolo (es. "1", "30")
- titolo: il titolo/rubrica se presente
- testo: il contenuto completo

Rispondi SOLO con JSON array valido:
[{{"numero": "1", "titolo": "...", "testo": "..."}}]

TESTO:
{text[:8000]}

JSON:"""

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4000,
            },
            timeout=120.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            articles = json.loads(match.group())
            chunks = []
            for i, art in enumerate(articles):
                text = art.get("testo", "")
                if text and len(text) > 10:
                    chunks.append(Chunk(
                        chunk_id=i,
                        text=text,
                        metadata={
                            "approach": "llm",
                            "articolo": art.get("numero"),
                            "titolo": art.get("titolo"),
                            "model": model
                        }
                    ))
            return chunks

    except Exception as e:
        print(f"  LLM Error: {e}")

    return []


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(chunks: list[Chunk]) -> None:
    """Genera embeddings con e5-large su GPU."""
    if not chunks:
        return

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=device)

    texts = [f"passage: {c.text[:1500]}" for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb.tolist()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test 3 chunking approaches")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--codice", default="TEST", help="Document code")
    parser.add_argument("--no-embeddings", action="store_true")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Testing 3 approaches on: {pdf_path.name}")
    print(f"{'='*60}")

    # Convert PDF
    print(f"\n[0] Converting PDF with Docling...")
    start = time.time()
    markdown = convert_pdf_docling(str(pdf_path))
    print(f"    {len(markdown)} chars in {time.time()-start:.1f}s")

    results = {}

    # Approach 1: Fixed Chunking
    print(f"\n[1] FIXED CHUNKING (500 chars, 100 overlap)...")
    start = time.time()
    chunks_1 = approach_1_fixed_chunking(markdown)
    print(f"    {len(chunks_1)} chunks in {time.time()-start:.3f}s")
    results["fixed_chunking"] = chunks_1

    # Approach 2: Regex
    print(f"\n[2] REGEX extraction...")
    start = time.time()
    chunks_2 = approach_2_regex(markdown)
    print(f"    {len(chunks_2)} articles in {time.time()-start:.3f}s")
    results["regex"] = chunks_2

    # Approach 3: LLM
    print(f"\n[3] LLM extraction (Llama 3.2 3B)...")
    start = time.time()
    chunks_3 = approach_3_llm(markdown)
    print(f"    {len(chunks_3)} articles in {time.time()-start:.1f}s")
    results["llm"] = chunks_3

    # Generate embeddings
    if not args.no_embeddings:
        print(f"\n[4] Generating embeddings...")
        for name, chunks in results.items():
            if chunks:
                start = time.time()
                generate_embeddings(chunks)
                print(f"    {name}: {len(chunks)} embeddings in {time.time()-start:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Approach':<20} {'Chunks':<10} {'Avg Length':<12} {'With Emb'}")
    print("-" * 55)

    for name, chunks in results.items():
        if chunks:
            avg_len = sum(len(c.text) for c in chunks) / len(chunks)
            with_emb = sum(1 for c in chunks if c.embedding)
            print(f"{name:<20} {len(chunks):<10} {avg_len:<12.0f} {with_emb}")
        else:
            print(f"{name:<20} {'0':<10} {'-':<12} {'-'}")

    # Show samples
    print(f"\n--- SAMPLES ---")

    for name, chunks in results.items():
        if chunks:
            print(f"\n{name.upper()} - First 2 chunks:")
            for c in chunks[:2]:
                art = c.metadata.get("articolo", "-")
                print(f"  [{c.chunk_id}] Art.{art}: {c.text[:80]}...")

    # Save results
    output_path = pdf_path.with_suffix('.3approaches.json')
    output_data = {
        "source": str(pdf_path),
        "codice": args.codice,
        "results": {
            name: [asdict(c) for c in chunks]
            for name, chunks in results.items()
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
