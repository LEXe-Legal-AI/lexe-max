#!/usr/bin/env python
"""
Ingestion Pipeline: PDF → Docling → Articoli → Embeddings (GPU)

Converte PDF legali italiani in articoli strutturati con embeddings vettoriali.

Features:
- Docling per estrazione text da PDF (GPU accelerated)
- Chunking per articoli con regex patterns robusti
- Embeddings con multilingual-e5-large-instruct (1024 dims, GPU)
- Output JSON con articoli + embeddings

Usage:
    uv run --no-sync python scripts/ingest_with_docling.py <pdf_path> --codice CC

Requires:
    - PyTorch CUDA nightly (per RTX 5080 Blackwell)
    - docling
    - sentence-transformers
"""

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class Articolo:
    """Articolo estratto con embedding."""
    articolo_num: str           # "17", "2043-bis"
    articolo_num_norm: int      # 17, 2043
    articolo_suffix: str | None # None, "bis", "ter"
    rubrica: str | None
    testo: str
    testo_context: str = ""     # Con overlap

    # Hierarchy
    libro: str | None = None
    titolo: str | None = None
    capo: str | None = None
    sezione: str | None = None

    # Provenance
    page_start: int = 0
    page_end: int = 0

    # Embedding
    embedding: list[float] = field(default_factory=list)
    content_hash: str = ""

    def compute_hash(self) -> str:
        normalized = self.testo.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        self.content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return self.content_hash


# Patterns per articoli
ARTICOLO_PATTERNS = [
    re.compile(r'(?:Articolo|Art\.?)\s+(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)', re.IGNORECASE),
]

ARTICOLO_NUM_PATTERN = re.compile(
    r'^(\d+)(?:[-\s]?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?$',
    re.IGNORECASE
)

# Hierarchy patterns
LIBRO_PATTERN = re.compile(r'Libro\s+([IVX]+)', re.IGNORECASE)
TITOLO_PATTERN = re.compile(r'Titolo\s+([IVX]+)', re.IGNORECASE)
CAPO_PATTERN = re.compile(r'Capo\s+([IVX]+)', re.IGNORECASE)
SEZIONE_PATTERN = re.compile(r'Sezione\s+([IVX]+)', re.IGNORECASE)


def parse_article_num(articolo_num: str) -> tuple[int, str | None]:
    """Parse numero articolo in (num_norm, suffix)."""
    match = ARTICOLO_NUM_PATTERN.match(articolo_num)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        if suffix:
            suffix = suffix.lower()
        return num, suffix

    num_match = re.match(r'(\d+)', articolo_num)
    if num_match:
        return int(num_match.group(1)), None

    return 0, None


def convert_pdf_docling(pdf_path: str, use_gpu: bool = True) -> str:
    """Converte PDF in markdown con Docling."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice
    )

    # Config GPU
    if use_gpu and torch.cuda.is_available():
        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accel = AcceleratorOptions(device=AcceleratorDevice.CPU)
        print("  Using CPU")

    pipeline_opts = PdfPipelineOptions(
        accelerator_options=accel,
        do_ocr=False,  # PDF Altalex sono text-based
        do_table_structure=False,  # Skip per velocità
    )

    converter = DocumentConverter(
        format_options={'pdf': PdfFormatOption(pipeline_options=pipeline_opts)}
    )

    result = converter.convert(pdf_path)

    # Export to markdown
    return result.document.export_to_markdown()


def extract_articles(markdown: str, codice: str) -> list[Articolo]:
    """Estrae articoli dal markdown."""
    articles = []

    # Split per articoli
    lines = markdown.split('\n')

    current_article = None
    current_text = []

    # Hierarchy
    current_libro = None
    current_titolo = None
    current_capo = None
    current_sezione = None

    for line in lines:
        # Check hierarchy
        libro_match = LIBRO_PATTERN.search(line)
        if libro_match:
            current_libro = f"Libro {libro_match.group(1)}"
            current_titolo = current_capo = current_sezione = None

        titolo_match = TITOLO_PATTERN.search(line)
        if titolo_match:
            current_titolo = f"Titolo {titolo_match.group(1)}"
            current_capo = current_sezione = None

        capo_match = CAPO_PATTERN.search(line)
        if capo_match:
            current_capo = f"Capo {capo_match.group(1)}"
            current_sezione = None

        sezione_match = SEZIONE_PATTERN.search(line)
        if sezione_match:
            current_sezione = f"Sezione {sezione_match.group(1)}"

        # Check article header
        article_match = None
        for pattern in ARTICOLO_PATTERNS:
            m = pattern.search(line)
            if m:
                article_match = m.group(1)
                break

        if article_match:
            # Save previous article
            if current_article and current_text:
                testo = '\n'.join(current_text).strip()
                if len(testo) > 10:
                    current_article.testo = testo
                    current_article.compute_hash()
                    articles.append(current_article)

            # Start new article
            num_norm, suffix = parse_article_num(article_match)
            current_article = Articolo(
                articolo_num=article_match,
                articolo_num_norm=num_norm,
                articolo_suffix=suffix,
                rubrica=None,
                testo="",
                libro=current_libro,
                titolo=current_titolo,
                capo=current_capo,
                sezione=current_sezione,
            )
            current_text = []

            # Rest of line might be rubrica
            rest = line[article_match.end():].strip() if hasattr(article_match, 'end') else ""
            if rest and len(rest) > 5:
                # Clean rubrica
                rest = re.sub(r'^[\.\s\-]+', '', rest)
                if rest:
                    current_article.rubrica = rest

        elif current_article is not None:
            current_text.append(line)

    # Don't forget last article
    if current_article and current_text:
        testo = '\n'.join(current_text).strip()
        if len(testo) > 10:
            current_article.testo = testo
            current_article.compute_hash()
            articles.append(current_article)

    return articles


def add_overlap(articles: list[Articolo], overlap_chars: int = 200) -> None:
    """Aggiunge overlap inter-articolo."""
    for i, article in enumerate(articles):
        parts = []

        if i > 0 and articles[i-1].testo:
            prev_text = articles[i-1].testo[-overlap_chars:]
            parts.append(f"[...] {prev_text}")

        parts.append(article.testo)

        if i < len(articles) - 1 and articles[i+1].testo:
            next_text = articles[i+1].testo[:overlap_chars]
            parts.append(f"{next_text} [...]")

        article.testo_context = "\n\n".join(parts)


def generate_embeddings(articles: list[Articolo], batch_size: int = 32) -> None:
    """Genera embeddings con sentence-transformers su GPU."""
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model on {device}...")

    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=device)

    # Prepare texts (use testo_context for better retrieval)
    texts = []
    for art in articles:
        # E5 instruction format
        text = f"passage: Art. {art.articolo_num}. {art.rubrica or ''}\n{art.testo_context or art.testo}"
        texts.append(text[:2000])  # Truncate

    print(f"  Embedding {len(texts)} articles...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} articles/sec)")

    # Assign embeddings
    for art, emb in zip(articles, embeddings):
        art.embedding = emb.tolist()


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF with Docling + Embeddings")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--codice", required=True, help="Document code (CC, CP, GDPR, etc.)")
    parser.add_argument("--output", help="Output JSON path (default: same as PDF with .json)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap chars between articles")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix('.json')

    print(f"\n{'='*60}")
    print(f"Ingesting: {pdf_path.name}")
    print(f"Codice: {args.codice}")
    print(f"GPU: {'enabled' if not args.no_gpu and torch.cuda.is_available() else 'disabled'}")
    print(f"{'='*60}")

    # 1. Convert PDF
    print(f"\n[1/4] Converting PDF with Docling...")
    start = time.time()
    markdown = convert_pdf_docling(str(pdf_path), use_gpu=not args.no_gpu)
    print(f"  Markdown: {len(markdown)} chars in {time.time()-start:.1f}s")

    # 2. Extract articles
    print(f"\n[2/4] Extracting articles...")
    start = time.time()
    articles = extract_articles(markdown, args.codice)
    print(f"  Found {len(articles)} articles in {time.time()-start:.1f}s")

    if not articles:
        print("ERROR: No articles found!")
        sys.exit(1)

    # 3. Add overlap
    print(f"\n[3/4] Adding context overlap ({args.overlap} chars)...")
    add_overlap(articles, args.overlap)

    # 4. Generate embeddings
    print(f"\n[4/4] Generating embeddings...")
    generate_embeddings(articles, args.batch_size)

    # Save
    print(f"\n{'='*60}")
    print(f"Saving to: {output_path}")

    output_data = {
        "source_file": str(pdf_path),
        "codice": args.codice,
        "total_articles": len(articles),
        "embedding_model": "intfloat/multilingual-e5-large-instruct",
        "embedding_dims": 1024,
        "articles": [asdict(art) for art in articles],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"{'='*60}")
    print(f"\nSUMMARY:")
    print(f"  Articles: {len(articles)}")
    print(f"  With rubrica: {sum(1 for a in articles if a.rubrica)}")
    print(f"  With embedding: {sum(1 for a in articles if a.embedding)}")
    print(f"  Output: {output_path}")

    # Show first 5 articles
    print(f"\nFirst 5 articles:")
    for art in articles[:5]:
        print(f"  Art. {art.articolo_num}: {art.rubrica or '(no rubrica)'}")
        print(f"    Testo: {art.testo[:60]}...")


if __name__ == "__main__":
    main()
