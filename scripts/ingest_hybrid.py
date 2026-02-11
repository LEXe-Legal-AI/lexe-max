#!/usr/bin/env python
"""
Ingestion Ibrida: Regex veloce + LLM per rubriche

Approccio:
1. Docling converte PDF → markdown (GPU)
2. Regex estrae articoli (veloce)
3. LLM arricchisce solo le rubriche mancanti (batch)
4. Embeddings su GPU

Usage:
    uv run --no-sync python scripts/ingest_hybrid.py <pdf_path> --codice CC
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
class Articolo:
    """Articolo estratto con embedding."""
    articolo_num: str
    articolo_num_norm: int
    articolo_suffix: str | None
    rubrica: str | None
    testo: str
    testo_context: str = ""

    libro: str | None = None
    titolo: str | None = None
    capo: str | None = None
    sezione: str | None = None

    page_start: int = 0
    page_end: int = 0

    embedding: list[float] = field(default_factory=list)
    content_hash: str = ""

    def compute_hash(self) -> str:
        normalized = self.testo.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        self.content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return self.content_hash


# Patterns
ARTICOLO_PATTERNS = [
    re.compile(r'[-•]\s*Art\.?\s*(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)\.\s*(.+)', re.IGNORECASE),
    re.compile(r'(?:Articolo|Art\.?)\s+(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)', re.IGNORECASE),
]

ARTICOLO_NUM_PATTERN = re.compile(
    r'^(\d+)(?:[-\s]?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?$',
    re.IGNORECASE
)

LIBRO_PATTERN = re.compile(r'Libro\s+([IVX]+)', re.IGNORECASE)
TITOLO_PATTERN = re.compile(r'Titolo\s+([IVX]+)', re.IGNORECASE)
CAPO_PATTERN = re.compile(r'Capo\s+([IVX]+)', re.IGNORECASE)
SEZIONE_PATTERN = re.compile(r'Sezione\s+([IVX]+)', re.IGNORECASE)


def convert_pdf_docling(pdf_path: str, use_gpu: bool = True) -> str:
    """Converte PDF in markdown con Docling."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice
    )

    if use_gpu and torch.cuda.is_available():
        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accel = AcceleratorOptions(device=AcceleratorDevice.CPU)
        print("  Using CPU")

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


def parse_article_num(articolo_num: str) -> tuple[int, str | None]:
    """Parse numero articolo."""
    match = ARTICOLO_NUM_PATTERN.match(str(articolo_num))
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        if suffix:
            suffix = suffix.lower()
        return num, suffix

    num_match = re.match(r'(\d+)', str(articolo_num))
    if num_match:
        return int(num_match.group(1)), None

    return 0, None


def extract_articles_regex(markdown: str) -> list[Articolo]:
    """Estrae articoli con regex."""
    articles = []
    lines = markdown.split('\n')

    current_article = None
    current_text = []

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
        rubrica = None

        for pattern in ARTICOLO_PATTERNS:
            m = pattern.search(line)
            if m:
                article_match = m.group(1)
                # Second pattern might have rubrica
                if len(m.groups()) > 1:
                    rubrica = m.group(2).strip() if m.group(2) else None
                break

        if article_match:
            # Save previous
            if current_article and current_text:
                testo = '\n'.join(current_text).strip()
                if len(testo) > 10:
                    current_article.testo = testo
                    current_article.compute_hash()
                    articles.append(current_article)

            # New article
            num_norm, suffix = parse_article_num(article_match)
            current_article = Articolo(
                articolo_num=article_match,
                articolo_num_norm=num_norm,
                articolo_suffix=suffix,
                rubrica=rubrica,
                testo="",
                libro=current_libro,
                titolo=current_titolo,
                capo=current_capo,
                sezione=current_sezione,
            )
            current_text = []

            # Rest of line as start of testo
            rest = line[line.find(article_match) + len(article_match):].strip()
            rest = re.sub(r'^[\.\s\-]+', '', rest)
            if rest and not rubrica:
                # This might be rubrica or start of text
                if len(rest) < 100 and not rest[0].islower():
                    current_article.rubrica = rest
                else:
                    current_text.append(rest)

        elif current_article is not None:
            current_text.append(line)

    # Last article
    if current_article and current_text:
        testo = '\n'.join(current_text).strip()
        if len(testo) > 10:
            current_article.testo = testo
            current_article.compute_hash()
            articles.append(current_article)

    return articles


def enrich_rubriche_llm(articles: list[Articolo], provider: str = "ollama") -> None:
    """Arricchisce rubriche mancanti con LLM (batch)."""
    missing = [a for a in articles if not a.rubrica]
    if not missing:
        print("  All articles have rubriche")
        return

    print(f"  {len(missing)} articles missing rubrica, asking LLM...")

    # Prepare batch prompt
    batch_text = "\n".join([
        f"Art. {a.articolo_num}: {a.testo[:200]}..."
        for a in missing[:20]  # Limit batch
    ])

    prompt = f"""Per ogni articolo, estrai la rubrica (titolo) se presente nel testo.
La rubrica è solitamente all'inizio dell'articolo, in forma breve (es. "Risarcimento del danno", "Diritto di rettifica").

Rispondi con JSON: {{"articolo_num": "rubrica"}}
Se non c'è rubrica, metti null.

Articoli:
{batch_text}

JSON rubriche:"""

    if provider == "ollama":
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "huihui_ai/deepseek-r1-Fusion:latest",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2000}
            },
            timeout=120.0
        )
        text = response.json().get("response", "")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60.0
        )
        text = response.json()["choices"][0]["message"]["content"]

    # Parse response
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            rubriche = json.loads(match.group())
            for art in missing:
                if art.articolo_num in rubriche and rubriche[art.articolo_num]:
                    art.rubrica = rubriche[art.articolo_num]
                    print(f"    Art. {art.articolo_num}: {art.rubrica}")
    except Exception as e:
        print(f"  WARNING: Could not parse LLM response: {e}")


def add_overlap(articles: list[Articolo], overlap_chars: int = 200) -> None:
    """Aggiunge overlap inter-articolo."""
    for i, article in enumerate(articles):
        parts = []
        if i > 0 and articles[i-1].testo:
            parts.append(f"[...] {articles[i-1].testo[-overlap_chars:]}")
        parts.append(article.testo)
        if i < len(articles) - 1 and articles[i+1].testo:
            parts.append(f"{articles[i+1].testo[:overlap_chars]} [...]")
        article.testo_context = "\n\n".join(parts)


def generate_embeddings(articles: list[Articolo], batch_size: int = 32) -> None:
    """Genera embeddings con sentence-transformers su GPU."""
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model on {device}...")

    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=device)

    texts = []
    for art in articles:
        rubrica = f" - {art.rubrica}" if art.rubrica else ""
        text = f"passage: Art. {art.articolo_num}{rubrica}\n{art.testo_context or art.testo}"
        texts.append(text[:2000])

    print(f"  Embedding {len(texts)} articles...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"  Done in {time.time()-start:.1f}s")

    for art, emb in zip(articles, embeddings):
        art.embedding = emb.tolist()


def main():
    parser = argparse.ArgumentParser(description="Hybrid PDF ingestion")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--codice", required=True, help="Document code")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM enrichment")
    parser.add_argument("--llm", choices=["ollama", "openrouter"], default="openrouter")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix('.hybrid.json')

    print(f"\n{'='*60}")
    print(f"Ingesting: {pdf_path.name}")
    print(f"Codice: {args.codice}")
    print(f"{'='*60}")

    # 1. Convert
    print(f"\n[1/5] Converting PDF with Docling...")
    start = time.time()
    markdown = convert_pdf_docling(str(pdf_path), use_gpu=not args.no_gpu)
    print(f"  {len(markdown)} chars in {time.time()-start:.1f}s")

    # 2. Regex extract
    print(f"\n[2/5] Extracting with regex...")
    start = time.time()
    articles = extract_articles_regex(markdown)
    print(f"  {len(articles)} articles in {time.time()-start:.1f}s")

    if not articles:
        print("ERROR: No articles found!")
        sys.exit(1)

    # 3. LLM enrichment
    if not args.no_llm:
        print(f"\n[3/5] Enriching rubriche with LLM ({args.llm})...")
        try:
            enrich_rubriche_llm(articles, args.llm)
        except Exception as e:
            print(f"  WARNING: LLM enrichment failed: {e}")
    else:
        print(f"\n[3/5] Skipping LLM (--no-llm)")

    # 4. Overlap
    print(f"\n[4/5] Adding overlap...")
    add_overlap(articles)

    # 5. Embeddings
    print(f"\n[5/5] Generating embeddings...")
    generate_embeddings(articles, args.batch_size)

    # Save
    print(f"\n{'='*60}")

    output_data = {
        "source_file": str(pdf_path),
        "codice": args.codice,
        "total_articles": len(articles),
        "with_rubrica": sum(1 for a in articles if a.rubrica),
        "embedding_model": "intfloat/multilingual-e5-large-instruct",
        "embedding_dims": 1024,
        "articles": [asdict(art) for art in articles],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"SUMMARY:")
    print(f"  Articles: {len(articles)}")
    print(f"  With rubrica: {sum(1 for a in articles if a.rubrica)}")
    print(f"  Output: {output_path}")

    # Show some
    print(f"\nSample articles:")
    for art in articles[:5]:
        rub = art.rubrica or "(no rubrica)"
        print(f"  Art. {art.articolo_num}: {rub}")


if __name__ == "__main__":
    main()
