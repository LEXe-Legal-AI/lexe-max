#!/usr/bin/env python
"""
Ingestion Pipeline con LLM: PDF → Docling → LLM Extract → Embeddings (GPU)

Usa un LLM per estrarre articoli strutturati dal markdown di Docling.
Molto più preciso dei regex per rubriche e struttura gerarchica.

Usage:
    uv run --no-sync python scripts/ingest_with_llm.py <pdf_path> --codice CC

    # Con Ollama (default)
    uv run --no-sync python scripts/ingest_with_llm.py file.pdf --codice CC --llm ollama

    # Con OpenRouter
    OPENROUTER_API_KEY=sk-or-... uv run --no-sync python scripts/ingest_with_llm.py file.pdf --codice CC --llm openrouter
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


EXTRACTION_PROMPT = """Sei un esperto di diritto italiano. Estrai TUTTI gli articoli dal seguente testo di legge.

Per OGNI articolo, estrai:
- articolo_num: numero dell'articolo (es. "1", "2043", "2043-bis")
- rubrica: titolo/nome dell'articolo se presente (es. "Risarcimento del danno")
- testo: il contenuto completo dell'articolo
- libro: se presente (es. "Libro IV")
- titolo: se presente (es. "Titolo IX")
- capo: se presente (es. "Capo I")
- sezione: se presente (es. "Sezione II")

IMPORTANTE:
- Estrai TUTTI gli articoli, non saltarne nessuno
- La rubrica è il titolo dell'articolo, NON il testo
- Il testo deve essere completo, inclusi tutti i commi
- Mantieni la numerazione originale (1, 2, 2-bis, etc.)

Rispondi SOLO con un JSON array valido, senza markdown o commenti.

Esempio formato:
[{{"articolo_num": "1", "rubrica": "Titolo", "testo": "...", "libro": null, "titolo": null, "capo": null, "sezione": null}}]

TESTO DA ANALIZZARE:
---
{text}
---

JSON degli articoli estratti:"""


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


def extract_with_ollama(markdown: str, model: str = "huihui_ai/deepseek-r1-Fusion:latest") -> list[dict]:
    """Estrae articoli usando Ollama."""
    print(f"  Using Ollama model: {model}")

    # Split markdown in chunks se troppo lungo (context limit)
    max_chars = 12000  # ~3000 tokens
    chunks = []
    current = ""

    for line in markdown.split('\n'):
        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += '\n' + line
    if current:
        chunks.append(current)

    print(f"  Split in {len(chunks)} chunks")

    all_articles = []

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")

        prompt = EXTRACTION_PROMPT.format(text=chunk)

        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 8000,
                }
            },
            timeout=300.0
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "")

        # Parse JSON from response
        articles = parse_llm_response(text)
        all_articles.extend(articles)

    return all_articles


def extract_with_openrouter(markdown: str, model: str = "anthropic/claude-3.5-sonnet") -> list[dict]:
    """Estrae articoli usando OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    print(f"  Using OpenRouter model: {model}")

    # Split in chunks
    max_chars = 30000  # Claude ha context più grande
    chunks = []
    current = ""

    for line in markdown.split('\n'):
        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += '\n' + line
    if current:
        chunks.append(current)

    print(f"  Split in {len(chunks)} chunks")

    all_articles = []

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")

        prompt = EXTRACTION_PROMPT.format(text=chunk)

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 16000,
            },
            timeout=300.0
        )
        response.raise_for_status()

        result = response.json()
        text = result["choices"][0]["message"]["content"]

        articles = parse_llm_response(text)
        all_articles.extend(articles)

    return all_articles


def parse_llm_response(text: str) -> list[dict]:
    """Parse JSON response from LLM."""
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        print(f"  WARNING: No JSON array found in response")
        return []

    try:
        articles = json.loads(match.group())
        return articles
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}")
        # Try to fix common issues
        fixed = match.group().replace("'", '"')
        try:
            return json.loads(fixed)
        except:
            return []


def parse_article_num(articolo_num: str) -> tuple[int, str | None]:
    """Parse numero articolo."""
    pattern = re.compile(
        r'^(\d+)(?:[-\s]?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?$',
        re.IGNORECASE
    )
    match = pattern.match(str(articolo_num))
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


def build_articles(raw_articles: list[dict]) -> list[Articolo]:
    """Converte raw dicts in Articolo objects."""
    articles = []

    for raw in raw_articles:
        articolo_num = str(raw.get("articolo_num", ""))
        if not articolo_num:
            continue

        num_norm, suffix = parse_article_num(articolo_num)

        testo = raw.get("testo", "")
        if not testo or len(testo) < 10:
            continue

        art = Articolo(
            articolo_num=articolo_num,
            articolo_num_norm=num_norm,
            articolo_suffix=suffix,
            rubrica=raw.get("rubrica"),
            testo=testo,
            libro=raw.get("libro"),
            titolo=raw.get("titolo"),
            capo=raw.get("capo"),
            sezione=raw.get("sezione"),
        )
        art.compute_hash()
        articles.append(art)

    # Sort by article number
    articles.sort(key=lambda x: (x.articolo_num_norm, x.articolo_suffix or ""))

    # Remove duplicates by content_hash
    seen = set()
    unique = []
    for art in articles:
        if art.content_hash not in seen:
            seen.add(art.content_hash)
            unique.append(art)

    return unique


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

    texts = []
    for art in articles:
        # E5 instruction format con rubrica
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

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} articles/sec)")

    for art, emb in zip(articles, embeddings):
        art.embedding = emb.tolist()


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF with LLM extraction")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--codice", required=True, help="Document code (CC, CP, GDPR, etc.)")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--llm", choices=["ollama", "openrouter"], default="ollama", help="LLM provider")
    parser.add_argument("--model", help="Model name (default depends on provider)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix('.llm.json')

    print(f"\n{'='*60}")
    print(f"Ingesting: {pdf_path.name}")
    print(f"Codice: {args.codice}")
    print(f"LLM: {args.llm}")
    print(f"GPU: {'enabled' if not args.no_gpu and torch.cuda.is_available() else 'disabled'}")
    print(f"{'='*60}")

    # 1. Convert PDF
    print(f"\n[1/4] Converting PDF with Docling...")
    start = time.time()
    markdown = convert_pdf_docling(str(pdf_path), use_gpu=not args.no_gpu)
    print(f"  Markdown: {len(markdown)} chars in {time.time()-start:.1f}s")

    # 2. Extract with LLM
    print(f"\n[2/4] Extracting articles with LLM...")
    start = time.time()

    if args.llm == "ollama":
        model = args.model or "huihui_ai/deepseek-r1-Fusion:latest"
        raw_articles = extract_with_ollama(markdown, model)
    else:
        model = args.model or "anthropic/claude-3.5-sonnet"
        raw_articles = extract_with_openrouter(markdown, model)

    print(f"  LLM extracted {len(raw_articles)} raw articles in {time.time()-start:.1f}s")

    # 3. Build articles
    print(f"\n[3/4] Building structured articles...")
    articles = build_articles(raw_articles)
    print(f"  Built {len(articles)} valid articles")

    if not articles:
        print("ERROR: No valid articles extracted!")
        sys.exit(1)

    # Add overlap
    add_overlap(articles)

    # 4. Generate embeddings
    if not args.no_embeddings:
        print(f"\n[4/4] Generating embeddings...")
        generate_embeddings(articles, args.batch_size)
    else:
        print(f"\n[4/4] Skipping embeddings (--no-embeddings)")

    # Save
    print(f"\n{'='*60}")
    print(f"Saving to: {output_path}")

    output_data = {
        "source_file": str(pdf_path),
        "codice": args.codice,
        "llm_provider": args.llm,
        "llm_model": model,
        "total_articles": len(articles),
        "embedding_model": "intfloat/multilingual-e5-large-instruct" if not args.no_embeddings else None,
        "embedding_dims": 1024 if not args.no_embeddings else None,
        "articles": [asdict(art) for art in articles],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"{'='*60}")
    print(f"\nSUMMARY:")
    print(f"  Articles: {len(articles)}")
    print(f"  With rubrica: {sum(1 for a in articles if a.rubrica)}")
    print(f"  With embedding: {sum(1 for a in articles if a.embedding)}")
    print(f"  Output: {output_path}")

    # Show articles with rubrica
    print(f"\nArticles with rubrica:")
    for art in articles[:10]:
        if art.rubrica:
            print(f"  Art. {art.articolo_num}: {art.rubrica}")


if __name__ == "__main__":
    main()
