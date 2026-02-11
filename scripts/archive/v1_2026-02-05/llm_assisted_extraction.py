#!/usr/bin/env python
"""
LLM-Assisted Article Extraction

Algoritmo in 4 fasi:
1. ANALYSIS: LLM analizza struttura documento
2. REGEX: Estrazione con pattern informato
3. CLASSIFY: LLM classifica HEADER vs REFERENCE
4. EXTRACT: Estrazione finale pulita

Usage:
    # Con modello free
    uv run --no-sync python scripts/llm_assisted_extraction.py <pdf_path> --codice CC --model free

    # Con modello low-cost
    OPENROUTER_API_KEY=sk-or-... uv run --no-sync python scripts/llm_assisted_extraction.py <pdf_path> --codice CC --model cheap
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
from typing import Literal

import httpx
import torch

# Suffissi latini ordinali
LATIN_SUFFIXES = [
    "bis", "ter", "quater", "quinquies", "sexies", "septies", "octies",
    "novies", "nonies", "decies", "undecies", "duodecies", "terdecies",
    "quaterdecies", "quinquiesdecies", "sexiesdecies", "septiesdecies", "octiesdecies"
]

SUFFIX_PATTERN = "|".join(LATIN_SUFFIXES)


@dataclass
class DocumentAnalysis:
    """Risultato analisi documento."""
    article_min: int = 1
    article_max: int = 100
    header_pattern: str = "Art. {N}"
    has_suffixes: bool = False
    reference_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ArticleMatch:
    """Match di un potenziale articolo."""
    article_num: str
    article_num_norm: int
    suffix: str | None
    position: int
    context: str
    classification: Literal["HEADER", "REFERENCE", "AMBIGUOUS"] = "AMBIGUOUS"
    confidence: float = 0.0


@dataclass
class ExtractedArticle:
    """Articolo estratto finale."""
    articolo_num: str
    articolo_num_norm: int
    articolo_suffix: str | None
    rubrica: str | None
    testo: str
    position: int
    content_hash: str = ""


# ==============================================================================
# LLM CLIENTS
# ==============================================================================

MODELS = {
    "free": "deepseek/deepseek-chat-v3-0324:free",       # Free tier
    "chimera": "tngtech/deepseek-r1t2-chimera:free",     # Free - R1T2 hybrid 671B
    "nemo": "mistralai/mistral-nemo",                     # $0.02/M tokens - 12B
    "cheap": "meta-llama/llama-3.2-3b-instruct",          # $0.02/M tokens
    "medium": "google/gemini-2.0-flash-001",              # $0.10/M tokens
    "good": "anthropic/claude-3.5-sonnet",                # $3.00/M tokens
}


def call_llm(prompt: str, model_key: str = "free", max_tokens: int = 2000) -> str:
    """Chiama LLM via OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = MODELS.get(model_key, MODELS["free"])

    # Per modelli free, non serve API key
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  LLM Error: {e}")
        return ""


# ==============================================================================
# FASE 1: DOCUMENT ANALYSIS
# ==============================================================================

ANALYSIS_PROMPT = """Analizza questo documento legale italiano e rispondi in JSON.

DOCUMENTO (inizio):
{head}

DOCUMENTO (fine):
{tail}

Rispondi SOLO con JSON valido:
{{
  "article_min": <numero primo articolo, es. 1>,
  "article_max": <numero ultimo articolo, es. 2969>,
  "header_pattern": "<pattern header articoli, es. '## Art. {{N}}.' o '- Art. {{N}}.''>",
  "has_suffixes": <true se ci sono art. bis/ter/etc, false altrimenti>,
  "reference_patterns": ["<pattern riferimenti ad altri articoli, es. 'art. {{N}} c.c.'>"]
}}

JSON:"""


def heuristic_analyze(text: str) -> DocumentAnalysis:
    """Fallback: Analisi euristica senza LLM."""
    # Trova tutti i numeri articolo nel testo
    pattern = re.compile(r'Art\.?\s*(\d+)', re.IGNORECASE)
    numbers = [int(m.group(1)) for m in pattern.finditer(text)]

    if not numbers:
        return DocumentAnalysis()

    # Range
    article_min = min(numbers)
    article_max = max(numbers)

    # Detect suffixes
    suffix_pattern = re.compile(rf'Art\.?\s*\d+[-\s]?({SUFFIX_PATTERN})', re.IGNORECASE)
    has_suffixes = bool(suffix_pattern.search(text))

    # Detect header pattern (sample first 20k chars)
    sample = text[:20000]
    if '## Art.' in sample:
        header_pattern = "## Art. {N}"
    elif '- Art.' in sample:
        header_pattern = "- Art. {N}"
    elif '<b>Art.' in sample:
        header_pattern = "<b>Art. {N}</b>"
    else:
        header_pattern = "Art. {N}"

    return DocumentAnalysis(
        article_min=article_min,
        article_max=article_max,
        header_pattern=header_pattern,
        has_suffixes=has_suffixes,
        reference_patterns=[],
        confidence=0.7  # Lower confidence for heuristic
    )


def analyze_document(text: str, model: str = "free") -> DocumentAnalysis:
    """Fase 1: Analizza struttura documento con LLM."""
    head = text[:5000]
    tail = text[-2000:] if len(text) > 7000 else ""

    prompt = ANALYSIS_PROMPT.format(head=head, tail=tail)
    response = call_llm(prompt, model)

    # Parse JSON
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())
            return DocumentAnalysis(
                article_min=data.get("article_min", 1),
                article_max=data.get("article_max", 100),
                header_pattern=data.get("header_pattern", "Art. {N}"),
                has_suffixes=data.get("has_suffixes", False),
                reference_patterns=data.get("reference_patterns", []),
                confidence=0.9
            )
    except Exception as e:
        print(f"  Analysis parse error: {e}")

    # Fallback to heuristic
    print("  Fallback to heuristic analysis...")
    return heuristic_analyze(text)


# ==============================================================================
# FASE 2: REGEX INFORMATO
# ==============================================================================

def build_informed_regex(analysis: DocumentAnalysis) -> re.Pattern:
    """Costruisce regex dal pattern analizzato."""
    # Pattern base per numeri articolo
    num_pattern = r"(\d+)"

    # Aggiungi suffissi se presenti
    if analysis.has_suffixes:
        num_pattern = rf"(\d+)(?:[-\s]?({SUFFIX_PATTERN}))?"

    # Pattern comuni per header - più flessibili per CCI e altri codici
    header_patterns = [
        rf"##\s*Art\.?\s*{num_pattern}",           # Markdown: ## Art. N
        rf"-\s*Art\.?\s*{num_pattern}",            # List: - Art. N
        rf"<b>Art\.?\s*{num_pattern}</b>",         # HTML bold
        rf"Articolo\s+{num_pattern}",              # Full: Articolo N
        rf"^Art\.?\s*{num_pattern}\.",             # Start of line: Art. N.
        rf"^\d+\.\s*Art\.?\s*{num_pattern}",       # Numbered: 1. Art. N (CCI format)
        rf"Art\.?\s*{num_pattern}\.\s+[A-Z]",      # Art. N. Title (uppercase start)
    ]

    combined = "|".join(f"(?:{p})" for p in header_patterns)
    return re.compile(combined, re.IGNORECASE | re.MULTILINE)


def extract_matches(text: str, pattern: re.Pattern, analysis: DocumentAnalysis) -> list[ArticleMatch]:
    """Estrae tutti i match con contesto."""
    matches = []

    for m in pattern.finditer(text):
        # Trova il gruppo che ha matchato
        groups = [g for g in m.groups() if g is not None]
        if not groups:
            continue

        # Primo gruppo numerico è il numero articolo
        num_str = groups[0]
        try:
            num = int(num_str)
        except ValueError:
            continue

        # Filtra per range
        if num < analysis.article_min or num > analysis.article_max:
            continue

        # Suffix se presente
        suffix = groups[1].lower() if len(groups) > 1 and groups[1] else None

        # Contesto
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 100)
        context = text[start:end]

        # Articolo formattato
        art_num = num_str
        if suffix:
            art_num = f"{num_str}-{suffix}"

        matches.append(ArticleMatch(
            article_num=art_num,
            article_num_norm=num,
            suffix=suffix,
            position=m.start(),
            context=context,
        ))

    return matches


# ==============================================================================
# FASE 3: CLASSIFICAZIONE LLM
# ==============================================================================

CLASSIFY_PROMPT = """Classifica ogni contesto come HEADER (inizio di un articolo) o REFERENCE (riferimento ad altro articolo).

Un HEADER è l'inizio vero dell'articolo, tipicamente:
- All'inizio di una riga
- Seguito dal testo dell'articolo
- Formato: "Art. N. Testo articolo..." o "## Art. N" o "Articolo N"

Un REFERENCE è un riferimento ad un altro articolo nel testo, tipicamente:
- Nel mezzo di una frase
- Formato: "v. art. N", "ai sensi dell'art. N", "art. N c.c."

CONTESTI DA CLASSIFICARE:
{contexts}

Rispondi con JSON array, un oggetto per ogni contesto:
[{{"id": 0, "class": "HEADER"}}, {{"id": 1, "class": "REFERENCE"}}, ...]

JSON:"""


def heuristic_classify(matches: list[ArticleMatch], text: str) -> None:
    """Fallback: Classificazione euristica senza LLM."""
    # Header markers: strong signals for headers (at the actual match location)
    header_markers = [
        r"^##\s*Art",                    # Markdown heading
        r"^-\s*Art",                     # List item
        r"^Art\.\s*\d+\.\s*$",           # Art. N. on its own line
        r"<b>Art\.",                     # Bold tag
        r"^\d+\.\s*Art\.",               # Numbered: 1. Art. (CCI format)
        r"^Art\.\s*\d+\.\s+[A-Z]",       # Art. N. Title (rubrica con maiuscola)
    ]
    header_pattern = re.compile("|".join(header_markers), re.IGNORECASE | re.MULTILINE)

    for m in matches:
        # Get the line where the match starts
        line_start = text.rfind('\n', 0, m.position) + 1
        line_end = text.find('\n', m.position)
        if line_end == -1:
            line_end = len(text)
        match_line = text[line_start:line_end]

        # Strong header signals: the match line itself starts with header pattern
        if header_pattern.match(match_line):
            m.classification = "HEADER"
            m.confidence = 0.95
            continue

        # Position-based: how many chars before Art. on this line?
        chars_before = m.position - line_start

        # Very strong header signal: at start of line (0-3 chars)
        if chars_before <= 3:
            m.classification = "HEADER"
            m.confidence = 0.85
            continue

        # Check for reference patterns in the context BEFORE the match
        ctx_before = text[max(0, m.position - 50):m.position]
        ref_signals = [
            r"ai sensi dell[''']",
            r"di cui all[''']",
            r"previsto dall[''']",
            r"secondo l[''']",
            r"v\.\s*$",
            r"cfr\.?\s*$",
            r"ex\s+$",
            r"dell[''']$",
        ]
        ref_pattern = re.compile("|".join(ref_signals), re.IGNORECASE)

        if ref_pattern.search(ctx_before):
            m.classification = "REFERENCE"
            m.confidence = 0.8
            continue

        # Check for reference patterns AFTER the match (like c.c., c.p.)
        ctx_after = text[m.position:min(len(text), m.position + 50)]
        after_ref = re.compile(r'Art\.?\s*\d+\s*(?:c\.c\.|c\.p\.|cost\.)', re.IGNORECASE)
        if after_ref.search(ctx_after):
            m.classification = "REFERENCE"
            m.confidence = 0.7
            continue

        # Check for "Art. N. Rubrica" pattern (title with uppercase)
        # This catches headers that appear mid-line after section titles
        rubrica_pattern = re.compile(
            rf'Art\.?\s*{m.article_num}(?:[-\s]?(?:{SUFFIX_PATTERN}))?\.\s+[A-Z][a-z]',
            re.IGNORECASE
        )
        if rubrica_pattern.search(ctx_after):
            m.classification = "HEADER"
            m.confidence = 0.75
            continue

        # Default: if at line start area (0-10 chars), treat as header
        if chars_before <= 10:
            m.classification = "HEADER"
            m.confidence = 0.6
        else:
            # Mid-line = likely reference
            m.classification = "REFERENCE"
            m.confidence = 0.5


def classify_matches(matches: list[ArticleMatch], model: str = "free", batch_size: int = 30, text: str = "") -> None:
    """Fase 3: Classifica matches con LLM."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    # If no API key, use heuristic
    if not api_key:
        print("  No API key, using heuristic classification...")
        heuristic_classify(matches, text)
        return

    for i in range(0, len(matches), batch_size):
        batch = matches[i:i + batch_size]

        # Prepara contesti
        contexts = "\n".join([
            f"[{j}] Art.{m.article_num}: ...{m.context}..."
            for j, m in enumerate(batch)
        ])

        prompt = CLASSIFY_PROMPT.format(contexts=contexts)
        response = call_llm(prompt, model, max_tokens=1000)

        # Parse response
        try:
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                classifications = json.loads(match.group())
                for c in classifications:
                    idx = c.get("id", -1)
                    cls = c.get("class", "AMBIGUOUS")
                    if 0 <= idx < len(batch):
                        batch[idx].classification = cls
                        batch[idx].confidence = 0.9
        except Exception as e:
            print(f"  Classify parse error: {e}")

        # Rate limit
        time.sleep(0.5)


# ==============================================================================
# FASE 4: ESTRAZIONE FINALE
# ==============================================================================

def extract_articles(text: str, matches: list[ArticleMatch]) -> list[ExtractedArticle]:
    """Fase 4: Estrai articoli dai match classificati come HEADER."""

    # Filtra solo HEADER
    headers = [m for m in matches if m.classification == "HEADER"]

    # Ordina per posizione
    headers.sort(key=lambda x: x.position)

    # Dedup per numero articolo (tieni primo per ogni numero)
    seen = set()
    unique_headers = []
    for h in headers:
        key = (h.article_num_norm, h.suffix)
        if key not in seen:
            seen.add(key)
            unique_headers.append(h)

    # Estrai testo tra headers
    articles = []
    for i, h in enumerate(unique_headers):
        # Fine = inizio prossimo header o fine documento
        end_pos = unique_headers[i + 1].position if i + 1 < len(unique_headers) else len(text)

        # Testo articolo
        article_text = text[h.position:end_pos].strip()

        # Estrai rubrica (prima riga dopo header)
        lines = article_text.split('\n')
        rubrica = None
        testo_start = 0

        for j, line in enumerate(lines[1:], 1):
            line = line.strip()
            if line and len(line) > 5:
                # Se sembra una rubrica (breve, capitalizzata)
                if len(line) < 150 and (line[0].isupper() or line.startswith('(')):
                    rubrica = line.rstrip('.')
                    testo_start = j + 1
                break

        testo = '\n'.join(lines[testo_start:]).strip() if testo_start > 0 else '\n'.join(lines[1:]).strip()

        # Hash
        content_hash = hashlib.sha256(testo.lower().encode()).hexdigest()[:16]

        articles.append(ExtractedArticle(
            articolo_num=h.article_num,
            articolo_num_norm=h.article_num_norm,
            articolo_suffix=h.suffix,
            rubrica=rubrica,
            testo=testo,
            position=h.position,
            content_hash=content_hash,
        ))

    return articles


# ==============================================================================
# PIPELINE COMPLETA
# ==============================================================================

def convert_pdf(pdf_path: str) -> str:
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


def run_pipeline(
    pdf_path: str,
    codice: str,
    model: str = "free",
    skip_classify: bool = False,
) -> tuple[list[ExtractedArticle], dict]:
    """Esegue pipeline completa."""

    stats = {"timings": {}, "counts": {}}

    # 0. Convert PDF
    print(f"\n[0] Converting PDF...")
    start = time.time()
    text = convert_pdf(pdf_path)
    stats["timings"]["convert"] = time.time() - start
    stats["counts"]["chars"] = len(text)
    print(f"    {len(text):,} chars in {stats['timings']['convert']:.1f}s")

    # 1. Analyze
    print(f"\n[1] Analyzing document structure (LLM: {model})...")
    start = time.time()
    analysis = analyze_document(text, model)
    stats["timings"]["analyze"] = time.time() - start
    print(f"    Range: Art. {analysis.article_min} - {analysis.article_max}")
    print(f"    Pattern: {analysis.header_pattern}")
    print(f"    Suffixes: {analysis.has_suffixes}")
    print(f"    Time: {stats['timings']['analyze']:.1f}s")

    # 2. Regex extraction
    print(f"\n[2] Extracting with informed regex...")
    start = time.time()
    pattern = build_informed_regex(analysis)
    matches = extract_matches(text, pattern, analysis)
    stats["timings"]["regex"] = time.time() - start
    stats["counts"]["matches"] = len(matches)
    print(f"    {len(matches)} matches in {stats['timings']['regex']:.3f}s")

    # 3. Classify (optional)
    if not skip_classify and matches:
        print(f"\n[3] Classifying matches (LLM: {model})...")
        start = time.time()
        classify_matches(matches, model, text=text)
        stats["timings"]["classify"] = time.time() - start

        headers = sum(1 for m in matches if m.classification == "HEADER")
        refs = sum(1 for m in matches if m.classification == "REFERENCE")
        ambig = sum(1 for m in matches if m.classification == "AMBIGUOUS")

        stats["counts"]["headers"] = headers
        stats["counts"]["references"] = refs
        stats["counts"]["ambiguous"] = ambig

        print(f"    HEADER: {headers}, REFERENCE: {refs}, AMBIGUOUS: {ambig}")
        print(f"    Time: {stats['timings']['classify']:.1f}s")
    else:
        # Se skip, assume tutti HEADER
        for m in matches:
            m.classification = "HEADER"
        stats["counts"]["headers"] = len(matches)
        print(f"\n[3] Skipping classification (--skip-classify)")

    # 4. Extract
    print(f"\n[4] Extracting final articles...")
    start = time.time()
    articles = extract_articles(text, matches)
    stats["timings"]["extract"] = time.time() - start
    stats["counts"]["articles"] = len(articles)

    with_rubrica = sum(1 for a in articles if a.rubrica)
    stats["counts"]["with_rubrica"] = with_rubrica

    print(f"    {len(articles)} unique articles")
    print(f"    {with_rubrica} with rubrica")

    return articles, stats


# ==============================================================================
# FILTRI DOCUMENTO-SPECIFICI
# ==============================================================================

# CCI (Codice Crisi Impresa) - Falsi positivi noti
# Questi "articoli" sono in realtà riferimenti nelle note di modifica a fine documento
CCI_FALSE_POSITIVES = {
    "69-septies",   # Appare a pos ~985k, testo: "- c) all'articolo 70..."
    "104-bis",      # Appare a pos ~1019k, testo: "luglio 1989, n. 271..."
}


def filter_false_positives(articles: list[ExtractedArticle], codice: str) -> list[ExtractedArticle]:
    """
    Rimuove falsi positivi noti per documento specifico.

    NOTA: Questi filtri sono specifici per documento e derivano da
    analisi manuale dei risultati di estrazione.
    """
    if codice.upper() == "CCI":
        return [a for a in articles if a.articolo_num not in CCI_FALSE_POSITIVES]

    # Altri codici: nessun filtro specifico (per ora)
    return articles


def validate_coverage(articles: list[ExtractedArticle], expected_max: int) -> dict:
    """Valida la copertura degli articoli estratti."""
    # Collect all base article numbers (ignoring suffixes for now)
    extracted_nums = set()
    for a in articles:
        extracted_nums.add(a.articolo_num_norm)

    # Expected range
    expected = set(range(1, expected_max + 1))

    # Analysis
    missing = sorted(expected - extracted_nums)
    extra = sorted(extracted_nums - expected)

    # Gap analysis - find consecutive gaps
    gaps = []
    if missing:
        gap_start = missing[0]
        gap_end = missing[0]
        for n in missing[1:]:
            if n == gap_end + 1:
                gap_end = n
            else:
                gaps.append((gap_start, gap_end))
                gap_start = n
                gap_end = n
        gaps.append((gap_start, gap_end))

    return {
        "expected_max": expected_max,
        "extracted_count": len(articles),
        "unique_base_nums": len(extracted_nums),
        "missing_count": len(missing),
        "missing_first_10": missing[:10] if missing else [],
        "gaps": gaps[:10] if gaps else [],
        "extra_count": len(extra),
        "extra_nums": extra[:10] if extra else [],
        "coverage_pct": (len(extracted_nums) / expected_max) * 100 if expected_max > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-Assisted Article Extraction")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--codice", required=True, help="Document code (CC, CP, etc.)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="free")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--skip-classify", action="store_true", help="Skip LLM classification")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix('.llm_extracted.json')

    print(f"\n{'='*60}")
    print(f"LLM-Assisted Article Extraction")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Model: {args.model} ({MODELS[args.model]})")

    # Run pipeline
    articles, stats = run_pipeline(
        str(pdf_path),
        args.codice,
        args.model,
        args.skip_classify,
    )

    # Apply document-specific filters
    original_count = len(articles)
    articles = filter_false_positives(articles, args.codice)
    filtered_count = original_count - len(articles)
    if filtered_count > 0:
        print(f"\n[5] Filtered {filtered_count} false positives (codice: {args.codice})")
        stats["counts"]["filtered"] = filtered_count

    # Save
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")

    # Validate coverage first
    coverage = None
    if articles:
        max_num = max(a.articolo_num_norm for a in articles)
        coverage = validate_coverage(articles, max_num)

    output_data = {
        "source": str(pdf_path),
        "codice": args.codice,
        "model": args.model,
        "stats": stats,
        "coverage": coverage,
        "articles": [asdict(a) for a in articles],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Total articles: {len(articles)}")
    print(f"Output: {output_path}")

    # Print coverage
    if coverage:
        print(f"\n--- COVERAGE ---")
        print(f"Max article number: {coverage['expected_max']}")
        print(f"Unique base numbers: {coverage['unique_base_nums']}")
        print(f"Coverage: {coverage['coverage_pct']:.1f}%")
        if coverage['missing_count'] > 0:
            print(f"Missing ({coverage['missing_count']}): {coverage['missing_first_10']}...")
            if coverage['gaps']:
                print(f"Gaps: {coverage['gaps']}")
        if coverage['extra_count'] > 0:
            print(f"Extra: {coverage['extra_nums']}")

    # Show sample
    print(f"\nFirst 5 articles:")
    for a in articles[:5]:
        rub = a.rubrica[:50] + "..." if a.rubrica and len(a.rubrica) > 50 else a.rubrica
        print(f"  Art. {a.articolo_num}: {rub or '(no rubrica)'}")

    print(f"\nLast 5 articles:")
    for a in articles[-5:]:
        rub = a.rubrica[:50] + "..." if a.rubrica and len(a.rubrica) > 50 else a.rubrica
        print(f"  Art. {a.articolo_num}: {rub or '(no rubrica)'}")


if __name__ == "__main__":
    main()
