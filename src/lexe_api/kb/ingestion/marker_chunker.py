"""Marker JSON Chunker for Altalex PDF ingestion.

Processes marker-pdf JSON output to extract structured articles (articoli).
Groups blocks by article headers and adds inter-article overlap.

Usage:
    chunker = MarkerChunker(overlap_chars=200)
    articles = chunker.process_file("gdpr.json", codice="GDPR")

    for art in articles:
        print(f"Art. {art.articolo_num}: {art.rubrica}")
        print(art.testo[:200])
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import structlog

logger = structlog.get_logger(__name__)


# Regex patterns for article detection
ARTICOLO_HEADER_PATTERNS = [
    # HTML-style headers from marker: <h1><b>Articolo 17</b></h1> or <h1><b>Art. 17</b></h1>
    # Also handles <span> tags before Art: <h3><span id="..."></span>ART. 5 (...)
    # Uses (?:<[^>]+>)* to skip any nested tags (span, b, etc.) before Art.
    re.compile(
        r'<h\d[^>]*>(?:<[^>]+>)*\s*(?:Articolo|Art\.?)\s*(\d+(?:-\w+)?)',
        re.IGNORECASE
    ),
    # Altalex format: <p block-type="Text"><b>Art. 3.</b> <b>Rubrica...</b></p>
    # The <b> tag wraps the article number
    re.compile(
        r'<p[^>]*>(?:<b>\s*)?(?:Articolo|Art\.?)\s*(\d+(?:-\w+)?)',
        re.IGNORECASE
    ),
    # Plain text header: Art. 17. or Articolo 17
    re.compile(r'^(?:Articolo|Art\.?)\s+(\d+(?:-\w+)?)\.\s*', re.IGNORECASE | re.MULTILINE),
    # Plain text without period: Articolo 17 or Art. 17
    re.compile(r'^(?:Articolo|Art\.?)\s+(\d+(?:-\w+)?)\s*$', re.IGNORECASE | re.MULTILINE),
    # Bold markdown: **Articolo 17** or **Art. 17**
    re.compile(r'^\*\*(?:Articolo|Art\.?)\s+(\d+(?:-\w+)?)\*\*', re.IGNORECASE | re.MULTILINE),
    # Paragraph with Art.: <p block-type="Text">Art. 3-bis.</p>
    re.compile(r'<p[^>]*>(?:Articolo|Art\.?)\s+(\d+(?:-\w+)?)\.</p>', re.IGNORECASE),
    # Art. anywhere followed by rubrica in parentheses: "...text... ART. 318-bis (Ambito di applicazione)"
    # Common in Text blocks where article header is mid-text
    re.compile(
        r'ART\.?\s*(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|octiesdecies))?)\s*\(',
        re.IGNORECASE
    ),
    # Bold Art. N. anywhere in text: "...section title <b>Art. 172.</b> <b>Rubrica</b>..."
    # Common in CCI where section headers precede article headers in same block
    re.compile(
        r'<b>Art\.?\s*(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|octiesdecies))?)\.</b>',
        re.IGNORECASE
    ),
]

# Pattern for extracting article number parts
# Extended Latin ordinals: bis(2), ter(3), quater(4), quinquies(5), sexies(6),
# septies(7), octies(8), novies(9), decies(10), undecies(11), duodecies(12),
# terdecies(13), quaterdecies(14), quinquiesdecies(15), sexiesdecies(16),
# septiesdecies(17), octiesdecies(18)
ARTICOLO_NUM_PATTERN = re.compile(
    r'^(\d+)(?:[-\s]?(bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|'
    r'decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|'
    r'sexiesdecies|septiesdecies|octiesdecies))?$',
    re.IGNORECASE
)

# Hierarchy patterns
LIBRO_PATTERN = re.compile(r'Libro\s+([IVX]+)', re.IGNORECASE)
TITOLO_PATTERN = re.compile(r'Titolo\s+([IVX]+)', re.IGNORECASE)
CAPO_PATTERN = re.compile(r'Capo\s+([IVX]+)', re.IGNORECASE)
SEZIONE_PATTERN = re.compile(r'Sezione\s+([IVX]+)', re.IGNORECASE)

# Rubrica detection (text immediately after article header)
RUBRICA_PATTERN = re.compile(
    r'^[A-Z][^.!?\n]{5,200}[.)]?\s*$',  # Capitalized, 5-200 chars, ends with . or )
    re.MULTILINE
)


@dataclass
class MarkerBlock:
    """A single block from marker JSON output."""
    block_id: str
    block_type: str  # SectionHeader, TextBlock, ListBlock, Table, etc.
    text: str
    html: str = ""
    page: int = 0
    position: int = 0  # Position in document

    @classmethod
    def from_dict(cls, data: dict, position: int = 0) -> "MarkerBlock":
        """Create from marker JSON block."""
        html = data.get("html", "")
        text = data.get("text", "")

        # If text is empty but html exists, extract text from html
        if not text and html:
            text = cls._html_to_text(html)

        return cls(
            block_id=data.get("id", ""),
            block_type=data.get("block_type", "Unknown"),
            text=text,
            html=html,
            page=data.get("page", data.get("pnum", 0)),
            position=position,
        )

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Extract plain text from HTML."""
        # Remove img tags
        text = re.sub(r'<img[^>]*>', '', html)
        # Remove HTML tags but keep content
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


@dataclass
class ExtractedArticle:
    """Extracted article from marker JSON."""

    # Article identification
    articolo_num: str           # "17", "2043-bis"
    articolo_num_norm: int      # 17, 2043
    articolo_suffix: str | None # None, "bis", "ter"

    # Content
    rubrica: str | None         # Article title
    testo: str                  # Clean text
    testo_context: str = ""     # Text with overlap

    # Hierarchy
    libro: str | None = None
    titolo: str | None = None
    capo: str | None = None
    sezione: str | None = None

    # Provenance
    page_start: int = 0
    page_end: int = 0
    source_blocks: list[str] = field(default_factory=list)

    # Computed
    content_hash: str = ""

    # Validation
    warnings: list[str] = field(default_factory=list)

    @property
    def articolo_sort_key(self) -> str:
        """Sort key for ordering: 000017.00 or 002043.bis"""
        suffix = self.articolo_suffix or "00"
        return f"{self.articolo_num_norm:06d}.{suffix}"

    def compute_hash(self) -> str:
        """SHA256 of normalized text."""
        normalized = self.testo.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        self.content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return self.content_hash


class MarkerChunker:
    """
    Process marker JSON output to extract articles.

    Strategy:
    1. Parse blocks from marker JSON
    2. Identify article headers (SectionHeader with "Articolo N")
    3. Group consecutive blocks until next article header
    4. Extract rubrica from first text block after header
    5. Add inter-article overlap for context
    """

    def __init__(
        self,
        overlap_chars: int = 200,
        min_article_chars: int = 50,
        max_rubrica_chars: int = 300,
    ):
        """
        Args:
            overlap_chars: Characters to overlap between consecutive articles
            min_article_chars: Minimum article text length
            max_rubrica_chars: Maximum rubrica length
        """
        self.overlap_chars = overlap_chars
        self.min_article_chars = min_article_chars
        self.max_rubrica_chars = max_rubrica_chars

    def process_file(
        self,
        json_path: str | Path,
        codice: str,
    ) -> list[ExtractedArticle]:
        """
        Process marker JSON file and extract articles.

        Args:
            json_path: Path to marker JSON output
            codice: Document code (CC, CP, GDPR, etc.)

        Returns:
            List of extracted articles
        """
        json_path = Path(json_path)
        logger.info("Processing marker JSON", path=str(json_path), codice=codice)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract blocks
        blocks = self._extract_blocks(data)
        logger.info("Extracted blocks", count=len(blocks))

        # Group into articles
        articles = self._group_into_articles(blocks, codice)
        logger.info("Grouped into articles", count=len(articles))

        # Add overlap
        self._add_overlap(articles)

        # Compute hashes
        for art in articles:
            art.compute_hash()

        return articles

    def _extract_blocks(self, data: dict) -> list[MarkerBlock]:
        """Extract blocks from marker JSON structure."""
        blocks = []

        # Marker JSON can have different structures
        # Try common patterns

        # Pattern 1: Direct "children" array
        if "children" in data:
            for i, child in enumerate(data["children"]):
                blocks.extend(self._flatten_block(child, i))

        # Pattern 2: "pages" array with blocks
        elif "pages" in data:
            position = 0
            for page in data["pages"]:
                for block in page.get("blocks", page.get("children", [])):
                    blocks.extend(self._flatten_block(block, position))
                    position += 1

        # Pattern 3: Direct "blocks" array
        elif "blocks" in data:
            for i, block in enumerate(data["blocks"]):
                blocks.extend(self._flatten_block(block, i))

        # Pattern 4: Root is the document with children
        else:
            # Try to iterate keys looking for block-like structures
            for key, value in data.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict) and ("block_type" in item or "text" in item):
                            blocks.extend(self._flatten_block(item, len(blocks)))

        return blocks

    def _flatten_block(self, block: dict, position: int) -> list[MarkerBlock]:
        """Flatten nested block structure."""
        blocks = []

        # Create block for this item if it has content
        if block.get("text") or block.get("html"):
            blocks.append(MarkerBlock.from_dict(block, position))

        # Recurse into children (handle null/None children)
        children = block.get("children") or []
        for child in children:
            if child:  # Skip null children
                blocks.extend(self._flatten_block(child, position + len(blocks)))

        return blocks

    def _group_into_articles(
        self,
        blocks: list[MarkerBlock],
        codice: str,
    ) -> list[ExtractedArticle]:
        """Group blocks into articles based on headers."""
        articles = []
        current_article: dict | None = None
        current_blocks: list[MarkerBlock] = []

        # Hierarchy tracking
        current_libro: str | None = None
        current_titolo: str | None = None
        current_capo: str | None = None
        current_sezione: str | None = None

        for block in blocks:
            # Check for hierarchy updates
            libro_match = LIBRO_PATTERN.search(block.text)
            if libro_match:
                current_libro = f"Libro {libro_match.group(1)}"
                current_titolo = None
                current_capo = None
                current_sezione = None

            titolo_match = TITOLO_PATTERN.search(block.text)
            if titolo_match:
                current_titolo = f"Titolo {titolo_match.group(1)}"
                current_capo = None
                current_sezione = None

            capo_match = CAPO_PATTERN.search(block.text)
            if capo_match:
                current_capo = f"Capo {capo_match.group(1)}"
                current_sezione = None

            sezione_match = SEZIONE_PATTERN.search(block.text)
            if sezione_match:
                current_sezione = f"Sezione {sezione_match.group(1)}"

            # Check if this block is an article header
            article_num = self._extract_article_num(block)

            if article_num:
                # Save previous article
                if current_article and current_blocks:
                    article = self._build_article(
                        current_article,
                        current_blocks,
                        codice,
                    )
                    if article:
                        articles.append(article)

                # Start new article
                num_norm, suffix = self._parse_article_num(article_num)
                current_article = {
                    "articolo_num": article_num,
                    "articolo_num_norm": num_norm,
                    "articolo_suffix": suffix,
                    "libro": current_libro,
                    "titolo": current_titolo,
                    "capo": current_capo,
                    "sezione": current_sezione,
                    "page_start": block.page,
                }
                current_blocks = [block]

            elif current_article is not None:
                # Add block to current article
                current_blocks.append(block)

        # Don't forget last article
        if current_article and current_blocks:
            article = self._build_article(
                current_article,
                current_blocks,
                codice,
            )
            if article:
                articles.append(article)

        return articles

    def _extract_article_num(self, block: MarkerBlock) -> str | None:
        """Extract article number from block if it's an article header."""
        # Check SectionHeader blocks first (most reliable)
        if block.block_type in ("SectionHeader", "Header", "Title"):
            text_to_check = block.html if block.html else block.text
            for pattern in ARTICOLO_HEADER_PATTERNS:
                match = pattern.search(text_to_check)
                if match:
                    return match.group(1)

        # Also check Text blocks that might be standalone article headers
        # (e.g., <p block-type="Text">Art. 3-bis.</p>)
        # Note: Altalex PDFs often have "Art. N. Rubrica (note) In vigore dal..."
        # all in one Text block, which can be 200-300+ chars
        # ListItem added: CCI has article headers inside <li> tags (e.g., Art. 39)
        if block.block_type in ("TextBlock", "Text", "ListItem"):
            text_to_check = block.html if block.html else block.text
            # Allow longer text blocks - Altalex often combines header+rubrica+date
            if len(block.text) < 400:
                for pattern in ARTICOLO_HEADER_PATTERNS:
                    match = pattern.search(text_to_check)
                    if match:
                        return match.group(1)

        return None

    def _parse_article_num(self, articolo_num: str) -> tuple[int, str | None]:
        """Parse article number into normalized form."""
        match = ARTICOLO_NUM_PATTERN.match(articolo_num)
        if match:
            num = int(match.group(1))
            suffix = match.group(2)
            if suffix:
                suffix = suffix.lower()
            return num, suffix

        # Fallback: try to extract just the number
        num_match = re.match(r'(\d+)', articolo_num)
        if num_match:
            return int(num_match.group(1)), None

        return 0, None

    def _build_article(
        self,
        article_data: dict,
        blocks: list[MarkerBlock],
        codice: str,
    ) -> ExtractedArticle | None:
        """Build article from accumulated blocks."""
        # Extract rubrica from first meaningful text block
        rubrica = None
        testo_blocks = []

        for i, block in enumerate(blocks):
            # Skip the header block itself
            if i == 0 and self._extract_article_num(block):
                continue

            text = block.text.strip()
            if not text:
                continue

            # First non-empty text might be rubrica
            if rubrica is None and len(text) <= self.max_rubrica_chars:
                # Check if it looks like a rubrica
                if RUBRICA_PATTERN.match(text):
                    rubrica = text.rstrip('.')
                    continue

            testo_blocks.append(text)

        # Build testo
        testo = "\n\n".join(testo_blocks)
        testo = self._clean_text(testo)

        # Validate
        warnings = []
        if len(testo) < self.min_article_chars:
            warnings.append(f"Testo too short: {len(testo)} chars")

        if not rubrica:
            warnings.append("No rubrica found")

        # Get page range
        page_end = blocks[-1].page if blocks else article_data.get("page_start", 0)

        return ExtractedArticle(
            articolo_num=article_data["articolo_num"],
            articolo_num_norm=article_data["articolo_num_norm"],
            articolo_suffix=article_data["articolo_suffix"],
            rubrica=rubrica,
            testo=testo,
            libro=article_data.get("libro"),
            titolo=article_data.get("titolo"),
            capo=article_data.get("capo"),
            sezione=article_data.get("sezione"),
            page_start=article_data.get("page_start", 0),
            page_end=page_end,
            source_blocks=[b.block_id for b in blocks if b.block_id],
            warnings=warnings,
        )

    def _clean_text(self, text: str) -> str:
        """Clean article text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        # Fix common encoding issues
        replacements = {
            '\ufffd': '"',
            '\u2019': "'",
            '\u2018': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2014': '-',
            '\xa0': ' ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

    def _add_overlap(self, articles: list[ExtractedArticle]) -> None:
        """Add inter-article overlap for context."""
        for i, article in enumerate(articles):
            context_parts = []

            # Previous article overlap
            if i > 0:
                prev = articles[i - 1]
                if prev.testo:
                    prev_text = prev.testo[-self.overlap_chars:]
                    context_parts.append(f"[...] {prev_text}")

            # Current article
            context_parts.append(article.testo)

            # Next article overlap
            if i < len(articles) - 1:
                next_art = articles[i + 1]
                if next_art.testo:
                    next_text = next_art.testo[:self.overlap_chars]
                    context_parts.append(f"{next_text} [...]")

            article.testo_context = "\n\n".join(context_parts)


def validate_article(article: ExtractedArticle) -> tuple[bool, list[str]]:
    """
    Validate extracted article.

    Returns:
        (is_valid, list of errors)
    """
    errors = []

    # 1. articolo_num must match pattern
    if not ARTICOLO_NUM_PATTERN.match(article.articolo_num):
        errors.append(f"Invalid articolo_num format: {article.articolo_num}")

    # 2. articolo_num_norm must be set
    if article.articolo_num_norm == 0:
        errors.append("articolo_num_norm is 0")

    # 3. testo must have content
    if len(article.testo) < 10:
        errors.append(f"Testo too short: {len(article.testo)} chars")

    # 4. rubrica max length
    if article.rubrica and len(article.rubrica) > 500:
        errors.append(f"Rubrica too long: {len(article.rubrica)} chars")

    # 5. Anti-hallucination: articolo num should appear in context
    # This is a WARNING, not an error
    art_pattern = re.compile(rf'art(?:icolo)?\.?\s*{article.articolo_num}', re.IGNORECASE)
    full_text = (article.rubrica or '') + ' ' + article.testo
    if not art_pattern.search(full_text):
        article.warnings.append(
            f"articolo_num '{article.articolo_num}' not found in text (check header extraction)"
        )

    return len(errors) == 0, errors


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python marker_chunker.py <json_file> [codice]")
        sys.exit(1)

    json_file = sys.argv[1]
    codice = sys.argv[2] if len(sys.argv) > 2 else "TEST"

    chunker = MarkerChunker()
    articles = chunker.process_file(json_file, codice)

    print(f"\n{'='*60}")
    print(f"Extracted {len(articles)} articles from {json_file}")
    print(f"{'='*60}\n")

    # Stats
    with_rubrica = sum(1 for a in articles if a.rubrica)
    with_warnings = sum(1 for a in articles if a.warnings)

    print(f"With rubrica: {with_rubrica}/{len(articles)}")
    print(f"With warnings: {with_warnings}/{len(articles)}")

    # Show first 5 and last 5
    print(f"\nFirst 5 articles:")
    for art in articles[:5]:
        warn = " [!]" if art.warnings else ""
        print(f"  Art. {art.articolo_num}{warn}: {art.rubrica or '(no rubrica)'}")
        print(f"    Testo: {art.testo[:80]}...")

    if len(articles) > 10:
        print(f"\nLast 5 articles:")
        for art in articles[-5:]:
            warn = " [!]" if art.warnings else ""
            print(f"  Art. {art.articolo_num}{warn}: {art.rubrica or '(no rubrica)'}")

    # Validation
    print(f"\nValidation:")
    invalid = 0
    for art in articles:
        valid, errors = validate_article(art)
        if not valid:
            invalid += 1
            print(f"  Art. {art.articolo_num}: INVALID - {errors}")

    print(f"\nValid: {len(articles) - invalid}/{len(articles)}")
