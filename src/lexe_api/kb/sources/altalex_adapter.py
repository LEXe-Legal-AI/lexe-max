"""Altalex Markdown Adapter.

Parses Codice Civile/Penale from Altalex PDF->MD exports.
Primary source for KB normativa with clean text and proper rubriche.

Format examples:
    **Art. 1. Indicazione delle fonti.**
    Sono fonti del diritto:
    1. le leggi;

    Art. 2043. Risarcimento per fatto illecito.
    Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto,
    obbliga colui che ha commesso il fatto a risarcire il danno.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AltalexArticle:
    """Parsed article from Altalex MD file."""

    codice: str  # 'CC', 'CP'
    articolo: str  # '2043', '1', '2043-bis'
    rubrica: str | None  # 'Risarcimento per fatto illecito'
    testo: str  # Full article text
    content_hash: str  # SHA256 of normalized text

    # Hierarchy
    libro: str | None = None
    titolo: str | None = None
    capo: str | None = None
    sezione: str | None = None

    # Preleggi flag (Art. 1-31 Disposizioni sulla legge in generale)
    is_preleggi: bool = False

    # Disposizioni attuazione/transitorie flag (Art. 1-N at end of file)
    is_attuazione: bool = False

    # Source tracking
    source_file: str = ""
    source_edition: str = "Altalex 2025"
    line_start: int = 0
    line_end: int = 0

    @property
    def canonical_id(self) -> str:
        """Canonical ID like CC:2043, CC:PREL:1, or CC:ATT:1."""
        if self.is_preleggi:
            return f"{self.codice}:PREL:{self.articolo}"
        if self.is_attuazione:
            return f"{self.codice}:ATT:{self.articolo}"
        return f"{self.codice}:{self.articolo}"


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Handles differences between sources:
    - Brocardi adds spaces before punctuation: "colposo , che" vs "colposo, che"
    - Brocardi adds article references: "[ 2058 ]", "[1]"
    - Different quote styles
    - Different dash styles
    """
    # Lowercase
    text = text.lower()

    # Remove Brocardi-style article references like [ 2058 ], [1], [ 2044, 2045 ]
    text = re.sub(r"\[\s*[\d,\s]+\s*\]", "", text)

    # Remove footnote markers like (1), (^1), (¹)
    text = re.sub(r"\(\^?\d+\s*\)", "", text)
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", text)

    # Normalize whitespace around punctuation (Brocardi adds spaces)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)  # Remove space before punct
    text = re.sub(r"([,.:;!?])\s+", r"\1 ", text)  # Single space after punct

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("\ufffd", '"')  # Fix corrupted quotes

    # Normalize dashes
    text = text.replace("–", "-").replace("—", "-")

    # Strip
    return text.strip()


def compute_hash(text: str) -> str:
    """SHA256 of normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def fix_encoding(text: str) -> str:
    """Fix common encoding issues from PDF->MD conversion."""
    replacements = {
        "\ufffd": '"',  # Same as above
        "\u2019": "'",  # Right single quote
        "\u2018": "'",  # Left single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\xa0": " ",  # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class AltalexAdapter:
    """Adapter for parsing Altalex MD exports.

    Usage:
        adapter = AltalexAdapter()

        # Parse single file
        articles = list(adapter.parse_file(
            "codice-civile-2025.md",
            codice="CC"
        ))

        # Get specific article
        art_2043 = adapter.get_article(articles, "2043")
    """

    # Regex patterns for article detection
    # Pattern 1: **Art. 2043. Rubrica.**
    PATTERN_BOLD = re.compile(r"^\*\*Art\.\s*(\d+(?:-\w+)?)\.\s*(.+?)\*\*", re.IGNORECASE)

    # Pattern 2: Art. 2043. Rubrica.
    PATTERN_PLAIN = re.compile(r"^Art\.\s*(\d+(?:-\w+)?)\.\s*(.+?)$", re.IGNORECASE)

    # Pattern 3: Art. 2043. (no rubrica, abrogated)
    PATTERN_NO_RUBRICA = re.compile(
        r"^(?:\*\*)?Art\.\s*(\d+(?:-\w+)?)\.\s*(?:\*\*)?$", re.IGNORECASE
    )

    # Section delimiters that should end an article
    SECTION_DELIMITERS = [
        re.compile(r"^#{1,6}\s*Libro\s+[IVX]+", re.IGNORECASE),  # ### Libro V
        re.compile(r"^Libro\s+[IVX]+\s*[-–]", re.IGNORECASE),  # Libro V - Del lavoro
        re.compile(r"^#{1,6}\s*TITOLO\s+[IVX]+", re.IGNORECASE),  # ### TITOLO IX
        re.compile(r"^TITOLO\s+[IVX]+\s*[-–]", re.IGNORECASE),  # TITOLO IX – DEI FATTI
        re.compile(r"^#{1,6}\s*CAPO\s+[IVX]+", re.IGNORECASE),  # ### CAPO I
        re.compile(r"^CAPO\s+[IVX]+\s*[-–]", re.IGNORECASE),  # CAPO I - DELLE FONTI
        re.compile(r"^Sommario\s*$", re.IGNORECASE),  # Sommario
        re.compile(r"^#{1,6}\s*Sommario", re.IGNORECASE),  # ### Sommario
        re.compile(r"^DISPOSIZIONI\s+", re.IGNORECASE),  # DISPOSIZIONI SULLA LEGGE
        re.compile(r"^#{1,6}\s*CODICE\s+CIVILE", re.IGNORECASE),  # ### CODICE CIVILE
        re.compile(r"^CODICE\s+CIVILE\s*$", re.IGNORECASE),  # CODICE CIVILE
        re.compile(r"^Altalex\s+eBook", re.IGNORECASE),  # Footer
    ]

    # Hierarchy patterns
    LIBRO_PATTERN = re.compile(r"^(?:#{1,6}\s*)?Libro\s+([IVX]+)\s*[-–]\s*(.+?)$", re.IGNORECASE)
    TITOLO_PATTERN = re.compile(
        r"^(?:#{1,6}\s*)?TITOLO\s+([IVX]+(?:-\w+)?)\s*[-–]\s*(.+?)$", re.IGNORECASE
    )
    CAPO_PATTERN = re.compile(r"^(?:#{1,6}\s*)?CAPO\s+([IVX]+)\s*[-–]\s*(.+?)$", re.IGNORECASE)
    SEZIONE_PATTERN = re.compile(
        r"^(?:#{1,6}\s*)?SEZIONE\s+([IVX]+)\s*[-–]\s*(.+?)$", re.IGNORECASE
    )

    # Preleggi detection (before Libro I)
    PRELEGGI_END_ARTICLE = 31  # Last article of Preleggi

    # Preleggi section detection (at start of file)
    PRELEGGI_PATTERN = re.compile(
        r"^(?:#{1,6}\s*)?DISPOSIZIONI\s+SULLA\s+LEGGE\s+IN\s+GENERALE\s*$", re.IGNORECASE
    )

    # Attuazione/Transitorie detection (after Libro VI)
    ATTUAZIONE_PATTERN = re.compile(
        r"^(?:#{1,6}\s*)?DISPOSIZIONI\s+PER\s+L.ATTUAZIONE", re.IGNORECASE
    )

    def __init__(self):
        self._cache: dict[str, list[AltalexArticle]] = {}

    def parse_file(
        self,
        filepath: str | Path,
        codice: str = "CC",
    ) -> Iterator[AltalexArticle]:
        """Parse MD file and yield articles.

        Args:
            filepath: Path to MD file
            codice: Code identifier (CC, CP, etc.)

        Yields:
            AltalexArticle for each article found
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info("Parsing Altalex MD", file=str(filepath), codice=codice)

        with open(filepath, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Current hierarchy context
        current_libro: str | None = None
        current_titolo: str | None = None
        current_capo: str | None = None
        current_sezione: str | None = None
        in_preleggi = False  # Becomes true when entering "DISPOSIZIONI SULLA LEGGE IN GENERALE"
        in_attuazione = False  # Becomes true after "DISPOSIZIONI PER L'ATTUAZIONE"
        found_first_libro = False  # Track if we've seen the real Libro I (not in sommario)

        current_article: dict | None = None
        current_text_lines: list[str] = []

        for i, line in enumerate(lines, 1):
            line = fix_encoding(line)
            line_stripped = line.strip()

            # Skip empty lines at start of article
            if not line_stripped and current_article is None:
                continue

            # Check for PRELEGGI section (at start of file, before Libro I)
            if (
                self.PRELEGGI_PATTERN.match(line_stripped)
                and not found_first_libro
                and not in_attuazione
            ):
                in_preleggi = True
                logger.debug("Entered PRELEGGI section", line=i)

            # Check for ATTUAZIONE section (at end of file)
            if self.ATTUAZIONE_PATTERN.match(line_stripped):
                in_attuazione = True
                in_preleggi = False
                current_libro = None  # Reset hierarchy for attuazione
                current_titolo = None
                current_capo = None
                current_sezione = None
                logger.debug("Entered ATTUAZIONE section", line=i)

            # Update hierarchy context
            libro_match = self.LIBRO_PATTERN.match(line_stripped)
            if libro_match and found_first_libro:
                # Only set libro if we've found at least one article
                # This skips the sommario Libro entries which appear before any articles
                current_libro = f"Libro {libro_match.group(1)} - {libro_match.group(2)}"
                current_titolo = None
                current_capo = None
                current_sezione = None
                in_preleggi = False  # After Libro, no longer preleggi
                logger.debug("Found Libro", libro=current_libro, line=i)

            titolo_match = self.TITOLO_PATTERN.match(line_stripped)
            if titolo_match:
                current_titolo = f"Titolo {titolo_match.group(1)} - {titolo_match.group(2)}"
                current_capo = None
                current_sezione = None

            capo_match = self.CAPO_PATTERN.match(line_stripped)
            if capo_match:
                current_capo = f"Capo {capo_match.group(1)} - {capo_match.group(2)}"
                current_sezione = None

            sezione_match = self.SEZIONE_PATTERN.match(line_stripped)
            if sezione_match:
                current_sezione = f"Sezione {sezione_match.group(1)} - {sezione_match.group(2)}"

            # Check if this line is a section delimiter (ends current article)
            is_section_delimiter = any(
                delim.match(line_stripped) for delim in self.SECTION_DELIMITERS
            )

            # Try to match article header
            match = None
            rubrica = None

            # Try bold pattern first
            match = self.PATTERN_BOLD.match(line_stripped)
            if match:
                art_num = match.group(1)
                rubrica = self._clean_rubrica(match.group(2))
            else:
                # Try plain pattern
                match = self.PATTERN_PLAIN.match(line_stripped)
                if match:
                    art_num = match.group(1)
                    rubrica = self._clean_rubrica(match.group(2))
                else:
                    # Try no-rubrica pattern
                    match = self.PATTERN_NO_RUBRICA.match(line_stripped)
                    if match:
                        art_num = match.group(1)
                        rubrica = None

            if match or is_section_delimiter:
                # Yield previous article if exists
                if current_article is not None:
                    text = self._clean_text("\n".join(current_text_lines))
                    if text:  # Only yield if has content
                        yield AltalexArticle(
                            codice=codice,
                            articolo=current_article["articolo"],
                            rubrica=current_article["rubrica"],
                            testo=text,
                            content_hash=compute_hash(text),
                            libro=current_article.get("libro"),
                            titolo=current_article.get("titolo"),
                            capo=current_article.get("capo"),
                            sezione=current_article.get("sezione"),
                            is_preleggi=current_article.get("is_preleggi", False),
                            is_attuazione=current_article.get("is_attuazione", False),
                            source_file=str(filepath),
                            line_start=current_article["line_start"],
                            line_end=i - 1,
                        )

                if match:
                    # Determine if this is a preleggi article
                    try:
                        art_num_int = int(re.match(r"\d+", art_num).group())
                        is_preleggi_art = in_preleggi and art_num_int <= self.PRELEGGI_END_ARTICLE
                    except (ValueError, AttributeError):
                        is_preleggi_art = False

                    # Mark that we've found articles (enables Libro detection)
                    if not found_first_libro:
                        found_first_libro = True

                    # Start new article
                    current_article = {
                        "articolo": art_num,
                        "rubrica": rubrica,
                        "line_start": i,
                        "libro": current_libro,
                        "titolo": current_titolo,
                        "capo": current_capo,
                        "sezione": current_sezione,
                        "is_preleggi": is_preleggi_art,
                        "is_attuazione": in_attuazione,
                    }
                    current_text_lines = []
                else:
                    # Section delimiter - reset state
                    current_article = None
                    current_text_lines = []

            elif current_article is not None:
                # Accumulate text for current article
                # Skip code blocks markers
                if line_stripped != "```":
                    current_text_lines.append(line)

        # Yield last article
        if current_article is not None:
            text = self._clean_text("\n".join(current_text_lines))
            if text:
                yield AltalexArticle(
                    codice=codice,
                    articolo=current_article["articolo"],
                    rubrica=current_article["rubrica"],
                    testo=text,
                    content_hash=compute_hash(text),
                    libro=current_article.get("libro"),
                    titolo=current_article.get("titolo"),
                    capo=current_article.get("capo"),
                    sezione=current_article.get("sezione"),
                    is_preleggi=current_article.get("is_preleggi", False),
                    is_attuazione=current_article.get("is_attuazione", False),
                    source_file=str(filepath),
                    line_start=current_article["line_start"],
                    line_end=len(lines),
                )

    def _clean_rubrica(self, rubrica: str) -> str | None:
        """Clean article rubrica (title)."""
        if not rubrica:
            return None

        # Strip trailing period
        rubrica = rubrica.rstrip(".")

        # Remove footnote markers like (^1), (^1 ), (1), (
        rubrica = re.sub(r"\s*\(\^?\d*\s*\)?\s*", "", rubrica)

        # Fix encoding
        rubrica = fix_encoding(rubrica)

        # Strip and check if empty or invalid
        rubrica = rubrica.strip()
        if not rubrica or rubrica in ("...", "(", ")"):
            return None

        return rubrica

    def _clean_text(self, text: str) -> str:
        """Clean article text."""
        # Fix encoding first
        text = fix_encoding(text)

        # Remove markdown code blocks
        text = re.sub(r"```\n?", "", text)

        # Remove page headers/footers
        text = re.sub(r"Altalex eBook \| Collana Codici Altalex \d+", "", text)
        text = re.sub(r"CODICE CIVILE\n", "", text)
        text = re.sub(r"CODICE PENALE\n", "", text)
        text = re.sub(r"Disposizioni sulla legge in generale\n", "", text)

        # Remove footnote content (lines starting with (1), (2), etc.)
        # But keep inline footnote markers for context
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Skip standalone footnote lines
            if re.match(r'^\s*\(\d+\)\s*[""\']', line):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def parse_and_cache(
        self,
        filepath: str | Path,
        codice: str = "CC",
    ) -> list[AltalexArticle]:
        """Parse file and cache results."""
        key = f"{codice}:{filepath}"
        if key not in self._cache:
            self._cache[key] = list(self.parse_file(filepath, codice))
            logger.info(
                "Cached Altalex articles",
                codice=codice,
                count=len(self._cache[key]),
            )
        return self._cache[key]

    def get_article(
        self,
        articles: list[AltalexArticle],
        articolo: str,
        is_preleggi: bool = False,
    ) -> AltalexArticle | None:
        """Get specific article from parsed list."""
        for art in articles:
            if art.articolo == articolo and art.is_preleggi == is_preleggi:
                return art
        # Fallback: try without preleggi filter
        for art in articles:
            if art.articolo == articolo:
                return art
        return None

    def compare_with_brocardi(
        self,
        altalex_article: AltalexArticle,
        brocardi_text: str,
    ) -> dict:
        """Compare Altalex article with Brocardi text.

        Returns:
            Dict with comparison results
        """
        altalex_hash = altalex_article.content_hash
        brocardi_hash = compute_hash(brocardi_text)

        hash_match = altalex_hash == brocardi_hash

        # Calculate Jaccard similarity
        altalex_words = set(normalize_text(altalex_article.testo).split())
        brocardi_words = set(normalize_text(brocardi_text).split())

        intersection = len(altalex_words & brocardi_words)
        union = len(altalex_words | brocardi_words)
        similarity = intersection / union if union > 0 else 0

        # Classify difference
        if hash_match:
            diff_type = "exact"
            status = "verified"
        elif similarity > 0.95:
            diff_type = "format_diff"
            status = "verified"
        elif similarity > 0.80:
            diff_type = "minor"
            status = "format_diff"
        else:
            diff_type = "substantive"
            status = "content_diff"

        return {
            "hash_match": hash_match,
            "altalex_hash": altalex_hash,
            "brocardi_hash": brocardi_hash,
            "similarity": similarity,
            "diff_type": diff_type,
            "status": status,
            "altalex_len": len(altalex_article.testo),
            "brocardi_len": len(brocardi_text),
        }


# Quick test function
async def test_parse():
    """Test parsing Altalex file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python altalex_adapter.py <md_file> [article_num]")
        return

    filepath = sys.argv[1]
    article_num = sys.argv[2] if len(sys.argv) > 2 else None

    adapter = AltalexAdapter()
    articles = list(adapter.parse_file(filepath, codice="CC"))

    print(f"Parsed {len(articles)} articles from {filepath}")

    # Count by section type
    preleggi = [a for a in articles if a.is_preleggi]
    attuazione = [a for a in articles if a.is_attuazione]
    regular = [a for a in articles if not a.is_preleggi and not a.is_attuazione]
    print(f"  - Preleggi (Art. 1-31): {len(preleggi)}")
    print(f"  - Regular CC articles: {len(regular)}")
    print(f"  - Attuazione/Transitorie: {len(attuazione)}")

    if article_num:
        art = adapter.get_article(articles, article_num)
        if art:
            print(f"\n=== Art. {art.articolo} ===")
            print(f"Rubrica: {art.rubrica}")
            print(f"Preleggi: {art.is_preleggi}")
            print(f"Libro: {art.libro}")
            print(f"Titolo: {art.titolo}")
            print(f"Hash: {art.content_hash[:16]}...")
            print(f"Lines: {art.line_start}-{art.line_end}")
            print(f"\nTesto:\n{art.testo[:500]}...")
        else:
            print(f"Article {article_num} not found")
    else:
        # Show first 10 and last 10
        print("\nFirst 10 articles:")
        for art in articles[:10]:
            tag = "[PREL]" if art.is_preleggi else "[ATT]" if art.is_attuazione else ""
            print(f"  Art. {art.articolo} {tag}: {art.rubrica or '(no rubrica)'}")

        print("\nLast 10 articles (Attuazione):")
        for art in articles[-10:]:
            tag = "[PREL]" if art.is_preleggi else "[ATT]" if art.is_attuazione else ""
            print(f"  Art. {art.articolo} {tag}: {art.rubrica or '(no rubrica)'}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_parse())
