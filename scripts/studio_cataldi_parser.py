#!/usr/bin/env python
"""
Studio Cataldi HTML Parser

Estrae articoli dai file HTML locali di Studio Cataldi.
Supporta sia file singoli (index.html) che file multipli (artt-N-M.html).

Usage:
    from studio_cataldi_parser import StudioCataldiParser

    parser = StudioCataldiParser()
    articles = parser.get_articles("TUB")
    article = parser.get_article("TUB", 106)
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

STUDIO_CATALDI_ROOT = Path("C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa")

# Mapping codice -> cartella
CODE_TO_FOLDER = {
    'TUB': 'testo-unico-bancario',
    'TUF': 'testo-unico-intermediazione-finanziaria',
    'TUIR': 'testo-unico-imposte-sui-redditi',
    'TUE': 'testo-unico-edilizia',
    'TUEL': 'testo-unico-enti-locali',
    'TUI': 'testo-unico-immigrazione',
    'TUSL': 'testo-unico-sicurezza-sul-lavoro',
    'TUSG': 'testo-unico-spese-giustizia',
    'DIVA': 'testo-unico-iva',
    'CAPP': 'codice-degli-appalti',
    'CBCP': 'codice-dei-beni-culturali',
    'CCONS': 'codice-del-consumo',
    'CAMB': 'codice-dell-ambiente',
    'CPRIV': 'codice-della-privacy',
    'CPI': 'codice-della-proprieta-industriale',
    'CDS': 'codicedellastrada',
    'CAP': 'codice-delle-assicurazioni-private',
    'CNAV': 'codice-della-navigazione',
    'SL': 'statuto-dei-lavoratori',
    'OP': 'ordinamento-penitenziario',
    'LF': 'legge-fallimentare',
    'LDIP': 'diritto-internazionale-privato',
    'LDIV': 'legge-divorzio',
    'LLOC': 'legge-locazioni-abitative',
}

# Suffissi latini
LATIN_SUFFIXES = [
    "bis", "ter", "quater", "quinquies", "sexies", "septies", "octies",
    "novies", "nonies", "decies", "undecies", "duodecies", "terdecies",
    "quaterdecies", "quinquiesdecies"
]
SUFFIX_PATTERN = "|".join(LATIN_SUFFIXES)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ParsedArticle:
    """Articolo estratto da Studio Cataldi."""
    code: str                    # es. TUB
    article_num: str             # es. "106" o "106-bis"
    base_num: int                # es. 106
    suffix: Optional[str]        # es. "bis" o None
    rubrica: Optional[str]       # Titolo/rubrica
    testo: str                   # Testo completo
    source_file: str             # File sorgente

    @property
    def article_key(self) -> str:
        if self.suffix:
            return f"{self.code}:{self.base_num}-{self.suffix}"
        return f"{self.code}:{self.base_num}"


# ==============================================================================
# PARSER
# ==============================================================================

class StudioCataldiParser:
    """Parser per file HTML Studio Cataldi."""

    def __init__(self, root: Path = STUDIO_CATALDI_ROOT):
        self.root = root
        self._cache: dict[str, list[ParsedArticle]] = {}

    def get_folder(self, code: str) -> Optional[Path]:
        """Restituisce path della cartella per un codice."""
        folder_name = CODE_TO_FOLDER.get(code.upper())
        if not folder_name:
            return None

        folder = self.root / folder_name
        if folder.exists():
            return folder
        return None

    def get_html_files(self, code: str) -> list[Path]:
        """Restituisce tutti i file HTML per un codice."""
        folder = self.get_folder(code)
        if not folder:
            return []

        # Cerca index.html o artt-*.html
        files = []

        index = folder / "index.html"
        if index.exists():
            files.append(index)

        # File articoli separati
        for f in folder.glob("artt*.html"):
            files.append(f)

        # Altri HTML nella cartella
        for f in folder.glob("*.html"):
            if f not in files:
                files.append(f)

        return sorted(files)

    def parse_html(self, html_path: Path, code: str) -> list[ParsedArticle]:
        """Parse un file HTML ed estrae gli articoli."""
        try:
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {html_path}: {e}")
            return []

        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        # Pattern per trovare articoli
        # Cerca <p> o altri elementi con "Art. N" o "Art. N-bis"
        art_pattern = re.compile(
            rf'Art\.?\s*(\d+)(?:[-\s]?({SUFFIX_PATTERN}))?',
            re.IGNORECASE
        )

        # Trova tutti i tag con potenziali articoli
        text_content = soup.get_text(separator='\n')

        # Trova tutte le occorrenze di articoli
        matches = list(art_pattern.finditer(text_content))

        for i, match in enumerate(matches):
            base_num = int(match.group(1))
            suffix = match.group(2).lower() if match.group(2) else None

            # Determina il testo dell'articolo (fino al prossimo "Art.")
            start = match.end()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text_content)

            raw_text = text_content[start:end].strip()

            # Pulisci il testo
            testo, rubrica = self._extract_rubrica_and_text(raw_text)

            if not testo or len(testo) < 10:
                continue

            article_num = str(base_num)
            if suffix:
                article_num = f"{base_num}-{suffix}"

            articles.append(ParsedArticle(
                code=code,
                article_num=article_num,
                base_num=base_num,
                suffix=suffix,
                rubrica=rubrica,
                testo=testo,
                source_file=str(html_path),
            ))

        return articles

    def _extract_rubrica_and_text(self, raw_text: str) -> tuple[str, Optional[str]]:
        """Estrae rubrica e testo dall'estratto grezzo."""
        lines = raw_text.strip().split('\n')
        if not lines:
            return "", None

        rubrica = None
        testo_start = 0

        # Prima riga potrebbe essere rubrica se breve e centrata
        first_line = lines[0].strip()
        if first_line and len(first_line) < 100:
            # Check se sembra una rubrica (breve, non inizia con numero)
            if not re.match(r'^\d+\.', first_line):
                rubrica = first_line
                testo_start = 1

        # Testo
        testo_lines = []
        for line in lines[testo_start:]:
            line = line.strip()
            if line:
                # Pulisci tag HTML residui
                line = re.sub(r'<[^>]+>', '', line)
                testo_lines.append(line)

        testo = '\n'.join(testo_lines)

        # Rimuovi spazi multipli
        testo = re.sub(r'\n{3,}', '\n\n', testo)
        testo = re.sub(r' {2,}', ' ', testo)

        return testo.strip(), rubrica

    def get_articles(self, code: str, use_cache: bool = True) -> list[ParsedArticle]:
        """
        Restituisce tutti gli articoli per un codice.

        Args:
            code: Codice documento (es. TUB)
            use_cache: Se usare cache in memoria

        Returns:
            Lista di ParsedArticle
        """
        code = code.upper()

        if use_cache and code in self._cache:
            return self._cache[code]

        html_files = self.get_html_files(code)
        if not html_files:
            logger.warning(f"No HTML files found for {code}")
            return []

        all_articles = []
        seen_keys = set()

        for html_path in html_files:
            articles = self.parse_html(html_path, code)
            for a in articles:
                if a.article_key not in seen_keys:
                    all_articles.append(a)
                    seen_keys.add(a.article_key)

        # Sort by base_num, then suffix
        all_articles.sort(key=lambda x: (x.base_num, x.suffix or ""))

        if use_cache:
            self._cache[code] = all_articles

        return all_articles

    def get_article(self, code: str, base_num: int, suffix: Optional[str] = None) -> Optional[ParsedArticle]:
        """
        Restituisce un singolo articolo.

        Args:
            code: Codice documento
            base_num: Numero articolo base
            suffix: Suffisso opzionale (bis, ter, etc.)

        Returns:
            ParsedArticle o None
        """
        articles = self.get_articles(code)

        for a in articles:
            if a.base_num == base_num:
                if suffix is None and a.suffix is None:
                    return a
                if suffix and a.suffix and suffix.lower() == a.suffix.lower():
                    return a

        return None

    def get_available_codes(self) -> list[str]:
        """Restituisce i codici disponibili."""
        available = []
        for code in CODE_TO_FOLDER:
            if self.get_folder(code):
                available.append(code)
        return sorted(available)


# ==============================================================================
# MAIN (test)
# ==============================================================================

if __name__ == "__main__":
    parser = StudioCataldiParser()

    print("=== STUDIO CATALDI PARSER ===")
    print(f"Root: {parser.root}")

    # Codici disponibili
    available = parser.get_available_codes()
    print(f"\nCodici disponibili: {len(available)}")
    for code in available[:10]:
        folder = parser.get_folder(code)
        files = parser.get_html_files(code)
        print(f"  {code}: {len(files)} files in {folder.name if folder else 'N/A'}")

    # Test parsing TUB
    print("\n=== TEST TUB ===")
    articles = parser.get_articles("TUB")
    print(f"Articoli estratti: {len(articles)}")

    if articles:
        print(f"\nPrimi 5:")
        for a in articles[:5]:
            rubrica = (a.rubrica[:40] + "...") if a.rubrica and len(a.rubrica) > 40 else a.rubrica
            print(f"  Art. {a.article_num}: {rubrica or '(no rubrica)'}")
            print(f"    Testo: {a.testo[:60]}...")

        # Test get_article
        print("\n=== TEST GET_ARTICLE ===")
        art106 = parser.get_article("TUB", 106)
        if art106:
            print(f"Art. 106: {art106.rubrica}")
            print(f"  Testo ({len(art106.testo)} chars): {art106.testo[:100]}...")
