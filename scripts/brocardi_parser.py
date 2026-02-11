#!/usr/bin/env python
"""
Brocardi Offline HTML Parser

Estrae articoli dai file HTML locali di Brocardi.
Supporta CC, CP, CPC, CPP, CCI, COST, GDPR.

Struttura Brocardi (mirror HTTrack):
    codice-civile/libro-primo/titolo-i/art1.html
    codice-civile/libro-primo/titolo-ii/capo-i/art11.html

Elementi HTML chiave:
    - h1.hbox-header: "Articolo N Codice X"
    - h3.hbox-content: Rubrica
    - div.corpoDelTesto.dispositivo: Testo principale
    - div.corpoDelTesto.nota: Note a piè

Usage:
    from brocardi_parser import BrocardiParser

    parser = BrocardiParser()
    articles = parser.get_articles("CC")
    article = parser.get_article("CC", 2043)
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

# Base path for all HTTrack mirrors
BROCARDI_BASE = Path("C:/Mie pagine Web")

# Mapping codice -> (httrack_folder, brocardi_subfolder)
# Ogni codice ha il suo mirror HTTrack separato
CODE_TO_PATH = {
    'CC':   ('broc-civ',    'codice-civile'),
    'CP':   ('broc-pen',    'codice-penale'),
    'CPC':  ('broc-prociv', 'codice-di-procedura-civile'),
    'CPP':  ('broc-propen', 'codice-di-procedura-penale'),
    'COST': ('broc-cost',   'costituzione'),
}

# Legacy mapping per compatibilità (non usato per i 5 principali)
CODE_TO_FOLDER = {
    'CCI': 'codice-della-crisi-dimpresa',
    'GDPR': 'privacy',
}

# Suffissi latini (come in studio_cataldi_parser)
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
class BrocardiArticle:
    """Articolo estratto da Brocardi."""
    code: str                    # es. CC
    article_num: str             # es. "2043" o "2043-bis"
    base_num: int                # es. 2043
    suffix: Optional[str]        # es. "bis" o None
    rubrica: Optional[str]       # Titolo/rubrica
    testo: str                   # Testo dispositivo
    note: list[str]              # Note a piè
    source_file: str             # File sorgente

    @property
    def article_key(self) -> str:
        if self.suffix:
            return f"{self.code}:{self.base_num}-{self.suffix}"
        return f"{self.code}:{self.base_num}"


# ==============================================================================
# PARSER
# ==============================================================================

class BrocardiParser:
    """Parser per file HTML Brocardi offline."""

    def __init__(self, base: Path = BROCARDI_BASE):
        self.base = base
        self._cache: dict[str, list[BrocardiArticle]] = {}

    def get_folder(self, code: str) -> Optional[Path]:
        """Restituisce path della cartella per un codice."""
        code_upper = code.upper()

        # Prima prova CODE_TO_PATH (5 codici principali con mirror separati)
        if code_upper in CODE_TO_PATH:
            httrack_folder, subfolder = CODE_TO_PATH[code_upper]
            folder = self.base / httrack_folder / "www.brocardi.it" / subfolder
            if folder.exists():
                return folder
            return None

        # Fallback per altri codici (legacy)
        folder_name = CODE_TO_FOLDER.get(code_upper)
        if not folder_name:
            return None

        # Per legacy, prova nella prima cartella disponibile
        for httrack_folder, _ in CODE_TO_PATH.values():
            folder = self.base / httrack_folder / "www.brocardi.it" / folder_name
            if folder.exists():
                return folder
        return None

    def get_html_files(self, code: str) -> list[Path]:
        """Restituisce tutti i file HTML articolo per un codice."""
        folder = self.get_folder(code)
        if not folder:
            return []

        # Pattern: art*.html (es. art1.html, art2043.html, art2043bis.html)
        files = list(folder.glob("**/art*.html"))

        # Filtra solo file articolo REALI (esclude HTTrack hashes come art209e6.html)
        # Pattern stretto: artN, artN-suffix, artNsuffix dove suffix è latino valido
        article_pattern = rf"^art(\d+)(?:[-]?({SUFFIX_PATTERN}))?$"
        article_files = []
        for f in files:
            if re.match(article_pattern, f.stem, re.IGNORECASE):
                article_files.append(f)

        return sorted(article_files)

    def parse_article_file(self, html_path: Path, code: str) -> Optional[BrocardiArticle]:
        """Parse un singolo file HTML articolo."""
        try:
            # Brocardi usa Windows-1252 (da HTTrack)
            with open(html_path, 'r', encoding='cp1252', errors='replace') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {html_path}: {e}")
            return None

        soup = BeautifulSoup(content, 'html.parser')

        # 1. Estrai numero articolo dal filename
        base_num, suffix = self._parse_filename(html_path.stem)
        if base_num is None:
            logger.warning(f"Cannot parse article number from {html_path.name}")
            return None

        # 2. Estrai rubrica da h3.hbox-content
        rubrica = None
        h3 = soup.find('h3', class_='hbox-content')
        if h3:
            rubrica = h3.get_text(strip=True)
            # Fix encoding issues
            rubrica = self._fix_encoding(rubrica)

        # 3. Estrai testo dispositivo da div.corpoDelTesto.dispositivo
        testo = ""
        dispositivo = soup.find('div', class_=re.compile(r'corpoDelTesto.*dispositivo'))
        if dispositivo:
            testo = self._extract_text(dispositivo)

        if not testo and not rubrica:
            logger.debug(f"Empty article: {html_path.name}")
            return None

        # 4. Estrai note
        note = []
        note_divs = soup.find_all('div', class_=re.compile(r'corpoDelTesto.*nota'))
        for nota_div in note_divs:
            nota_text = self._extract_text(nota_div)
            if nota_text:
                note.append(nota_text)

        article_num = str(base_num)
        if suffix:
            article_num = f"{base_num}-{suffix}"

        return BrocardiArticle(
            code=code,
            article_num=article_num,
            base_num=base_num,
            suffix=suffix,
            rubrica=rubrica,
            testo=testo,
            note=note,
            source_file=str(html_path),
        )

    def _parse_filename(self, stem: str) -> tuple[Optional[int], Optional[str]]:
        """
        Parse article number from filename.

        Examples:
            art1 -> (1, None)
            art2043 -> (2043, None)
            art2043bis -> (2043, "bis")
            art106-bis -> (106, "bis")
        """
        # Pattern: artN, artN-suffix, artNsuffix
        pattern = rf"art(\d+)[-]?({SUFFIX_PATTERN})?"
        match = re.match(pattern, stem, re.IGNORECASE)
        if match:
            base_num = int(match.group(1))
            suffix = match.group(2).lower() if match.group(2) else None
            return base_num, suffix
        return None, None

    def _extract_text(self, element) -> str:
        """Estrae testo pulito da un elemento BeautifulSoup."""
        # Rimuovi tag script, style
        for tag in element.find_all(['script', 'style', 'ins']):
            tag.decompose()

        # Sostituisci <p> e <br> con newline
        for p in element.find_all('p'):
            p.insert_before('\n')
            p.insert_after('\n')
        for br in element.find_all('br'):
            br.replace_with('\n')

        # Estrai testo
        text = element.get_text(separator=' ')

        # Fix encoding
        text = self._fix_encoding(text)

        # Pulisci
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues and HTML entities."""
        import html
        # First decode HTML entities
        text = html.unescape(text)

        # Windows-1252 quirks that might slip through
        replacements = {
            '\x92': "'",
            '\x93': '"',
            '\x94': '"',
            '\x96': '–',
            '\x97': '—',
            '\ufffd': '',  # Unicode replacement char
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def get_articles(self, code: str, use_cache: bool = True, max_article: Optional[int] = None) -> list[BrocardiArticle]:
        """
        Restituisce tutti gli articoli per un codice.

        Args:
            code: Codice documento (es. CC)
            use_cache: Se usare cache in memoria
            max_article: Filtra articoli con base_num > max_article (HTTrack anomalies)

        Returns:
            Lista di BrocardiArticle
        """
        code = code.upper()

        # Default max per codice (da document_codes.json)
        default_max = {
            'COST': 139,
            'CC': 2969,
            'CP': 734,
            'CPC': 840,
            'CPP': 746,
        }

        if max_article is None:
            max_article = default_max.get(code)

        cache_key = f"{code}:{max_article or 'all'}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        html_files = self.get_html_files(code)
        if not html_files:
            logger.warning(f"No HTML files found for {code}")
            return []

        articles = []
        seen_keys = set()
        skipped_anomalies = 0

        for html_path in html_files:
            article = self.parse_article_file(html_path, code)
            if article and article.article_key not in seen_keys:
                # Filter HTTrack anomalies (es. art425392.html)
                if max_article and article.base_num > max_article:
                    skipped_anomalies += 1
                    logger.debug(f"Skipping HTTrack anomaly: {article.article_num} (base_num={article.base_num})")
                    continue
                articles.append(article)
                seen_keys.add(article.article_key)

        if skipped_anomalies:
            logger.info(f"{code}: skipped {skipped_anomalies} HTTrack anomalies (base_num > {max_article})")

        # Sort by base_num, then suffix
        articles.sort(key=lambda x: (x.base_num, x.suffix or ""))

        if use_cache:
            self._cache[cache_key] = articles

        return articles

    def get_article(self, code: str, base_num: int, suffix: Optional[str] = None) -> Optional[BrocardiArticle]:
        """
        Restituisce un singolo articolo.

        Args:
            code: Codice documento
            base_num: Numero articolo base
            suffix: Suffisso opzionale (bis, ter, etc.)

        Returns:
            BrocardiArticle o None
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
        """Restituisce i codici disponibili (con folder esistente)."""
        available = []
        # Check CODE_TO_PATH first (5 codici principali)
        for code in CODE_TO_PATH:
            if self.get_folder(code):
                available.append(code)
        # Check CODE_TO_FOLDER (legacy)
        for code in CODE_TO_FOLDER:
            if code not in available and self.get_folder(code):
                available.append(code)
        return sorted(available)


# ==============================================================================
# MAIN (test)
# ==============================================================================

if __name__ == "__main__":
    parser = BrocardiParser()

    print("=== BROCARDI PARSER ===")
    print(f"Base: {parser.base}")
    print(f"Base exists: {parser.base.exists()}")

    # Codici disponibili
    available = parser.get_available_codes()
    print(f"\nCodici disponibili: {len(available)}")
    for code in available:
        folder = parser.get_folder(code)
        files = parser.get_html_files(code)
        print(f"  {code}: {len(files)} article files in {folder.name if folder else 'N/A'}")

    # Test parsing CC
    if 'CC' in available:
        print("\n=== TEST CC ===")
        articles = parser.get_articles("CC")
        print(f"Articoli estratti: {len(articles)}")

        if articles:
            print(f"\nPrimi 5:")
            for a in articles[:5]:
                rubrica = (a.rubrica[:40] + "...") if a.rubrica and len(a.rubrica) > 40 else a.rubrica
                print(f"  Art. {a.article_num}: {rubrica or '(no rubrica)'}")
                print(f"    Testo: {a.testo[:60]}...")

            # Test get_article specifico
            print("\n=== TEST GET_ARTICLE ===")
            art1 = parser.get_article("CC", 1)
            if art1:
                print(f"Art. 1: {art1.rubrica}")
                print(f"  Testo ({len(art1.testo)} chars): {art1.testo[:100]}...")
                print(f"  Note: {len(art1.note)}")
            else:
                print("Art. 1 non trovato")
    else:
        print("\nCC non disponibile - download in corso?")
