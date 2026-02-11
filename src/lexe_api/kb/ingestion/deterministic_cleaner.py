# lexe_api/kb/ingestion/deterministic_cleaner.py
"""
Deterministic HTML Cleaner - NO LLM REQUIRED.

Pulisce HTML da mirror StudioCataldi (e simili) usando solo BeautifulSoup.
Questo step PRIMA di qualsiasi LLM riduce i token del 70%!

Pipeline:
1. Parse HTML con BeautifulSoup
2. Rimuovi elementi inutili (nav, ads, scripts, etc.)
3. Estrai contenuto principale
4. Normalizza whitespace
5. Output: testo pulito pronto per structure extraction

Costo: $0 (puro Python, nessun LLM)
"""

import re
from dataclasses import dataclass
from pathlib import Path

import structlog
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

logger = structlog.get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Tags da rimuovere completamente (incluso contenuto)
REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "object",
    "embed",
    "applet",
    "form",
    "input",
    "button",
    "select",
    "textarea",
    "svg",
    "canvas",
    "video",
    "audio",
    "source",
    "track",
    "map",
    "area",
}

# Tags di navigazione/layout da rimuovere
REMOVE_NAV_TAGS = {
    "nav",
    "header",
    "footer",
    "aside",
    "menu",
    "menuitem",
}

# Classi CSS che indicano contenuto da rimuovere
REMOVE_CLASSES = {
    # Navigation
    "nav",
    "navbar",
    "navigation",
    "menu",
    "sidebar",
    "side-bar",
    "header",
    "footer",
    "breadcrumb",
    "breadcrumbs",
    "pagination",
    # Ads
    "ad",
    "ads",
    "advert",
    "advertisement",
    "banner",
    "sponsor",
    "promo",
    "promotion",
    # Social
    "social",
    "share",
    "sharing",
    "follow",
    "like",
    "tweet",
    # Comments
    "comment",
    "comments",
    "disqus",
    # Misc
    "cookie",
    "popup",
    "modal",
    "overlay",
    "widget",
    "related",
    "related-posts",
    "signup",
    "newsletter",
}

# IDs che indicano contenuto da rimuovere
REMOVE_IDS = {
    "nav",
    "navbar",
    "navigation",
    "header",
    "footer",
    "sidebar",
    "comments",
    "disqus",
    "cookie",
    "popup",
    "modal",
}

# Selettori CSS per contenuto principale (in ordine di priorità)
MAIN_CONTENT_SELECTORS = [
    # StudioCataldi specific
    "div.corpo_articolo",
    "div.testo_articolo",
    "div.contenuto_articolo",
    "div.article-body",
    "div.articleBody",
    # Generic
    "article",
    "main",
    "[role='main']",
    "div.content",
    "div.main-content",
    "div.post-content",
    "div.entry-content",
    "div.article-content",
    # Fallback
    "div.container",
    "div#content",
    "div#main",
]


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class CleanedContent:
    """Risultato della pulizia HTML."""

    # Contenuto pulito
    text: str
    html_clean: str  # HTML semplificato (solo tags strutturali)

    # Metadata estratti
    title: str | None = None
    meta_description: str | None = None

    # Stats
    original_size: int = 0
    cleaned_size: int = 0

    @property
    def reduction_ratio(self) -> float:
        """Percentuale di riduzione dimensione."""
        if self.original_size == 0:
            return 0.0
        return 1 - (self.cleaned_size / self.original_size)

    @property
    def token_estimate(self) -> int:
        """Stima token (1 token ~ 4 caratteri in italiano)."""
        return len(self.text) // 4


# ============================================================
# DETERMINISTIC CLEANER
# ============================================================


class DeterministicCleaner:
    """
    Pulisce HTML senza LLM usando BeautifulSoup.

    Strategie:
    1. AGGRESSIVE: Rimuove tutto tranne testo puro
    2. STRUCTURAL: Mantiene struttura (h1-h6, p, ul, ol, li)
    3. PRESERVE: Mantiene più markup (anche table, blockquote)
    """

    def __init__(self, mode: str = "structural"):
        """
        Args:
            mode: 'aggressive', 'structural', or 'preserve'
        """
        self.mode = mode

    def clean_html(self, html: str) -> CleanedContent:
        """
        Pulisce HTML e restituisce contenuto pulito.

        Args:
            html: HTML raw da pulire

        Returns:
            CleanedContent con testo pulito e metadata
        """
        original_size = len(html)

        # Parse HTML
        soup = BeautifulSoup(html, "lxml")

        # Extract metadata before cleaning
        title = self._extract_title(soup)
        meta_desc = self._extract_meta_description(soup)

        # Remove unwanted elements
        self._remove_unwanted_elements(soup)

        # Find main content
        main_content = self._find_main_content(soup)

        if main_content is None:
            # Fallback: usa body
            main_content = soup.body or soup

        # Clean the main content
        self._clean_content(main_content)

        # Extract text
        text = self._extract_text(main_content)
        html_clean = str(main_content) if self.mode != "aggressive" else ""

        return CleanedContent(
            text=text,
            html_clean=html_clean,
            title=title,
            meta_description=meta_desc,
            original_size=original_size,
            cleaned_size=len(text),
        )

    def clean_file(self, file_path: str | Path) -> CleanedContent:
        """
        Pulisce file HTML da disco.

        Args:
            file_path: Path al file HTML

        Returns:
            CleanedContent
        """
        path = Path(file_path)

        # Try different encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                html = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Last resort: ignore errors
            html = path.read_bytes().decode("utf-8", errors="ignore")

        return self.clean_html(html)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        """Estrae titolo dalla pagina."""
        # Try <title>
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            # Remove site name suffix
            title = re.sub(r"\s*[|\-–]\s*[^|\-–]+$", "", title)
            return title

        # Try <h1>
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"]

        return None

    def _extract_meta_description(self, soup: BeautifulSoup) -> str | None:
        """Estrae meta description."""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"]

        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc["content"]

        return None

    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Rimuove elementi inutili dal DOM."""
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove script, style, etc.
        for tag_name in REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove nav elements
        for tag_name in REMOVE_NAV_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove by class
        for class_name in REMOVE_CLASSES:
            for tag in soup.find_all(
                class_=lambda c, cn=class_name: c and cn in c.lower() if c else False
            ):
                tag.decompose()

        # Remove by ID
        for id_name in REMOVE_IDS:
            for tag in soup.find_all(
                id=lambda i, idn=id_name: i and idn in i.lower() if i else False
            ):
                tag.decompose()

        # Remove hidden elements
        for tag in soup.find_all(
            style=lambda s: s and "display:none" in s.replace(" ", "") if s else False
        ):
            tag.decompose()
        for tag in soup.find_all(
            style=lambda s: s and "visibility:hidden" in s.replace(" ", "") if s else False
        ):
            tag.decompose()

        # Remove empty divs
        for div in soup.find_all("div"):
            if not div.get_text(strip=True):
                div.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> Tag | None:
        """Trova il contenitore principale del contenuto."""
        for selector in MAIN_CONTENT_SELECTORS:
            try:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 100:
                    return element
            except Exception:
                continue

        return None

    def _clean_content(self, element: Tag) -> None:
        """Pulisce il contenuto principale."""
        # Remove remaining nav/footer within content
        for tag in element.find_all(["nav", "footer", "aside"]):
            tag.decompose()

        # Remove empty paragraphs
        for p in element.find_all("p"):
            if not p.get_text(strip=True):
                p.decompose()

        # Remove excessive br tags
        for br in element.find_all("br"):
            # Keep single br, remove if many in a row
            next_sibling = br.next_sibling
            if isinstance(next_sibling, Tag) and next_sibling.name == "br":
                br.decompose()

        # Clean links (keep text, remove href in aggressive mode)
        if self.mode == "aggressive":
            for a in element.find_all("a"):
                a.unwrap()

        # Remove all attributes except essentials in structural mode
        if self.mode in ("structural", "aggressive"):
            for tag in element.find_all(True):
                # Keep only id and class for debugging
                attrs_to_keep = {}
                if self.mode == "structural" and tag.get("id"):
                    attrs_to_keep["id"] = tag["id"]
                tag.attrs = attrs_to_keep

    def _extract_text(self, element: Tag) -> str:
        """Estrae testo pulito dall'elemento."""
        if self.mode == "aggressive":
            # Solo testo, nessuna formattazione
            text = element.get_text(separator=" ")
        else:
            # Mantieni un po' di struttura con newlines
            text = self._get_text_with_structure(element)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        return text

    def _get_text_with_structure(self, element: Tag) -> str:
        """Estrae testo mantenendo struttura base."""
        lines = []

        for child in element.descendants:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    lines.append(text)
            elif isinstance(child, Tag):
                if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    lines.append("\n\n" + child.get_text(strip=True) + "\n")
                elif child.name == "p":
                    lines.append("\n" + child.get_text(strip=True))
                elif child.name == "br":
                    lines.append("\n")
                elif child.name == "li":
                    lines.append("\n- " + child.get_text(strip=True))

        return "".join(lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalizza spazi bianchi."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace on each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove leading/trailing whitespace overall
        text = text.strip()

        return text


# ============================================================
# STUDIOCATALDI-SPECIFIC CLEANER
# ============================================================


class StudioCataldiCleaner(DeterministicCleaner):
    """
    Cleaner ottimizzato per mirror StudioCataldi.

    Conosce la struttura specifica dei file StudioCataldi e usa
    selettori mirati per estrarre contenuto legale.
    """

    # Selettori specifici StudioCataldi
    ARTICLE_SELECTORS = [
        "div.corpo_articolo",
        "div.testo_articolo",
        "div.articolo_codice",
        "div.testo_legge",
        "div#testo_articolo",
        "div.main-content",
    ]

    # Pattern per riconoscere intestazioni articolo
    ARTICLE_HEADER_PATTERN = re.compile(
        r"^(?:Art(?:icolo)?\.?\s*)?(\d+[\-bis\-ter\-quater\-quinquies\-sexies\-septies\-octies\-novies\-decies]*)"
        r"(?:\s*[\.\-\–]\s*(.+))?$",
        re.IGNORECASE,
    )

    def __init__(self):
        super().__init__(mode="structural")

    def clean_article_page(self, html: str) -> tuple[CleanedContent, dict]:
        """
        Pulisce pagina articolo StudioCataldi.

        Returns:
            (CleanedContent, metadata_dict)
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract structured metadata
        metadata = self._extract_article_metadata(soup)

        # Standard cleaning
        self._remove_unwanted_elements(soup)

        # Find article content with specific selectors
        content = None
        for selector in self.ARTICLE_SELECTORS:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 50:
                break

        if content is None:
            content = self._find_main_content(soup) or soup.body or soup

        self._clean_content(content)

        text = self._extract_text(content)
        html_clean = str(content)

        return CleanedContent(
            text=text,
            html_clean=html_clean,
            title=metadata.get("rubrica"),
            meta_description=None,
            original_size=len(html),
            cleaned_size=len(text),
        ), metadata

    def _extract_article_metadata(self, soup: BeautifulSoup) -> dict:
        """Estrae metadata strutturati da pagina articolo."""
        metadata = {}

        # Try to find article number from h1/h2
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text(strip=True)
            match = self.ARTICLE_HEADER_PATTERN.match(text)
            if match:
                metadata["articolo"] = match.group(1)
                if match.group(2):
                    metadata["rubrica"] = match.group(2).strip()
                break

        # Try to find codice from breadcrumb or title
        title = soup.title.string if soup.title else ""
        if "codice civile" in title.lower():
            metadata["codice"] = "CC"
        elif "codice penale" in title.lower():
            metadata["codice"] = "CP"
        elif "costituzione" in title.lower():
            metadata["codice"] = "COST"
        elif "procedura civile" in title.lower():
            metadata["codice"] = "CPC"
        elif "procedura penale" in title.lower():
            metadata["codice"] = "CPP"
        elif "codice della strada" in title.lower():
            metadata["codice"] = "CDS"

        return metadata


# ============================================================
# BATCH PROCESSING
# ============================================================


def clean_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "*.html",
    mode: str = "structural",
) -> dict:
    """
    Pulisce tutti i file HTML in una directory.

    Args:
        input_dir: Directory input
        output_dir: Directory output (opzionale, default: input_dir/cleaned)
        pattern: Glob pattern per file
        mode: Modalità pulizia

    Returns:
        Stats dict con conteggi e metriche
    """
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "cleaned"
    output_path.mkdir(parents=True, exist_ok=True)

    cleaner = DeterministicCleaner(mode=mode)

    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "total_original_size": 0,
        "total_cleaned_size": 0,
        "tokens_saved": 0,
    }

    for file_path in input_path.glob(pattern):
        try:
            result = cleaner.clean_file(file_path)

            # Write cleaned text
            out_file = output_path / f"{file_path.stem}.txt"
            out_file.write_text(result.text, encoding="utf-8")

            stats["files_processed"] += 1
            stats["total_original_size"] += result.original_size
            stats["total_cleaned_size"] += result.cleaned_size

        except Exception as e:
            logger.error("Failed to clean file", file=str(file_path), error=str(e))
            stats["files_failed"] += 1

    # Calculate token savings
    stats["tokens_saved"] = (stats["total_original_size"] - stats["total_cleaned_size"]) // 4

    stats["reduction_ratio"] = (
        1 - (stats["total_cleaned_size"] / stats["total_original_size"])
        if stats["total_original_size"] > 0
        else 0
    )

    return stats
