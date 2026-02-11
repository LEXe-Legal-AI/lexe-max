# lexe_api/kb/sources/brocardi_adapter.py
"""
Brocardi.it Adapter - Editorial Source for Italian Legal Codes.

Brocardi.it è una fonte EDITORIALE (trust level: editorial) con:
- Ottima struttura articolo per articolo
- Rubriche, indici, aggiornamenti rapidi
- Brocardi latini + Dizionario giuridico

URLs:
- CC: https://www.brocardi.it/codice-civile/
- CP: https://www.brocardi.it/codice-penale/
- CPC: https://www.brocardi.it/codice-di-procedura-civile/
- CPP: https://www.brocardi.it/codice-di-procedura-penale/
- COST: https://www.brocardi.it/costituzione-italiana/

Throttling: 1 req/sec per rispettare ToS
"""

import asyncio
import re
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
import structlog

from lexe_api.kb.sources.base_adapter import (
    BaseLegalSourceAdapter,
    FetchError,
    ParseError,
    ProgressCallback,
    RateLimitError,
    TrustLevel,
)
from lexe_api.kb.sources.models import (
    ArticleExtract,
    BrocardiExtract,
    DizionarioExtract,
)
from lexe_api.kb.ingestion.urn_generator import URNGenerator
from lexe_api.kb.ingestion.legal_numbers_extractor import extract_canonical_ids

logger = structlog.get_logger(__name__)


# ============================================================
# BROCARDI URL MAPPING
# ============================================================

BROCARDI_BASE_URL = "https://www.brocardi.it"

CODICE_URLS = {
    "CC": f"{BROCARDI_BASE_URL}/codice-civile/",
    "CP": f"{BROCARDI_BASE_URL}/codice-penale/",
    "CPC": f"{BROCARDI_BASE_URL}/codice-di-procedura-civile/",
    "CPP": f"{BROCARDI_BASE_URL}/codice-di-procedura-penale/",
    "COST": f"{BROCARDI_BASE_URL}/costituzione-italiana/",
    "CDS": f"{BROCARDI_BASE_URL}/codice-della-strada/",
    "CCONS": f"{BROCARDI_BASE_URL}/codice-del-consumo/",
    "CDPR": f"{BROCARDI_BASE_URL}/codice-della-privacy/",
}

# Pattern per estrarre numero articolo da URL
ARTICLE_URL_PATTERN = re.compile(r"/art(\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)", re.I)


# ============================================================
# BROCARDI ADAPTER
# ============================================================

class BrocardiAdapter(BaseLegalSourceAdapter):
    """
    Adapter per Brocardi.it - fonte editoriale di alta qualità.

    Supporta:
    - Fetch articoli singoli
    - Fetch codice completo
    - Brocardi latini
    - Dizionario giuridico
    """

    def __init__(
        self,
        requests_per_second: float = 1.0,
        timeout: float = 30.0,
    ):
        """
        Args:
            requests_per_second: Rate limit (default 1.0 per ToS)
            timeout: HTTP timeout in seconds
        """
        self._rps = requests_per_second
        self._timeout = timeout
        self._last_request_time: float = 0
        self._client: httpx.AsyncClient | None = None
        self._urn_gen = URNGenerator()

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def source_name(self) -> str:
        return "brocardi"

    @property
    def trust_level(self) -> TrustLevel:
        return TrustLevel.EDITORIAL

    @property
    def base_url(self) -> str:
        return BROCARDI_BASE_URL

    @property
    def requires_throttling(self) -> bool:
        return True

    @property
    def requests_per_second(self) -> float:
        return self._rps

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> "BrocardiAdapter":
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "LEXE-Legal-AI/1.0 (legal research; contact@lexe.pro)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # ABSTRACT METHODS IMPLEMENTATION
    # =========================================================================

    async def fetch_article(
        self,
        codice: str,
        articolo: str,
        comma: str | None = None,
    ) -> ArticleExtract | None:
        """Fetch singolo articolo da Brocardi."""
        codice_upper = codice.upper()

        if codice_upper not in CODICE_URLS:
            logger.warning("Unsupported codice", codice=codice)
            return None

        # Brocardi has hierarchical URLs, we need to find the correct URL
        # by crawling or using their search
        base_url = CODICE_URLS[codice_upper]
        art_normalized = articolo.lower().replace("-", "").replace(" ", "")

        # Get all article URLs (cached if possible)
        article_urls = await self._get_article_urls(base_url)

        # Find the matching article
        article_url = None
        for art_num, url in article_urls:
            if art_num.lower() == art_normalized:
                article_url = url
                break

        if not article_url:
            logger.debug("Article not found in index", codice=codice, articolo=articolo)
            return None

        try:
            html = await self._fetch_html(article_url)
            return self._parse_article_page(html, codice_upper, articolo, article_url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise FetchError(self.source_name, f"HTTP {e.response.status_code} for {article_url}")
        except Exception as e:
            logger.error("Failed to fetch article", url=article_url, error=str(e))
            raise FetchError(self.source_name, str(e))

    async def fetch_codice(
        self,
        codice: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ArticleExtract]:
        """Fetch tutti gli articoli di un codice."""
        codice_upper = codice.upper()

        if codice_upper not in CODICE_URLS:
            logger.warning("Unsupported codice", codice=codice)
            return []

        # First, get the index page to find all articles
        base_url = CODICE_URLS[codice_upper]

        try:
            article_urls = await self._get_article_urls(base_url)
        except Exception as e:
            logger.error("Failed to get article index", url=base_url, error=str(e))
            raise FetchError(self.source_name, f"Failed to get index: {e}")

        total = len(article_urls)
        logger.info("Found articles", codice=codice, count=total)

        articles = []
        for i, (art_num, art_url) in enumerate(article_urls):
            try:
                html = await self._fetch_html(art_url)
                article = self._parse_article_page(html, codice_upper, art_num, art_url)

                if article:
                    articles.append(article)

                if progress_callback:
                    progress_callback(i + 1, total, f"Art. {art_num}")

            except Exception as e:
                logger.warning("Failed to fetch article", url=art_url, error=str(e))
                continue

        return articles

    async def stream_codice(
        self,
        codice: str,
        progress_callback: ProgressCallback | None = None,
    ) -> AsyncIterator[ArticleExtract]:
        """Stream articoli (memory efficient)."""
        codice_upper = codice.upper()

        if codice_upper not in CODICE_URLS:
            return

        base_url = CODICE_URLS[codice_upper]
        article_urls = await self._get_article_urls(base_url)
        total = len(article_urls)

        for i, (art_num, art_url) in enumerate(article_urls):
            try:
                html = await self._fetch_html(art_url)
                article = self._parse_article_page(html, codice_upper, art_num, art_url)

                if article:
                    yield article

                if progress_callback:
                    progress_callback(i + 1, total, f"Art. {art_num}")

            except Exception as e:
                logger.warning("Failed to fetch article", url=art_url, error=str(e))
                continue

    async def list_codici(self) -> list[str]:
        """Lista codici supportati."""
        return list(CODICE_URLS.keys())

    # =========================================================================
    # BROCARDI-SPECIFIC METHODS
    # =========================================================================

    async def fetch_brocardo(self, latino: str) -> BrocardiExtract | None:
        """Fetch singolo brocardo latino."""
        # Brocardi are at /brocardi/
        # This would need a search or specific URL pattern
        # TODO: Implement brocardi search
        return None

    async def fetch_all_brocardi(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> list[BrocardiExtract]:
        """Fetch tutti i brocardi."""
        brocardi_url = f"{BROCARDI_BASE_URL}/brocardi/"

        try:
            html = await self._fetch_html(brocardi_url)
            soup = BeautifulSoup(html, "lxml")

            # Find all brocardi links
            brocardi = []
            brocardi_links = soup.select("a[href*='/brocardi/']")

            total = len(brocardi_links)
            for i, link in enumerate(brocardi_links):
                href = link.get("href")
                if not href or href == "/brocardi/":
                    continue

                try:
                    brocardo_html = await self._fetch_html(urljoin(BROCARDI_BASE_URL, href))
                    brocardo = self._parse_brocardo_page(brocardo_html, href)
                    if brocardo:
                        brocardi.append(brocardo)

                    if progress_callback:
                        progress_callback(i + 1, total, brocardo.latino if brocardo else None)

                except Exception as e:
                    logger.warning("Failed to fetch brocardo", url=href, error=str(e))
                    continue

            return brocardi

        except Exception as e:
            logger.error("Failed to fetch brocardi index", error=str(e))
            return []

    async def fetch_voce_dizionario(self, voce: str) -> DizionarioExtract | None:
        """Fetch voce dizionario."""
        # Dizionario at /dizionario/
        voce_slug = voce.lower().replace(" ", "-")
        url = f"{BROCARDI_BASE_URL}/dizionario/{voce_slug}.html"

        try:
            html = await self._fetch_html(url)
            return self._parse_dizionario_page(html, voce, url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error("Failed to fetch dizionario", voce=voce, error=str(e))
            return None

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    async def _fetch_html(self, url: str) -> str:
        """Fetch HTML with throttling."""
        await self._throttle()

        if self._client is None:
            # Create temporary client
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text

        response = await self._client.get(url)
        response.raise_for_status()
        return response.text

    async def _throttle(self) -> None:
        """Apply rate limiting."""
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self._rps

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    async def _get_article_urls(self, base_url: str) -> list[tuple[str, str]]:
        """Get all article URLs by crawling through hierarchy."""
        articles = []
        visited = set()

        async def crawl_page(url: str, depth: int = 0) -> None:
            """Recursively crawl pages to find article links."""
            if url in visited or depth > 4:
                return
            visited.add(url)

            try:
                html = await self._fetch_html(url)
            except Exception as e:
                logger.warning("Failed to fetch", url=url, error=str(e))
                return

            soup = BeautifulSoup(html, "lxml")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                # Skip external links
                if not full_url.startswith(BROCARDI_BASE_URL):
                    continue

                # Check if it's an article link
                match = ARTICLE_URL_PATTERN.search(href)
                if match:
                    art_num = match.group(1)
                    if (art_num, full_url) not in articles:
                        articles.append((art_num, full_url))
                    continue

                # Check if it's a hierarchy link (libro/titolo/capo)
                if any(x in href.lower() for x in ["/libro", "/titolo", "/capo", "/sezione"]):
                    if full_url not in visited:
                        await crawl_page(full_url, depth + 1)

        # Start crawling from base URL
        logger.info("Crawling hierarchy", base_url=base_url)
        await crawl_page(base_url)

        # Sort by article number
        def sort_key(item):
            num = item[0]
            num_match = re.match(r"(\d+)", num)
            return int(num_match.group(1)) if num_match else 999999

        articles.sort(key=sort_key)

        logger.info("Found articles", count=len(articles))
        return articles

    def _parse_article_page(
        self,
        html: str,
        codice: str,
        articolo: str,
        source_url: str,
    ) -> ArticleExtract | None:
        """Parse article page HTML."""
        soup = BeautifulSoup(html, "lxml")

        # Find article content - Brocardi uses div.corpoDelTesto or div.dispositivo
        content_div = soup.select_one(
            "div.corpoDelTesto, div.dispositivo, div.corpo-articolo, "
            "div.art-testo, div.testo-articolo"
        )

        if not content_div:
            # Broader fallback
            content_div = soup.select_one("article, main, div.content")

        if not content_div:
            logger.debug("No content found", url=source_url)
            return None

        # Extract rubrica from h1
        rubrica = None
        h1 = soup.select_one("h1")
        if h1:
            rubrica_text = h1.get_text(strip=True)
            # Extract rubrica: "Articolo 2043 Codice Civile" -> might have subtitle
            # Sometimes rubrica is in a separate element
            rubrica_el = soup.select_one("h2.rubrica, .rubrica, .titolo-articolo")
            if rubrica_el:
                rubrica = rubrica_el.get_text(strip=True)
            else:
                # Clean up h1: remove "Articolo X Codice Y"
                rubrica = re.sub(r"^Articolo\s+\d+\S*\s+Codice\s+\w+", "", rubrica_text).strip()
                if not rubrica:
                    rubrica = None

        # Extract article text - clean up annotation numbers like (1), (2)
        testo = content_div.get_text(separator=" ", strip=True)
        # Remove inline annotation markers
        testo = re.sub(r"\(\d+\)", "", testo)
        testo = re.sub(r"\[\d+\]", "", testo)
        testo = re.sub(r"\s+", " ", testo).strip()

        if not testo or len(testo) < 10:
            return None

        # Extract hierarchy from breadcrumb or URL
        libro = titolo = capo = sezione = None

        breadcrumb = soup.select_one(".breadcrumb, nav[aria-label='breadcrumb']")
        if breadcrumb:
            crumbs = breadcrumb.get_text(separator=" > ")
            # Parse hierarchy from breadcrumb
            if "Libro" in crumbs:
                libro_match = re.search(r"Libro\s+([IVX\d]+)", crumbs)
                if libro_match:
                    libro = f"Libro {libro_match.group(1)}"
            if "Titolo" in crumbs:
                titolo_match = re.search(r"Titolo\s+([IVX\d]+)", crumbs)
                if titolo_match:
                    titolo = f"Titolo {titolo_match.group(1)}"

        # Generate URN
        urn = self._urn_gen.generate_for_codice(codice, articolo)

        # Extract citations
        citations = list(extract_canonical_ids(testo))

        return ArticleExtract(
            codice=codice,
            articolo=articolo,
            comma=None,
            urn_nir=urn,
            rubrica=rubrica,
            testo=testo,
            testo_normalizzato=None,
            libro=libro,
            titolo=titolo,
            capo=capo,
            sezione=sezione,
            source="brocardi",
            source_url=source_url,
            retrieved_at=datetime.now(),
            citations_raw=citations if citations else None,
        )

    def _parse_brocardo_page(self, html: str, url: str) -> BrocardiExtract | None:
        """Parse brocardo page."""
        soup = BeautifulSoup(html, "lxml")

        # Find latino text
        latino_el = soup.select_one("h1, .brocardo-latino")
        if not latino_el:
            return None

        latino = latino_el.get_text(strip=True)

        # Find Italian translation
        italiano = None
        ita_el = soup.select_one(".brocardo-italiano, .traduzione")
        if ita_el:
            italiano = ita_el.get_text(strip=True)

        # Find significato
        significato = None
        sig_el = soup.select_one(".significato, .spiegazione, article p")
        if sig_el:
            significato = sig_el.get_text(strip=True)

        return BrocardiExtract(
            latino=latino,
            italiano=italiano,
            significato=significato,
            source="brocardi",
            source_url=urljoin(BROCARDI_BASE_URL, url),
            retrieved_at=datetime.now(),
        )

    def _parse_dizionario_page(
        self,
        html: str,
        voce: str,
        url: str,
    ) -> DizionarioExtract | None:
        """Parse dizionario page."""
        soup = BeautifulSoup(html, "lxml")

        # Find definition
        def_el = soup.select_one(".definizione, article, .content p")
        if not def_el:
            return None

        definizione = def_el.get_text(strip=True)

        return DizionarioExtract(
            voce=voce,
            definizione=definizione,
            source="brocardi",
            source_url=url,
            retrieved_at=datetime.now(),
        )


# ============================================================
# FACTORY & CONVENIENCE
# ============================================================

def create_brocardi_adapter(
    requests_per_second: float = 1.0,
) -> BrocardiAdapter:
    """Create Brocardi adapter instance."""
    return BrocardiAdapter(requests_per_second=requests_per_second)


async def fetch_codice_civile(
    progress_callback: ProgressCallback | None = None,
) -> list[ArticleExtract]:
    """Convenience function to fetch entire Codice Civile."""
    async with BrocardiAdapter() as adapter:
        return await adapter.fetch_codice("CC", progress_callback)
