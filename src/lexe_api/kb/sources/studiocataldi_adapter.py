# lexe_api/kb/sources/studiocataldi_adapter.py
"""
StudioCataldi Mirror Adapter.

Parser per i file HTML del mirror locale di StudioCataldi.
Questo è un MIRROR (trust level: mirror), richiede SEMPRE cross-check
con fonti canoniche (Normattiva).

Il mirror contiene:
- normativa/: Codici italiani (1,598 files, 87 MB)
- juris/: Sentenze (42,000 files) - FASE 2

Questo adapter gestisce solo la cartella normativa/ per FASE 1.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import structlog

from lexe_api.kb.sources.base_adapter import (
    BaseLegalSourceAdapter,
    FetchError,
    ParseError,
    ProgressCallback,
    TrustLevel,
)
from lexe_api.kb.sources.models import ArticleExtract
from lexe_api.kb.ingestion.deterministic_cleaner import StudioCataldiCleaner
from lexe_api.kb.ingestion.structure_extractor import StructureExtractor
from lexe_api.kb.ingestion.urn_generator import URNGenerator
from lexe_api.kb.ingestion.legal_numbers_extractor import extract_canonical_ids

logger = structlog.get_logger(__name__)


# ============================================================
# FILE PATH MAPPING
# ============================================================

CODICE_DIRECTORIES = {
    "CC": "codice-civile",
    "CP": "codice-penale",
    "CPC": "codice-procedura-civile",
    "CPP": "codice-procedura-penale",
    "COST": "costituzione",
    "CDS": "codicedellastrada",
    "CDPR": "codice-della-privacy",
    "CCONS": "codice-del-consumo",
    "CNAV": "codice-della-navigazione",
}

# Pattern per identificare file articolo
ARTICLE_FILE_PATTERN = re.compile(
    r"art(?:icolo)?[\-_]?(\d+[\-_]?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)",
    re.IGNORECASE
)


# ============================================================
# STUDIOCATALDI ADAPTER
# ============================================================

class StudioCataldiAdapter(BaseLegalSourceAdapter):
    """
    Adapter per mirror locale StudioCataldi.

    Legge file HTML da disco e li converte in ArticleExtract.
    Trust level: MIRROR - richiede SEMPRE cross-validation.
    """

    def __init__(self, base_path: str | Path):
        """
        Args:
            base_path: Path alla cartella normativa/ del mirror
        """
        self.base_path = Path(base_path)
        self.cleaner = StudioCataldiCleaner()
        self.extractor = StructureExtractor()
        self.urn_gen = URNGenerator()

        if not self.base_path.exists():
            logger.warning("Mirror path does not exist", path=str(self.base_path))

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def source_name(self) -> str:
        return "studiocataldi"

    @property
    def trust_level(self) -> TrustLevel:
        return TrustLevel.MIRROR

    @property
    def requires_throttling(self) -> bool:
        # No throttling needed - local files
        return False

    # =========================================================================
    # ABSTRACT METHODS IMPLEMENTATION
    # =========================================================================

    async def fetch_article(
        self,
        codice: str,
        articolo: str,
        comma: str | None = None,
    ) -> ArticleExtract | None:
        """
        Fetch singolo articolo dal mirror.
        """
        # Find the file
        file_path = self._find_article_file(codice, articolo)

        if file_path is None:
            logger.debug(
                "Article file not found",
                codice=codice,
                articolo=articolo,
            )
            return None

        # Parse the file
        try:
            return await self._parse_article_file(file_path, codice, articolo)
        except Exception as e:
            logger.error(
                "Failed to parse article file",
                file=str(file_path),
                error=str(e),
            )
            return None

    async def fetch_codice(
        self,
        codice: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ArticleExtract]:
        """
        Fetch tutti gli articoli di un codice.
        """
        articles = []

        # Get directory for this codice
        codice_dir = self._get_codice_directory(codice)
        if codice_dir is None or not codice_dir.exists():
            logger.warning("Codice directory not found", codice=codice)
            return articles

        # Find all article files
        article_files = self._find_article_files(codice_dir)
        total = len(article_files)

        logger.info(
            "Processing codice",
            codice=codice,
            files=total,
        )

        for i, file_path in enumerate(article_files):
            try:
                # Extract article number from filename
                articolo = self._extract_article_from_filename(file_path)

                if articolo is None:
                    continue

                article = await self._parse_article_file(file_path, codice, articolo)

                if article:
                    articles.append(article)

                if progress_callback:
                    progress_callback(i + 1, total, f"Art. {articolo}")

            except Exception as e:
                logger.warning(
                    "Failed to parse file",
                    file=str(file_path),
                    error=str(e),
                )
                continue

        logger.info(
            "Codice processed",
            codice=codice,
            articles=len(articles),
        )

        return articles

    async def stream_codice(
        self,
        codice: str,
        progress_callback: ProgressCallback | None = None,
    ) -> AsyncIterator[ArticleExtract]:
        """
        Stream articoli per memory efficiency.
        """
        codice_dir = self._get_codice_directory(codice)
        if codice_dir is None or not codice_dir.exists():
            return

        article_files = self._find_article_files(codice_dir)
        total = len(article_files)

        for i, file_path in enumerate(article_files):
            try:
                articolo = self._extract_article_from_filename(file_path)
                if articolo is None:
                    continue

                article = await self._parse_article_file(file_path, codice, articolo)

                if article:
                    yield article

                if progress_callback:
                    progress_callback(i + 1, total, f"Art. {articolo}")

            except Exception as e:
                logger.warning(
                    "Failed to parse file",
                    file=str(file_path),
                    error=str(e),
                )
                continue

    async def list_codici(self) -> list[str]:
        """
        Lista codici disponibili nel mirror.
        """
        available = []

        for codice, dir_name in CODICE_DIRECTORIES.items():
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                available.append(codice)

        # Also check for other directories
        for subdir in self.base_path.iterdir():
            if subdir.is_dir():
                # Try to infer codice from directory name
                name_lower = subdir.name.lower()
                if "costituzione" in name_lower and "COST" not in available:
                    available.append("COST")

        return available

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_codice_directory(self, codice: str) -> Path | None:
        """Get directory path for a codice."""
        codice_upper = codice.upper()

        if codice_upper in CODICE_DIRECTORIES:
            return self.base_path / CODICE_DIRECTORIES[codice_upper]

        # Try to find by scanning directories
        for subdir in self.base_path.iterdir():
            if subdir.is_dir():
                name_lower = subdir.name.lower().replace("-", "").replace("_", "")

                if codice_upper == "CC" and "codicecivile" in name_lower:
                    return subdir
                if codice_upper == "CP" and "codicepenale" in name_lower:
                    return subdir
                if codice_upper == "COST" and "costituzione" in name_lower:
                    return subdir

        return None

    def _find_article_file(self, codice: str, articolo: str) -> Path | None:
        """Find specific article file."""
        codice_dir = self._get_codice_directory(codice)
        if codice_dir is None:
            return None

        # Normalize articolo for filename matching
        art_normalized = articolo.lower().replace("-", "").replace(" ", "")

        # Try different filename patterns
        patterns = [
            f"art{art_normalized}*.html",
            f"articolo{art_normalized}*.html",
            f"art-{art_normalized}*.html",
            f"art_{art_normalized}*.html",
            f"*art{art_normalized}*.html",
        ]

        for pattern in patterns:
            matches = list(codice_dir.glob(pattern))
            if matches:
                return matches[0]

            # Also check subdirectories
            matches = list(codice_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]

        return None

    def _find_article_files(self, codice_dir: Path) -> list[Path]:
        """Find all article files in a codice directory."""
        files = []

        # Get all HTML files
        for html_file in codice_dir.glob("**/*.html"):
            # Check if it's an article file
            if ARTICLE_FILE_PATTERN.search(html_file.stem):
                files.append(html_file)

        # Sort by article number
        def sort_key(f: Path) -> tuple:
            match = ARTICLE_FILE_PATTERN.search(f.stem)
            if match:
                art = match.group(1).replace("-", "").replace("_", "")
                # Extract numeric part
                num_match = re.match(r"(\d+)", art)
                if num_match:
                    return (int(num_match.group(1)), art)
            return (999999, f.stem)

        files.sort(key=sort_key)

        return files

    def _extract_article_from_filename(self, file_path: Path) -> str | None:
        """Extract article number from filename."""
        match = ARTICLE_FILE_PATTERN.search(file_path.stem)
        if match:
            art = match.group(1)
            # Normalize: 2043bis -> 2043-bis
            art = re.sub(r"(\d+)(bis|ter|quater|quinquies)", r"\1-\2", art, flags=re.I)
            return art
        return None

    async def _parse_article_file(
        self,
        file_path: Path,
        codice: str,
        articolo: str,
    ) -> ArticleExtract | None:
        """Parse an article HTML file."""
        try:
            # Read and clean HTML
            cleaned, metadata = self.cleaner.clean_article_page(
                file_path.read_text(encoding="utf-8", errors="ignore")
            )

            if not cleaned.text or len(cleaned.text) < 10:
                return None

            # Extract structure
            structure = self.extractor.extract_single_article(cleaned.text)

            # Use extracted or provided values
            final_articolo = structure.articolo if structure else articolo
            final_rubrica = structure.rubrica if structure else metadata.get("rubrica")

            # Generate URN
            urn = self.urn_gen.generate_for_codice(codice, final_articolo)

            # Extract citations
            citations = list(extract_canonical_ids(cleaned.text))

            return ArticleExtract(
                codice=codice.upper(),
                articolo=final_articolo,
                comma=None,
                urn_nir=urn,
                rubrica=final_rubrica,
                testo=cleaned.text,
                testo_normalizzato=None,  # Will be computed
                libro=structure.libro if structure else None,
                titolo=structure.titolo if structure else None,
                capo=structure.capo if structure else None,
                sezione=structure.sezione if structure else None,
                source="studiocataldi",
                source_file=str(file_path.relative_to(self.base_path)),
                retrieved_at=datetime.now(),
                citations_raw=citations if citations else None,
            )

        except Exception as e:
            raise ParseError(
                "studiocataldi",
                f"Failed to parse {file_path}: {e}"
            )


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_studiocataldi_adapter(
    mirror_path: str | Path | None = None,
) -> StudioCataldiAdapter:
    """
    Create StudioCataldi adapter with default or custom path.

    Args:
        mirror_path: Path to normativa/ folder, or None to use default

    Returns:
        StudioCataldiAdapter instance
    """
    if mirror_path is None:
        # Try common locations
        candidates = [
            Path("/opt/lexe-platform/data/studiocataldi/normativa"),
            Path("C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa"),
            Path.cwd() / "data" / "studiocataldi" / "normativa",
        ]
        for candidate in candidates:
            if candidate.exists():
                mirror_path = candidate
                break

        if mirror_path is None:
            mirror_path = candidates[0]  # Default even if not exists

    return StudioCataldiAdapter(mirror_path)
