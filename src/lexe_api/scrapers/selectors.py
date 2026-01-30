"""CSS Selectors for legal document scraping.

Centralized selectors for maintainability when sources change their HTML structure.
"""


class Selectors:
    """CSS selectors for different sources."""

    # =========================================================================
    # Normattiva Selectors
    # =========================================================================

    class Normattiva:
        """Selectors for normattiva.it."""

        # Article content
        ARTICLE_BODY = "#articolo_body"
        ARTICLE_TEXT = "#articolo_corpo"
        ARTICLE_RUBRICA = "#articolo_rubrica"

        # Metadata
        METADATA_URN = "meta[name='DC.identifier']"
        METADATA_TITLE = "meta[name='DC.title']"
        METADATA_DATE = "meta[name='DC.date']"

        # Vigenza info
        VIGENZA_BOX = ".vigenza-box"
        VIGENZA_STATUS = ".vigenza-status"
        ABROGATO_INFO = ".abrogato-info"
        MODIFICHE_INFO = ".modifiche-elenco li"

        # Tree navigation
        TREE_CONTAINER = "#indice-atto"
        TREE_ITEMS = ".indice-item"
        TREE_ARTICLE_LINK = "a[href*='articolo']"

        # Error messages
        ERROR_NOT_FOUND = ".messaggio-errore"

    # =========================================================================
    # Brocardi Selectors
    # =========================================================================

    class Brocardi:
        """Selectors for brocardi.it."""

        # Article content
        ARTICLE_CONTAINER = ".articolo-container"
        ARTICLE_TEXT = ".testo-articolo"
        ARTICLE_RUBRICA = ".rubrica-articolo"

        # Brocardi commentary
        SPIEGAZIONE = ".spiegazione-container"
        RATIO = ".ratio-articolo"

        # Massime
        MASSIME_CONTAINER = ".massime-container"
        MASSIMA_ITEM = ".massima-item"
        MASSIMA_AUTORITA = ".massima-autorita"
        MASSIMA_DATA = ".massima-data"
        MASSIMA_NUMERO = ".massima-numero"
        MASSIMA_TESTO = ".massima-testo"

        # Related content
        RELAZIONI_CONTAINER = ".relazioni-container"
        RELAZIONE_ITEM = ".relazione-item"

        # Footnotes
        FOOTNOTES_CONTAINER = ".note-container"
        FOOTNOTE_ITEM = ".nota-item"

    # =========================================================================
    # EUR-Lex Selectors (for fallback scraping)
    # =========================================================================

    class EurLex:
        """Selectors for eur-lex.europa.eu (fallback scraping)."""

        # Document content
        DOCUMENT_BODY = "#document1"
        DOCUMENT_TITLE = ".doc-ti"
        DOCUMENT_PREAMBLE = ".eli-preamble"

        # Article content
        ARTICLE_CONTAINER = ".eli-subdivision"
        ARTICLE_NUMBER = ".eli-title"
        ARTICLE_TEXT = ".eli-content"

        # Metadata
        CELEX_ID = ".celex-number"
        IN_FORCE_STATUS = ".in-force"
        PUBLICATION_DATE = ".pub-date"

        # Navigation
        TOC_CONTAINER = "#toc"
        TOC_ITEM = ".toc-item"

    # =========================================================================
    # URL Patterns
    # =========================================================================

    class URLs:
        """URL templates for different sources."""

        # Normattiva
        NORMATTIVA_BASE = "https://www.normattiva.it"
        NORMATTIVA_URN = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:{urn}"
        NORMATTIVA_ARTICLE = (
            "https://www.normattiva.it/uri-res/N2Ls?"
            "urn:nir:{act_type}:{date};{number}~art{article}"
        )
        NORMATTIVA_SEARCH = "https://www.normattiva.it/ricerca/semplice"

        # Brocardi
        BROCARDI_BASE = "https://www.brocardi.it"
        BROCARDI_CODICE_CIVILE = "https://www.brocardi.it/codice-civile/art{article}.html"
        BROCARDI_CODICE_PENALE = "https://www.brocardi.it/codice-penale/art{article}.html"
        BROCARDI_SEARCH = "https://www.brocardi.it/cerca?q={query}"

        # EUR-Lex
        EURLEX_BASE = "https://eur-lex.europa.eu"
        EURLEX_CELEX = (
            "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/"
            "?uri=CELEX:{celex}"
        )
        EURLEX_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"

        @classmethod
        def brocardi_url(cls, act_type: str, article: str) -> str:
            """Build Brocardi URL for an article."""
            act_map = {
                "codice civile": cls.BROCARDI_CODICE_CIVILE,
                "codice penale": cls.BROCARDI_CODICE_PENALE,
                "cc": cls.BROCARDI_CODICE_CIVILE,
                "cp": cls.BROCARDI_CODICE_PENALE,
            }
            template = act_map.get(act_type.lower())
            if template:
                return template.format(article=article)
            return f"{cls.BROCARDI_BASE}/cerca?q={act_type}+art+{article}"

        @classmethod
        def eurlex_celex_url(cls, celex: str, lang: str = "IT") -> str:
            """Build EUR-Lex URL for a CELEX ID."""
            return cls.EURLEX_CELEX.format(celex=celex, lang=lang)
