"""
LEXE Knowledge Base - Hierarchical Parser

Parsing gerarchico deterministico per massimari Cassazione.
Struttura: Volume > Parte > Capitolo > Sezione > Sottosezione > Massima
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

from .extractor import ExtractedElement

logger = structlog.get_logger(__name__)


# ============================================================
# Pattern Recognition per Struttura Gerarchica
# ============================================================

# Pattern per identificare livelli gerarchici
HIERARCHY_PATTERNS = {
    # Livello 1: PARTE
    "parte": re.compile(
        r"^(?:PARTE|Parte)\s+([IVXLCDM]+|[0-9]+)\s*[-–—]?\s*(.+)?$",
        re.IGNORECASE,
    ),
    # Livello 2: CAPITOLO
    "capitolo": re.compile(
        r"^(?:CAPITOLO|Capitolo|Cap\.?)\s+([IVXLCDM]+|[0-9]+)\s*[-–—]?\s*(.+)?$",
        re.IGNORECASE,
    ),
    # Livello 3: SEZIONE
    "sezione": re.compile(
        r"^(?:SEZIONE|Sezione|Sez\.?)\s+([IVXLCDM]+|[0-9]+)\s*[-–—]?\s*(.+)?$",
        re.IGNORECASE,
    ),
    # Livello 4: SOTTOSEZIONE / PARAGRAFO
    "sottosezione": re.compile(
        r"^(?:SOTTOSEZIONE|Sottosezione|§|Par\.?)\s*([0-9]+(?:\.[0-9]+)?)\s*[-–—]?\s*(.+)?$",
        re.IGNORECASE,
    ),
}

# Pattern per numeri romani
ROMAN_NUMERALS = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
}


@dataclass
class SectionNode:
    """Nodo della struttura gerarchica."""

    level: int  # 1=parte, 2=capitolo, 3=sezione, 4=sottosezione
    type_name: str  # "parte", "capitolo", etc.
    number: str  # "I", "1", "2.3", etc.
    title: str
    page_start: int | None = None
    page_end: int | None = None
    parent: Optional["SectionNode"] = None
    children: list["SectionNode"] = field(default_factory=list)
    elements: list[ExtractedElement] = field(default_factory=list)

    @property
    def section_path(self) -> str:
        """Path completo dalla root (es. 'PARTE I > Cap. 2 > Sez. 1')."""
        parts = []
        node: SectionNode | None = self
        while node:
            if node.type_name == "parte":
                parts.append(f"PARTE {node.number}")
            elif node.type_name == "capitolo":
                parts.append(f"Cap. {node.number}")
            elif node.type_name == "sezione":
                parts.append(f"Sez. {node.number}")
            elif node.type_name == "sottosezione":
                parts.append(f"§{node.number}")
            node = node.parent
        return " > ".join(reversed(parts))

    @property
    def full_title(self) -> str:
        """Titolo completo con tipo e numero."""
        type_labels = {
            "parte": "PARTE",
            "capitolo": "CAPITOLO",
            "sezione": "SEZIONE",
            "sottosezione": "§",
        }
        label = type_labels.get(self.type_name, self.type_name.upper())
        if self.title:
            return f"{label} {self.number} - {self.title}"
        return f"{label} {self.number}"


@dataclass
class ParsedDocument:
    """Documento parsato con struttura gerarchica."""

    root_sections: list[SectionNode]
    all_sections: list[SectionNode]
    orphan_elements: list[ExtractedElement]  # Elementi senza sezione
    total_pages: int = 0

    def get_section_by_path(self, path: str) -> SectionNode | None:
        """Trova sezione per path."""
        for section in self.all_sections:
            if section.section_path == path:
                return section
        return None

    def get_sections_at_level(self, level: int) -> list[SectionNode]:
        """Ottieni tutte le sezioni a un livello."""
        return [s for s in self.all_sections if s.level == level]


def roman_to_int(roman: str) -> int:
    """Converti numero romano in intero."""
    return ROMAN_NUMERALS.get(roman.upper(), 0)


def normalize_section_number(number: str) -> str:
    """Normalizza numero sezione (romano -> arabo se necessario)."""
    number = number.strip().upper()
    if number in ROMAN_NUMERALS:
        return str(ROMAN_NUMERALS[number])
    return number


def detect_section_type(text: str) -> tuple[str, str, str] | None:
    """
    Rileva tipo di sezione dal testo.

    Returns:
        Tuple (type_name, number, title) o None se non e' un header
    """
    text = text.strip()

    # Prova ogni pattern in ordine di priorita' (dal piu' specifico)
    for type_name, pattern in HIERARCHY_PATTERNS.items():
        match = pattern.match(text)
        if match:
            number = match.group(1)
            title = match.group(2) if match.lastindex >= 2 else ""
            return type_name, number, (title or "").strip()

    return None


def is_likely_section_header(element: ExtractedElement) -> bool:
    """
    Euristica per identificare header di sezione.

    Considera:
    - Categoria "Title" da unstructured
    - Lunghezza testo (header sono corti)
    - Pattern riconosciuto
    """
    if not element.text:
        return False

    text = element.text.strip()

    # Header sono tipicamente corti
    if len(text) > 200:
        return False

    # Se unstructured l'ha marcato come Title
    if element.category == "Title":
        return True

    # Se matcha un pattern gerarchico
    return bool(detect_section_type(text))


def parse_document_structure(
    elements: list[ExtractedElement],
) -> ParsedDocument:
    """
    Parsa struttura gerarchica da elementi estratti.

    Args:
        elements: Lista elementi da extractor

    Returns:
        ParsedDocument con struttura gerarchica
    """
    root_sections: list[SectionNode] = []
    all_sections: list[SectionNode] = []
    orphan_elements: list[ExtractedElement] = []

    # Stack per tracciare contesto gerarchico corrente
    # {level: SectionNode}
    current_context: dict[int, SectionNode] = {}
    current_section: SectionNode | None = None
    max_page = 0

    for element in elements:
        # Aggiorna max page
        if element.page_number and element.page_number > max_page:
            max_page = element.page_number

        # Verifica se e' un header di sezione
        section_info = detect_section_type(element.text) if element.text else None

        if section_info:
            type_name, number, title = section_info
            level = ["parte", "capitolo", "sezione", "sottosezione"].index(type_name) + 1

            # Crea nuovo nodo
            new_section = SectionNode(
                level=level,
                type_name=type_name,
                number=number,
                title=title,
                page_start=element.page_number,
            )

            # Trova parent appropriato
            parent = None
            for parent_level in range(level - 1, 0, -1):
                if parent_level in current_context:
                    parent = current_context[parent_level]
                    break

            if parent:
                new_section.parent = parent
                parent.children.append(new_section)
            else:
                # Nodo root
                root_sections.append(new_section)

            # Aggiorna contesto
            current_context[level] = new_section
            # Rimuovi contesti di livello inferiore
            for lvl in list(current_context.keys()):
                if lvl > level:
                    del current_context[lvl]

            # Chiudi pagina sezione precedente allo stesso livello
            if (
                current_section
                and current_section.level == level
                and element.page_number
                and current_section.page_start
            ):
                current_section.page_end = element.page_number - 1

            current_section = new_section
            all_sections.append(new_section)

            logger.debug(
                "Section detected",
                type=type_name,
                number=number,
                title=title[:50] if title else "",
                page=element.page_number,
                path=new_section.section_path,
            )

        else:
            # Elemento normale - assegna alla sezione corrente
            if current_section:
                current_section.elements.append(element)
            else:
                orphan_elements.append(element)

    # Chiudi ultima sezione
    if current_section and max_page:
        current_section.page_end = max_page

    # Propaga page_end ai parent
    for section in reversed(all_sections):
        if section.children:
            max_child_page = max((c.page_end or c.page_start or 0) for c in section.children)
            if max_child_page and (not section.page_end or max_child_page > section.page_end):
                section.page_end = max_child_page

    logger.info(
        "Document structure parsed",
        root_sections=len(root_sections),
        total_sections=len(all_sections),
        orphan_elements=len(orphan_elements),
        total_pages=max_page,
    )

    return ParsedDocument(
        root_sections=root_sections,
        all_sections=all_sections,
        orphan_elements=orphan_elements,
        total_pages=max_page,
    )


def get_section_context(section: SectionNode, include_siblings: bool = False) -> str:
    """
    Ottieni contesto testuale per una sezione.

    Args:
        section: Sezione di riferimento
        include_siblings: Includere sezioni sorelle

    Returns:
        Stringa con contesto editoriale
    """
    parts = []

    # Aggiungi path gerarchico
    parts.append(f"[{section.section_path}]")

    # Aggiungi titolo completo
    parts.append(section.full_title)

    # Se richiesto, aggiungi sibling titles
    if include_siblings and section.parent:
        siblings = [s for s in section.parent.children if s != section]
        if siblings:
            sibling_titles = ", ".join(s.full_title for s in siblings[:3])
            parts.append(f"(Sezioni correlate: {sibling_titles})")

    return " | ".join(parts)


def flatten_sections_with_elements(
    parsed: ParsedDocument,
) -> list[tuple[SectionNode | None, ExtractedElement]]:
    """
    Appiattisci struttura in lista (sezione, elemento).

    Utile per iterare su tutti gli elementi mantenendo contesto.
    """
    result: list[tuple[SectionNode | None, ExtractedElement]] = []

    # Prima gli orfani
    for elem in parsed.orphan_elements:
        result.append((None, elem))

    # Poi ricorsivamente le sezioni
    def process_section(section: SectionNode) -> None:
        for elem in section.elements:
            result.append((section, elem))
        for child in section.children:
            process_section(child)

    for root in parsed.root_sections:
        process_section(root)

    return result


def extract_toc_from_first_pages(
    elements: list[ExtractedElement],
    max_pages: int = 10,
) -> list[tuple[str, int]]:
    """
    Estrai indice (TOC) dalle prime pagine.

    Molti massimari hanno TOC con "Titolo ... pagina".

    Returns:
        Lista di (titolo, pagina)
    """
    toc_entries: list[tuple[str, int]] = []

    # Pattern per entry TOC: "Titolo ..... 123" o "Titolo - 123"
    toc_pattern = re.compile(r"^(.+?)[\s.…]+(\d{1,4})\s*$")

    for elem in elements:
        if elem.page_number and elem.page_number > max_pages:
            break

        if not elem.text:
            continue

        # Cerca pattern TOC
        for line in elem.text.split("\n"):
            match = toc_pattern.match(line.strip())
            if match:
                title = match.group(1).strip()
                page = int(match.group(2))
                # Filtra entry troppo corte o numeri di pagina improbabili
                if len(title) > 5 and page > 0:
                    toc_entries.append((title, page))

    logger.debug(
        "TOC entries extracted",
        count=len(toc_entries),
    )

    return toc_entries
