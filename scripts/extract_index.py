"""
Extract Index/TOC from Massimari PDFs
Estrae la struttura dei capitoli e materie dall'indice.
"""
import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

import asyncpg
import httpx

# Config
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
PDF_DIR = Path("C:/Users/Fra/Documents/lexe/collezione zecca/New folder (2)/Massimario_PDF/")

TEST_FILES = [
    "2021_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2023_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2018_MASSIMARIO CIVILE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTIDELLE SEZIONI CIVILI.pdf",
]


@dataclass
class Chapter:
    """Rappresenta un capitolo/sezione del massimario."""
    id: str
    level: int  # 0=parte, 1=capitolo, 2=sezione, 3=sottosezione, 4=paragrafo
    number: str  # "PRIMA", "I", "1", "1.1", etc.
    title: str
    chapter_type: str  # "parte", "capitolo", "sezione", "sottosezione", "paragrafo"
    page_start: Optional[int] = None
    parent_id: Optional[str] = None


@dataclass
class IndexStructure:
    """Struttura completa dell'indice."""
    document_title: str
    anno: int
    tipo: str  # civile/penale
    volume: int
    chapters: list[Chapter]


def extract_year_and_type(filename: str) -> tuple[int, str, int]:
    """Extract year, type, and volume from filename."""
    match = re.match(r"(\d{4})_MASSIMARIO\s+(CIVILE|PENALE)\s+VOL\.?\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower(), int(match.group(3))
    return 2020, "unknown", 1


def parse_chapter_number(text: str) -> tuple[Optional[str], int, str]:
    """Parse chapter number and determine level.

    Returns (number, level, type) where:
    - level 0 = PARTE (PARTE PRIMA, SECONDA...)
    - level 1 = CAPITOLO (CAPITOLO I, II, III...)
    - level 2 = Sezione principale (1., 2., 3.)
    - level 3 = Sottosezione (1.1., 1.2.)
    - level 4 = Paragrafo (1.1.1.)
    - type = "parte", "capitolo", "sezione", "sottosezione", "paragrafo"
    """
    text = text.strip()

    # PARTE PRIMA, SECONDA... (highest level)
    match = re.match(r"PARTE\s+(\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), 0, "parte"

    # CAPITOLO I, II, III... (second level)
    match = re.match(r"CAPITOLO\s+([IVXLC]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), 1, "capitolo"

    # Sottosezione 1.1., 1.2., etc
    match = re.match(r"^(\d+\.\d+(?:\.\d+)*)\.", text)
    if match:
        num = match.group(1)
        dots = num.count('.')
        if dots == 1:
            return num, 3, "sottosezione"
        else:
            return num, 4, "paragrafo"

    # Simple number: 1., 2., 3. (section within chapter)
    match = re.match(r"^(\d+)\.", text)
    if match:
        return match.group(1), 2, "sezione"

    return None, -1, ""


def parse_page_number(text: str) -> Optional[int]:
    """Extract page number from TOC line (after dots)."""
    # Pattern: ". . . . . 123" or "...123" at end
    match = re.search(r"[.\s]+(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return None


def clean_title(text: str) -> str:
    """Clean title by removing page numbers, dots, and extra content."""
    # First, split on multiple dots followed by number (TOC pattern)
    # This handles "Premessa . . . . . 37 2. Il contrasto..."
    parts = re.split(r"\s*\.[\s.]+\d+\s*\d*\.", text)
    if parts:
        text = parts[0]  # Take only the first part (the actual title)

    # Remove trailing dots and page numbers
    text = re.sub(r"[.\s]+\d+\s*$", "", text)
    # Remove leading CAPITOLO/PARTE markers
    text = re.sub(r"^(CAPITOLO|caPITOLO)\s+[IVXLC]+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(PARTE)\s+\w+\s*", "", text, flags=re.IGNORECASE)
    # Remove leading numbers like "1.", "2.3."
    text = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", text)
    # Remove isolated dots
    text = re.sub(r"\.{2,}", "", text)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def extract_with_unstructured(pdf_path: Path) -> list[dict]:
    """Extract PDF elements using Unstructured API."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            data = {"strategy": "fast", "languages": ["ita"]}
            response = await client.post(UNSTRUCTURED_URL, files=files, data=data)

    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return []

    return response.json()


def find_toc_elements(elements: list[dict]) -> list[dict]:
    """Find elements that are part of the Table of Contents."""
    toc_elements = []

    # TOC is usually in first 20-30 pages, look for ListItems and Titles
    for e in elements:
        page = e.get("metadata", {}).get("page_number", 999)
        etype = e.get("type", "")
        text = e.get("text", "")

        # Skip if not in TOC range (usually pages 8-30)
        if page < 8 or page > 40:
            continue

        # Look for TOC patterns
        if etype in ("ListItem", "Title"):
            # Has dots pattern (TOC indicator)
            if ". . ." in text or re.search(r"\.\s*\d+\s*$", text):
                toc_elements.append(e)
            # Or is a chapter/section header
            elif re.match(r"^(CAPITOLO|PARTE|SEZIONE|\d+\.)", text, re.IGNORECASE):
                toc_elements.append(e)

    return toc_elements


def parse_toc_to_chapters(toc_elements: list[dict]) -> list[Chapter]:
    """Parse TOC elements into chapter structure."""
    chapters = []
    parent_stack = []  # Stack of (level, chapter_id) for parent tracking
    seen_titles = set()  # Avoid duplicates

    for e in toc_elements:
        text = e.get("text", "").strip()

        if not text or len(text) < 3:
            continue

        # Parse chapter number, level and type
        number, level, chapter_type = parse_chapter_number(text)

        if level < 0:
            continue  # Skip non-chapter elements

        # Extract title and page
        title = clean_title(text)
        page_num = parse_page_number(text)

        # Skip if no meaningful title
        if not title or len(title) < 3:
            # For PARTE/CAPITOLO, use type as title if no other title
            if chapter_type in ("parte", "capitolo"):
                title = f"{chapter_type.upper()} {number}"
            else:
                continue

        # Skip duplicates (same level + number + similar title)
        dedup_key = f"{level}:{number}:{title[:30]}"
        if dedup_key in seen_titles:
            continue
        seen_titles.add(dedup_key)

        # Determine parent
        parent_id = None
        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()
        if parent_stack:
            parent_id = parent_stack[-1][1]

        # Create chapter
        chapter = Chapter(
            id=str(uuid4()),
            level=level,
            number=number or "",
            title=title[:500],  # Limit title length
            chapter_type=chapter_type,
            page_start=page_num,
            parent_id=parent_id,
        )
        chapters.append(chapter)

        # Push to parent stack
        parent_stack.append((level, chapter.id))

    return chapters


async def extract_index(pdf_path: Path) -> IndexStructure:
    """Extract complete index structure from PDF."""
    filename = pdf_path.name
    anno, tipo, volume = extract_year_and_type(filename)

    print(f"\nExtracting index from: {filename[:60]}...")
    print(f"  Anno: {anno}, Tipo: {tipo}, Volume: {volume}")

    # Extract elements
    elements = await extract_with_unstructured(pdf_path)
    print(f"  Total elements: {len(elements)}")

    # Find TOC elements
    toc_elements = find_toc_elements(elements)
    print(f"  TOC elements: {len(toc_elements)}")

    # Parse to chapters
    chapters = parse_toc_to_chapters(toc_elements)
    print(f"  Chapters found: {len(chapters)}")

    return IndexStructure(
        document_title=filename,
        anno=anno,
        tipo=tipo,
        volume=volume,
        chapters=chapters,
    )


def print_index_tree(index: IndexStructure):
    """Print index as tree structure."""
    print(f"\n{'='*70}")
    print(f"INDEX TREE: {index.document_title[:50]}...")
    print(f"Anno: {index.anno} | Tipo: {index.tipo} | Volume: {index.volume}")
    print("="*70)

    for chapter in index.chapters:
        indent = "  " * chapter.level
        page = f"p.{chapter.page_start}" if chapter.page_start else ""
        type_tag = f"[{chapter.chapter_type[:3].upper()}]" if chapter.chapter_type else ""
        num = f"{chapter.number}." if chapter.number and chapter.chapter_type not in ("parte", "capitolo") else chapter.number
        print(f"{indent}{type_tag} {num} {chapter.title[:50]} {page}")


async def find_document_id(conn: asyncpg.Connection, filename: str) -> Optional[str]:
    """Find document_id by source_path filename."""
    from uuid import UUID
    row = await conn.fetchrow(
        "SELECT id FROM kb.documents WHERE source_path = $1",
        filename
    )
    if row:
        doc_id = row["id"]
        # asyncpg returns UUID object, convert to string for use
        return str(doc_id) if isinstance(doc_id, UUID) else doc_id
    return None


async def save_to_database(conn: asyncpg.Connection, index: IndexStructure, doc_id: str) -> int:
    """Save index structure to database sections table."""
    from uuid import UUID

    # Convert doc_id string to UUID
    doc_uuid = UUID(doc_id)

    # First, delete existing sections for this document
    await conn.execute(
        "DELETE FROM kb.sections WHERE document_id = $1",
        doc_uuid
    )

    saved = 0
    for chapter in index.chapters:
        # Build section_path for hierarchy
        section_path = f"{index.tipo}/{index.anno}/{chapter.chapter_type}/{chapter.number}"

        # Convert UUIDs
        chapter_uuid = UUID(chapter.id)
        parent_uuid = UUID(chapter.parent_id) if chapter.parent_id else None

        try:
            await conn.execute("""
                INSERT INTO kb.sections (id, document_id, parent_id, level, numero, titolo, tipo, pagina_inizio, section_path)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                chapter_uuid,
                doc_uuid,
                parent_uuid,
                chapter.level,
                chapter.number,
                chapter.title,
                chapter.chapter_type,
                chapter.page_start,
                section_path,
            )
            saved += 1
        except Exception as e:
            print(f"  ERROR saving section: {e}")

    return saved


async def main():
    print("="*70)
    print("KB Massimari - Index Extraction")
    print("="*70)

    # Check Unstructured API
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:8500/healthcheck")
            print(f"Unstructured API: {resp.json()['healthcheck']}")
        except Exception as e:
            print(f"ERROR: Unstructured API not available: {e}")
            return

    # Connect to database
    conn = None
    try:
        conn = await asyncpg.connect(DB_URL)
        print(f"Database: Connected to lexe_kb")
    except Exception as e:
        print(f"WARNING: Database not available: {e}")
        print("Continuing without database save...")

    all_indices = []
    total_saved = 0

    for filename in TEST_FILES:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            print(f"SKIP: {filename} not found")
            continue

        index = await extract_index(pdf_path)
        all_indices.append(index)
        print_index_tree(index)

        # Save to database if connected
        if conn:
            doc_id = await find_document_id(conn, filename)
            if doc_id:
                saved = await save_to_database(conn, index, doc_id)
                total_saved += saved
                print(f"  DATABASE: Saved {saved} sections for document {doc_id[:8]}...")
            else:
                print(f"  WARNING: Document not found in database: {filename[:50]}...")

    # Summary
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print("="*70)

    total_chapters = sum(len(i.chapters) for i in all_indices)
    print(f"Documents processed: {len(all_indices)}")
    print(f"Total chapters extracted: {total_chapters}")

    # Type distribution
    types = {}
    for idx in all_indices:
        for ch in idx.chapters:
            types[ch.chapter_type] = types.get(ch.chapter_type, 0) + 1

    print("\nChapters by type:")
    for ctype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ctype.capitalize()}: {count}")

    # Save to JSON for review
    output_path = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/extracted_indices.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = []
    for idx in all_indices:
        output_data.append({
            "document_title": idx.document_title,
            "anno": idx.anno,
            "tipo": idx.tipo,
            "volume": idx.volume,
            "chapters": [asdict(c) for c in idx.chapters],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nJSON saved to: {output_path}")

    if conn:
        print(f"DATABASE: Total sections saved: {total_saved}")
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
