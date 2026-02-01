#!/usr/bin/env python3
"""
Smart PDF Split - Find chapter boundaries for splitting large PDFs.

Trova i punti di divisione intelligenti nei PDF grandi (>300 pagine)
basandosi su pattern di inizio capitolo/sezione.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF


# Pattern per identificare inizio capitolo/sezione
CHAPTER_PATTERNS = [
    r"^SEZIONE\s+[IVXLCDM]+",           # SEZIONE I, SEZIONE II, etc.
    r"^Sez\.\s*[IVXLCDM]+",              # Sez. I, Sez. II, etc.
    r"^CAPITOLO\s+[IVXLCDM\d]+",         # CAPITOLO I, CAPITOLO 1
    r"^Cap\.\s*[IVXLCDM\d]+",            # Cap. I, Cap. 1
    r"^PARTE\s+[IVXLCDM]+",              # PARTE I, PARTE PRIMA
    r"^TITOLO\s+[IVXLCDM]+",             # TITOLO I, TITOLO II
    r"^LIBRO\s+[IVXLCDM]+",              # LIBRO I, LIBRO II
    r"^\d+\.\s+[A-Z]{2,}",               # 1. DIRITTO CIVILE
    r"^[IVXLCDM]+\s*[\.\-]\s*[A-Z]",     # I. DIRITTO, I - DIRITTO
]


def sanitize_for_print(text: str) -> str:
    """Rimuovi caratteri non-ASCII per evitare errori di encoding."""
    return text.encode('ascii', 'replace').decode('ascii')


def find_chapter_boundaries(pdf_path: Path, verbose: bool = True) -> list[dict]:
    """
    Trova i VERI inizi di capitolo nel PDF (non header ripetuti).

    Returns:
        Lista di dict con: page_num, line, pattern_matched
    """
    boundaries = []
    seen_chapters = {}  # Traccia capitoli già visti per evitare header ripetuti

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if verbose:
        print(f"PDF: {pdf_path.name}")
        print(f"Pagine totali: {total_pages}")
        print(f"Cercando chapter boundaries (solo primi inizi, no header ripetuti)...")
        print()

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()

        # Controlla le prime 15 righe di ogni pagina
        lines = text.strip().split('\n')[:15]

        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or len(line_clean) < 3:
                continue

            for pattern in CHAPTER_PATTERNS:
                if re.match(pattern, line_clean, re.IGNORECASE):
                    # Estrai identificatore capitolo (es. "CAPITOLO XIII", "Sez. IV")
                    chapter_id = extract_chapter_id(line_clean)

                    # Solo se NON abbiamo già visto questo capitolo
                    # O se è un capitolo "semplice" (senza sottotitolo) - probabile vero inizio
                    is_simple = len(line_clean) < 30 and ' - ' not in line_clean

                    if chapter_id not in seen_chapters:
                        seen_chapters[chapter_id] = page_num + 1
                        boundaries.append({
                            'page': page_num + 1,  # 1-based
                            'line': line_clean[:80],
                            'pattern': pattern,
                            'chapter_id': chapter_id,
                            'is_simple': is_simple,
                        })
                    break  # Una sola match per riga

    doc.close()

    if verbose:
        print(f"Trovati {len(boundaries)} chapter boundaries UNICI:")
        for b in boundaries:
            marker = "*" if b.get('is_simple') else " "
            print(f"  {marker} Pag {b['page']:4d}: {sanitize_for_print(b['line'])}")
        print()
        print("(* = titolo semplice, probabile vero inizio capitolo)")
        print()

    return boundaries


def extract_chapter_id(line: str) -> str:
    """Estrae l'identificatore del capitolo dalla riga (es. 'CAPITOLO XIII')."""
    # Prova a estrarre "CAPITOLO X", "Sez. I", "PARTE II", etc.
    patterns = [
        r"(CAPITOLO\s+[IVXLCDM\d]+)",
        r"(Cap\.\s*[IVXLCDM\d]+)",
        r"(SEZIONE\s+[IVXLCDM]+)",
        r"(Sez\.\s*[IVXLCDM]+)",
        r"(PARTE\s+[IVXLCDM]+)",
        r"(TITOLO\s+[IVXLCDM]+)",
        r"(LIBRO\s+[IVXLCDM]+)",
        r"(\d+\.\s+[A-Z]{2,})",
    ]

    for p in patterns:
        match = re.search(p, line, re.IGNORECASE)
        if match:
            return match.group(1).upper().strip()

    # Fallback: prime 20 chars
    return line[:20].upper().strip()


def find_best_split_point(pdf_path: Path, verbose: bool = True, only_chapters: bool = True) -> dict | None:
    """
    Trova il miglior punto di split vicino alla metà del PDF.

    Args:
        only_chapters: Se True, considera solo veri inizi capitolo (CAPITOLO, Sez., etc.)
                       non paragrafi numerati

    Returns:
        Dict con: page, line, distance_from_middle, part1_pages, part2_pages
    """
    boundaries = find_chapter_boundaries(pdf_path, verbose=verbose)

    if not boundaries:
        print("ERRORE: Nessun chapter boundary trovato!")
        return None

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    middle = total_pages / 2

    # Filtra: solo veri inizi capitolo se richiesto
    if only_chapters:
        # Solo entries che iniziano con CAPITOLO, SEZIONE, PARTE, TITOLO, LIBRO
        chapter_keywords = ['CAPITOLO', 'SEZIONE', 'SEZ.', 'PARTE', 'TITOLO', 'LIBRO']
        valid_boundaries = [
            b for b in boundaries
            if any(b['line'].upper().startswith(kw) for kw in chapter_keywords)
        ]
        if verbose and valid_boundaries:
            print(f"Filtrati {len(valid_boundaries)} VERI inizi capitolo:")
            for b in valid_boundaries:
                print(f"  Pag {b['page']:4d}: {sanitize_for_print(b['line'])}")
            print()
    else:
        valid_boundaries = boundaries

    if not valid_boundaries:
        print("ERRORE: Nessun vero inizio capitolo trovato!")
        return None

    # Trova il boundary più vicino alla metà
    best = None
    best_distance = float('inf')

    for b in valid_boundaries:
        distance = abs(b['page'] - middle)
        if distance < best_distance:
            best_distance = distance
            best = b

    if best:
        result = {
            'page': best['page'],
            'line': best['line'],
            'distance_from_middle': best_distance,
            'part1_pages': best['page'] - 1,
            'part2_pages': total_pages - best['page'] + 1,
            'total_pages': total_pages,
        }

        if verbose:
            print("=" * 60)
            print("MIGLIOR PUNTO DI SPLIT")
            print("=" * 60)
            print(f"Pagina: {result['page']} (meta ideale: {middle:.0f})")
            print(f"Distanza dalla meta: {result['distance_from_middle']:.1f} pagine")
            print(f"Titolo: {sanitize_for_print(result['line'])}")
            print()
            print(f"PARTE 1: pagine 1-{result['part1_pages']} ({result['part1_pages']} pagine)")
            print(f"PARTE 2: pagine {result['page']}-{result['total_pages']} ({result['part2_pages']} pagine)")
            print()

            # Warning se una parte è ancora > 300
            max_part = max(result['part1_pages'], result['part2_pages'])
            if max_part > 300:
                print(f"[!] ATTENZIONE: Una parte ha {max_part} pagine (>300)")
                print("    Potrebbe servire un secondo split.")

        return result

    return None


def analyze_large_pdfs(pdf_dir: Path, min_pages: int = 300) -> list[dict]:
    """
    Analizza tutti i PDF grandi in una directory.

    Returns:
        Lista di risultati per PDF > min_pages
    """
    results = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    print(f"Analizzando {len(pdf_files)} PDF in {pdf_dir}")
    print(f"Filtro: solo PDF con > {min_pages} pagine")
    print("=" * 60)
    print()

    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        if total_pages <= min_pages:
            continue

        print(f"\n{'='*60}")
        print(f"PDF GRANDE: {pdf_path.name} ({total_pages} pagine)")
        print(f"{'='*60}\n")

        result = find_best_split_point(pdf_path, verbose=True)
        if result:
            result['filename'] = pdf_path.name
            results.append(result)

    # Summary finale
    print("\n" + "=" * 60)
    print("SUMMARY - PDF CHE RICHIEDONO SPLIT")
    print("=" * 60)

    for r in results:
        print(f"\n{r['filename']}:")
        print(f"  Split a pagina {r['page']}: {sanitize_for_print(r['line'][:50])}...")
        print(f"  Parti: {r['part1_pages']} + {r['part2_pages']} pagine")

    return results


def normalize_filename(name: str) -> str:
    """Normalizza nome file: rimuovi 'pagg XXX', spazi -> underscore."""
    # Rimuovi 'pagg XXX' o 'pag XXX'
    name = re.sub(r'\s*pagg?\s*\d+', '', name)
    # Rimuovi caratteri problematici
    name = re.sub(r'[()[\]{}]', '', name)
    # Spazi -> underscore
    name = name.replace(' ', '_')
    # Underscore multipli -> singolo
    name = re.sub(r'_+', '_', name)
    # Rimuovi underscore finali prima di estensione
    name = re.sub(r'_+\.', '.', name)
    return name.strip('_')


def split_pdf(pdf_path: Path, split_page: int, output_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Divide il PDF in due parti al punto specificato.

    Args:
        pdf_path: Path del PDF originale
        split_page: Numero pagina (1-based) dove inizia la parte 2
        output_dir: Directory output (default: stessa del PDF originale)

    Returns:
        Tuple con i path dei due PDF creati (part1, part2)
    """
    if output_dir is None:
        output_dir = pdf_path.parent

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if split_page < 2 or split_page > total_pages:
        raise ValueError(f"split_page deve essere tra 2 e {total_pages}")

    stem = normalize_filename(pdf_path.stem)

    # Part 1: pagine 0 to split_page-2 (0-indexed)
    part1_path = output_dir / f"{stem}_PART1.pdf"
    part1 = fitz.open()
    part1.insert_pdf(doc, from_page=0, to_page=split_page - 2)
    part1.save(part1_path)
    part1.close()

    # Part 2: pagine split_page-1 to end (0-indexed)
    part2_path = output_dir / f"{stem}_PART2.pdf"
    part2 = fitz.open()
    part2.insert_pdf(doc, from_page=split_page - 1, to_page=total_pages - 1)
    part2.save(part2_path)
    part2.close()

    doc.close()

    print(f"PDF diviso con successo!")
    print(f"  PART1: {part1_path.name} ({split_page - 1} pagine)")
    print(f"  PART2: {part2_path.name} ({total_pages - split_page + 1} pagine)")

    return part1_path, part2_path


def split_pdf_smart(pdf_path: Path, output_dir: Path | None = None) -> tuple[Path, Path] | None:
    """
    Trova automaticamente il miglior punto di split e divide il PDF.

    Returns:
        Tuple con i path dei due PDF creati, o None se split non necessario/possibile
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    if total_pages <= 300:
        print(f"PDF ha solo {total_pages} pagine, split non necessario")
        return None

    result = find_best_split_point(pdf_path, verbose=True)

    if not result:
        print("Impossibile trovare un punto di split valido")
        return None

    return split_pdf(pdf_path, result['page'], output_dir)


def test_single_pdf(pdf_path: Path):
    """Test su un singolo PDF."""
    print(f"Analisi split per: {pdf_path.name}")
    print("=" * 60)
    print()

    result = find_best_split_point(pdf_path, verbose=True)

    if result:
        print("\nPuoi verificare manualmente nel PDF:")
        print(f"  - Apri pagina {result['page']}")
        print(f"  - Verifica che '{sanitize_for_print(result['line'][:40])}...' sia un inizio capitolo")
        print(f"  - Verifica che il contenuto prima sia completo")

    return result


if __name__ == "__main__":
    import sys

    # Default: directory massimari su staging
    pdf_dir = Path(r"C:\PROJECTS\lexe-genesis\data\raccolta")

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        path = Path(arg)

        if path.is_file() and path.suffix.lower() == '.pdf':
            # Singolo PDF
            test_single_pdf(path)
        elif path.is_dir():
            # Directory
            analyze_large_pdfs(path)
        else:
            print(f"Errore: {arg} non è un PDF o directory valida")
            sys.exit(1)
    else:
        # Analizza tutti i PDF grandi nella directory di default
        if pdf_dir.exists():
            analyze_large_pdfs(pdf_dir)
        else:
            print(f"Directory non trovata: {pdf_dir}")
            print("Uso: python smart_pdf_split.py [pdf_file|pdf_dir]")
