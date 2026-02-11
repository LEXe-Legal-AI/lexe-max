#!/usr/bin/env python
"""
Generate Quality Report

Applica classificazione qualità ai JSON estratti e genera:
1. Tabella standard "dal, al, totale, validi, deboli, invalidi"
2. JSON aggiornati con campo 'quality' per ogni articolo
3. Report riassuntivo CSV

Usage:
    uv run --no-sync python scripts/generate_quality_report.py [--root DIR] [--update-json]
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

from article_quality import (
    classify_articles_batch,
    generate_quality_summary,
    ArticleQuality,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_document(json_path: Path, update_json: bool = False) -> dict:
    """
    Processa un documento JSON e genera report qualità.

    Args:
        json_path: Path al file JSON estratto
        update_json: Se True, aggiorna il JSON con le classificazioni

    Returns:
        Dict con metriche di qualità
    """
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get('articles', [])
    codice = data.get('codice', json_path.stem.split('.')[0])

    if not articles:
        return {
            'json_path': str(json_path),
            'pdf_name': json_path.name.replace('.llm_extracted.json', '.pdf'),
            'codice': codice,
            'dal': 0,
            'al': 0,
            'totale': 0,
            'validi': 0,
            'deboli': 0,
            'invalidi': 0,
            'by_warning_type': {},
            'weak_articles': [],
            'invalid_articles': [],
        }

    # Classifica articoli
    classified = classify_articles_batch(articles)
    summary = generate_quality_summary(classified)

    # Raccogli articoli problematici
    weak_articles = []
    invalid_articles = []

    for a in classified:
        q = a.get('quality', {})
        quality = q.get('quality', 'VALID')

        if quality == 'WEAK':
            weak_articles.append({
                'articolo_num': a.get('articolo_num'),
                'warnings': q.get('warnings', []),
            })
        elif quality == 'INVALID':
            invalid_articles.append({
                'articolo_num': a.get('articolo_num'),
                'warnings': q.get('warnings', []),
                'text_preview': (a.get('testo', '') or '')[:100],
            })

    # Aggiorna JSON se richiesto
    if update_json:
        data['articles'] = classified
        data['quality_summary'] = summary
        data['quality_generated'] = datetime.now().isoformat()

        with json_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Updated: {json_path.name}")

    return {
        'json_path': str(json_path),
        'pdf_name': json_path.name.replace('.llm_extracted.json', '.pdf'),
        'codice': codice,
        **summary,
        'weak_articles': weak_articles,
        'invalid_articles': invalid_articles,
    }


def generate_report(root: Path, update_json: bool = False) -> list[dict]:
    """
    Genera report per tutti i JSON nella directory.

    Args:
        root: Directory root da cercare
        update_json: Se True, aggiorna i JSON con le classificazioni

    Returns:
        Lista di report per ogni documento
    """
    json_files = list(root.rglob('*.llm_extracted.json'))
    # Escludi i .old.json
    json_files = [f for f in json_files if '.old.' not in f.name]

    logger.info(f"Trovati {len(json_files)} file JSON")

    if not json_files:
        return []

    reports = []
    for json_path in sorted(json_files):
        try:
            report = process_document(json_path, update_json)
            reports.append(report)
        except Exception as e:
            logger.error(f"Errore processando {json_path}: {e}")

    return reports


def print_table(reports: list[dict]):
    """Stampa tabella formattata."""
    print()
    print('=' * 100)
    print(f'{"Documento":<35} {"dal":>5} {"al":>6} {"totale":>7} {"validi":>7} {"deboli":>7} {"invalidi":>9}')
    print('=' * 100)

    for r in reports:
        name = r['pdf_name'][:35]
        print(f'{name:<35} {r["dal"]:>5} {r["al"]:>6} {r["totale"]:>7} {r["validi"]:>7} {r["deboli"]:>7} {r["invalidi"]:>9}')

    print('=' * 100)

    # Totali
    total_tot = sum(r['totale'] for r in reports)
    total_val = sum(r['validi'] for r in reports)
    total_weak = sum(r['deboli'] for r in reports)
    total_inv = sum(r['invalidi'] for r in reports)

    print(f'{"TOTALE":<35} {"":>5} {"":>6} {total_tot:>7} {total_val:>7} {total_weak:>7} {total_inv:>9}')
    print()

    # Percentuali
    if total_tot > 0:
        pct_val = total_val / total_tot * 100
        pct_weak = total_weak / total_tot * 100
        pct_inv = total_inv / total_tot * 100
        print(f'Percentuali: {pct_val:.1f}% validi, {pct_weak:.1f}% deboli, {pct_inv:.1f}% invalidi')


def save_csv(reports: list[dict], output_path: Path):
    """Salva report in CSV."""
    fieldnames = [
        'pdf_name', 'codice', 'dal', 'al', 'totale', 'validi', 'deboli', 'invalidi'
    ]

    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', extrasaction='ignore')
        writer.writeheader()
        for r in reports:
            writer.writerow(r)

    logger.info(f"CSV: {output_path}")


def save_json_report(reports: list[dict], output_path: Path):
    """Salva report dettagliato in JSON."""
    output = {
        'generated': datetime.now().isoformat(),
        'total_documents': len(reports),
        'summary': {
            'total_articles': sum(r['totale'] for r in reports),
            'total_valid': sum(r['validi'] for r in reports),
            'total_weak': sum(r['deboli'] for r in reports),
            'total_invalid': sum(r['invalidi'] for r in reports),
        },
        'reports': reports,
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Quality Report')
    parser.add_argument('--root', default=r'C:\PROJECTS\lexe-genesis\altalex pdf',
                       help='Directory root con i JSON estratti')
    parser.add_argument('--output-dir', default=None,
                       help='Directory output (default: stesso di root)')
    parser.add_argument('--update-json', action='store_true',
                       help='Aggiorna i JSON con le classificazioni di qualità')

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root

    # Genera report
    reports = generate_report(root, args.update_json)

    if not reports:
        logger.warning('Nessun documento trovato')
        return

    # Stampa tabella
    print_table(reports)

    # Mostra articoli problematici
    print('\n--- ARTICOLI DEBOLI ---')
    for r in reports:
        if r['weak_articles']:
            print(f"\n{r['codice']} ({len(r['weak_articles'])} deboli):")
            for a in r['weak_articles'][:5]:
                warnings_str = ', '.join(w['type'] for w in a['warnings'])
                print(f"  Art. {a['articolo_num']}: {warnings_str}")
            if len(r['weak_articles']) > 5:
                print(f"  ... e altri {len(r['weak_articles']) - 5}")

    print('\n--- ARTICOLI INVALIDI ---')
    for r in reports:
        if r['invalid_articles']:
            print(f"\n{r['codice']} ({len(r['invalid_articles'])} invalidi):")
            for a in r['invalid_articles']:
                warnings_str = ', '.join(w['type'] for w in a['warnings'])
                print(f"  Art. {a['articolo_num']}: {warnings_str}")
                print(f"    Preview: {a['text_preview'][:60]}...")

    # Salva output
    save_csv(reports, output_dir / 'lexe_quality_report.csv')
    save_json_report(reports, output_dir / 'lexe_quality_report.json')


if __name__ == '__main__':
    main()
