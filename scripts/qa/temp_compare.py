#!/usr/bin/env python3
"""Temporary compare script for A/B test results."""

import asyncio
from collections import Counter

import asyncpg


async def compare():
    conn = await asyncpg.connect('postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb')

    rows = await conn.fetch('''
        SELECT
            a.filename,
            a.doc_type as a_doc_type, b.doc_type as b_doc_type,
            a.profile as a_profile, b.profile as b_profile,
            a.chunking_strategy as a_chunking, b.chunking_strategy as b_chunking,
            a.confidence as a_conf, b.confidence as b_conf,
            a.min_length as a_min_len, b.min_length as b_min_len
        FROM kb.doc_intel_ab_results a
        JOIN kb.doc_intel_ab_results b ON a.manifest_id = b.manifest_id
        WHERE a.run_name = 'A' AND b.run_name = 'B'
        ORDER BY a.filename
    ''')

    total = len(rows)
    print(f'Documenti confrontati: {total}')
    print()

    doc_flips = [r for r in rows if r['a_doc_type'] != r['b_doc_type']]
    prof_flips = [r for r in rows if r['a_profile'] != r['b_profile']]
    chunk_flips = [r for r in rows if r['a_chunking'] != r['b_chunking']]

    print('=== LABEL FLIP RATE ===')
    print(f'doc_type flip:  {len(doc_flips)}/{total} ({100*len(doc_flips)/total:.1f}%)')
    print(f'profile flip:   {len(prof_flips)}/{total} ({100*len(prof_flips)/total:.1f}%)')
    print(f'chunking flip:  {len(chunk_flips)}/{total} ({100*len(chunk_flips)/total:.1f}%)')
    print()

    if prof_flips:
        print('=== PROFILE FLIPS ===')
        for r in prof_flips:
            print(f'  {r["filename"][:45]:45}')
            print(f'    A: {r["a_profile"]}')
            print(f'    B: {r["b_profile"]}')
        print()

    if chunk_flips:
        print('=== CHUNKING FLIPS ===')
        for r in chunk_flips:
            print(f'  {r["filename"][:45]:45}')
            print(f'    A: {r["a_chunking"]}')
            print(f'    B: {r["b_chunking"]}')
        print()

    # Confidence analysis
    conf_diffs = [abs(r['a_conf'] - r['b_conf']) for r in rows if r['a_conf'] and r['b_conf']]
    if conf_diffs:
        avg_diff = sum(conf_diffs) / len(conf_diffs)
        max_diff = max(conf_diffs)
        print('=== CONFIDENCE STABILITY ===')
        print(f'Mean diff:  {avg_diff:.4f}')
        print(f'Max diff:   {max_diff:.4f}')
        print()

    # Distributions
    a_types = Counter(r['a_doc_type'] for r in rows)
    b_types = Counter(r['b_doc_type'] for r in rows)

    print('=== doc_type DISTRIBUTION ===')
    print(f'{"Type":30} {"Run A":>6} {"Run B":>6}')
    print('-' * 44)
    for t in sorted(set(a_types) | set(b_types)):
        print(f'{t:30} {a_types.get(t,0):>6} {b_types.get(t,0):>6}')
    print()

    a_profs = Counter(r['a_profile'] for r in rows)
    b_profs = Counter(r['b_profile'] for r in rows)

    print('=== profile DISTRIBUTION ===')
    print(f'{"Profile":30} {"Run A":>6} {"Run B":>6}')
    print('-' * 44)
    for p in sorted(set(a_profs) | set(b_profs)):
        print(f'{p:30} {a_profs.get(p,0):>6} {b_profs.get(p,0):>6}')
    print()

    # High confidence disagreement
    high_conf_disagree = [
        r for r in rows
        if r['a_conf'] and r['b_conf']
        and r['a_conf'] > 0.8 and r['b_conf'] > 0.8
        and r['a_profile'] != r['b_profile']
    ]

    print('=== HIGH CONFIDENCE DISAGREEMENT (conf > 0.8) ===')
    print(f'Count: {len(high_conf_disagree)}')
    for r in high_conf_disagree[:5]:
        print(f'  {r["filename"][:40]}')
        print(f'    A (conf={r["a_conf"]:.2f}): {r["a_profile"]}')
        print(f'    B (conf={r["b_conf"]:.2f}): {r["b_profile"]}')
    print()

    # RECOMMENDATION
    print('=' * 60)
    print('RACCOMANDAZIONE')
    print('=' * 60)
    total_flips = len(doc_flips) + len(prof_flips)
    if total_flips == 0:
        print('RUN A e B IDENTICI - Usa temperature=0.0 (deterministico)')
    elif total_flips <= total * 0.05:
        print(f'{total_flips} flip su {total} ({100*total_flips/total:.1f}%)')
        print('Differenza minima - Usa temperature=0.0 per riproducibilita')
    else:
        print(f'{total_flips} flip su {total} ({100*total_flips/total:.1f}%)')
        print('Differenza significativa - Usa temperature=0.0')

    await conn.close()


if __name__ == "__main__":
    asyncio.run(compare())
