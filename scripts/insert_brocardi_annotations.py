#!/usr/bin/env python3
"""Insert Brocardi notes into kb.annotation and kb.annotation_link."""

import asyncio
import sys
sys.path.insert(0, 'C:/PROJECTS/lexe-genesis/lexe-max')

import asyncpg
from datetime import datetime
from scripts.brocardi_parser import BrocardiParser

# Config
DB_URL = "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"


def fix_encoding(text: str) -> str:
    """Fix common encoding issues."""
    replacements = {
        '\ufffd': '',
        '�': '',
        '\x92': "'",
        '\x93': '"',
        '\x94': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


async def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to DB...")
    conn = await asyncpg.connect(DB_URL)
    
    # Get Brocardi source_system_id
    brocardi_id = await conn.fetchval(
        "SELECT id FROM kb.source_system WHERE code = 'BROCARDI_OFFLINE'"
    )
    if not brocardi_id:
        print("ERROR: BROCARDI_OFFLINE source_system not found")
        return
    print(f"  Brocardi source_system_id: {brocardi_id}")
    
    # Load work codes -> work_id mapping
    works = await conn.fetch("SELECT id, code FROM kb.work")
    work_map = {w["code"]: w["id"] for w in works}
    print(f"  Works loaded: {len(work_map)}")
    
    # Load normativa -> for matching articles
    normativa_rows = await conn.fetch("""
        SELECT n.id, w.code, n.articolo, n.articolo_num, n.articolo_suffix
        FROM kb.normativa n
        JOIN kb.work w ON w.id = n.work_id
    """)
    
    # Build lookup: (code, base_num, suffix) -> normativa_id
    normativa_map = {}
    for r in normativa_rows:
        key = (r["code"], r["articolo_num"], r["articolo_suffix"] or "")
        normativa_map[key] = r["id"]
    print(f"  Normativa loaded: {len(normativa_map)}")
    
    # Check existing annotations to avoid duplicates
    existing = await conn.fetch("""
        SELECT al.normativa_id, a.source_ref
        FROM kb.annotation_link al
        JOIN kb.annotation a ON a.id = al.annotation_id
        WHERE a.source_system_id = $1
    """, brocardi_id)
    existing_set = {(e["normativa_id"], e["source_ref"]) for e in existing}
    print(f"  Existing annotations: {len(existing_set)}")
    
    # Parse Brocardi files
    parser = BrocardiParser()
    codes = parser.get_available_codes()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing {len(codes)} codes...")
    
    total_inserted = 0
    total_skipped = 0
    total_not_found = 0
    
    for code in codes:
        if code not in work_map:
            print(f"  {code}: skipped (not in DB)")
            continue
        
        articles = parser.get_articles(code)
        code_inserted = 0
        code_skipped = 0
        code_not_found = 0
        
        for art in articles:
            if not art.note:
                continue
            
            # Find normativa_id
            suffix = art.suffix.lower() if art.suffix else ""
            key = (code, art.base_num, suffix)
            normativa_id = normativa_map.get(key)
            
            if not normativa_id:
                code_not_found += 1
                continue
            
            # Insert each note
            for i, note_text in enumerate(art.note):
                note_text = fix_encoding(note_text.strip())
                if not note_text or len(note_text) < 10:
                    continue
                
                source_ref = f"brocardi:{code}:{art.article_num}:note:{i+1}"
                
                # Check if already exists
                if (normativa_id, source_ref) in existing_set:
                    code_skipped += 1
                    continue
                
                # Insert annotation
                ann_id = await conn.fetchval("""
                    INSERT INTO kb.annotation (source_system_id, type, content, source_ref)
                    VALUES ($1, 'NOTE', $2, $3)
                    RETURNING id
                """, brocardi_id, note_text, source_ref)
                
                # Insert link
                await conn.execute("""
                    INSERT INTO kb.annotation_link (annotation_id, normativa_id, relation)
                    VALUES ($1, $2, 'footnote')
                """, ann_id, normativa_id)
                
                code_inserted += 1
                existing_set.add((normativa_id, source_ref))
        
        print(f"  {code}: +{code_inserted} notes, {code_skipped} skipped, {code_not_found} not found")
        total_inserted += code_inserted
        total_skipped += code_skipped
        total_not_found += code_not_found
    
    # Final stats
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === FINAL STATS ===")
    print(f"  Inserted: {total_inserted}")
    print(f"  Skipped (duplicates): {total_skipped}")
    print(f"  Not found in DB: {total_not_found}")
    
    total_ann = await conn.fetchval("SELECT COUNT(*) FROM kb.annotation")
    total_links = await conn.fetchval("SELECT COUNT(*) FROM kb.annotation_link")
    print(f"\n  Total annotations: {total_ann}")
    print(f"  Total annotation_links: {total_links}")
    
    await conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
