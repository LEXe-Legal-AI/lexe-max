#!/usr/bin/env python3
"""Chunk normativa on staging - runs directly on server."""
import re
import psycopg2
from psycopg2.extras import execute_values

TARGET_CHARS = 1000
OVERLAP_CHARS = 150
MIN_CHUNK_LEN = 30

def normalize(text):
    if not text: return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def find_split(text, target, window=100):
    start = max(0, target - window)
    end = min(len(text), target + window)
    search = text[start:end]
    idx = search.rfind("\n\n")
    if idx != -1: return start + idx + 2
    for p in [". ", ".\n"]:
        idx = search.rfind(p)
        if idx != -1: return start + idx + len(p)
    idx = search.rfind(", ")
    if idx != -1: return start + idx + 2
    idx = search.rfind(" ")
    if idx != -1: return start + idx + 1
    return target

def create_chunks(text):
    text = normalize(text)
    if len(text) < MIN_CHUNK_LEN: return []
    chunks, pos, num = [], 0, 0
    while pos < len(text):
        end = min(pos + TARGET_CHARS, len(text))
        if end < len(text): end = find_split(text, end)
        chunk_text = text[pos:end].strip()
        if len(chunk_text) >= MIN_CHUNK_LEN:
            chunks.append((num, pos, end, chunk_text, len(chunk_text)//4))
            num += 1
        pos = end - OVERLAP_CHARS if end < len(text) else end
    return chunks

conn = psycopg2.connect(
    host="localhost", port=5435,
    user="lexe", password="lexe_stage_cc07b664a88cb8e6",
    dbname="lexe_kb"
)
cur = conn.cursor()

cur.execute("SELECT id, codice, articolo, testo FROM kb.normativa_altalex ORDER BY codice, articolo")
articles = cur.fetchall()
print(f"Articles: {len(articles)}")

total_chunks = 0
batch = []
for art_id, codice, articolo, testo in articles:
    chunks = create_chunks(testo)
    for chunk_no, char_start, char_end, text, token_est in chunks:
        batch.append((art_id, codice, articolo, chunk_no, char_start, char_end, text, token_est))
        total_chunks += 1
    if len(batch) >= 500:
        execute_values(cur, """
            INSERT INTO kb.normativa_chunk (normativa_id, codice, articolo, chunk_no, char_start, char_end, text, token_est)
            VALUES %s
        """, batch)
        conn.commit()
        print(f"  Inserted {total_chunks} chunks...")
        batch = []

if batch:
    execute_values(cur, """
        INSERT INTO kb.normativa_chunk (normativa_id, codice, articolo, chunk_no, char_start, char_end, text, token_est)
        VALUES %s
    """, batch)
    conn.commit()

cur.execute("SELECT COUNT(*) FROM kb.normativa_chunk")
final = cur.fetchone()[0]
print(f"DONE! Total chunks: {final}")
conn.close()
