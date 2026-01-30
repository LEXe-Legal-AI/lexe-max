"""Fix the two Approfondimenti Tematici files that failed due to unknown tipo."""
import asyncio
import hashlib
import re
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx

DB_URL = "postgresql://leo:stage_postgres_2026_secure@localhost:5432/leo"
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
PDF_DIR = Path("/opt/leo-platform/lexe-api/data/massimari")
MIN_LENGTH = 150


async def main():
    conn = await asyncpg.connect(DB_URL)

    files_to_fix = [
        ("Volume III_2016_Approfond_Tematici.pdf", 2016, "civile"),
        ("Volume III_2017_Approfond_Tematici.pdf", 2017, "civile"),
    ]

    for pdf_name, anno, tipo in files_to_fix:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"[SKIP] File not found: {pdf_name}")
            continue

        source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        existing = await conn.fetchval(
            "SELECT id FROM kb.documents WHERE source_hash = $1", source_hash
        )
        if existing:
            print(f"[SKIP] {pdf_name} already exists")
            continue

        print(f"[PROCESS] {pdf_name}")

        async with httpx.AsyncClient() as client:
            with open(pdf_path, "rb") as f:
                files = {"files": (pdf_path.name, f, "application/pdf")}
                response = await client.post(
                    UNSTRUCTURED_URL,
                    files=files,
                    data={"strategy": "fast", "output_format": "application/json"},
                    timeout=300.0,
                )
            elements = response.json()

        print(f"  Extracted {len(elements)} elements")

        doc_id = uuid4()
        await conn.execute(
            """
            INSERT INTO kb.documents (id, source_path, source_hash, anno, volume, tipo, titolo, processed_at)
            VALUES ($1, $2, $3, $4, 3, $5, $6, NOW())
            """,
            doc_id,
            str(pdf_path),
            source_hash,
            anno,
            tipo,
            pdf_name,
        )

        count = 0
        current = []
        for elem in elements:
            text = elem.get("text", "").strip()
            if elem.get("type") in ["Header", "Footer", "PageNumber"]:
                continue
            if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+)", text, re.IGNORECASE):
                if current:
                    full = " ".join(current)
                    if len(full) >= MIN_LENGTH:
                        testo_norm = re.sub(r"\s+", " ", full.lower().strip())
                        content_hash = hashlib.sha256(testo_norm.encode()).hexdigest()
                        exists = await conn.fetchval(
                            "SELECT 1 FROM kb.massime WHERE content_hash = $1",
                            content_hash,
                        )
                        if not exists:
                            await conn.execute(
                                """
                                INSERT INTO kb.massime (id, document_id, testo, testo_normalizzato, content_hash, anno, tipo)
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                                """,
                                uuid4(),
                                doc_id,
                                full,
                                testo_norm,
                                content_hash,
                                anno,
                                tipo,
                            )
                            count += 1
                current = [text]
            elif text:
                current.append(text)

        # Last massima
        if current:
            full = " ".join(current)
            if len(full) >= MIN_LENGTH:
                testo_norm = re.sub(r"\s+", " ", full.lower().strip())
                content_hash = hashlib.sha256(testo_norm.encode()).hexdigest()
                exists = await conn.fetchval(
                    "SELECT 1 FROM kb.massime WHERE content_hash = $1", content_hash
                )
                if not exists:
                    await conn.execute(
                        """
                        INSERT INTO kb.massime (id, document_id, testo, testo_normalizzato, content_hash, anno, tipo)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        uuid4(),
                        doc_id,
                        full,
                        testo_norm,
                        content_hash,
                        anno,
                        tipo,
                    )
                    count += 1

        print(f"  [OK] {pdf_name}: {anno} {tipo} vol.3 - {count} massime")

    total = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")
    docs = await conn.fetchval("SELECT COUNT(*) FROM kb.documents")
    await conn.close()
    print(f"\nTotal: {docs} documents, {total} massime")


if __name__ == "__main__":
    asyncio.run(main())
