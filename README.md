# LEXe API

Legal Tools API for Italian and European legislation search.

## Features

- **Normattiva**: Italian legislation (leggi, decreti, codici)
- **EUR-Lex**: European legislation (regolamenti, direttive)
- **InfoLex**: Case law and commentary (Brocardi.it)

## Quick Start

```bash
# Install dependencies
uv sync

# Run development server
uvicorn lexe_api.main:app --reload --port 8020

# Run with Docker
docker compose -f docker-compose.lexe.yml --env-file .env.lexe up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tools/normattiva/search` | POST | Search Italian legislation |
| `/api/v1/tools/eurlex/search` | POST | Search EU legislation |
| `/api/v1/tools/infolex/search` | POST | Search case law |
| `/health/status` | GET | Service health |

## Architecture

```
lexe-api:8020      ← FastAPI service
lexe-postgres:5433 ← Document storage
lexe-valkey:6380   ← Cache
```

## License

Proprietary - ITC Consulting SRL
