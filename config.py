"""
config.py – Centralized configuration loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── OpenRouter / Embedding ──────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# ── PostgreSQL ──────────────────────────────────────────────────────
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "fengshui_user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "fengshui_pass")
PG_DATABASE = os.getenv("PG_DATABASE", "fengshui_db")

PG_URL = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)

# ── OpenSearch ──────────────────────────────────────────────────────
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", "9200"))
OS_INDEX = os.getenv("OS_INDEX", "fengshui_products")

# ── Data file paths ─────────────────────────────────────────────────
DATA_FILE_1 = os.getenv("DATA_FILE_1", "data/40_san_pham_numbered.txt")
DATA_FILE_2 = os.getenv("DATA_FILE_2", "data/--41--.txt")
