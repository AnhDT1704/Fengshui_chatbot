"""Create the PostgreSQL schema required by the chatbot.

Run from the repository root:
    python scripts/init_database.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db_service
from models import get_engine


MIGRATIONS = [
    "017_auth_and_user_history.py",
    "018_session_handoff_status.py",
    "019_conversation_product_ref.py",
]


def run_migration(path: Path) -> None:
    print(f"  Running {path.name}...")
    runpy.run_path(str(path), run_name="__main__")


def run_sql_migration(path: Path) -> None:
    print(f"  Running {path.name}...")
    statements = [statement.strip() for statement in path.read_text(encoding="utf-8").split(";")]
    with get_engine().begin() as connection:
        for statement in statements:
            if statement:
                connection.execute(text(statement))


def main() -> None:
    print("Creating base tables...")
    db_service.init_db()

    migrations_dir = ROOT / "migrations"
    run_sql_migration(migrations_dir / "001_chatbot_tables.sql")
    for filename in MIGRATIONS:
        run_migration(migrations_dir / filename)

    with get_engine().connect() as connection:
        connection.exec_driver_sql("SELECT 1")
    print("Database schema is ready.")


if __name__ == "__main__":
    main()