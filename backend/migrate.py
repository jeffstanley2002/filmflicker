"""Serialized, versioned database migrations."""
from pathlib import Path

from sqlalchemy import Engine, text

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
LOCK_ID = 173528441


def run_migrations(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": LOCK_ID})
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS cinematch_v2"))
        conn.execute(
            text(
                """CREATE TABLE IF NOT EXISTS cinematch_v2.schema_migrations (
                       version TEXT PRIMARY KEY,
                       applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                   )"""
            )
        )
        applied = {
            row[0]
            for row in conn.execute(text("SELECT version FROM cinematch_v2.schema_migrations"))
        }
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            conn.exec_driver_sql(path.read_text())
            conn.execute(
                text("INSERT INTO cinematch_v2.schema_migrations (version) VALUES (:version)"),
                {"version": path.name},
            )
