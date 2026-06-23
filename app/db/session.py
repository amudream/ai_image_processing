from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def create_configured_engine(
    database_url: str,
    *,
    sqlite_busy_timeout_ms: int | None = None,
    sqlite_wal_enabled: bool | None = None,
) -> Engine:
    if not database_url.startswith("sqlite"):
        return create_engine(database_url)

    timeout_seconds = (sqlite_busy_timeout_ms or settings.sqlite_busy_timeout_ms) / 1000
    engine = create_engine(database_url, connect_args={"timeout": timeout_seconds})
    busy_timeout_ms = sqlite_busy_timeout_ms or settings.sqlite_busy_timeout_ms
    wal_enabled = settings.sqlite_wal_enabled if sqlite_wal_enabled is None else sqlite_wal_enabled

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            if wal_enabled and ":memory:" not in database_url:
                cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    return engine


engine = create_configured_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
