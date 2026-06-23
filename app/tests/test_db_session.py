from __future__ import annotations

from pathlib import Path

from app.db.session import create_configured_engine


def test_sqlite_engine_uses_busy_timeout_and_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "factory.db"
    engine = create_configured_engine(f"sqlite:///{db_path}", sqlite_busy_timeout_ms=45000)

    with engine.connect() as connection:
        busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar()

    assert busy_timeout == 45000
    assert str(journal_mode).lower() == "wal"
