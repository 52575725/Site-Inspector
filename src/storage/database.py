from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import Settings


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None


def get_engine(settings: Settings | None = None):
    global _engine
    if _engine is None:
        db_path = (settings.data_dir if settings else Path("data")) / "site_inspector.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = get_engine(settings)
        _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def init_db(settings: Settings | None = None) -> None:
    engine = get_engine(settings)

    # Enable WAL mode outside any transaction (must be done on a raw connection)
    async with engine.connect() as raw_conn:
        await raw_conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await raw_conn.exec_driver_sql("PRAGMA busy_timeout=5000")
        await raw_conn.commit()

    async with engine.begin() as conn:
        from src.storage.models import (  # noqa: F401
            AuditLog,
            Fix,
            Issue,
            PageScan,
            Scan,
            Target,
            Verification,
        )
        await conn.run_sync(Base.metadata.create_all)
        await _upgrade_sqlite_schema(conn)


async def _upgrade_sqlite_schema(conn) -> None:
    """Apply additive SQLite upgrades for installations created before migrations existed."""
    result = await conn.exec_driver_sql("PRAGMA table_info(fixes)")
    columns = {row[1] for row in result.fetchall()}
    additions = {
        "status": "VARCHAR(30) NOT NULL DEFAULT 'proposed'",
        "plain_summary": "TEXT",
        "impact_explanation": "TEXT",
        "change_explanation": "TEXT",
        "risk_level": "VARCHAR(10) NOT NULL DEFAULT 'medium'",
        "approved_at": "DATETIME",
        "rejected_at": "DATETIME",
    }
    for name, definition in additions.items():
        if name not in columns:
            await conn.exec_driver_sql(f"ALTER TABLE fixes ADD COLUMN {name} {definition}")

    # Earlier versions called generated suggestions "auto_fixed" even when
    # nothing had been written. Preserve honest semantics after upgrading.
    await conn.exec_driver_sql(
        "UPDATE issues SET status = 'proposed' "
        "WHERE status = 'auto_fixed' AND id IN "
        "(SELECT issue_id FROM fixes WHERE applied_at IS NULL)"
    )


async def get_session() -> AsyncSession:
    factory = get_session_factory()
    return factory()
