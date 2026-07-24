from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.storage.database import Base


MOCK_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Home</title>
</head>
<body>
    <h1>Welcome</h1>
    <h3>Skip H2</h3>
    <p>Thin content here.</p>
    <img src="/images/hero.jpg">
    <a href="/broken-page">Broken Link</a>
</body>
</html>"""

MOCK_PRODUCTS_HTML = """<!DOCTYPE html>
<html>
<head><title>Products - Helin Silver</title></head>
<body>
    <h1>Our Silver Products</h1>
    <h2>Silver Bars</h2>
    <p>High quality 99.99% pure silver bars for industrial use. Available in various sizes and weights. Our silver bars meet LBMA standards and come with SGS certification.</p>
    <h2>Silver Grains</h2>
    <p>Premium silver grains suitable for chemical applications. Consistent particle size distribution ensures reliable performance in your manufacturing processes.</p>
    <h2>Silver Powder</h2>
    <p>Ultra-fine silver powder for electronics and specialized applications. Our powder is produced under strict quality control to meet the highest industry standards.</p>
    <img src="/images/bar.jpg" alt="">
    <a href="/about/">About Us</a>
    <a href="/dead-link">Dead Link</a>
</body>
</html>"""


@pytest.fixture
def mock_site(tmp_path) -> Path:
    """Create a minimal static site with known issues for testing."""
    site_dir = tmp_path / "mock_site"
    site_dir.mkdir()

    (site_dir / "index.html").write_text(MOCK_INDEX_HTML, encoding="utf-8")
    (site_dir / "products").mkdir()
    (site_dir / "products" / "index.html").write_text(MOCK_PRODUCTS_HTML, encoding="utf-8")
    (site_dir / "about").mkdir()
    (site_dir / "about" / "index.html").write_text(
        "<!DOCTYPE html><html><head><title>About</title></head>"
        "<body><h1>About Us</h1><p>About content here.</p></body></html>",
        encoding="utf-8",
    )

    return site_dir


@pytest.fixture
async def test_db():
    """In-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session(test_db):
    """Async session for test database."""
    factory = async_sessionmaker(test_db, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
