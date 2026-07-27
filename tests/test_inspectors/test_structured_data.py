from __future__ import annotations

import pytest

from src.inspectors.structured_data import (
    REQUIRED_FIELDS,
    REQUIRED_SCHEMAS,
    StructuredDataValidator,
)


@pytest.fixture
def sd_validator():
    return StructuredDataValidator()


@pytest.mark.asyncio
async def test_detects_no_jsonld(sd_validator):
    html = "<html><head><title>Test</title></head><body></body></html>"
    findings = await sd_validator.inspect("https://example.com/", html)
    categories = {f.category for f in findings}
    assert "schema_missing_type" in categories
    assert any("Organization" in f.description for f in findings)


@pytest.mark.asyncio
async def test_detects_missing_required_type(sd_validator):
    html = """<html><head>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "Organization", "name": "Test Corp"}
    </script>
    </head><body></body></html>"""
    findings = await sd_validator.inspect("https://example.com/", html)
    categories = {f.category for f in findings}
    # Homepage needs Organization, WebSite, BreadcrumbList - it has Org, missing WebSite and BreadcrumbList
    assert "schema_missing_type" in categories
    missing_types = {f.suggested_value for f in findings if f.category == "schema_missing_type"}
    assert "WebSite" in missing_types or "BreadcrumbList" in missing_types


@pytest.mark.asyncio
async def test_detects_duplicate_type(sd_validator):
    html = """<html><head>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "Organization", "name": "Test Corp", "url": "https://example.com"}
    </script>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "Organization", "name": "Test Corp Again"}
    </script>
    </head><body></body></html>"""
    findings = await sd_validator.inspect("https://example.com/", html)
    categories = {f.category for f in findings}
    assert "schema_duplicate" in categories


@pytest.mark.asyncio
async def test_detects_missing_field(sd_validator):
    html = """<html><head>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "Organization"}
    </script>
    </head><body></body></html>"""
    findings = await sd_validator.inspect("https://example.com/", html)
    categories = {f.category for f in findings}
    assert "schema_missing_field" in categories


@pytest.mark.asyncio
async def test_empty_html_returns_no_findings(sd_validator):
    findings = await sd_validator.inspect("https://example.com/", "")
    assert len(findings) == 0


@pytest.mark.asyncio
async def test_invalid_json_reports_error(sd_validator):
    html = """<html><head>
    <script type="application/ld+json">
    {invalid json!!!
    </script>
    </head><body></body></html>"""
    findings = await sd_validator.inspect("https://example.com/", html)
    assert any(f.category == "invalid_jsonld" for f in findings)


@pytest.mark.asyncio
async def test_complete_schema_no_findings(sd_validator):
    html = """<html><head>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "Organization", "name": "Test Corp", "url": "https://example.com"}
    </script>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "WebSite", "name": "Test Site", "url": "https://example.com"}
    </script>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": []}
    </script>
    </head><body></body></html>"""
    findings = await sd_validator.inspect("https://example.com/", html)
    # Should not report missing_type for homepage
    missing_type_findings = [f for f in findings if f.category == "schema_missing_type"]
    assert len(missing_type_findings) == 0


@pytest.mark.asyncio
async def test_handles_graph_array(sd_validator):
    html = """<html><head>
    <script type="application/ld+json">
    [{"@type": "Organization", "name": "Test Corp", "url": "https://example.com"},
     {"@type": "WebSite", "name": "Test Site", "url": "https://example.com"}]
    </script>
    </head><body></body></html>"""
    findings = await sd_validator.inspect("https://example.com/", html)
    # Should parse array format, missing BreadcrumbList for homepage
    categories = {f.category for f in findings}
    assert "schema_missing_type" in categories


def test_classify_page_homepage():
    assert StructuredDataValidator._classify_page("https://example.com/") == "homepage"
    assert StructuredDataValidator._classify_page("https://example.com/jp") == "homepage"


def test_classify_page_product():
    assert StructuredDataValidator._classify_page("https://example.com/products/silver/") == "product"


def test_classify_page_blog_article():
    assert StructuredDataValidator._classify_page("https://example.com/blog/2024/my-post/") == "blog_article"


def test_classify_page_blog_list():
    assert StructuredDataValidator._classify_page("https://example.com/blog") == "blog_list"


def test_classify_page_about():
    assert StructuredDataValidator._classify_page("https://example.com/about/us/") == "about"


def test_classify_page_contact():
    assert StructuredDataValidator._classify_page("https://example.com/contact/us/") == "contact"


def test_required_schemas_defined():
    assert "homepage" in REQUIRED_SCHEMAS
    assert "product" in REQUIRED_SCHEMAS
    assert "Organization" in REQUIRED_FIELDS
    assert "name" in REQUIRED_FIELDS["Organization"]
