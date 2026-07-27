"""Compare your pages against competitor pages for SEO gaps.

Integrates competitor analysis into the scan pipeline. For each of your
pages, finds the most relevant competitor page and compares title, meta
description, word count, heading structure, schema types, and keywords.
"""

from __future__ import annotations

import json as json_mod
import logging
import re
from collections import Counter
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

MIN_WORDS_FOR_COMPARISON = 200
_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "this", "that", "these", "those", "it", "its",
    "not", "no", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so",
    "than", "too", "very", "just", "will", "can", "has", "have",
    "had", "been", "being", "also", "new",
}


class CompetitorGapInspector(BaseInspector):
    """Compare your website pages against competitors for SEO gaps.

    Fetches competitor pages once per scan, extracts their SEO profile,
    and compares against every crawled page to surface actionable gaps.

    When a DeepSeekClient is provided, also performs AI-powered semantic
    gap analysis — identifying topics, angles, and data points the
    competitor covers that your page does not.
    """

    inspector_name = "competitor_gap"

    def __init__(self, competitor_urls: list[str] | None = None,
                 timeout: int = 15,
                 deepseek=None,
                 business_config: dict | None = None):
        self.competitor_urls = competitor_urls or []
        self.timeout = timeout
        self.deepseek = deepseek
        self.business_config = business_config or {}
        self._competitor_profiles: dict[str, dict] = {}
        self._competitor_html: dict[str, str] = {}  # raw HTML for AI analysis
        self._page_data: list[dict] = []
        self._fetched = False
        self._deepseek_available: bool | None = None

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    # ── Public API: called by scan orchestrator ───────────────────

    def set_page_data(self, pages: list[dict]) -> None:
        """Receive crawled page metadata for keyword extraction.

        Called by ScanOrchestrator before inspect() to enable
        auto-discovery when competitor_urls is empty.
        """
        self._page_data = pages

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        if not self._fetched:
            # Auto-discover competitors if none configured
            if not self.competitor_urls:
                await self._auto_discover()
            if self.competitor_urls:
                await self._fetch_all_competitors()
            self._fetched = True

        if not self._competitor_profiles:
            return findings

        # ── Report competitor changes on homepage ────────────────
        # Only emit once per scan — use the first (homepage) URL
        is_homepage = url.rstrip("/").split("/")[-1] in ("", "jp", "index.html")
        if is_homepage:
            for comp_url, profile in self._competitor_profiles.items():
                recent = profile.get("_recent_changes")
                if recent:
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="competitor_page_changed",
                        description=(
                            f"Competitor {urlparse(comp_url).netloc} has changed: "
                            f"{'; '.join(recent)}"
                        ),
                        current_value=f"Changes detected on {urlparse(comp_url).netloc}",
                        suggested_value=(
                            "Review competitor changes and adjust your content "
                            "strategy if needed."
                        ),
                        raw_metadata={
                            "competitor_url": comp_url,
                            "changes": recent,
                        },
                    ))

        your_soup = BeautifulSoup(html_content, "html.parser")
        your = self._extract_profile(url, your_soup)

        if your["word_count"] < MIN_WORDS_FOR_COMPARISON:
            return findings

        best = self._find_best_match(your)
        if not best:
            return findings

        comp_url, comp = best

        # ── 1. Title length comparison ──────────────────────────
        yt_len = len(your["title"])
        ct_len = len(comp["title"])
        if ct_len >= 40 and yt_len < 30:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="title_too_short",
                description=(
                    f"Title too short ({yt_len} chars) vs competitor "
                    f"({urlparse(comp_url).netloc}: {ct_len} chars). "
                    f"Competitor title: '{comp['title'][:100]}'"
                ),
                current_value=f"{yt_len} chars: '{your['title']}'",
                suggested_value=(
                    f"Expand title to 50-60 chars. "
                    f"Competitor example: '{comp['title'][:120]}'"
                ),
            ))

        # ── 2. Meta description comparison ──────────────────────
        your_desc = your.get("meta_description", "")
        comp_desc = comp.get("meta_description", "")
        yd_len = len(your_desc)
        cd_len = len(comp_desc)
        if cd_len >= 120 and yd_len < 80:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="meta_description_too_short",
                description=(
                    f"Meta description too short ({yd_len} chars) vs "
                    f"competitor ({cd_len} chars)"
                ),
                current_value=your_desc[:100] or "(none)",
                suggested_value=(
                    f"Expand to 120-160 chars. "
                    f"Competitor example: '{comp_desc[:160]}'"
                ),
            ))
        elif not your_desc and comp_desc:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_meta_description",
                description=(
                    f"Missing meta description — competitor has one "
                    f"({cd_len} chars)"
                ),
                suggested_value=f"Add a meta description. Example: '{comp_desc[:160]}'",
            ))

        # ── 3. Content depth (word count) ───────────────────────
        yw = your["word_count"]
        cw = comp["word_count"]
        if cw > yw * 1.5 and cw > MIN_WORDS_FOR_COMPARISON:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="thin_content",
                description=(
                    f"Content thinner than competitor: {yw} vs {cw} words "
                    f"({urlparse(comp_url).netloc})"
                ),
                current_value=f"{yw} words",
                suggested_value=(
                    f"Expand to at least {int(cw * 0.7)} words. "
                    f"Competitor H2 sections: {', '.join(comp['h2_topics'][:5])}"
                ),
                raw_metadata={
                    "your_words": yw, "comp_words": cw,
                    "comp_h2_topics": comp.get("h2_topics", []),
                },
            ))

        # ── 4. Schema type gap ──────────────────────────────────
        your_schemas = set(your.get("schema_types", []))
        comp_schemas = set(comp.get("schema_types", []))
        missing_schemas = comp_schemas - your_schemas
        if missing_schemas:
            high_value = {"FAQ", "HowTo", "Product", "Review",
                          "Article", "BreadcrumbList"}
            actionable = missing_schemas & high_value
            target = list(actionable)[0] if actionable else list(missing_schemas)[0]
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="schema_missing_type",
                description=(
                    f"Competitor uses '{target}' schema but your page doesn't. "
                    f"Missing: {', '.join(sorted(missing_schemas))}"
                ),
                current_value=", ".join(sorted(your_schemas)) if your_schemas else "none",
                suggested_value=(
                    f"Add '{target}' JSON-LD schema. "
                    f"Competitor schemas: {', '.join(sorted(comp_schemas))}"
                ),
                raw_metadata={
                    "your_schemas": sorted(your_schemas),
                    "comp_schemas": sorted(comp_schemas),
                    "missing": sorted(missing_schemas),
                },
            ))

        # ── 5. H2 section coverage ──────────────────────────────
        yh2 = len(your.get("h2_topics", []))
        ch2 = len(comp.get("h2_topics", []))
        if ch2 > yh2 * 1.5 and ch2 >= 3:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="content_gap_section",
                description=(
                    f"Page has {yh2} H2 sections vs competitor's {ch2} — "
                    f"competitor covers more subtopics"
                ),
                current_value=f"{yh2} sections",
                suggested_value=(
                    f"Add {ch2 - yh2}+ H2 sections. "
                    f"Competitor sections: {', '.join(comp['h2_topics'][:8])}"
                ),
                raw_metadata={
                    "your_h2s": yh2, "comp_h2s": ch2,
                    "comp_h2_topics": comp.get("h2_topics", []),
                },
            ))

        # ── 6. AI-powered semantic gap analysis ──────────────────
        ai_findings = await self._deep_semantic_compare(
            url, your_soup, comp_url, comp,
        )
        findings.extend(ai_findings)

        return findings

    # ── Fetch + Profile ──────────────────────────────────────────────

    async def _auto_discover(self) -> None:
        """Auto-discover competitors when none are manually configured."""
        from urllib.parse import urlparse
        from src.inspectors.competitor_discovery import (
            discover_competitors, extract_keywords_from_pages,
        )
        page_data = self._page_data if self._page_data else []

        # Extract domain from the first page's URL
        your_domain = "example.com"
        if page_data and page_data[0].get("url"):
            your_domain = urlparse(page_data[0]["url"]).netloc.lower().replace("www.", "")

        keywords = extract_keywords_from_pages(
            pages=page_data,
            your_domain=your_domain,
        )
        if not keywords:
            logger.info("No keywords extracted, skipping competitor auto-discovery")
            return

        discovered = await discover_competitors(
            keywords=keywords,
            your_domain=your_domain,
            max_competitors=5,
        )
        if discovered:
            self.competitor_urls = discovered
            logger.info(
                f"Auto-discovered {len(discovered)} competitors: {discovered}"
            )

    async def _fetch_all_competitors(self) -> None:
        """Fetch, profile, and snapshot all competitor URLs once per scan.

        Also saves snapshots for change tracking — so you know when a
        competitor updates their title, meta description, or content.
        """
        from src.inspectors.competitor import CompetitorTracker
        tracker = CompetitorTracker(timeout=self.timeout)
        # Snapshot before fetching to detect changes
        snapshots_before: dict[str, dict] = {}
        for url in self.competitor_urls:
            try:
                history = tracker.get_history(url, limit=1)
                if history:
                    snapshots_before[url] = history[0]
            except Exception:
                pass

        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True,
            headers={"User-Agent": "SiteInspector/1.0"},
        ) as client:
            # Fetch all competitors concurrently instead of sequentially
            async def _fetch_one(url: str):
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return url, resp
                except Exception as e:
                    logger.warning(f"Competitor fetch failed {url}: {e}")
                    return url, None

            results = await asyncio.gather(
                *[_fetch_one(u) for u in self.competitor_urls],
            )
            for url, resp in results:
                if resp is None:
                    continue
                try:
                    self._competitor_html[url] = resp.text
                    soup = BeautifulSoup(resp.text, "html.parser")
                    profile = self._extract_profile(url, soup)
                    self._competitor_profiles[url] = profile
                    logger.info(
                        f"Competitor profiled: {urlparse(url).netloc} "
                        f"({profile['word_count']} words, "
                        f"{len(profile.get('h2_topics', []))} H2s)"
                    )

                    # ── Snapshot + change detection ──────────────
                    snapshot = await tracker.snapshot(url)
                    prev = snapshots_before.get(url)
                    if prev and snapshot.get("changes"):
                        changes = snapshot["changes"]
                        change_descs = []
                        for ch in changes:
                            field = ch["field"]
                            before = str(ch.get("before", ""))[:60]
                            after = str(ch.get("after", ""))[:60]
                            if before or after:
                                change_descs.append(
                                    f"{field}: '{before}' → '{after}'"
                                )
                        if change_descs:
                            logger.info(
                                f"Competitor changed: {urlparse(url).netloc} — "
                                f"{'; '.join(change_descs)}"
                            )
                            # Attach changes to profile for reporting
                            self._competitor_profiles[url]["_recent_changes"] = change_descs

                except Exception as e:
                    logger.warning(f"Competitor fetch failed {url}: {e}")

    def _find_best_match(self, your: dict) -> tuple[str, dict] | None:
        """Find the most relevant competitor page by keyword overlap."""
        your_keywords = set(your.get("top_keywords", []))
        best = None
        best_score = -1
        for url, cp in self._competitor_profiles.items():
            comp_keywords = set(cp.get("top_keywords", []))
            kw_overlap = len(your_keywords & comp_keywords)
            # More keyword overlap + longer content = better match
            score = kw_overlap * 10 + min(cp.get("word_count", 0), 2000) // 200
            if score > best_score:
                best_score = score
                best = (url, cp)
        return best

    # ── AI Semantic Comparison ───────────────────────────────────────

    async def _deep_semantic_compare(
        self, your_url: str, your_soup: BeautifulSoup,
        comp_url: str, comp_profile: dict,
    ) -> list[RawFinding]:
        """AI-powered semantic gap analysis between our page and competitor.

        Only runs when DeepSeek is configured and available.  Compares the
        actual body content (not just metadata) to find topics, angles,
        and data points the competitor covers that we don't.
        """
        if self.deepseek is None:
            return []

        if self._deepseek_available is None:
            try:
                self._deepseek_available = await self.deepseek.health_check()
            except Exception:
                self._deepseek_available = False
        if not self._deepseek_available:
            return []

        # Extract our body text
        for tag in your_soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        your_text = your_soup.get_text(separator=" ", strip=True)[:4000]

        # Extract competitor body text from stored HTML
        comp_html = self._competitor_html.get(comp_url, "")
        comp_text = ""
        if comp_html:
            comp_soup = BeautifulSoup(comp_html, "html.parser")
            for tag in comp_soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            comp_text = comp_soup.get_text(separator=" ", strip=True)[:4000]

        if not comp_text:
            return []

        business_desc = self.business_config.get(
            "description", "a business website"
        )

        system = (
            "You are a competitive content strategist.  Compare two pages "
            "on the same topic and identify what the competitor covers that "
            "we don't.  Be specific — name concrete topics, data points, "
            "and angles.  Return ONLY valid JSON."
        )

        prompt = (
            f"Context: {business_desc}\n\n"
            f"Our page ({your_url}):\n{your_text}\n\n"
            f"Competitor page ({comp_url}):\n{comp_text}\n\n"
            f"Return JSON:\n"
            f'{{\n'
            f'  "our_strengths": ["..."],\n'
            f'  "competitor_strengths": ["..."],\n'
            f'  "topics_competitor_covers_we_dont": [\n'
            f'    {{"topic": "...", "importance": "high/medium/low", '
            f'"suggested_approach": "..."}}\n'
            f'  ],\n'
            f'  "depth_comparison": "ours_deeper/similar/competitor_deeper",\n'
            f'  "unique_angles_we_should_add": ["..."],\n'
            f'  "data_points_competitor_uses": ["..."],\n'
            f'  "overall_gap_severity": "critical/moderate/minimal"\n'
            f'}}'
        )

        try:
            result = await self.deepseek.generate_json(
                prompt=prompt, system=system, temperature=0.3, max_tokens=1500,
            )
        except Exception as e:
            logger.debug(f"Semantic competitor comparison failed: {e}")
            return []

        if result.get("error") or result.get("raw"):
            return []

        findings: list[RawFinding] = []

        # Critical gap
        if result.get("overall_gap_severity") == "critical":
            topics = result.get("topics_competitor_covers_we_dont", [])
            topic_desc = "; ".join(
                f"{t.get('topic', '?')} ({t.get('importance', '?')})"
                for t in topics[:5]
            )
            findings.append(RawFinding(
                url=your_url, inspector=self.inspector_name,
                category="competitor_semantic_gap_critical",
                description=(
                    f"Critical content gap vs competitor ({urlparse(comp_url).netloc}). "
                    f"Missing topics: {topic_desc}. "
                    f"Depth: {result.get('depth_comparison', 'unknown')}."
                ),
                current_value="significant content gap detected",
                suggested_value=(
                    f"Add coverage for: "
                    f"{', '.join(t.get('topic', '') for t in topics[:5])}"
                ),
                raw_metadata=result,
            ))
        elif result.get("overall_gap_severity") == "moderate":
            topics = result.get("topics_competitor_covers_we_dont", [])
            if topics:
                findings.append(RawFinding(
                    url=your_url, inspector=self.inspector_name,
                    category="competitor_semantic_gap_moderate",
                    description=(
                        f"Moderate content gap vs competitor: missing "
                        f"{len(topics)} topic(s). "
                        f"Key missing: {topics[0].get('topic', '?') if topics else '?'}"
                    ),
                    current_value=f"{len(topics)} topics missing",
                    suggested_value=(
                        f"Address gap: {topics[0].get('suggested_approach', '') if topics else ''}"
                    ),
                    raw_metadata=result,
                ))

        # Data point gap
        data_points = result.get("data_points_competitor_uses", [])
        if data_points and len(data_points) >= 3:
            findings.append(RawFinding(
                url=your_url, inspector=self.inspector_name,
                category="competitor_data_gap",
                description=(
                    f"Competitor uses {len(data_points)} data points/statistics "
                    f"that your page lacks. Adding data improves credibility."
                ),
                current_value="no data points found in your content",
                suggested_value=f"Add data points like: {', '.join(data_points[:3])}",
                raw_metadata=result,
            ))

        return findings

    # ── SEO Profile Extraction ───────────────────────────────────────

    @staticmethod
    def _extract_profile(url: str, soup: BeautifulSoup) -> dict:
        """Extract SEO profile from a parsed page."""
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = desc_tag.get("content", "").strip() if desc_tag else ""

        # Body text minus boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body = soup.find("body")
        body_text = body.get_text(separator=" ", strip=True) if body else ""
        words = body_text.split()
        word_count = len(words)

        # Schema types
        schema_types: list[str] = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json_mod.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    if isinstance(block, dict) and "@type" in block:
                        t = block["@type"]
                        schema_types.append(t) if isinstance(t, str) else schema_types.extend(t)
            except (json_mod.JSONDecodeError, TypeError):
                pass

        # H2 topics
        h2_topics: list[str] = []
        for h2 in soup.find_all("h2"):
            text = h2.get_text(strip=True)
            if text and len(text) > 3:
                h2_topics.append(text[:80])

        # Top keywords
        word_list = re.findall(r"[a-z]{4,}", body_text.lower())
        filtered = [w for w in word_list if w not in _STOPWORDS]
        top_keywords = [w for w, _ in Counter(filtered).most_common(15)]

        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "word_count": word_count,
            "schema_types": schema_types,
            "h2_topics": h2_topics,
            "top_keywords": top_keywords,
        }
