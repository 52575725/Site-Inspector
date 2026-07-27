"""Internal link graph analysis with PageRank-like scoring.

Cross-page inspector that builds a directed graph from all crawled
pages, computes link-equity scores, and surfaces structural issues:
- Pages with dangerously few internal links (orphans / near-orphans)
- Important pages buried too deep
- Link equity distribution imbalances
- Crawl depth outliers
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# A page is "near-orphan" if it has fewer than this many internal inlinks
MIN_INLINKS_WARNING = 2
MIN_INLINKS_CRITICAL = 1

# Pages this deep from homepage should be flagged
MAX_RECOMMENDED_DEPTH = 3

# Damping factor for PageRank
DAMPING = 0.85
PR_CONVERGENCE = 1e-5
PR_MAX_ITERS = 100


class LinkGraphInspector(BaseInspector):
    """Analyze internal link structure: PageRank, orphans, depth, equity.

    Collects link data during per-page inspect() calls, then runs graph
    analysis on teardown().  Cross-page findings are collected via
    the same pattern as ContentClusterInspector.
    """

    inspector_name = "link_graph"

    def __init__(self):
        super().__init__()
        # url -> set of outbound URLs (internal only)
        self._outlinks: dict[str, set[str]] = {}
        # url -> set of inbound URLs (reverse of outlinks)
        self._inlinks: dict[str, set[str]] = {}
        # url -> page metadata
        self._titles: dict[str, str] = {}
        self._findings: list[RawFinding] = []

    async def setup(self) -> None:
        self._outlinks.clear()
        self._inlinks.clear()
        self._titles.clear()
        self._findings.clear()

    async def teardown(self) -> None:
        await self._run_link_analysis()

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        """Collect outlinks from this page. Real findings emitted in teardown()."""
        if not html_content:
            return []

        from urllib.parse import urljoin, urlparse
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # Store title
        title_tag = soup.find("title")
        self._titles[url] = (
            title_tag.string.strip() if title_tag and title_tag.string else ""
        )[:120]

        base_host = urlparse(url).hostname
        base_host_alt = (
            "www." + base_host if not base_host.startswith("www.") else base_host[4:]
        )

        def _same_host(h: str) -> bool:
            return h == base_host or h == base_host_alt

        out: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full = urljoin(url, href).split("#")[0].rstrip("/")
            parsed = urlparse(full)
            if not _same_host(parsed.hostname):
                continue
            if parsed.scheme not in ("http", "https"):
                continue
            out.add(full)

        self._outlinks[url] = out

        # Build reverse index incrementally
        if url not in self._inlinks:
            self._inlinks[url] = set()
        for target in out:
            self._inlinks.setdefault(target, set()).add(url)

        return []

    # ── Graph analysis (runs once after all pages inspected) ───────────

    async def _run_link_analysis(self) -> None:
        all_urls = set(self._outlinks.keys()) | set(self._inlinks.keys())
        if len(all_urls) < 3:
            return

        n = len(all_urls)
        url_list = sorted(all_urls)
        url_to_idx = {u: i for i, u in enumerate(url_list)}

        # ── 1. Compute crawl depth from homepage ──────────────────────
        homepage = self._find_homepage(all_urls)
        depths = self._bfs_depths(homepage, url_to_idx)

        # ── 2. PageRank ───────────────────────────────────────────────
        pagerank = self._pagerank(url_list, url_to_idx)

        # ── 3. Generate findings ──────────────────────────────────────

        # Compute stats
        inlink_counts = {
            u: len(self._inlinks.get(u, set())) for u in all_urls
        }
        avg_pr = sum(pagerank.values()) / max(n, 1)
        max_pr = max(pagerank.values()) if pagerank else 0
        pr_threshold_high = avg_pr * 2.0
        pr_threshold_low = avg_pr * 0.3

        # 3a. Orphan / near-orphan pages
        for url in all_urls:
            ic = inlink_counts.get(url, 0)
            if ic == 0:
                self._findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="link_graph_orphan",
                    description=(
                        f"Orphan page: no other page links to '{self._titles.get(url, url)[:80]}'. "
                        f"Search engines may not discover it."
                    ),
                    current_value="0 internal inlinks",
                    suggested_value=(
                        "Add internal links from at least 2-3 relevant pages"
                    ),
                    raw_metadata={
                        "inlinks": ic,
                        "title": self._titles.get(url, ""),
                    },
                ))
            elif ic == MIN_INLINKS_CRITICAL:
                self._findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="link_graph_near_orphan",
                    description=(
                        f"Near-orphan page: only {ic} internal link pointing to "
                        f"'{self._titles.get(url, url)[:80]}'. Weak discoverability."
                    ),
                    current_value=f"{ic} internal inlinks",
                    suggested_value="Add 3-5 contextual internal links from related pages",
                    raw_metadata={
                        "inlinks": ic,
                        "linked_from": list(self._inlinks.get(url, set()))[:5],
                        "title": self._titles.get(url, ""),
                    },
                ))
            elif ic <= MIN_INLINKS_WARNING and n > 10:
                self._findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="link_graph_low_inlinks",
                    description=(
                        f"Low internal link count ({ic} inlinks) for "
                        f"'{self._titles.get(url, url)[:80]}'. May benefit from additional links."
                    ),
                    current_value=f"{ic} internal inlinks",
                    suggested_value="Add 2-3 more internal links from contextually relevant pages",
                    raw_metadata={"inlinks": ic, "title": self._titles.get(url, "")},
                ))

        # 3b. Pages buried too deep
        deep_pages = sorted(
            [(u, d) for u, d in depths.items() if d > MAX_RECOMMENDED_DEPTH],
            key=lambda x: -x[1],
        )[:10]
        for url, depth in deep_pages:
            self._findings.append(RawFinding(
                url=url,
                inspector=self.inspector_name,
                category="link_graph_deep_page",
                description=(
                    f"Page is {depth} clicks from homepage — "
                    f"'{self._titles.get(url, url)[:80]}'. "
                    f"Search engines may consider it less important."
                ),
                current_value=f"depth={depth}",
                suggested_value=(
                    "Link to this page from higher-level pages "
                    "(homepage, category pages, or site-wide navigation)"
                ),
                raw_metadata={
                    "crawl_depth": depth,
                    "title": self._titles.get(url, ""),
                    "inlinks": inlink_counts.get(url, 0),
                },
            ))

        # 3c. Link equity concentration
        high_pr_urls = [
            (u, pagerank[u]) for u in all_urls if pagerank[u] > pr_threshold_high
        ]
        low_pr_urls = [
            (u, pagerank[u]) for u in all_urls
            if pagerank[u] < pr_threshold_low and inlink_counts.get(u, 0) > 0
        ]

        if high_pr_urls and len(high_pr_urls) <= max(1, n * 0.05):
            # Too few pages hoarding link equity
            representative = high_pr_urls[0][0]
            self._findings.append(RawFinding(
                url=representative,
                inspector=self.inspector_name,
                category="link_graph_equity_concentration",
                description=(
                    f"Link equity is concentrated on {len(high_pr_urls)} page(s) "
                    f"({len(high_pr_urls)/n:.0%} of site). Consider distributing "
                    f"links more evenly to strengthen deeper pages."
                ),
                current_value=f"{len(high_pr_urls)} pages with high PageRank",
                suggested_value=(
                    "Add links from high-PageRank pages to important but "
                    "under-linked content"
                ),
                raw_metadata={
                    "high_pr_pages": [
                        {"url": u, "pagerank": round(pr, 4)}
                        for u, pr in high_pr_urls[:5]
                    ],
                    "low_pr_count": len(low_pr_urls),
                    "total_pages": n,
                },
            ))

        # 3d. Pages with no outlinks (dead ends)
        dead_ends = [
            u for u in all_urls
            if len(self._outlinks.get(u, set())) == 0
            and self._titles.get(u, "")  # only pages with content
        ]
        if dead_ends and len(dead_ends) > n * 0.1:
            representative = dead_ends[0]
            self._findings.append(RawFinding(
                url=representative,
                inspector=self.inspector_name,
                category="link_graph_dead_ends",
                description=(
                    f"{len(dead_ends)}/{n} pages have zero internal outlinks "
                    f"(dead ends). Link equity that flows in cannot flow out."
                ),
                current_value=f"{len(dead_ends)} dead-end pages",
                suggested_value="Add relevant internal links from dead-end pages to other content",
                raw_metadata={
                    "dead_end_count": len(dead_ends),
                    "dead_end_examples": dead_ends[:5],
                    "total_pages": n,
                },
            ))

    def get_findings(self) -> list[RawFinding]:
        return self._findings

    # ── Graph algorithms ────────────────────────────────────────────

    def _find_homepage(self, all_urls: set[str]) -> str:
        """Heuristically identify the homepage URL."""
        for u in all_urls:
            parsed = __import__("urllib.parse").urlparse(u)
            path = parsed.path.rstrip("/")
            if path in ("", "/", "/index.html", "/index.htm", "/index.php"):
                return u
        # Fallback: shortest URL
        return min(all_urls, key=len) if all_urls else ""

    def _bfs_depths(self, start: str,
                    url_to_idx: dict[str, int]) -> dict[str, int]:
        """BFS from homepage to compute crawl depth for each page."""
        depths: dict[str, int] = {}
        if not start:
            return depths

        queue = [(start, 0)]
        visited: set[str] = set()

        while queue:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            depths[url] = depth

            for out in self._outlinks.get(url, set()):
                if out not in visited:
                    queue.append((out, depth + 1))

        return depths

    def _pagerank(
        self, url_list: list[str], url_to_idx: dict[str, int],
    ) -> dict[str, float]:
        """Compute PageRank for the internal link graph."""
        n = len(url_list)
        if n == 0:
            return {}

        # Initial uniform distribution
        pr = [1.0 / n] * n

        for _ in range(PR_MAX_ITERS):
            new_pr = [(1.0 - DAMPING) / n] * n

            for i, url in enumerate(url_list):
                out = self._outlinks.get(url, set())
                if not out:
                    # Dangling node: distribute evenly
                    for j in range(n):
                        new_pr[j] += DAMPING * pr[i] / n
                else:
                    out_indices = [
                        url_to_idx[o] for o in out if o in url_to_idx
                    ]
                    if out_indices:
                        share = DAMPING * pr[i] / len(out_indices)
                        for j in out_indices:
                            new_pr[j] += share

            # Check convergence
            delta = sum(abs(new_pr[i] - pr[i]) for i in range(n))
            pr = new_pr
            if delta < PR_CONVERGENCE:
                break

        return {url_list[i]: pr[i] for i in range(n)}
