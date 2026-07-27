"""Embedding-based content clustering and gap analysis.

Uses Ollama embeddings to discover content clusters across the site,
detect orphan pages that don't fit any cluster, and identify overcrowded
topic areas that may indicate cannibalization risk.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.ollama_client import OllamaClient
from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Cosine similarity thresholds
CLUSTER_SIMILARITY_THRESHOLD = 0.82   # pages this similar belong to the same cluster
CANNIBALIZATION_THRESHOLD = 0.90      # pages this similar may be cannibalizing
ISOLATION_THRESHOLD = 0.45            # below this, page has no semantic neighbors

# Minimum word count to generate a meaningful embedding
MIN_WORDS_FOR_EMBED = 100

# Max characters sent to embedding model
MAX_EMBED_CHARS = 8000


class ContentClusterInspector(BaseInspector):
    """Discover content clusters, orphans, and overcrowded topics via embeddings.

    Runs as a cross-page inspector — collects all page embeddings first,
    then runs clustering on teardown.  Individual inspect() calls collect data;
    the actual analysis fires once in teardown().
    """

    inspector_name = "content_cluster"

    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama
        self._pages: list[dict] = []          # {url, title, embedding, word_count}
        self._findings: list[RawFinding] = []
        self._embed_available: bool | None = None

    async def setup(self) -> None:
        self._pages.clear()
        self._findings.clear()
        self._embed_available = None

    async def teardown(self) -> None:
        """Run clustering analysis after all pages have been collected."""
        await self._run_clustering()

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        """Collect page data for clustering. Returns [] — real findings
        are emitted by teardown() and attached to the first page."""
        if not html_content or self.ollama is None:
            return []

        if self._embed_available is None:
            try:
                self._embed_available = await self.ollama.health_check()
            except Exception:
                self._embed_available = False

        if not self._embed_available:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else ""

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body_text = soup.get_text(separator=" ", strip=True)
        word_count = len(body_text.split())

        if word_count < MIN_WORDS_FOR_EMBED:
            self._pages.append({
                "url": url, "title": title, "embedding": None, "word_count": word_count,
            })
            return []

        try:
            embedding = await self.ollama.embed(body_text[:MAX_EMBED_CHARS])
            self._pages.append({
                "url": url, "title": title,
                "embedding": embedding, "word_count": word_count,
            })
        except Exception as e:
            logger.debug(f"Embedding failed for {url}: {e}")
            self._pages.append({
                "url": url, "title": title, "embedding": None, "word_count": word_count,
            })

        return []  # Findings emitted in teardown

    # ── Clustering logic ──────────────────────────────────────────────

    async def _run_clustering(self) -> None:
        """Analyze collected embeddings for clusters, orphans, and overlaps."""
        embed_pages = [p for p in self._pages if p.get("embedding")]

        if len(embed_pages) < 3:
            return

        # ── 1. Build similarity matrix ─────────────────────────────────
        n = len(embed_pages)
        similarity_matrix: list[list[float]] = [
            [0.0] * n for _ in range(n)
        ]

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(
                    embed_pages[i]["embedding"], embed_pages[j]["embedding"]
                )
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        # ── 2. Find clusters (connected components above threshold) ────
        visited: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(n):
            if i in visited:
                continue
            # BFS to find all pages in this cluster
            cluster: list[int] = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in range(n):
                    if neighbor not in visited and similarity_matrix[node][neighbor] > CLUSTER_SIMILARITY_THRESHOLD:
                        stack.append(neighbor)
            if cluster:
                clusters.append(cluster)

        # ── 3. Generate findings ───────────────────────────────────────

        # 3a. Overcrowded clusters → cannibalization risk
        for cluster in clusters:
            if len(cluster) >= 4:
                cluster_urls = [embed_pages[i]["url"] for i in cluster]
                # Check for very-near-duplicate pairs within cluster
                near_dupes: list[tuple[str, str, float]] = []
                for a_idx in range(len(cluster)):
                    for b_idx in range(a_idx + 1, len(cluster)):
                        ai = cluster[a_idx]
                        bi = cluster[b_idx]
                        sim = similarity_matrix[ai][bi]
                        if sim > CANNIBALIZATION_THRESHOLD:
                            near_dupes.append((
                                embed_pages[ai]["url"],
                                embed_pages[bi]["url"],
                                sim,
                            ))

                representative = embed_pages[cluster[0]]["url"]
                self._findings.append(RawFinding(
                    url=representative,
                    inspector=self.inspector_name,
                    category="cluster_overcrowded",
                    description=(
                        f"Found {len(cluster)} pages clustered around the same topic. "
                        f"High cannibalization risk — consider consolidating into a "
                        f"pillar page with supporting detail pages. "
                        f"URLs: {', '.join(cluster_urls[:5])}"
                        f"{' and more...' if len(cluster_urls) > 5 else ''}"
                    ),
                    current_value=f"{len(cluster)} pages in cluster",
                    suggested_value=(
                        "Designate one pillar page and redirect or differentiate "
                        "the others, or merge overlapping content"
                    ),
                    raw_metadata={
                        "cluster_size": len(cluster),
                        "cluster_urls": cluster_urls[:10],
                        "near_duplicate_pairs": [
                            {"url_a": ua, "url_b": ub, "similarity": round(s, 3)}
                            for ua, ub, s in near_dupes[:5]
                        ],
                    },
                ))

        # 3b. Isolated pages → no semantic neighbors
        for i in range(n):
            max_sim = max(
                similarity_matrix[i][j] for j in range(n) if j != i
            ) if n > 1 else 0
            if max_sim < ISOLATION_THRESHOLD:
                page = embed_pages[i]
                self._findings.append(RawFinding(
                    url=page["url"],
                    inspector=self.inspector_name,
                    category="cluster_isolated_page",
                    description=(
                        f"Page '{page['title'][:100]}' is semantically isolated — "
                        f"no other page on the site covers a related topic "
                        f"(max similarity: {max_sim:.2f}). Consider adding related "
                        f"content or linking to this page from relevant sections."
                    ),
                    current_value=f"max_similarity={max_sim:.2f}",
                    suggested_value=(
                        "Add internal links from topically related pages, "
                        "or create supporting content around this topic"
                    ),
                    raw_metadata={
                        "max_similarity": round(max_sim, 3),
                        "word_count": page["word_count"],
                    },
                ))

        # 3c. Site-wide content diversity check
        cluster_sizes = [len(c) for c in clusters]
        pages_in_clusters = sum(cluster_sizes)
        orphan_count = n - pages_in_clusters
        if n > 10 and orphan_count > n * 0.4:
            representative = embed_pages[0]["url"]
            self._findings.append(RawFinding(
                url=representative,
                inspector=self.inspector_name,
                category="cluster_low_cohesion",
                description=(
                    f"Low site-wide content cohesion: {orphan_count}/{n} pages "
                    f"({orphan_count/n:.0%}) don't belong to any topical cluster. "
                    f"The site may lack a clear content strategy."
                ),
                current_value=f"{orphan_count} orphan pages",
                suggested_value=(
                    "Audit content strategy: group related pages into topic "
                    "clusters with clear pillar pages and internal linking"
                ),
                raw_metadata={
                    "total_pages": n,
                    "pages_in_clusters": pages_in_clusters,
                    "orphan_count": orphan_count,
                    "clusters_found": len(clusters),
                    "avg_cluster_size": round(pages_in_clusters / max(len(clusters), 1), 1),
                },
            ))

    def get_findings(self) -> list[RawFinding]:
        """Return collected cross-page findings (called by orchestrator)."""
        return self._findings

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
