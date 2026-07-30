"""User-intent research: discover how real users ask questions around site keywords.

Given a website's keywords and profile, researches:
1. Real search queries — how users actually phrase questions (long-tail, voice-search,
   PAA-style, comparison, problem-solving)
2. Search-intent clustering — informational, commercial, transactional, navigational
3. Content-gap opportunities — what users are asking that the site might not answer yet

The output feeds directly into article planning so content matches real demand.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config.settings import Settings
from src.ai.deepseek_client import DeepSeekClient
from src.integrations.web_search import search_public_web, suggest_public_queries
from src.web.security import validate_public_http_url

logger = logging.getLogger(__name__)


# ── Data types ──────────────────────────────────────────────────────────


@dataclass
class UserQuery:
    """A single real-user search query discovered for a keyword."""

    query: str  # the full query as a user would type
    keyword: str  # which keyword this relates to
    intent: str  # informational, commercial, transactional, navigational
    format: str  # question, comparison, how-to, definition, review, price-check, etc.
    estimated_volume: str = ""  # qualitative: high/medium/low/none
    source: str = ""  # where this query was discovered (PAA, related, AI-simulated, etc.)
    validation_status: str = "unverified"
    evidence_urls: list[str] = field(default_factory=list)


@dataclass
class KeywordIntent:
    """A keyword enriched with real-user query intelligence."""

    keyword: str
    search_volume_hint: str = ""  # qualitative signal from search results
    dominant_intent: str = "informational"
    user_queries: list[UserQuery] = field(default_factory=list)
    content_gaps: list[str] = field(default_factory=list)


@dataclass
class ArticleIdea:
    """An article concept derived from user-intent research."""

    topic: str
    page_type: str  # blog, guide, market_analysis, product_review, news, landing
    target_keywords: list[str] = field(default_factory=list)
    answers_questions: list[str] = field(default_factory=list)
    search_intent: str = "informational"
    estimated_demand: str = ""  # high/medium/low
    headline_options: list[str] = field(default_factory=list)
    outline_sections: list[str] = field(default_factory=list)


@dataclass
class UserIntentReport:
    """Full user-intent research report for a website."""

    website_url: str
    site_name: str
    generated_at: str = ""
    keywords_intents: list[KeywordIntent] = field(default_factory=list)
    article_ideas: list[ArticleIdea] = field(default_factory=list)
    summary: str = ""
    total_queries_discovered: int = 0

    def to_dict(self) -> dict:
        return {
            "website_url": self.website_url,
            "site_name": self.site_name,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "total_queries_discovered": self.total_queries_discovered,
            "keywords_intents": [
                {
                    "keyword": ki.keyword,
                    "search_volume_hint": ki.search_volume_hint,
                    "dominant_intent": ki.dominant_intent,
                    "user_queries": [asdict(q) for q in ki.user_queries],
                    "content_gaps": ki.content_gaps,
                }
                for ki in self.keywords_intents
            ],
            "article_ideas": [asdict(ai) for ai in self.article_ideas],
        }


# ── Intent research engine ───────────────────────────────────────────


class UserIntentResearcher:
    """Discover how real users search around a website's topic space."""

    # Question prefixes that real users commonly use in search
    QUESTION_PREFIXES = [
        "what is", "how to", "how do", "why is", "why do",
        "when will", "where to", "which is best", "who is",
        "is it worth", "can i", "should i", "does", "do i need",
        "what are the best", "how much does", "where can i",
        "what is the difference between", "is there a",
    ]

    # Chinese question patterns for bilingual support
    CN_QUESTION_PATTERNS = [
        "是什么", "怎么", "如何", "为什么", "哪里", "哪个",
        "多少钱", "值得吗", "好不好", "怎么样", "什么意思",
        "区别", "推荐", "攻略", "教程", "方法", "流程",
        "注意事项", "需要什么", "如何选择", "怎么判断",
    ]

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        ai_client: DeepSeekClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_http = http_client is None
        self.http = http_client or httpx.AsyncClient(
            timeout=settings.crawl_timeout,
            headers={"User-Agent": settings.crawl_user_agent},
            follow_redirects=False,
        )
        self._owns_ai = ai_client is None
        self.ai = ai_client or DeepSeekClient(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            timeout=settings.deepseek_timeout,
        )
        self._network_semaphore = asyncio.Semaphore(max(1, settings.crawl_max_concurrent))

    async def close(self) -> None:
        if self._owns_http:
            await self.http.aclose()
        if self._owns_ai:
            await self.ai.close()

    # ── Main pipeline ───────────────────────────────────────────────

    async def run(
        self,
        website_url: str,
        keywords: list[str],
        *,
        site_name: str = "",
        niche: str = "",
        language: str = "en",
        max_queries_per_keyword: int = 8,
    ) -> UserIntentReport:
        """Run the full user-intent research pipeline."""
        website_url = await validate_public_http_url(website_url)
        keywords = keywords[:8]  # bound network work while covering more distinct site themes

        # Phase 1: Search for real user queries per keyword
        all_keyword_intents = list(await asyncio.gather(*[
            self._research_keyword_intent(
                keyword,
                language=language,
                max_queries=max_queries_per_keyword,
            )
            for keyword in keywords
        ]))

        # Phase 2: Use AI to generate realistic user-query variants
        ai_queries = await self._generate_ai_user_queries(
            keywords, language=language, site_name=site_name, niche=niche
        )
        self._merge_ai_queries(all_keyword_intents, ai_queries)
        await self._validate_simulated_queries(all_keyword_intents, limit=12)

        # Phase 3: Generate article ideas from the enriched intents
        article_ideas = await self._generate_article_ideas(
            all_keyword_intents,
            language=language,
            site_name=site_name,
            niche=niche,
            website_url=website_url,
        )

        total = sum(len(ki.user_queries) for ki in all_keyword_intents)
        summary = self._build_summary(all_keyword_intents, article_ideas, total)

        return UserIntentReport(
            website_url=website_url,
            site_name=site_name or urlparse(website_url).hostname or "",
            generated_at=datetime.now(UTC).isoformat(),
            keywords_intents=all_keyword_intents,
            article_ideas=article_ideas,
            summary=summary,
            total_queries_discovered=total,
        )

    # ── Phase 1: Search-based query discovery ───────────────────────

    async def _research_keyword_intent(
        self,
        keyword: str,
        *,
        language: str,
        max_queries: int,
    ) -> KeywordIntent:
        """Discover real user queries for a single keyword via search."""
        queries: list[UserQuery] = []

        suggestions = await suggest_public_queries(
            self.http,
            keyword,
            semaphore=self._network_semaphore,
            language=language,
            limit=max_queries * 2,
        )
        for suggestion in suggestions:
            inferred = self._infer_user_query(
                suggestion,
                keyword,
                source="google-autocomplete",
            )
            if inferred and len(queries) < max_queries:
                if not any(q.query.casefold() == inferred.query.casefold() for q in queries):
                    queries.append(inferred)

        # Deduplicate and categorize
        seen = set()
        unique: list[UserQuery] = []
        for q in queries:
            key = q.query.casefold().strip("? ")
            if key not in seen:
                seen.add(key)
                unique.append(q)

        # Detect dominant intent
        intents = [q.intent for q in unique] if unique else ["informational"]
        dominant = max(set(intents), key=intents.count)

        return KeywordIntent(
            keyword=keyword,
            dominant_intent=dominant,
            user_queries=unique[:max_queries],
        )

    def _infer_user_query(
        self,
        title: str,
        keyword: str,
        *,
        source: str,
        evidence_url: str = "",
    ) -> UserQuery | None:
        """Build a UserQuery from a search result title."""
        title = title.strip()
        if len(title) < 10 or len(title) > 200:
            return None
        keyword_terms = self._matching_terms(keyword)
        title_terms = self._matching_terms(title)
        required_overlap = min(2, len(keyword_terms))
        if not keyword_terms or len(keyword_terms & title_terms) < required_overlap:
            return None

        # Classify intent
        intent = "informational"
        if any(w in title.casefold() for w in ("price", "cost", "buy", "cheap", "for sale", "order", "quote")):
            intent = "commercial"
        elif any(w in title.casefold() for w in ("review", "best", "top", "vs", "compare", "which")):
            intent = "commercial"
        elif any(w in title.casefold() for w in ("how to", "how do", "what is", "why", "guide", "tutorial")):
            intent = "informational"

        # Classify format
        fmt = "question"
        if any(w in title.casefold() for w in ("how to", "how do")):
            fmt = "how-to"
        elif any(w in title.casefold() for w in ("vs", "compare", "difference", "versus")):
            fmt = "comparison"
        elif any(w in title.casefold() for w in ("review", "best", "top")):
            fmt = "review"
        elif any(w in title.casefold() for w in ("price", "cost", "how much")):
            fmt = "price-check"
        elif any(w in title.casefold() for w in ("guide", "complete", "ultimate", "comprehensive")):
            fmt = "guide"

        return UserQuery(
            query=title.rstrip(".")[:200],
            keyword=keyword,
            intent=intent,
            format=fmt,
            source=source,
            validation_status="search-observed",
            evidence_urls=[evidence_url] if evidence_url else [],
        )

    @staticmethod
    def _matching_terms(value: str) -> set[str]:
        stopwords = {
            "and", "for", "from", "how", "much", "near", "the", "to", "what",
            "where", "with",
        }
        terms = set()
        for token in re.findall(r"[a-z0-9]+", value.casefold()):
            if len(token) < 3 or token in stopwords or token.isdigit():
                continue
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            terms.add(token)
        return terms

    # ── Phase 2: AI-simulated user queries ──────────────────────────

    async def _generate_ai_user_queries(
        self,
        keywords: list[str],
        *,
        language: str,
        site_name: str,
        niche: str,
    ) -> dict[str, list[dict]]:
        """Use AI to generate realistic user search queries for keywords."""
        lang_label = "Chinese (zh-CN)" if language == "zh" else "English"
        prompt = f"""You are a search-behavior researcher. Given a website's keywords, generate
realistic search queries that real users would actually type into Google or other
search engines. Think like a real person with a problem to solve or a question to answer.

Website: {site_name or 'Unknown'}
Industry: {niche or 'General'}
Keywords: {json.dumps(keywords[:12], ensure_ascii=False)}
Language: {lang_label}

For EACH keyword, generate 3-5 REALISTIC user queries that:
- Use natural language (questions, problem statements, comparison requests)
- Reflect different search intents (informational, commercial, transactional)
- Sound like something a real person would type or speak (voice search)
- Vary in specificity (broad → narrow)

Include query types like:
- Direct questions: "What is the current silver price per ounce?"
- How-to: "How to import silver bars from Hong Kong to the USA"
- Comparison: "Silver bars vs silver coins which is better for investment"
- Problem-solving: "Why are COMEX silver inventories declining in 2026"
- Purchase intent: "Buy LBMA certified silver bars wholesale price"
- Local intent: "Silver suppliers Hong Kong LBMA certified"

Return a JSON object keyed by keyword, each containing an array of objects:
{{"query": "the query text", "intent": "informational|commercial|transactional|navigational",
  "format": "question|how-to|comparison|definition|review|price-check|problem-solving",
  "estimated_volume": "unknown"}}

Make queries diverse — avoid repeating the same structure. Prioritize queries
that reveal genuine user needs, not just keyword variations."""
        try:
            data = await self.ai.generate_json(
                prompt,
                system="You are a search-intent researcher who thinks like a real Google user.",
                temperature=0.6,
                max_tokens=2500,
            )
            if isinstance(data, dict) and not data.get("error"):
                return data
        except Exception as exc:
            logger.warning("AI user-query generation failed: %s", exc)
        return {}

    def _merge_ai_queries(
        self,
        keyword_intents: list[KeywordIntent],
        ai_data: dict[str, list[dict]],
    ) -> None:
        """Merge AI-generated queries into keyword intents, avoiding duplicates."""
        existing_texts = {
            q.query.casefold().strip("? ")
            for ki in keyword_intents
            for q in ki.user_queries
        }
        for ki in keyword_intents:
            candidates = ai_data.get(ki.keyword, [])
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                q_text = str(item.get("query", "")).strip()
                if not q_text or len(q_text) < 8:
                    continue
                if q_text.casefold().strip("? ") in existing_texts:
                    continue
                existing_texts.add(q_text.casefold().strip("? "))
                ki.user_queries.append(UserQuery(
                    query=q_text[:200],
                    keyword=ki.keyword,
                    intent=str(item.get("intent", "informational")),
                    format=str(item.get("format", "question")),
                    estimated_volume="",
                    source="AI-simulated",
                    validation_status="unverified",
                ))

    async def _validate_simulated_queries(
        self,
        keyword_intents: list[KeywordIntent],
        *,
        limit: int,
    ) -> None:
        """Validate AI query phrasing against live search results without inventing volume."""
        simulated = [
            query
            for intent in keyword_intents
            for query in intent.user_queries
            if query.source == "AI-simulated"
        ][:limit]
        semaphore = asyncio.Semaphore(4)

        async def validate(query: UserQuery) -> None:
            async with semaphore:
                results = await self._search_web(query.query)
            if results:
                query.validation_status = "search-validated"
                query.evidence_urls = [item.url for item in results[:3]]

        await asyncio.gather(*(validate(query) for query in simulated))

    # ── Phase 3: Article ideation ──────────────────────────────────

    async def _generate_article_ideas(
        self,
        keyword_intents: list[KeywordIntent],
        *,
        language: str,
        site_name: str,
        niche: str,
        website_url: str,
    ) -> list[ArticleIdea]:
        """Generate article ideas from user-intent research."""
        # Prepare evidence for AI
        query_evidence: list[dict] = []
        for ki in keyword_intents:
            for q in ki.user_queries:
                query_evidence.append({
                    "keyword": ki.keyword,
                    "query": q.query,
                    "intent": q.intent,
                    "format": q.format,
                })

        if not query_evidence:
            return self._fallback_article_ideas(keyword_intents)

        lang_label = "Chinese (zh-CN)" if language == "zh" else "English"
        prompt = f"""You are a content strategist. Given real user search queries discovered
for a website, propose article ideas that directly answer what users are asking for.

Website: {site_name or urlparse(website_url).hostname}
Industry: {niche or 'General'}
Language: {lang_label}

User queries discovered (these are what real people actually search for):
{json.dumps(query_evidence[:40], ensure_ascii=False, indent=2)}

Propose 4-8 article ideas. Each idea should:
- Directly answer 2-4 of the discovered user queries
- Have a clear, specific topic (not generic)
- Target a clear search intent
- Include an outline of 4-7 sections

Return an array of objects, each with:
- topic: the specific article topic
- page_type: blog | guide | market_analysis | product_review | news | landing
- target_keywords: 2-4 keywords this article targets
- answers_questions: 2-4 specific user queries this article answers
- search_intent: informational | commercial | transactional
- estimated_demand: high | medium | low
- headline_options: 3 descriptive headline options (no clickbait)
- outline_sections: 4-7 section headings that form the article outline

Prioritize ideas that answer MULTIPLE user queries and fill clear content gaps.
Deduplicate — do not propose two articles that cover the same ground."""
        try:
            data = await self.ai.generate_json(
                prompt,
                system="You are a content strategist who turns user search data into actionable article plans.",
                temperature=0.4,
                max_tokens=3000,
            )
            if isinstance(data, list):
                ideas: list[ArticleIdea] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    ideas.append(ArticleIdea(
                        topic=str(item.get("topic", ""))[:300],
                        page_type=str(item.get("page_type", "blog")),
                        target_keywords=self._clean_string_list(item.get("target_keywords"), 4),
                        answers_questions=self._clean_string_list(item.get("answers_questions"), 4),
                        search_intent=str(item.get("search_intent", "informational")),
                        estimated_demand=str(item.get("estimated_demand", "medium")),
                        headline_options=self._clean_string_list(item.get("headline_options"), 3),
                        outline_sections=self._clean_string_list(item.get("outline_sections"), 7),
                    ))
                return ideas
        except Exception as exc:
            logger.warning("AI article ideation failed: %s", exc)
        return self._fallback_article_ideas(keyword_intents)

    def _fallback_article_ideas(self, keyword_intents: list[KeywordIntent]) -> list[ArticleIdea]:
        """Generate basic article ideas without AI."""
        ideas: list[ArticleIdea] = []
        for ki in keyword_intents[:8]:
            queries = [q.query for q in ki.user_queries[:3]]
            ideas.append(ArticleIdea(
                topic=f"A practical guide to {ki.keyword}",
                page_type="guide" if "how" in " ".join(queries).casefold() else "blog",
                target_keywords=[ki.keyword],
                answers_questions=queries[:3] if queries else [f"What is {ki.keyword}?"],
                search_intent=ki.dominant_intent,
                estimated_demand="medium",
                headline_options=[f"Everything You Need to Know About {ki.keyword}"],
                outline_sections=["Introduction", f"Understanding {ki.keyword}", "Key Considerations", "Conclusion"],
            ))
        return ideas

    # ── Helpers ───────────────────────────────────────────────────

    async def _search_web(self, query: str) -> list[TrendCandidate]:
        """Search public result providers for a query."""
        results = await search_public_web(
            self.http,
            query,
            semaphore=self._network_semaphore,
            timeout=12,
            limit=8,
        )
        return [TrendCandidate(**item) for item in results]

    @staticmethod
    def _unwrap_search_url(href: str) -> str:
        absolute = urljoin("https://html.duckduckgo.com", href)
        parsed = urlparse(absolute)
        redirected = parse_qs(parsed.query).get("uddg")
        return unquote(redirected[0]) if redirected else absolute

    async def _fetch_public(
        self,
        url: str,
        *,
        check_robots: bool,
        params: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response | None:
        try:
            current = await validate_public_http_url(url)
        except Exception:
            return None
        options: dict = {"params": params}
        if timeout is not None:
            options["timeout"] = timeout
        try:
            async with self._network_semaphore:
                response = await self.http.get(current, **options)
        except Exception:
            return None
        if response.status_code >= 400:
            return None
        return response

    def _build_summary(
        self,
        keyword_intents: list[KeywordIntent],
        article_ideas: list[ArticleIdea],
        total_queries: int,
    ) -> str:
        lines = [
            f"Discovered {total_queries} real user queries across {len(keyword_intents)} keywords.",
            f"Generated {len(article_ideas)} article ideas.",
        ]
        intents = [ki.dominant_intent for ki in keyword_intents]
        if intents:
            top = max(set(intents), key=intents.count)
            lines.append(f"Dominant search intent: {top}.")
        return " ".join(lines)

    @staticmethod
    def _clean_string_list(value: object, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            cleaned = " ".join(str(item).split())[:200]
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result[:limit]


# Re-use TrendCandidate from automatic_article for DuckDuckGo results
@dataclass
class TrendCandidate:
    query: str
    title: str
    url: str
    snippet: str = ""
    provider: str = ""
