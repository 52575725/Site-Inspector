from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.storage.models import Issue

logger = logging.getLogger(__name__)

# Ranking opportunity multipliers — higher = fix this page first
# Pages near page 1 get the biggest boost because a small fix can
# push them onto page 1, where CTR is ~10x higher than page 2.
RANKING_OPPORTUNITY: dict[str, float] = {
    "top3":      0.7,   # already winning — don't risk breaking it
    "page1":     1.3,   # position 4-10: close to top 3, good upside
    "page2":     1.5,   # position 11-20: highest upside, small push = page 1
    "page3":     1.15,  # position 21-30: moderate upside
    "deep":      0.8,   # position 31+: needs fundamental work
    "unindexed": 0.5,   # not in GSC at all — fix won't help rankings yet
    "no_data":   1.0,   # GSC unavailable — neutral, don't change score
}


class Analyzer:
    """Three-dimensional priority scoring and issue classification."""

    SEVERITY_MAP = {
        # SEO
        "missing_title": 0.95,
        "title_too_short": 0.55,
        "title_too_long": 0.50,
        "missing_meta_description": 0.50,
        "meta_description_too_short": 0.35,
        "meta_description_too_long": 0.30,
        "missing_h1": 0.90,
        "multiple_h1": 0.75,
        "h_tag_skip": 0.70,
        "missing_canonical": 0.70,
        "missing_hreflang": 0.80,
        "incomplete_hreflang": 0.75,
        "missing_og_tags": 0.45,
        "missing_structured_data": 0.60,
        "invalid_jsonld": 0.55,
        # Performance
        "poor_lcp": 0.85,
        "poor_cls": 0.85,
        "poor_tti": 0.70,
        "poor_speed_index": 0.65,
        "needs_improvement_lcp": 0.55,
        "needs_improvement_cls": 0.55,
        "needs_improvement_tti": 0.40,
        "needs_improvement_speed_index": 0.35,
        "optimize_render_blocking": 0.60,
        "optimize_unused_css": 0.45,
        "optimize_unused_js": 0.45,
        "optimize_offscreen_images": 0.40,
        "optimize_total_byte_weight": 0.50,
        "optimize_uses_webp": 0.35,
        "too_many_resources": 0.40,
        "excessive_inline_styles": 0.30,
        "large_inline_images": 0.45,
        # Mobile
        "missing_viewport_meta": 0.90,
        "horizontal_scroll": 0.80,
        "small_font_size": 0.65,
        "small_touch_targets": 0.70,
        # Accessibility
        "missing_alt_text": 0.70,
        "empty_alt_text": 0.60,
        "missing_lang_attribute": 0.55,
        "missing_form_label": 0.75,
        "missing_iframe_title": 0.50,
        # Content
        "thin_content": 0.60,
        "duplicate_content": 0.70,
        "low_readability": 0.30,
        "low_content_quality_ai": 0.40,
        "empty_page": 0.95,
        # Links
        "http_404": 0.70,
        "http_500": 0.85,
        "link_timeout": 0.50,
        "mixed_content": 0.85,
        "redirect_chain": 0.50,
        # Sitemap
        "sitemap_missing": 0.90,
        "sitemap_dead_url": 0.85,
        "sitemap_missing_url": 0.80,
        "sitemap_stale_lastmod": 0.30,
        "sitemap_missing_hreflang": 0.55,
        # Structured Data
        "schema_missing_type": 0.70,
        "schema_missing_field": 0.50,
        "schema_invalid_value": 0.45,
        "schema_duplicate": 0.65,
        # Content Gap
        "content_gap_section": 0.60,
        "content_gap_word_count": 0.50,
        "content_gap_links": 0.40,
        # Twitter & Image SEO
        "missing_twitter_cards": 0.30,
        "image_missing_alt": 0.55,
        "image_empty_alt": 0.60,
        # Internal Links
        "internal_orphan_page": 0.50,
        "internal_deep_page": 0.30,
        # Keyword Analysis
        "keyword_density_low": 0.55,
        "keyword_not_in_title": 0.65,
        "keyword_not_in_h1": 0.60,
        "keyword_not_in_first_paragraph": 0.40,
        "keyword_not_in_url": 0.35,
        # Image SEO
        "image_no_alt": 0.55,
        "image_missing_dimensions": 0.60,
        "image_no_lazy_loading": 0.45,
        "image_not_webp": 0.35,
        "image_no_async_decoding": 0.25,
        # Geo tags
        "missing_geo_region": 0.45,
        "missing_geo_placename": 0.40,
        "missing_geo_position": 0.50,
        # Platform Verification
        "platform_missing_google_verify": 0.75,
        "platform_missing_baidu_verify": 0.70,
        "platform_missing_bing_verify": 0.55,
        "platform_missing_yandex_verify": 0.30,
        "platform_missing_head": 0.90,
        # Robots.txt
        "robots_txt_missing": 0.80,
        "robots_txt_empty": 0.75,
        "robots_txt_no_sitemap": 0.55,
        "robots_txt_disallow_all": 0.90,
        "robots_txt_high_crawl_delay": 0.30,
        # Inspector errors
        "inspector_error": 0.30,
        # HTTP Headers – Security
        "missing_strict_transport_security": 0.85,
        "missing_content_security_policy": 0.75,
        "missing_x_frame_options": 0.70,
        "missing_x_frame_options_with_csp": 0.35,
        "missing_x_content_type_options": 0.60,
        "missing_referrer_policy": 0.45,
        "missing_permissions_policy": 0.40,
        "missing_cross_origin_opener_policy": 0.25,
        "missing_cross_origin_resource_policy": 0.25,
        "missing_cross_origin_embedder_policy": 0.20,
        "hsts_max_age_too_short": 0.35,
        # HTTP Headers – Caching
        "missing_cache_control": 0.55,
        "cache_control_conflict": 0.50,
        "missing_etag": 0.40,
        "missing_vary": 0.35,
        "vary_missing_accept_encoding": 0.35,
        # HTTP Headers – Compression
        "missing_compression": 0.65,
        # HTTP Headers – Info Leak
        "info_leak_server": 0.30,
        "info_leak_x_powered_by": 0.30,
        "info_leak_x_aspnet_version": 0.30,
        "info_leak_x_generator": 0.25,
        "info_leak_x_drupal_cache": 0.20,
        "info_leak_x_drupal_dynamic_cache": 0.20,
        # HTTP Headers – Other
        "headers_no_response_headers": 0.90,
        "x_robots_tag_blocks_indexing": 0.85,
        "x_robots_tag_meta_conflict": 0.60,
        "missing_content_type": 0.85,
        "content_type_missing_charset": 0.55,
        # Cannibalization
        "cannibalization_title_duplicate": 0.90,
        "cannibalization_title_similar": 0.65,
        "cannibalization_topic_overlap": 0.80,
        "cannibalization_taxonomy_vs_detail": 0.75,
        "cannibalization_taxonomy_title_overlap": 0.60,
        # JavaScript SEO
        "js_csr_empty_mount": 0.90,
        "js_low_visible_text": 0.90,
        "js_low_content_ratio": 0.75,
        "js_redirect_detected": 0.80,
        "js_excessive_scripts": 0.70,
        "js_many_scripts": 0.40,
        "js_blocking_scripts": 0.55,
        "js_large_inline_script": 0.50,
        "js_missing_noscript": 0.60,
        "js_thin_noscript": 0.35,
        "js_no_ssr_indicator": 0.70,
        # Content Freshness
        "freshness_very_stale": 0.85,
        "freshness_stale": 0.65,
        "freshness_aging": 0.40,
        "freshness_no_date_modified": 0.50,
        "freshness_schema_date_modified_old": 0.60,
        "freshness_date_mismatch": 0.70,
        "freshness_modified_date_mismatch": 0.65,
        "freshness_outdated_year_refs": 0.55,
        "freshness_relative_time_refs": 0.35,
        # URL Audit
        "url_too_long": 0.50,
        "url_long": 0.35,
        "url_uppercase": 0.60,
        "url_case_duplicate": 0.70,
        "url_underscores": 0.40,
        "url_non_ascii": 0.55,
        "url_unsafe_chars": 0.65,
        "url_too_deep": 0.55,
        "url_deep_with_long_slug": 0.35,
        "url_dynamic_extension": 0.55,
        "url_numeric_only": 0.60,
        "url_many_query_params": 0.45,
        "url_stop_words": 0.30,
        "url_date_in_slug": 0.40,
        "url_short_slug": 0.45,
        "url_trailing_slash_inconsistent": 0.70,
        "url_double_encoded": 0.60,
        "url_unencoded_spaces": 0.55,
        # Crawl Budget
        "crawl_budget_session_params": 0.85,
        "crawl_budget_tracking_params": 0.70,
        "crawl_budget_facet_params": 0.55,
        "crawl_budget_many_params": 0.60,
        "crawl_budget_faceted_url": 0.65,
        "crawl_budget_pagination_no_rel": 0.55,
        "crawl_budget_pagination_self_canonical": 0.60,
        "crawl_budget_thin_page": 0.55,
        "crawl_budget_orphan_page": 0.65,
        "crawl_budget_high_param_ratio": 0.70,
        "crawl_budget_tracking_param_scale": 0.60,
        "crawl_budget_faceted_scale": 0.60,
        "crawl_budget_top_params": 0.50,
        # E-E-A-T
        "eeat_no_author": 0.80,
        "eeat_author_no_schema": 0.50,
        "eeat_author_no_credentials": 0.55,
        "eeat_no_date": 0.65,
        "eeat_date_no_schema": 0.40,
        "eeat_date_schema_only": 0.35,
        "eeat_no_references": 0.50,
        "eeat_no_authoritative_refs": 0.45,
        "eeat_ymyl_no_disclaimer": 0.75,
        "eeat_ymyl_no_reviewer": 0.65,
        "eeat_no_about_page": 0.55,
        "eeat_no_contact_page": 0.55,
        "eeat_no_privacy_page": 0.60,
        "eeat_no_terms_page": 0.40,
    }

    IMPACT_FACTORS = {
        "sitewide": 1.0,
        "all_of_type": 0.8,
        "landing_page": 0.7,
        "multiple_pages": 0.5,
        "single_page": 0.3,
    }

    ROI_FACTORS = {
        "fully_auto": 0.9,
        "semi_auto": 0.5,
        "manual_required": 0.1,
    }

    def __init__(self, settings: Settings, session: AsyncSession):
        self.settings = settings
        self.session = session
        # Lazy-init GSC to avoid crash when credentials aren't configured
        self._gsc = None

    def _init_gsc(self):
        """Lazy GSC init — only when credentials are configured."""
        if self._gsc is not None:
            return
        from src.integrations.google_search_console import GoogleSearchConsole
        self._gsc = GoogleSearchConsole(
            credentials_path=self.settings.google_credentials_path,
            site_url=self.settings.gsc_property,
        )

    async def analyze_scan(self, scan_id: int, total_pages: int) -> Sequence[Issue]:
        """Score and prioritize all open issues from a scan."""
        # Fetch issues
        from sqlalchemy import select
        result = await self.session.execute(
            select(Issue).where(Issue.scan_id == scan_id, Issue.status == "open")
        )
        issues = list(result.scalars().all())

        if not issues:
            return []

        # Group issues by category for impact calculation
        category_counts: dict[str, int] = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

        # ── Fetch GSC ranking data for ranking-aware prioritization ──
        gsc_positions: dict[str, float] = await self._fetch_ranking_positions(issues)

        # Score each issue
        gsc_hits = 0
        for issue in issues:
            impact = self._calculate_impact(issue, category_counts[issue.category], total_pages)
            severity = self._calculate_severity(issue)
            fix_roi = self._calculate_roi(issue)

            issue.impact_scope = impact
            issue.severity = severity
            issue.fix_roi = fix_roi

            base_score = (
                impact * self.settings.priority_impact_weight
                + severity * self.settings.priority_severity_weight
                + fix_roi * self.settings.priority_fix_roi_weight
            )

            # Apply ranking opportunity multiplier
            position = gsc_positions.get(issue.url)
            opportunity = self._ranking_opportunity(position)
            if position is not None:
                gsc_hits += 1
            issue.priority_score = min(1.0, round(base_score * opportunity, 3))
            issue.priority_tier = self._classify_tier(round(issue.priority_score, 3))

        await self.session.flush()
        logger.info(
            f"Analyzed {len(issues)} issues "
            f"(GSC data available for {gsc_hits}): "
            f"P0={sum(1 for i in issues if i.priority_tier == 'P0')}, "
            f"P1={sum(1 for i in issues if i.priority_tier == 'P1')}, "
            f"P2={sum(1 for i in issues if i.priority_tier == 'P2')}, "
            f"P3={sum(1 for i in issues if i.priority_tier == 'P3')}"
        )

        return issues

    def _calculate_impact(self, issue: Issue, category_count: int, total_pages: int) -> float:
        if category_count >= total_pages:
            return self.IMPACT_FACTORS["sitewide"]
        if "landing" in issue.url.lower() or issue.url.rstrip("/") in (
            self.settings.target_base_url,
            self.settings.target_base_url + "/jp",
        ):
            return self.IMPACT_FACTORS["landing_page"]
        ratio = category_count / total_pages
        if ratio > 0.5:
            return self.IMPACT_FACTORS["all_of_type"]
        if category_count > 1:
            return self.IMPACT_FACTORS["multiple_pages"]
        return self.IMPACT_FACTORS["single_page"]

    def _calculate_severity(self, issue: Issue) -> float:
        # Check specific category first
        if issue.category in self.SEVERITY_MAP:
            return self.SEVERITY_MAP[issue.category]
        # Check prefix match (e.g., "wcag_*" → 0.6 on average)
        if issue.category.startswith("wcag_"):
            return 0.65
        if issue.category.startswith("poor_"):
            return 0.75
        if issue.category.startswith("needs_improvement_"):
            return 0.45
        return 0.4  # default

    def _calculate_roi(self, issue: Issue) -> float:
        category = issue.category

        # Fully auto-fixable categories
        fully_auto = {
            "missing_title", "title_too_short", "title_too_long",
            "missing_meta_description", "meta_description_too_short", "meta_description_too_long",
            "missing_alt_text", "empty_alt_text",
            "missing_og_tags", "missing_og_image",
            "missing_structured_data", "invalid_jsonld",
            "missing_hreflang", "incomplete_hreflang",
            "missing_canonical",
            "http_404", "mixed_content", "redirect_chain",
            "sitemap_dead_url", "sitemap_missing_url", "sitemap_missing_hreflang",
            "schema_missing_type", "schema_duplicate",
            "missing_twitter_cards", "image_missing_alt", "image_empty_alt",
            "missing_geo_region", "missing_geo_placename", "missing_geo_position",
            "image_no_alt", "image_missing_dimensions",
            "image_no_lazy_loading", "image_not_webp", "image_no_async_decoding",
            "robots_txt_missing", "robots_txt_empty",
            "robots_txt_no_sitemap", "robots_txt_disallow_all",
            "robots_txt_high_crawl_delay",
            "missing_breadcrumb",
            # URL Audit – auto-fixable redirects
            "url_uppercase",
            "url_case_duplicate",
            "url_trailing_slash_inconsistent",
            "url_double_encoded",
            "url_unencoded_spaces",
            "url_underscores",
            # HTTP Headers – ones that a headers fixer can auto-generate as config
            "missing_strict_transport_security",
            "missing_x_frame_options",
            "missing_x_frame_options_with_csp",
            "missing_x_content_type_options",
            "missing_referrer_policy",
            "missing_permissions_policy",
            "missing_cache_control",
            "cache_control_conflict",
            "missing_etag",
            "missing_vary",
            "vary_missing_accept_encoding",
            "missing_compression",
            "info_leak_server",
            "info_leak_x_powered_by",
            "info_leak_x_aspnet_version",
            "info_leak_x_generator",
            "content_type_missing_charset",
            # Crawl Budget – semi-automatic (config generation)
            "crawl_budget_session_params",
            "crawl_budget_tracking_params",
            "crawl_budget_facet_params",
            "crawl_budget_many_params",
            "crawl_budget_faceted_url",
            "crawl_budget_pagination_no_rel",
            "crawl_budget_pagination_self_canonical",
            "crawl_budget_thin_page",
            "crawl_budget_orphan_page",
            "crawl_budget_high_param_ratio",
            "crawl_budget_tracking_param_scale",
            "crawl_budget_faceted_scale",
            "crawl_budget_top_params",
        }
        # Semi-auto categories
        semi_auto = {
            "h_tag_skip", "multiple_h1", "missing_h1",
            "thin_content", "duplicate_content", "low_readability",
            "low_content_quality_ai",
            "horizontal_scroll", "small_font_size", "small_touch_targets",
            "sitemap_missing", "sitemap_stale_lastmod",
            "schema_missing_field", "schema_invalid_value",
            "content_gap_section", "content_gap_word_count", "content_gap_links",
            "internal_orphan_page", "internal_deep_page",
            "keyword_density_low",
            "keyword_not_in_title", "keyword_not_in_h1",
            "keyword_not_in_first_paragraph", "keyword_not_in_url",
            # Content Freshness
            "freshness_very_stale",
            "freshness_stale",
            "freshness_aging",
            "freshness_no_date_modified",
            "freshness_schema_date_modified_old",
            "freshness_date_mismatch",
            "freshness_modified_date_mismatch",
            "freshness_outdated_year_refs",
            "freshness_relative_time_refs",
            "platform_missing_google_verify", "platform_missing_baidu_verify",
            "platform_missing_bing_verify", "platform_missing_yandex_verify",
            "platform_missing_head",
            # URL Audit – needs planning
            "url_too_long",
            "url_long",
            "url_non_ascii",
            "url_unsafe_chars",
            "url_too_deep",
            "url_deep_with_long_slug",
            "url_dynamic_extension",
            "url_numeric_only",
            "url_many_query_params",
            "url_stop_words",
            "url_date_in_slug",
            "url_short_slug",
            # HTTP Headers – needs application-specific review
            "missing_content_security_policy",
            "missing_cross_origin_opener_policy",
            "missing_cross_origin_resource_policy",
            "missing_cross_origin_embedder_policy",
            "hsts_max_age_too_short",
            "headers_no_response_headers",
            "x_robots_tag_blocks_indexing",
            "x_robots_tag_meta_conflict",
            "missing_content_type",
            "info_leak_x_drupal_cache",
            "info_leak_x_drupal_dynamic_cache",
            # Cannibalization – needs human decision
            "cannibalization_title_duplicate",
            "cannibalization_title_similar",
            "cannibalization_topic_overlap",
            "cannibalization_taxonomy_vs_detail",
            "cannibalization_taxonomy_title_overlap",
            # JavaScript SEO – requires developer effort
            "js_csr_empty_mount",
            "js_low_visible_text",
            "js_low_content_ratio",
            "js_redirect_detected",
            "js_excessive_scripts",
            "js_many_scripts",
            "js_blocking_scripts",
            "js_large_inline_script",
            "js_missing_noscript",
            "js_thin_noscript",
            "js_no_ssr_indicator",
            # E-E-A-T – requires human content decisions
            "eeat_no_author",
            "eeat_author_no_schema",
            "eeat_author_no_credentials",
            "eeat_no_date",
            "eeat_date_no_schema",
            "eeat_date_schema_only",
            "eeat_no_references",
            "eeat_no_authoritative_refs",
            "eeat_ymyl_no_disclaimer",
            "eeat_ymyl_no_reviewer",
            "eeat_no_about_page",
            "eeat_no_contact_page",
            "eeat_no_privacy_page",
            "eeat_no_terms_page",
        }

        if category in fully_auto:
            return self.ROI_FACTORS["fully_auto"]
        if category in semi_auto:
            return self.ROI_FACTORS["semi_auto"]
        if category.startswith("wcag_"):
            return self.ROI_FACTORS["semi_auto"]
        return self.ROI_FACTORS["manual_required"]

    async def _fetch_ranking_positions(
        self, issues: Sequence[Issue],
    ) -> dict[str, float]:
        """Fetch GSC average position for each unique issue URL.

        Returns {url: position} or empty dict if GSC is unavailable.
        """
        self._init_gsc()
        if self._gsc is None or not self._gsc.available:
            return {}

        # Deduplicate URLs
        unique_urls = list({issue.url for issue in issues if issue.url})
        if not unique_urls:
            return {}

        today = date.today()
        start = (today - timedelta(days=28)).isoformat()
        end = today.isoformat()

        try:
            positions = await self._gsc.get_average_position(start, end, unique_urls)
            if positions:
                logger.info(
                    f"GSC: got positions for {len(positions)}/{len(unique_urls)} URLs"
                )
            return positions
        except Exception as e:
            logger.debug(f"GSC position fetch skipped: {e}")
            return {}

    @staticmethod
    def _ranking_opportunity(position: float | None) -> float:
        """Convert a GSC ranking position to a priority multiplier.

        Pages close to page 1 get the highest boost because a small SEO
        fix can push them onto page 1, where CTR is dramatically higher.
        """
        if position is None:
            return RANKING_OPPORTUNITY["no_data"]
        if position <= 0:
            return RANKING_OPPORTUNITY["unindexed"]
        if position <= 3:
            return RANKING_OPPORTUNITY["top3"]
        if position <= 10:
            return RANKING_OPPORTUNITY["page1"]
        if position <= 20:
            return RANKING_OPPORTUNITY["page2"]
        if position <= 30:
            return RANKING_OPPORTUNITY["page3"]
        return RANKING_OPPORTUNITY["deep"]

    @staticmethod
    def _classify_tier(score: float) -> str:
        if score >= 0.70:
            return "P0"
        if score >= 0.45:
            return "P1"
        if score >= 0.25:
            return "P2"
        return "P3"
