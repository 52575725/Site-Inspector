"""Auto-fix: find and insert relevant images into blog articles.

Searches free image APIs based on article content, downloads
images locally, and inserts them at natural positions in the article body.
"""
from __future__ import annotations

import asyncio
import difflib
import html
import logging
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.integrations.image_search import (
    download_image,
    extract_keywords_from_html,
    search_images,
)
from src.integrations.image_generation import OpenAIImageGenerator
from src.integrations.image_webp import convert_to_webp
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)


class ArticleImageFixer(BaseFixer):
    """Fix articles with no images by searching and inserting relevant images.

    Searches free image APIs (Unsplash → Pexels → Pixabay) using article
    title and headings as queries. Downloads images to the local images/
    directory and inserts <img> tags at SEO-friendly positions:
      - First image after H1 (hero image)
      - Remaining images after H2 sections

    Fix type is "semi_auto" because image selection should be reviewed
    by a human before publishing.
    """

    fixer_name = "article_image_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "article_no_images",
        "article_image_shortage",
        "blog_no_images",
        "thin_content_images",
    ]

    # Maximum images to insert per article
    MAX_IMAGES = 4
    # Minimum word count to consider inserting images
    MIN_WORDS = 200

    def __init__(
        self,
        unsplash_key: str = "",
        pexels_key: str = "",
        pixabay_key: str = "",
        openai_api_key: str = "",
        ai_fallback_enabled: bool = False,
        image_generation_model: str = "gpt-image-2",
        max_images: int = 4,
    ):
        self.unsplash_key = unsplash_key
        self.pexels_key = pexels_key
        self.pixabay_key = pixabay_key
        self.ai_fallback_enabled = ai_fallback_enabled
        self.max_images = min(4, max(3, max_images))
        self.image_generator = OpenAIImageGenerator(
            openai_api_key, image_generation_model,
        ) if ai_fallback_enabled and openai_api_key else None

    async def generate_fix(
        self,
        issue: dict,
        source: BaseSource,
        page_content: str,
    ) -> FixResult:
        url = issue.get("url", "")
        file_path = issue.get("file_path", "")
        issue_id = issue.get("id", 0)

        if not page_content:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content="",
                after_content="",
                error_message="No page content provided",
            )

        soup = BeautifulSoup(page_content, "html.parser")

        main = self._find_main(soup)
        if main is None:
            return FixResult(
                success=False, issue_id=issue_id, fixer_name=self.fixer_name,
                fix_type=self.fix_type, file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message="Could not identify article content area",
            )

        word_count = len(main.get_text(" ", strip=True).split())
        if word_count < self.MIN_WORDS:
            return FixResult(
                success=False, issue_id=issue_id, fixer_name=self.fixer_name,
                fix_type=self.fix_type, file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message=f"Article is too short for automatic images ({word_count} words)",
            )

        existing_images = main.find_all("img")
        desired_count = 4 if word_count >= 900 and len(main.find_all("h2")) >= 3 else 3
        desired_count = min(desired_count, self.max_images)
        if len(existing_images) >= desired_count:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message=f"Article already has {len(existing_images)} content images",
            )

        needed = desired_count - len(existing_images)

        # Extract keywords for image search
        queries = extract_keywords_from_html(page_content, max_queries=desired_count)
        if not queries:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="Could not extract keywords from article",
            )

        # Search for images
        primary_results = []
        backup_results = []
        seen_urls = set()
        for query in queries:
            results = await asyncio.to_thread(
                search_images, query, 3, self.unsplash_key,
                self.pexels_key, self.pixabay_key,
            )
            for result in results:
                key = result.url.split("?")[0]
                if not result.url or key in seen_urls:
                    continue
                seen_urls.add(key)
                candidate = (query, result)
                if not any(existing_query == query for existing_query, _ in primary_results):
                    primary_results.append(candidate)
                else:
                    backup_results.append(candidate)

        all_results = primary_results + backup_results

        if not all_results and not (self.ai_fallback_enabled and self.image_generator):
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="No licensed search images found and AI fallback is disabled",
            )

        # Download images to local directory
        work_dir = self._get_work_dir(source)
        images_dir = Path(work_dir) / "images" if work_dir else Path("images")
        images_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        for idx, (query, result) in enumerate(all_results):
            if len(downloaded) >= needed:
                break
            filename = self._generate_filename(result, idx, url)
            local_path = await asyncio.to_thread(
                download_image, result.url, images_dir, filename,
            )
            if local_path:
                webp_path = await asyncio.to_thread(convert_to_webp, local_path, 82)
                asset_path = Path(webp_path or local_path)
                original_path = Path(local_path)
                if webp_path and asset_path != original_path:
                    original_path.unlink(missing_ok=True)
                result_dict = {
                    "local_path": f"/images/{asset_path.name}",
                    "query": query,
                    "alt_text": result.alt_text or query,
                    "caption": result.alt_text or query,
                    "width": result.width or 800,
                    "height": result.height or 600,
                    "source": result.source,
                    "photographer": result.photographer,
                    "page_url": result.page_url,
                    "license_name": result.license_name,
                    "license_url": result.license_url,
                }
                downloaded.append(result_dict)

        if len(downloaded) < needed and self.ai_fallback_enabled and self.image_generator:
            for idx in range(len(downloaded), needed):
                query = queries[idx % len(queries)]
                filename = self._generate_ai_filename(idx, url)
                generated_path = await self.image_generator.generate(
                    self._build_generation_prompt(query), images_dir / filename,
                )
                if generated_path:
                    downloaded.append({
                        "local_path": f"/images/{generated_path.name}",
                        "query": query,
                        "alt_text": query,
                        "caption": query,
                        "width": 1536,
                        "height": 1024,
                        "source": "ai-generated",
                        "photographer": "",
                    })

        if not downloaded:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="Failed to download any images",
            )

        # Insert images into article
        modified_count = self._insert_images(
            soup,
            downloaded[:needed],
            include_hero=not existing_images,
        )

        if modified_count == 0:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="Could not find suitable insertion points in article",
            )

        integrity_error = self._validate_document_integrity(
            page_content,
            soup,
            modified_count,
        )
        if integrity_error:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message=f"Image insertion failed integrity validation: {integrity_error}",
            )

        new_content = str(soup)
        diff = "\n".join(difflib.unified_diff(
            page_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))

        logger.info(
            f"ArticleImageFixer: inserted {modified_count} images into {url} "
            f"(queries: {', '.join(queries[:3])})"
        )

        return FixResult(
            success=True,
            issue_id=issue_id,
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=file_path,
            before_content=page_content,
            after_content=new_content,
            diff=diff[:5000],
        )

    def _insert_images(
        self,
        soup: BeautifulSoup,
        images: list[dict],
        include_hero: bool = True,
    ) -> int:
        """Insert images at natural positions in the article body.

        Match each image's query and description to the most relevant H2.
        Unmatched images use the introduction and distributed sections.
        """
        inserted = 0

        main = self._find_main(soup)
        if main is None:
            return 0

        self._ensure_image_styles(soup)

        sections = [
            {
                "heading": heading,
                "anchor": self._section_anchor(heading),
                "text": self._section_text(heading),
            }
            for heading in main.find_all("h2")
        ]
        assignments = self._match_images_to_sections(images, sections)
        placed_images = set()
        used_anchors = set()

        for image_index, section_index in assignments.items():
            image = images[image_index]
            section = sections[section_index]
            anchor = section["anchor"]
            image["target_heading"] = section["heading"].get_text(" ", strip=True)
            figure = BeautifulSoup(
                self._build_figure_tag(image, is_hero=False), "html.parser",
            )
            anchor.insert_after(figure)
            image["placement_heading"] = section["heading"].get_text(" ", strip=True)
            placed_images.add(image_index)
            used_anchors.add(id(anchor))
            inserted += 1

        remaining = [
            (index, image) for index, image in enumerate(images)
            if index not in placed_images
        ]
        if remaining and include_hero:
            index, image = remaining.pop(0)
            h1 = main.find("h1")
            intro = h1.find_next("p") if h1 else main.find("p")
            anchor = intro or h1
            image["target_heading"] = h1.get_text(" ", strip=True) if h1 else "Introduction"
            figure = BeautifulSoup(
                self._build_figure_tag(image, is_hero=True), "html.parser",
            )
            if anchor:
                anchor.insert_after(figure)
                used_anchors.add(id(anchor))
            else:
                main.insert(0, figure)
            image["placement_heading"] = h1.get_text(" ", strip=True) if h1 else "Introduction"
            placed_images.add(index)
            inserted += 1

        if remaining:
            available_h2 = [
                section["heading"] for section in sections
                if id(section["anchor"]) not in used_anchors
            ]
            anchors = self._distributed_anchors(main, available_h2, len(remaining))
            for (_, image), anchor in zip(remaining, anchors):
                heading = (
                    anchor if getattr(anchor, "name", None) == "h2"
                    else anchor.find_previous("h2")
                )
                image["target_heading"] = (
                    heading.get_text(" ", strip=True) if heading else "Article body"
                )
                figure = BeautifulSoup(
                    self._build_figure_tag(image, is_hero=False), "html.parser",
                )
                anchor.insert_after(figure)
                image["placement_heading"] = (
                    heading.get_text(" ", strip=True) if heading else "Article body"
                )
                inserted += 1

        return inserted

    _SEMANTIC_STOPWORDS = {
        "article", "complete", "content", "guide", "image", "industry",
        "international", "photo", "silver", "the", "this", "with",
    }
    _SEMANTIC_CONCEPTS = {
        "air-freight": {
            "air cargo", "air freight", "aircraft", "airplane", "airport",
            "aviation", "flight",
        },
        "sea-freight": {
            "cargo ship", "container", "ocean freight", "port", "sea freight",
            "ship", "vessel",
        },
        "customs": {"border", "clearance", "customs", "inspection", "tariff"},
        "logistics": {
            "cargo", "freight", "logistics", "shipping", "supply chain", "transport",
        },
        "storage": {"storage", "vault", "warehouse", "warehousing"},
        "mining": {"geology", "mine", "mining", "ore"},
        "refining": {"assay", "laboratory", "refinery", "refining", "smelter"},
        "solar": {"photovoltaic", "solar panel", "solar panels"},
        "jewelry": {"jewellery", "jewelry", "necklace", "ring"},
        "market": {
            "benchmark", "chart", "exchange", "futures", "inventory", "price", "trading",
        },
    }

    @classmethod
    def _semantic_terms(cls, value: str) -> set[str]:
        lowered = value.lower()
        terms = {
            token for token in re.findall(r"[a-z0-9]+", lowered)
            if len(token) >= 3
            and token not in cls._SEMANTIC_STOPWORDS
            and not token.isdigit()
        }
        for concept, phrases in cls._SEMANTIC_CONCEPTS.items():
            if any(
                re.search(rf"\b{re.escape(phrase)}\b", lowered)
                for phrase in phrases
            ):
                terms.add(f"concept:{concept}")
        return terms

    @staticmethod
    def _section_text(heading) -> str:
        parts = [heading.get_text(" ", strip=True)]
        current = heading.find_next_sibling()
        while current is not None and getattr(current, "name", None) != "h2":
            if getattr(current, "get_text", None):
                parts.append(current.get_text(" ", strip=True))
            current = current.find_next_sibling()
        return " ".join(parts)

    @classmethod
    def _match_images_to_sections(
        cls,
        images: list[dict],
        sections: list[dict],
    ) -> dict[int, int]:
        scored_pairs = []
        for image_index, image in enumerate(images):
            target_heading = " ".join(str(image.get("target_heading", "")).split()).casefold()
            try:
                target_section_index = int(image.get("target_section_index", -1))
            except (TypeError, ValueError):
                target_section_index = -1
            query_terms = cls._semantic_terms(str(image.get("query", "")))
            description_text = " ".join(
                str(image.get(key, "")) for key in ("alt_text", "caption")
            )
            description_terms = cls._semantic_terms(description_text)
            image_terms = query_terms | description_terms
            for section_index, section in enumerate(sections):
                heading_text = section["heading"].get_text(" ", strip=True)
                if target_heading and heading_text.casefold() == target_heading:
                    scored_pairs.append((10_000, image_index, section_index))
                    continue
                if target_section_index == section_index:
                    scored_pairs.append((9_000, image_index, section_index))
                    continue
                heading_terms = cls._semantic_terms(heading_text)
                section_terms = cls._semantic_terms(section["text"])
                query_heading_overlap = query_terms & heading_terms
                description_heading_overlap = description_terms & heading_terms
                body_overlap = image_terms & section_terms
                query_concepts = {
                    term for term in query_terms if term.startswith("concept:")
                }
                heading_concepts = {
                    term for term in heading_terms if term.startswith("concept:")
                }
                query_heading_concepts = query_concepts & heading_concepts
                broad_concepts = {"concept:logistics"}
                specific_query_concepts = query_concepts - broad_concepts
                specific_heading_concepts = heading_concepts - broad_concepts
                conflicting_heading_concepts = set()
                if specific_query_concepts:
                    conflicting_heading_concepts = (
                        specific_heading_concepts - specific_query_concepts
                    )
                score = (
                    len(query_heading_overlap) * 12
                    + len(query_heading_concepts) * 12
                    + len(description_heading_overlap) * 3
                    + min(len(body_overlap), 5)
                    - len(conflicting_heading_concepts) * 30
                )
                if score >= 4:
                    scored_pairs.append((score, image_index, section_index))

        assignments = {}
        used_sections = set()
        for _, image_index, section_index in sorted(scored_pairs, reverse=True):
            if image_index in assignments or section_index in used_sections:
                continue
            assignments[image_index] = section_index
            used_sections.add(section_index)
        return assignments

    @classmethod
    def _validate_document_integrity(
        cls,
        original_html: str,
        modified_soup: BeautifulSoup,
        expected_inserted: int,
    ) -> str | None:
        """Return an error when insertion changed article content or core structure."""
        original = BeautifulSoup(original_html, "html.parser")
        reparsed = BeautifulSoup(str(modified_soup), "html.parser")
        original_figures = original.select("figure.article-media")
        figures = reparsed.select("figure.article-media")
        if len(figures) != len(original_figures) + expected_inserted:
            return "inserted figure count changed after HTML parsing"
        for figure in figures:
            image = figure.find("img", recursive=False)
            caption = figure.find("figcaption", recursive=False)
            if not image or not image.get("src") or image.get("alt") is None or not caption:
                return "an inserted figure is missing its image, alt text, or caption"
            if not str(image.get("width", "")).isdigit() or not str(
                image.get("height", ""),
            ).isdigit():
                return "an inserted image has invalid dimensions"

        original_main = cls._find_main(original)
        modified_main = cls._find_main(reparsed)
        if original_main is None or modified_main is None:
            return "article content container is missing"
        for figure in original_main.select("figure.article-media"):
            figure.decompose()
        for figure in modified_main.select("figure.article-media"):
            figure.decompose()
        original_style = original.find("style", id="site-inspector-article-images")
        if original_style:
            original_style.decompose()
        style = reparsed.find("style", id="site-inspector-article-images")
        if style:
            style.decompose()

        def normalize(value: str) -> str:
            return " ".join(value.split())

        if normalize(original_main.get_text(" ", strip=True)) != normalize(
            modified_main.get_text(" ", strip=True),
        ):
            return "article text changed during image insertion"

        protected_tags = (
            "h1", "h2", "h3", "p", "ul", "ol", "li", "table", "form", "script", "iframe",
        )
        for tag_name in protected_tags:
            if len(original.find_all(tag_name)) != len(reparsed.find_all(tag_name)):
                return f"{tag_name} structure changed during image insertion"
        baseline_markup = str(BeautifulSoup(str(original), "html.parser"))
        preserved_markup = str(BeautifulSoup(str(reparsed), "html.parser"))
        if baseline_markup != preserved_markup:
            return "existing HTML elements or attributes changed during image insertion"
        return None

    @staticmethod
    def _build_figure_tag(image: dict, is_hero: bool = False) -> str:
        """Build semantic, attributable and responsive article media."""
        src = image["local_path"]
        alt = html.escape(image.get("alt_text", "Article image"), quote=True)
        caption = html.escape(image.get("caption", ""))
        width = image.get("width", 800)
        height = image.get("height", 600)
        source = image.get("source", "")
        target_heading = html.escape(str(image.get("target_heading", "")), quote=True)
        image_type = html.escape(str(image.get("image_type", "photo")), quote=True)
        image_purpose = html.escape(str(image.get("insertion_reason", "")), quote=True)
        target_section_index = html.escape(
            str(image.get("target_section_index", -1)), quote=True,
        )
        photographer = image.get("photographer", "")
        page_url = html.escape(image.get("page_url", ""), quote=True)
        license_name = html.escape(image.get("license_name", ""))
        license_url = html.escape(image.get("license_url", ""), quote=True)
        loading = "eager" if is_hero else "lazy"
        priority = ' fetchpriority="high"' if is_hero else ""

        credits = []
        if photographer:
            label = html.escape(photographer)
            credits.append(
                f'<a href="{page_url}" rel="noopener noreferrer" target="_blank">{label}</a>'
                if page_url else label
            )
        if license_name:
            credits.append(
                f'<a href="{license_url}" rel="noopener noreferrer" target="_blank">{license_name}</a>'
                if license_url else license_name
            )
        credit_html = f'<span class="article-image-credit">{" / ".join(credits)}</span>' if credits else ""
        return (
            f'<figure class="article-media{(" article-media-hero" if is_hero else "")}" '
            f'data-target-heading="{target_heading}" data-image-type="{image_type}" '
            f'data-image-purpose="{image_purpose}" '
            f'data-target-section-index="{target_section_index}">'
            f'<img src="{src}" alt="{alt}" width="{width}" height="{height}" '
            f'loading="{loading}" decoding="async"{priority}>'
            f'<figcaption>{caption}{credit_html}</figcaption></figure>'
        )

    @staticmethod
    def _find_main(soup: BeautifulSoup):
        body = soup.find("body")
        if not body:
            return None
        return (
            body.find("article")
            or body.find("main")
            or body.find("div", class_=re.compile(r"article|post-content|entry-content", re.I))
            or body
        )

    @staticmethod
    def _distributed_anchors(main, h2_tags: list, count: int) -> list:
        if count <= 0:
            return []
        if h2_tags:
            indexes = []
            for position in range(count):
                idx = round(position * (len(h2_tags) - 1) / max(count - 1, 1))
                if idx not in indexes:
                    indexes.append(idx)
            anchors = [ArticleImageFixer._section_anchor(h2_tags[idx]) for idx in indexes]
        else:
            anchors = []
        if len(anchors) < count:
            paragraphs = [p for p in main.find_all("p") if not p.find_parent("figcaption")]
            for position in range(len(anchors), count):
                if not paragraphs:
                    break
                idx = round((position + 1) * (len(paragraphs) - 1) / (count + 1))
                if paragraphs[idx] not in anchors:
                    anchors.append(paragraphs[idx])
        return anchors[:count]

    @staticmethod
    def _section_anchor(heading):
        current = heading.find_next_sibling()
        while current is not None:
            if getattr(current, "name", None) == "h2":
                break
            if getattr(current, "name", None) == "p" and current.get_text(" ", strip=True):
                return current
            current = current.find_next_sibling()
        return heading

    @staticmethod
    def _ensure_image_styles(soup: BeautifulSoup) -> None:
        if soup.find("style", id="site-inspector-article-images"):
            return
        head = soup.find("head")
        if not head:
            return
        style = soup.new_tag("style", id="site-inspector-article-images")
        style.string = (
            ".article-media{margin:32px auto;max-width:960px}.article-media img{display:block;"
            "width:100%;height:auto;aspect-ratio:3/2;object-fit:cover}.article-media figcaption{"
            "margin-top:8px;color:#5f6368;font-size:14px;line-height:1.5}.article-image-credit{"
            "display:block;font-size:12px}.article-image-credit a{color:inherit}"
        )
        head.append(style)

    @staticmethod
    def _build_generation_prompt(
        query: str,
        *,
        article_title: str = "",
        section_headings: list[str] | None = None,
        avoid_concepts: list[str] | None = None,
        variation_index: int = 0,
    ) -> str:
        visual_directions = (
            "a specific real-world subject in its working environment",
            "a detailed process-focused scene with meaningful objects and human scale",
            "a location-rich wide scene that makes the operational context unmistakable",
            "a precise editorial still life using objects directly discussed in the section",
        )
        sections = "; ".join((section_headings or [])[:6]) or "not provided"
        avoid = "; ".join((avoid_concepts or [])[-6:]) or "none"
        return (
            "Use case: photorealistic-natural. Asset type: editorial blog illustration. "
            f"Article title: {article_title or query}. Current section visual brief: {query}. "
            f"Other article sections for context only: {sections}. "
            "The image must visibly connect the article's primary subject with the current section; "
            "do not create a generic business, office, handshake, abstract finance, or unrelated stock scene. "
            f"Visual direction: {visual_directions[variation_index % len(visual_directions)]}. "
            f"Avoid repeating these previously used scene concepts: {avoid}. "
            "Composition: wide 3:2 editorial framing, one clear focal subject, natural light, "
            "credible materials and scale, and a composition distinct from earlier images. "
            "Constraints: factual, professional, no logos, no certificates, no text, no watermark."
        )

    @staticmethod
    def _generate_ai_filename(idx: int, url: str) -> str:
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        return f"article-{url_hash}-ai-{idx + 1}.webp"

    @staticmethod
    def _generate_filename(result, idx: int, url: str) -> str:
        """Generate a descriptive filename for a downloaded image."""
        import hashlib

        alt_slug = result.alt_text.lower().replace(" ", "-")[:40]
        alt_slug = re.sub(r"[^a-z0-9-]", "", alt_slug)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        source_abbr = result.source[:4]
        return f"article-{url_hash}-{source_abbr}-{idx + 1}-{alt_slug}.jpg"

    @staticmethod
    def _get_work_dir(source: BaseSource | None) -> str | None:
        """Extract the working directory from a source for file I/O."""
        if source is None:
            return None
        return (
            getattr(source, "root", None)
            or getattr(source, "_work_dir", None)
        )
