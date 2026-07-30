from __future__ import annotations

from bs4 import BeautifulSoup

from src.agents.models import ImagePlacementSlot, ImagePlan


class ArticleImageAgent:
    """Decide how many images an article needs and where they belong."""

    MIN_IMAGES = 3
    MAX_IMAGES = 10

    def plan(
        self,
        html: str,
        *,
        research_report: dict | None = None,
        requested_target: int | None = None,
    ) -> ImagePlan:
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article") or soup.find("main") or soup.body or soup
        title_tag = article.find("h1") or soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else "Article"
        headings = [
            heading.get_text(" ", strip=True)
            for heading in article.find_all("h2")
            if heading.get_text(" ", strip=True)
        ]
        existing_count = len(article.find_all("img"))
        capacity = max(self.MIN_IMAGES, min(self.MAX_IMAGES, len(headings) + 1))
        brief = ((research_report or {}).get("writing_brief") or {})
        recommended_min = self._bounded_int(brief.get("image_count_min"), 4)
        recommended_max = self._bounded_int(
            brief.get("image_count_max"), recommended_min
        )
        recommended = round((recommended_min + recommended_max) / 2)
        desired = requested_target if requested_target is not None else recommended
        target = max(self.MIN_IMAGES, min(self.MAX_IMAGES, int(desired)))
        target = min(target, capacity)
        target = max(target, min(existing_count, self.MAX_IMAGES))
        needed = max(0, target - existing_count)

        slots: list[ImagePlacementSlot] = []
        if needed and existing_count == 0:
            slots.append(ImagePlacementSlot(
                kind="hero",
                heading=title,
                visual_brief=f"A concrete lead image representing {title}",
                slot_id="hero",
                image_type="photo",
                insertion_reason="Establish the article's main subject after the introduction.",
            ))
        section_slots = max(0, needed - len(slots))
        for heading in self._spread_headings(headings, section_slots):
            slots.append(ImagePlacementSlot(
                kind="section",
                heading=heading,
                visual_brief=f"A specific real-world scene illustrating {heading} in {title}",
                slot_id=f"section-{len(slots) + 1}",
                image_type="photo",
                insertion_reason=f"Clarify the concrete idea discussed in {heading}.",
                section_index=headings.index(heading),
            ))

        rationale = (
            f"Selected {target} total images from {len(headings)} sections; "
            f"{existing_count} already exist and {needed} new images are needed."
        )
        return ImagePlan(
            target_count=target,
            existing_count=existing_count,
            needed_count=needed,
            article_title=title,
            section_count=len(headings),
            placement_slots=slots,
            rationale=rationale,
        )

    @classmethod
    def _bounded_int(cls, value, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(cls.MIN_IMAGES, min(cls.MAX_IMAGES, number))

    @staticmethod
    def _spread_headings(headings: list[str], count: int) -> list[str]:
        if count <= 0 or not headings:
            return []
        if count >= len(headings):
            return headings[:count]
        indexes = [round(index * (len(headings) - 1) / max(1, count - 1)) for index in range(count)]
        return [headings[index] for index in dict.fromkeys(indexes)]
