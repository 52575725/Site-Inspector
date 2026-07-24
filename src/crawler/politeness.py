from __future__ import annotations

import logging
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)


async def check_robots_txt(base_url: str, user_agent: str,
                           client: httpx.AsyncClient) -> dict:
    """Check robots.txt and return crawl parameters."""
    robots_url = urljoin(base_url, "/robots.txt")
    result = {
        "allowed": True,
        "crawl_delay": None,
        "sitemaps": [],
        "disallowed_paths": [],
    }

    try:
        resp = await client.get(robots_url, follow_redirects=True)
        resp.raise_for_status()
        content = resp.text
    except Exception:
        logger.info(f"No robots.txt found at {robots_url}, assuming full access")
        return result

    current_agent = "*"
    agent_specific_rules = False

    for line in content.splitlines():
        line = line.strip()

        # Track which user-agent section we're in
        lower_line = line.lower()
        if lower_line.startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip().lower()
            if agent == user_agent.lower():
                current_agent = user_agent.lower()
                agent_specific_rules = True
            elif agent == "*":
                current_agent = "*"
            else:
                current_agent = None

        if current_agent is None:
            continue

        if lower_line.startswith("crawl-delay:"):
            try:
                result["crawl_delay"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

        elif lower_line.startswith("sitemap:"):
            result["sitemaps"].append(line.split(":", 1)[1].strip())

        elif lower_line.startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            result["disallowed_paths"].append(path)

    return result
