from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web and return a compact list of results.

    Args:
        query: Search query.
        max_results: Maximum number of web results.

    Returns:
        List of dicts with title, url, snippet.
    """
    hits: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for r in results:
            hits.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return hits


def fetch_url_text(url: str, timeout: int = 12, max_chars: int = 4000) -> str:
    """Fetch and clean main text from a URL for grounding.

    Args:
        url: Page URL.
        timeout: Request timeout seconds.
        max_chars: Truncate output to this length.

    Returns:
        Cleaned page text (possibly truncated).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AskTheWebAgent/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]
