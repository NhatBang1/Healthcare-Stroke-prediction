from __future__ import annotations

import httpx
from langchain_core.tools import tool
from urllib.parse import urlparse

from app.core.config import settings


_IMAGE_TERMS = (
    "đau đầu",
    "dấu hiệu đột quỵ",
    "triệu chứng đột quỵ",
    "fast",
    "méo miệng",
    "yếu tay",
    "yếu chân",
    "nói khó",
    "tê một bên",
    "stroke",
)

_WEB_SEARCH_TERMS = (
    "bài báo",
    "bai bao",
    "nghiên cứu",
    "nghien cuu",
    "tài liệu",
    "tai lieu",
    "nguồn",
    "nguon",
    "khẳng định",
    "khang dinh",
    "evidence",
    "paper",
    "article",
    "guideline",
    "hút thuốc",
    "thuốc lá",
    "uống rượu",
    "rượu bia",
    "bệnh tim",
    "suy tim",
    "ảnh hưởng",
)

_AUTHORITATIVE_DOMAINS = (
    "stroke.org",
    "cdc.gov",
    "nhs.uk",
    "who.int",
    "nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "heart.org",
)


def should_search_stroke_images(message: str) -> bool:
    normalized = str(message or "").casefold()
    return any(term in normalized for term in _IMAGE_TERMS)


def should_search_stroke_web(message: str) -> bool:
    normalized = str(message or "").casefold()
    return any(term in normalized for term in _WEB_SEARCH_TERMS)


@tool("search_stroke_education_images")
def search_stroke_education_images(query: str) -> list[dict[str, str]]:
    """Search authoritative stroke-education images through Tavily when configured."""
    if not settings.tavily_api_key:
        return []
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "topic": "general",
                "search_depth": "basic",
                "max_results": 4,
                "include_images": True,
                "include_image_descriptions": True,
                "include_answer": False,
                "include_raw_content": False,
                "include_domains": ["stroke.org", "cdc.gov", "nhs.uk", "who.int"],
            },
            timeout=4.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    results = payload.get("results") or []
    first_source = results[0] if results and isinstance(results[0], dict) else {}
    source_url = str(first_source.get("url") or "")
    if not source_url.startswith(("https://", "http://")):
        source_url = ""
    media: list[dict[str, str]] = []
    for item in payload.get("images") or []:
        if isinstance(item, str):
            url = item
            title = "Minh họa dấu hiệu đột quỵ"
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("image_url") or "")
            title = str(item.get("description") or item.get("title") or "Minh họa dấu hiệu đột quỵ")
        else:
            continue
        if not url.startswith(("https://", "http://")):
            continue
        media.append({
            "type": "image",
            "url": url,
            "title": title[:180],
            "source_url": source_url,
        })
        if len(media) == 2:
            break
    return media


@tool("search_stroke_web_sources")
def search_stroke_web_sources(query: str) -> list[dict[str, str]]:
    """Search authoritative web sources for stroke education and evidence."""
    if not settings.tavily_api_key:
        return []
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "topic": "general",
                "search_depth": "advanced",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
                "include_domains": ["stroke.org", "cdc.gov", "nhs.uk", "who.int", "nih.gov"],
            },
            timeout=4.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    sources: list[dict[str, str]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not url.startswith(("https://", "http://")):
            continue
        hostname = urlparse(url).hostname or ""
        if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in _AUTHORITATIVE_DOMAINS):
            continue
        sources.append(
            {
                "title": str(item.get("title") or "Nguồn tham khảo đột quỵ")[:180],
                "url": url,
                "content": " ".join(str(item.get("content") or "").split())[:700],
                "published_date": str(item.get("published_date") or "")[:40],
            }
        )
        if len(sources) == 5:
            break
    return sources


def search_images_for_message(message: str) -> list[dict[str, str]]:
    if not should_search_stroke_images(message):
        return []
    return search_stroke_education_images.invoke({
        "query": "stroke warning signs FAST headache medical education illustration",
    })


def search_sources_for_message(message: str) -> list[dict[str, str]]:
    if not should_search_stroke_web(message):
        return []
    return search_stroke_web_sources.invoke(
        {"query": f"stroke risk smoking alcohol heart disease evidence guideline {message}"}
    )


def search_risk_factor_sources(profile: dict[str, object]) -> list[dict[str, str]]:
    """Fetch concise authoritative evidence for explicitly supplied risk factors."""
    factors: list[str] = []
    if profile.get("heart_disease"):
        factors.append("heart disease heart failure atrial fibrillation")
    if profile.get("smoke_flag"):
        factors.append("smoking tobacco")
    if profile.get("alco_flag"):
        factors.append("alcohol consumption")
    if not factors or not settings.tavily_api_key:
        return []
    query = (
        "stroke risk association evidence guideline systematic review "
        + " ".join(factors)
    )
    return search_stroke_web_sources.invoke({"query": query})[:3]
