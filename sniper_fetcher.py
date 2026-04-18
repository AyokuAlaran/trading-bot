"""
sniper_fetcher.py — Breaking News Feed Monitor

Fetches headlines from free RSS feeds and returns only articles published
in the last NEWS_WINDOW_MINUTES minutes.

Sources:
  Reuters Top News, AP News, BBC News, Reuters Politics

Raises SnifferAPIError if ALL sources return no data.
"""

import re
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import feedparser

NEWS_WINDOW_MINUTES = 15

RSS_FEEDS = {
    "BBC News":          "http://feeds.bbci.co.uk/news/rss.xml",
    "BBC World":         "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Guardian World":    "https://www.theguardian.com/world/rss",
    "Al Jazeera":        "https://www.aljazeera.com/xml/rss/all.xml",
}


class SnifferAPIError(Exception):
    """Raised when all news sources fail or return nothing."""


@dataclass
class NewsItem:
    headline:     str
    summary:      str
    source:       str
    published_at: Optional[datetime]
    url:          str


def _parse_pub_date(entry) -> Optional[datetime]:
    """Extract UTC datetime from a feedparser entry's published_parsed field."""
    t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if t is None:
        return None
    try:
        return datetime(*t[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:500]


def _fetch_feed(name: str, url: str, cutoff: datetime) -> list[NewsItem]:
    """Parse one RSS feed and return items newer than cutoff."""
    try:
        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            return []

        items = []
        for entry in feed.entries:
            pub = _parse_pub_date(entry)
            if pub is None or pub < cutoff:
                continue

            headline = getattr(entry, "title", "").strip()
            raw_sum  = getattr(entry, "summary", "") or getattr(entry, "description", "")
            summary  = _strip_html(raw_sum)
            link     = getattr(entry, "link", "")

            if not headline or not link:
                continue

            items.append(NewsItem(
                headline=headline,
                summary=summary,
                source=name,
                published_at=pub,
                url=link,
            ))
        return items
    except Exception:
        return []


def fetch_breaking_news() -> list[NewsItem]:
    """
    Return articles from all RSS feeds published in the last NEWS_WINDOW_MINUTES.
    Results are sorted newest-first.
    Raises SnifferAPIError if all sources fail AND no articles were found.
    """
    cutoff    = datetime.now(timezone.utc) - timedelta(minutes=NEWS_WINDOW_MINUTES)
    all_items: list[NewsItem] = []
    failures:  list[str]      = []

    for name, url in RSS_FEEDS.items():
        try:
            batch = _fetch_feed(name, url, cutoff)
            if batch:
                print(f"  [{name}] {len(batch)} article(s) in last {NEWS_WINDOW_MINUTES}m")
            else:
                print(f"  [{name}] 0 articles in last {NEWS_WINDOW_MINUTES}m")
            all_items.extend(batch)
        except Exception as e:
            failures.append(f"{name}: {e}")
            print(f"  [{name}] ERROR: {e}")

    if not all_items and len(failures) == len(RSS_FEEDS):
        raise SnifferAPIError(f"All {len(RSS_FEEDS)} RSS feeds failed: {failures[0]}")

    all_items.sort(
        key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return all_items
