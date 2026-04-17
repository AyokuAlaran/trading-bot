"""
market_fetcher.py — Live Polymarket Market Fetcher

Fetches active binary prediction markets from Polymarket's Gamma API,
applies liquidity/volume filters, and maps them to the same schema as
mock_markets.json so bot.py and executor.py work without any changes.

Enable with:  USE_LIVE_MARKETS=true  in your .env file.

Caches the last successful fetch to markets_cache.json so the bot can
still run if the API is temporarily unavailable.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
GAMMA_API_URL      = "https://gamma-api.polymarket.com/markets"
CACHE_FILE         = "markets_cache.json"
REQUEST_TIMEOUT    = 15          # seconds before giving up on the API call
CACHE_MAX_AGE_S    = 3600        # treat cache as fresh for 1 hour
MIN_LIQUIDITY      = 1_000.0     # skip markets with less than $1 000 liquidity
MIN_VOLUME_24H     = 1_000.0     # skip markets with less than $1 000 daily volume
MAX_MARKETS        = 20          # cap how many markets we pass to Claude
FETCH_BATCH        = 200         # how many raw markets to pull before filtering


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value (string, int, float, None) to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_field(value, default=None):
    """
    Polymarket stores some list fields as JSON strings (e.g. outcomes, outcomePrices).
    Accept either a pre-parsed list or a JSON string.
    """
    if default is None:
        default = []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else default
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def _extract_yes_price(market: dict) -> Optional[float]:
    """
    Find the YES token price from a raw Polymarket market object.

    Polymarket stores outcomes as a JSON-stringified array, e.g.:
      outcomes:      '["Yes", "No"]'
      outcomePrices: '["0.62", "0.38"]'
    Index positions match, so we find the "Yes" index and return that price.
    """
    outcomes = _parse_json_field(market.get("outcomes"))
    prices   = _parse_json_field(market.get("outcomePrices"))

    if not outcomes or not prices or len(outcomes) != len(prices):
        return None

    for i, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() in ("yes", "true", "1"):
            price = _safe_float(prices[i], -1.0)
            if 0.0 < price < 1.0:
                return round(price, 4)

    return None


def _extract_category(market: dict) -> str:
    """Pull the first tag slug as the category, falling back to 'general'."""
    tags = market.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, dict):
            return str(first.get("slug") or first.get("label") or "general").lower()
    return "general"


def _parse_end_date(raw: str) -> str:
    """Normalise an ISO-8601 datetime to a YYYY-MM-DD date string."""
    if not raw:
        return ""
    try:
        # Handle both "2026-06-30T00:00:00Z" and bare "2026-06-30"
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return str(raw)[:10]  # best-effort slice


# ── Market mapping ─────────────────────────────────────────────────────────────

def _map_market(raw: dict) -> Optional[dict]:
    """
    Convert one raw Polymarket API market to the bot's internal schema.
    Returns None if the market should be skipped (resolved, low liquidity,
    non-binary, or missing required fields).
    """
    # Skip inactive / closed / archived / negRisk markets
    if not raw.get("active"):
        return None
    if raw.get("closed") or raw.get("archived"):
        return None
    if raw.get("negRisk"):
        # Negative-risk markets have different pricing mechanics; skip for now
        return None

    # Only handle binary (2-outcome) markets
    outcomes = _parse_json_field(raw.get("outcomes"))
    if len(outcomes) != 2:
        return None

    yes_price = _extract_yes_price(raw)
    if yes_price is None:
        return None

    # Volume / liquidity filters
    liquidity  = _safe_float(raw.get("liquidity", 0))
    volume_24h = _safe_float(
        raw.get("volume24hr") or raw.get("volume24h") or raw.get("oneDayVolume") or 0
    )

    if liquidity < MIN_LIQUIDITY:
        return None
    if volume_24h < MIN_VOLUME_24H:
        return None

    # Require a non-empty question
    question = str(raw.get("question") or "").strip()
    if not question:
        return None

    market_id = str(raw.get("id") or raw.get("conditionId") or "").strip()
    if not market_id:
        return None

    return {
        "market_id":        market_id,
        "question":         question,
        "category":         _extract_category(raw),
        "current_yes_price": yes_price,
        "volume_24h":       round(volume_24h, 2),
        "liquidity":        round(liquidity, 2),
        "end_date":         _parse_end_date(str(raw.get("endDate") or "")),
        "description":      str(raw.get("description") or "").strip(),
        # Extra fields kept for reference but ignored by bot.py / executor.py
        "slug":             str(raw.get("slug") or ""),
        "_source":          "live",
    }


# ── API fetch ──────────────────────────────────────────────────────────────────

def _fetch_raw_markets(limit: int = FETCH_BATCH) -> list:
    """
    Call the Polymarket Gamma API and return the raw list of market objects.
    Raises requests.RequestException on network / HTTP errors.
    """
    params = {
        "limit":     limit,
        "active":    "true",
        "closed":    "false",
        "archived":  "false",
        "order":     "volume24hr",   # sort by 24h volume descending
        "ascending": "false",
    }
    headers = {
        "Accept":     "application/json",
        "User-Agent": "polymarket-trading-bot/1.0",
    }

    resp = requests.get(
        GAMMA_API_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected Polymarket API response type: {type(data).__name__} "
            f"(expected list). First 200 chars: {str(data)[:200]}"
        )
    return data


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache(allow_stale: bool = False) -> Optional[list]:
    """
    Return cached markets if the cache file exists and is fresh enough.
    If allow_stale=True, return even an expired cache (used as last resort).
    """
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        markets = cache.get("markets")
        if not isinstance(markets, list) or not markets:
            return None

        age_s = time.time() - float(cache.get("fetched_at", 0))
        if not allow_stale and age_s > CACHE_MAX_AGE_S:
            return None  # stale — caller must decide whether to use it

        return markets
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _save_cache(markets: list):
    """Write a successful fetch to markets_cache.json."""
    payload = {
        "fetched_at":     time.time(),
        "fetched_at_iso": datetime.now(timezone.utc).isoformat(),
        "market_count":   len(markets),
        "markets":        markets,
    }
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        print(f"  WARNING: Could not write market cache ({CACHE_FILE}): {e}")


# ── Public interface ───────────────────────────────────────────────────────────

def fetch_live_markets(max_markets: int = MAX_MARKETS) -> list:
    """
    Return up to `max_markets` live Polymarket markets mapped to the bot's schema.

    Flow:
      1. Try the Gamma API (sorted by 24h volume, filtered for quality).
      2. On any error, fall back to a fresh-enough cache.
      3. If the cache is stale but the only option, use it with a warning.
      4. If nothing works, return [] so bot.py can fall back to mock data.
    """
    print("Fetching live markets from Polymarket Gamma API...")

    try:
        raw = _fetch_raw_markets(limit=FETCH_BATCH)
        print(f"  API returned {len(raw)} raw markets — filtering...")

        mapped = []
        skipped = 0
        for item in raw:
            try:
                m = _map_market(item)
                if m:
                    mapped.append(m)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
            if len(mapped) >= max_markets:
                break

        print(f"  {len(mapped)} markets passed filters, {skipped} skipped "
              f"(inactive / low volume / non-binary / negRisk)")

        if not mapped:
            raise ValueError("No markets survived the quality filter.")

        _save_cache(mapped)
        return mapped

    except Exception as e:
        print(f"  WARNING: Live fetch failed — {e}")

        # Fresh cache?
        markets = _load_cache(allow_stale=False)
        if markets:
            print(f"  Using cached markets ({len(markets)} markets, <1 h old)")
            return markets

        # Stale cache as absolute last resort
        markets = _load_cache(allow_stale=True)
        if markets:
            print(f"  WARNING: Cache is stale but it is the only data available. "
                  f"({len(markets)} markets)")
            return markets

        print(f"  ERROR: No live data and no cache found.")
        return []
