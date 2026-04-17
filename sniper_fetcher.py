"""
sniper_fetcher.py — Live Sports & Esports Result Fetcher

Fetches CONFIRMED match outcomes from:
  • PandaScore  (esports: CS2, LoL, Dota 2, Valorant)
  • API-Football (football/soccer)

STRICT no-fallback policy: if an API is down or returns no usable data,
this module raises SnifferAPIError. The sniper bot catches this and pauses
the current poll cycle — it never falls back to cached or mock data.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

REQUEST_TIMEOUT = 12   # seconds

PANDASCORE_BASE  = "https://api.pandascore.co"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# How far back to look for recently-finished matches
LOOKBACK_MINUTES = 30


class SnifferAPIError(Exception):
    """Raised when a live data API is unavailable or returns unusable data."""


@dataclass
class MatchResult:
    source:     str           # 'pandascore' | 'api_football'
    sport:      str           # 'cs2' | 'lol' | 'dota2' | 'valorant' | 'football'
    match_id:   str
    team_a:     str           # first/home team
    team_b:     str           # second/away team
    winner:     Optional[str] # winning team name, None if draw or unknown
    status:     str           # 'finished' | 'running'
    started_at: Optional[datetime]


# ── PandaScore ─────────────────────────────────────────────────────────────────

def _pandascore_headers() -> dict:
    key = os.environ.get("PANDASCORE_API_KEY", "")
    if not key:
        raise SnifferAPIError(
            "PANDASCORE_API_KEY is not set. "
            "Get a free key at pandascore.co and add it to .env"
        )
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _parse_pandascore_match(raw: dict, sport: str) -> Optional[MatchResult]:
    """Convert one raw PandaScore match dict into a MatchResult."""
    status = str(raw.get("status", "")).lower()
    if status not in ("finished", "running"):
        return None

    opponents = raw.get("opponents", [])
    if len(opponents) < 2:
        return None

    team_a = (opponents[0].get("opponent") or {}).get("name", "")
    team_b = (opponents[1].get("opponent") or {}).get("name", "")
    if not team_a or not team_b:
        return None

    winner_obj = raw.get("winner")
    winner = winner_obj.get("name") if isinstance(winner_obj, dict) else None

    begin_at = raw.get("begin_at") or raw.get("scheduled_at")
    started_at = None
    if begin_at:
        try:
            started_at = datetime.fromisoformat(begin_at.replace("Z", "+00:00"))
        except ValueError:
            pass

    return MatchResult(
        source="pandascore",
        sport=sport,
        match_id=str(raw.get("id", "")),
        team_a=team_a,
        team_b=team_b,
        winner=winner,
        status=status,
        started_at=started_at,
    )


def fetch_esports_results() -> list[MatchResult]:
    """
    Fetch recently-finished and currently-running esports matches from PandaScore.

    Raises SnifferAPIError if the API key is missing, the request fails,
    or the response contains no usable data.
    """
    headers = _pandascore_headers()
    since   = (datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)).isoformat()

    results: list[MatchResult] = []
    errors: list[str] = []

    # Videogames we care about and their PandaScore slug
    videogames = {
        "cs2":      "cs-go",
        "lol":      "lol",
        "dota2":    "dota-2",
        "valorant": "valorant",
    }

    for sport, slug in videogames.items():
        # Recently-finished matches
        for endpoint in (
            f"{PANDASCORE_BASE}/matches/past?filter[videogame]={slug}&range[end_at]={since},",
            f"{PANDASCORE_BASE}/{slug}/matches/running",
        ):
            try:
                resp = requests.get(
                    endpoint,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                    params={"per_page": 50},
                )
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                for raw in resp.json():
                    m = _parse_pandascore_match(raw, sport)
                    if m:
                        results.append(m)
            except requests.RequestException as e:
                errors.append(f"PandaScore/{sport}: {e}")

    if errors and not results:
        raise SnifferAPIError(
            f"PandaScore API unavailable — {len(errors)} error(s): {errors[0]}"
        )

    finished = [r for r in results if r.status == "finished" and r.winner]
    if not finished and not any(r.status == "running" for r in results):
        # API responded but no usable data right now — treat as transient
        raise SnifferAPIError(
            "PandaScore returned no finished or live matches at this time."
        )

    return results


# ── API-Football ───────────────────────────────────────────────────────────────

def _football_headers() -> dict:
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise SnifferAPIError(
            "API_FOOTBALL_KEY is not set. "
            "Get a free key at api-football.com and add it to .env"
        )
    return {"x-apisports-key": key, "Accept": "application/json"}


def _parse_football_fixture(raw: dict) -> Optional[MatchResult]:
    """Convert one API-Football fixture response item into a MatchResult."""
    fixture = raw.get("fixture", {})
    status  = (fixture.get("status") or {}).get("short", "")

    if status not in ("FT", "AET", "PEN", "1H", "HT", "2H", "ET", "BT"):
        return None

    is_finished = status in ("FT", "AET", "PEN")

    teams = raw.get("teams", {})
    home  = teams.get("home", {})
    away  = teams.get("away", {})

    team_a = home.get("name", "")
    team_b = away.get("name", "")
    if not team_a or not team_b:
        return None

    winner = None
    if is_finished:
        if home.get("winner") is True:
            winner = team_a
        elif away.get("winner") is True:
            winner = team_b
        # winner stays None for draws

    date_str = fixture.get("date")
    started_at = None
    if date_str:
        try:
            started_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    return MatchResult(
        source="api_football",
        sport="football",
        match_id=str(fixture.get("id", "")),
        team_a=team_a,
        team_b=team_b,
        winner=winner,
        status="finished" if is_finished else "running",
        started_at=started_at,
    )


def fetch_football_results() -> list[MatchResult]:
    """
    Fetch live and recently-finished football fixtures from API-Football.

    Raises SnifferAPIError if the API key is missing, the request fails,
    or the response contains no usable data.
    """
    headers = _football_headers()
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: list[MatchResult] = []
    errors:  list[str] = []

    for params in (
        {"live": "all"},
        {"date": today, "status": "FT"},
    ):
        try:
            resp = requests.get(
                f"{API_FOOTBALL_BASE}/fixtures",
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()

            # API-Football wraps results in response[]
            for item in body.get("response", []):
                m = _parse_football_fixture(item)
                if m:
                    results.append(m)
        except requests.RequestException as e:
            errors.append(str(e))

    if errors and not results:
        raise SnifferAPIError(
            f"API-Football unavailable — {errors[0]}"
        )

    if not results:
        raise SnifferAPIError(
            "API-Football returned no live or finished fixtures right now."
        )

    return results


# ── Combined fetch ─────────────────────────────────────────────────────────────

def fetch_all_results() -> list[MatchResult]:
    """
    Fetch from both PandaScore and API-Football.
    Returns combined list of results.
    Raises SnifferAPIError only if BOTH sources fail with zero results.
    Partial failures are logged but tolerated if at least one source works.
    """
    results: list[MatchResult] = []
    failures: list[str] = []

    for fetcher, label in (
        (fetch_esports_results, "PandaScore"),
        (fetch_football_results, "API-Football"),
    ):
        try:
            batch = fetcher()
            results.extend(batch)
            print(f"  [{label}] {len(batch)} match result(s) fetched")
        except SnifferAPIError as e:
            failures.append(f"{label}: {e}")
            print(f"  [{label}] WARNING: {e}")

    if not results:
        raise SnifferAPIError(
            f"All live data sources failed — cannot trade this cycle.\n"
            + "\n".join(failures)
        )

    return results
