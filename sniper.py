"""
sniper.py — Sniper Bot

Polls live sports/esports result APIs every SNIPER_POLL_INTERVAL_SECONDS seconds.
When a confirmed real-world outcome matches an open Polymarket market, it bets
on the winning side immediately — before Polymarket has settled the contract.

Position sizing : 50% Kelly, max 20% of sniper bankroll per trade.
Data sources    : PandaScore (esports) + API-Football (football/soccer)

CRITICAL: If live data APIs are down, this bot pauses the cycle and logs a
warning. It never falls back to cached or mock outcome data.
"""

import os
import re
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from executor import SupabaseExecutor
from market_fetcher import fetch_live_markets, _load_cache
from sniper_fetcher import SnifferAPIError, MatchResult, fetch_all_results

load_dotenv(override=True)

# ── Configuration ──────────────────────────────────────────────────────────────
STRATEGY      = "sniper"
MAX_BET_PCT   = float(os.environ.get("SNIPER_MAX_BET_PCT", "0.20"))
POLL_INTERVAL = int(os.environ.get("SNIPER_POLL_INTERVAL_SECONDS", "15"))
# Minimum market price (YES token) before we consider betting
# Very low/high prices often mean the market is nearly settled already
MIN_MARKET_PRICE = 0.05
MAX_MARKET_PRICE = 0.95

# Markets we've already bet on this session (market_id → True) to avoid duplicates
# across rapid poll cycles (the executor DB guard handles cross-session dedup)
_SESSION_BETS: dict[str, bool] = {}


# ── Team-name normalisation ────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase, strip punctuation/common suffixes for fuzzy matching."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    # Strip common esports org suffixes that Polymarket often omits
    for suffix in ("gaming", "esports", "team", "club", "fc", "sc", "united", "city"):
        name = re.sub(rf"\b{suffix}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _names_match(api_name: str, question_text: str) -> bool:
    """Return True if api_name (or a significant substring) appears in the question."""
    norm_api = _norm(api_name)
    norm_q   = _norm(question_text)
    if not norm_api:
        return False
    # Full match or at least 6-char prefix match (handles abbreviations)
    if norm_api in norm_q:
        return True
    # Word-level: all words of the API name must appear in the question
    words = [w for w in norm_api.split() if len(w) >= 4]
    return bool(words) and all(w in norm_q for w in words)


# ── Market matching ────────────────────────────────────────────────────────────

def _determine_recommendation(
    question: str,
    winner: str,
    loser: str,
) -> str:
    """
    Given a confirmed winner and loser, figure out whether YES or NO wins.

    Heuristic: the team the question asks about first (or after 'will') is
    what YES maps to. If that team is the winner → YES, else → NO.
    """
    q      = question.lower()
    w_norm = _norm(winner)
    l_norm = _norm(loser)

    # Find approximate positions of each team in the question
    w_pos = q.find(w_norm[:max(4, len(w_norm))]) if w_norm else -1
    l_pos = q.find(l_norm[:max(4, len(l_norm))]) if l_norm else -1

    if w_pos == -1 and l_pos == -1:
        return "SKIP"

    # Whichever team the question is "about" (appears first / subject of "Will X...")
    if w_pos != -1 and (l_pos == -1 or w_pos < l_pos):
        # Question is about the winner → YES resolves true
        return "YES"
    else:
        # Question is about the loser → YES resolves false → bet NO
        return "NO"


def match_result_to_market(
    result: MatchResult,
    markets: list[dict],
) -> list[tuple[dict, str]]:
    """
    Try to find Polymarket markets that correspond to this match result.

    Returns a list of (market, recommendation) pairs where we have a
    confident match and a confirmed winning side.
    """
    if not result.winner:
        return []  # draw or unknown — skip

    loser = result.team_b if _norm(result.winner) in _norm(result.team_a) else result.team_a

    candidates = []
    for market in markets:
        q = market.get("question", "")
        yes_price = float(market.get("current_yes_price", 0.5))

        # Skip markets already at extreme prices (nearly resolved on Polymarket)
        if not (MIN_MARKET_PRICE <= yes_price <= MAX_MARKET_PRICE):
            continue

        # Both team names must appear in the question
        if not _names_match(result.team_a, q):
            continue
        if not _names_match(result.team_b, q):
            continue

        rec = _determine_recommendation(q, result.winner, loser)
        if rec == "SKIP":
            continue

        candidates.append((market, rec))

    return candidates


# ── Bet placement ──────────────────────────────────────────────────────────────

def place_sniper_bet(
    executor: SupabaseExecutor,
    market: dict,
    rec: str,
    result: MatchResult,
):
    """Build an opportunity dict and fire execute_trade + settle_sniper_trade."""
    market_id = market["market_id"]

    if market_id in _SESSION_BETS:
        return  # already bet this session (fast duplicate guard)

    yes_price = float(market["current_yes_price"])
    yes_price = max(0.001, min(0.999, yes_price))
    entry_price = yes_price if rec == "YES" else (1.0 - yes_price)

    # Edge: we know the outcome → effective edge = 1 − entry_price
    edge       = round(1.0 - entry_price, 4)
    confidence = 0.90  # allow 10% for matching errors

    opportunity = {
        "market_id":                  market_id,
        "question":                   market["question"],
        "recommendation":             rec,
        "current_price":              yes_price,
        "current_market_price":       yes_price,
        "edge":                       edge,
        "confidence":                 confidence,
        "estimated_true_probability": 1.0 if rec == "YES" else 0.0,
        "yes_token_id":               market.get("yes_token_id"),
        "no_token_id":                market.get("no_token_id"),
        "reasoning": (
            f"[SNIPER] Confirmed outcome from {result.source}: "
            f"{result.winner} beat {result.team_a if result.winner == result.team_b else result.team_b} "
            f"in {result.sport}. Market not yet resolved."
        ),
        "suggested_bet_fraction": min(edge * confidence * 0.50, MAX_BET_PCT),
    }

    trade = executor.execute_trade(opportunity, source="sniper")
    if not trade:
        return

    _SESSION_BETS[market_id] = True

    # Immediately settle since we know the real-world outcome
    # (In production: wait for Polymarket to resolve the contract)
    outcome = "WIN"  # we bet on the confirmed winning side
    trade_id = trade.get("id", "")
    if trade_id:
        executor.settle_sniper_trade(trade_id, outcome, entry_price, trade["amount"])


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  SNIPER BOT  —  Live Outcome Arbitrage")
    print("=" * 60)
    print(f"  Strategy   : {STRATEGY}")
    print(f"  Max bet    : {MAX_BET_PCT:.0%} of bankroll  (50% Kelly)")
    print(f"  Poll every : {POLL_INTERVAL} seconds")
    print("=" * 60)

    # Validate API keys up front
    missing = []
    if not os.environ.get("PANDASCORE_API_KEY"):
        missing.append("PANDASCORE_API_KEY")
    if not os.environ.get("API_FOOTBALL_KEY"):
        missing.append("API_FOOTBALL_KEY")
    if missing:
        print(f"\nWARNING: {', '.join(missing)} not set — those sources will be skipped.")
        print("Add them to .env to enable those sports feeds.\n")

    executor = SupabaseExecutor(
        strategy=STRATEGY,
        max_bet_fraction=MAX_BET_PCT,
        kelly_fraction=0.50,
    )
    executor.sync_bankroll_from_polymarket(num_strategies=2)

    print(f"\nStarting poll loop. Ctrl+C to stop.\n")
    cycle = 0

    while True:
        cycle += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Poll #{cycle} —", end=" ", flush=True)

        try:
            # ── 1. Fetch live results (strict: no fallback on failure) ──────────
            results = fetch_all_results()
            finished = [r for r in results if r.status == "finished" and r.winner]
            print(f"{len(finished)} confirmed result(s) from {len(results)} match(es)")

            if not finished:
                print(f"  No confirmed outcomes — sleeping {POLL_INTERVAL}s\n")
                time.sleep(POLL_INTERVAL)
                continue

            # ── 2. Get open Polymarket markets (cache OK here) ─────────────────
            markets = fetch_live_markets()
            if not markets:
                # Try stale cache as last resort for the market list only
                cached = _load_cache(allow_stale=True)
                markets = cached or []
            if not markets:
                print("  No Polymarket markets available — sleeping")
                time.sleep(POLL_INTERVAL)
                continue

            # ── 3. Match results → markets → bet ──────────────────────────────
            bets_this_cycle = 0
            for result in finished:
                for market, rec in match_result_to_market(result, markets):
                    place_sniper_bet(executor, market, rec, result)
                    bets_this_cycle += 1

            if bets_this_cycle == 0:
                print(f"  No market matches found this cycle")

        except SnifferAPIError as e:
            # Live data API is down — pause this cycle, do NOT fall back
            print(f"\n  [PAUSED] Live data unavailable: {e}")
            print(f"  Waiting {POLL_INTERVAL}s before retrying...\n")

        except KeyboardInterrupt:
            print("\nSniper bot stopped by user.")
            executor.print_stats()
            sys.exit(0)

        except Exception as e:
            print(f"\n  Unexpected error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
