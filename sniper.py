"""
sniper.py — News/Political Event Sniper Bot

Polls RSS news feeds every SNIPER_POLL_INTERVAL_SECONDS.
When a breaking headline definitively resolves an open Polymarket market,
Claude Haiku identifies the match and the bot bets immediately.

Also pre-positions on markets resolving within 6 hours if Claude Haiku
assesses the likely outcome with >80% confidence.

Requirements: confidence > 0.80 before any trade is placed.
Position sizing: 50% Kelly × liquidity multiplier, max 20% per trade.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import anthropic
from dotenv import load_dotenv

from executor import SupabaseExecutor
from market_fetcher import fetch_live_markets, _load_cache
from sniper_fetcher import SnifferAPIError, NewsItem, fetch_breaking_news

load_dotenv(override=True)

STRATEGY       = "sniper"
MAX_BET_PCT    = float(os.environ.get("SNIPER_MAX_BET_PCT", "0.20"))
POLL_INTERVAL  = int(os.environ.get("SNIPER_POLL_INTERVAL_SECONDS", "15"))
HAIKU_MODEL    = "claude-haiku-4-5-20251001"
MIN_CONFIDENCE = 0.80

_PROCESSED_URLS: set[str]  = set()   # news URLs already sent to Claude
_SESSION_BETS: dict[str, bool] = {}  # market_ids bet on this session
_ASSESSED_CALENDAR: set[str] = set() # market_ids checked via calendar this session


# ── Claude Haiku matching ──────────────────────────────────────────────────────

def _match_news_to_market(
    client: anthropic.Anthropic,
    item: NewsItem,
    markets: list[dict],
) -> Optional[dict]:
    """Ask Haiku if this headline definitively resolves any market (conf > MIN_CONFIDENCE)."""
    market_list = [
        {"market_id": m["market_id"], "question": m["question"], "yes_price": m["current_yes_price"]}
        for m in markets
    ]
    prompt = (
        f"You are a prediction market analyst.\n\n"
        f"Breaking news:\nHeadline: {item.headline}\nSummary: {item.summary}\n"
        f"Source: {item.source}\nPublished: {item.published_at}\n\n"
        f"Open Polymarket markets:\n{json.dumps(market_list, indent=2)}\n\n"
        f"Does this news DEFINITIVELY confirm the outcome of any listed market?\n"
        f"Rules:\n"
        f"- Only match if the news conclusively proves YES or NO resolution\n"
        f"- Do NOT match on speculation or partial information\n"
        f"- confidence > {MIN_CONFIDENCE} is required\n\n"
        f'If a confident match exists, return ONLY valid JSON:\n'
        f'{{"market_id":"...","question":"...","side":"YES" or "NO","confidence":0.0-1.0,"reasoning":"..."}}\n\n'
        f"If no confident match, return ONLY the word: NONE"
    )
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.upper().startswith("NONE"):
            return None
        if "```json" in raw:
            raw = raw[raw.find("```json") + 7 : raw.find("```", raw.find("```json") + 7)].strip()
        elif "```" in raw:
            raw = raw[raw.find("```") + 3 : raw.find("```", raw.find("```") + 3)].strip()
        if not raw.lstrip().startswith("{"):
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e > s:
                raw = raw[s : e + 1]
            else:
                return None
        match = json.loads(raw)
        if float(match.get("confidence", 0)) < MIN_CONFIDENCE:
            return None
        if match.get("side") not in ("YES", "NO"):
            return None
        return match
    except Exception:
        return None


# ── Scheduled resolution calendar ─────────────────────────────────────────────

def check_resolution_calendar(
    executor: SupabaseExecutor,
    client: anthropic.Anthropic,
    markets: list[dict],
):
    """
    Find markets resolving in the next 6 hours. For each unassessed market,
    ask Claude Haiku for the most likely outcome. Pre-position if conf > MIN_CONFIDENCE.
    """
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=6)

    soon = []
    for m in markets:
        dt_str = m.get("end_datetime", "")
        if not dt_str:
            continue
        try:
            end_dt = datetime.fromisoformat(dt_str)
            if now <= end_dt <= cutoff and m["market_id"] not in _ASSESSED_CALENDAR:
                soon.append((m, end_dt))
        except (ValueError, TypeError):
            continue

    if not soon:
        return

    print(f"\n  [CALENDAR] {len(soon)} market(s) resolving within 6 hours:")
    for m, end_dt in soon:
        mins = int((end_dt - now).total_seconds() / 60)
        print(f"    {m['market_id']}  ({mins}m)  {m['question'][:60]}")
        _ASSESSED_CALENDAR.add(m["market_id"])

        prompt = (
            f"A Polymarket prediction market resolves in {mins} minutes.\n"
            f"Market: {m['question']}\n"
            f"Current YES price: {m['current_yes_price']:.2f}\n\n"
            f"Based on current events and your knowledge, what is the most likely outcome?\n"
            f"If confidence > {MIN_CONFIDENCE}, return JSON:\n"
            f'{{"side":"YES" or "NO","confidence":0.0-1.0,"reasoning":"..."}}\n'
            f"Otherwise return ONLY: NONE"
        )
        try:
            resp = client.messages.create(
                model=HAIKU_MODEL, max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.upper().startswith("NONE"):
                print(f"    -> No confident assessment")
                continue
            if not raw.lstrip().startswith("{"):
                s, e = raw.find("{"), raw.rfind("}")
                if s != -1 and e > s:
                    raw = raw[s : e + 1]
                else:
                    continue
            rec = json.loads(raw)
            conf = float(rec.get("confidence", 0))
            side = rec.get("side", "")
            if conf < MIN_CONFIDENCE or side not in ("YES", "NO"):
                print(f"    -> Low confidence ({conf:.0%}) — skipping")
                continue
            print(f"    -> {side} at {conf:.0%}: {rec.get('reasoning','')[:80]}")
            fake_item = NewsItem(
                headline=f"[CALENDAR] {m['question']}",
                summary=rec.get("reasoning", ""),
                source="calendar",
                published_at=now,
                url=f"calendar://{m['market_id']}",
            )
            _place_sniper_bet(executor, m, {"side": side, "confidence": conf,
                                             "reasoning": rec.get("reasoning", ""),
                                             "market_id": m["market_id"],
                                             "question": m["question"]}, fake_item)
        except Exception as e:
            print(f"    -> Calendar assessment error: {e}")


# ── Bet placement ──────────────────────────────────────────────────────────────

def _place_sniper_bet(
    executor: SupabaseExecutor,
    market: dict,
    match: dict,
    item: NewsItem,
):
    market_id = market["market_id"]
    if market_id in _SESSION_BETS:
        return

    side        = match["side"]
    yes_price   = max(0.001, min(0.999, float(market["current_yes_price"])))
    entry_price = yes_price if side == "YES" else (1.0 - yes_price)
    edge        = round(1.0 - entry_price, 4)
    confidence  = float(match.get("confidence", 0.90))

    opportunity = {
        "market_id":                  market_id,
        "question":                   market["question"],
        "recommendation":             side,
        "current_price":              yes_price,
        "current_market_price":       yes_price,
        "edge":                       edge,
        "confidence":                 confidence,
        "estimated_true_probability": 1.0 if side == "YES" else 0.0,
        "yes_token_id":               market.get("yes_token_id"),
        "no_token_id":                market.get("no_token_id"),
        "volume_24h":                 market.get("volume_24h", 0),
        "reasoning": (
            f"[SNIPER] {item.source}: \"{item.headline[:120]}\" | "
            f"{match.get('reasoning', '')}"
        ),
        "suggested_bet_fraction": min(edge * confidence * 0.50, MAX_BET_PCT),
    }

    trade = executor.execute_trade(opportunity, source="sniper")
    if trade:
        _SESSION_BETS[market_id] = True
        print(f"    [BET PLACED] {side} on '{market['question'][:60]}'")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  SNIPER BOT  —  News / Political Event Arbitrage")
    print("=" * 60)
    print(f"  Strategy    : {STRATEGY}")
    print(f"  Model       : {HAIKU_MODEL}")
    print(f"  Min conf    : {MIN_CONFIDENCE:.0%}")
    print(f"  Max bet     : {MAX_BET_PCT:.0%} of bankroll  (50% Kelly)")
    print(f"  Poll every  : {POLL_INTERVAL}s")
    print(f"  News window : 15 minutes")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    executor = SupabaseExecutor(
        strategy=STRATEGY,
        max_bet_fraction=MAX_BET_PCT,
        kelly_fraction=0.50,
    )
    executor.sync_bankroll_from_polymarket(num_strategies=2)
    client = anthropic.Anthropic(api_key=api_key)

    run_once = "--once" in sys.argv
    print(f"\nStarting news poll loop. Ctrl+C to stop.\n")
    cycle = 0

    while True:
        cycle += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Poll #{cycle}")

        try:
            # 1. Settle any resolved positions
            print("  Settling open positions...")
            executor.settle_open_positions()

            # 2. Fetch open markets
            markets = fetch_live_markets()
            if not markets:
                cached  = _load_cache(allow_stale=True)
                markets = cached or []
            if not markets:
                print("  No Polymarket markets available — sleeping")
                if run_once:
                    break
                time.sleep(POLL_INTERVAL)
                continue

            print(f"  {len(markets)} Polymarket markets loaded")

            # 3. Calendar: pre-position on markets resolving soon
            check_resolution_calendar(executor, client, markets)

            # 4. Fetch breaking news
            news_items = fetch_breaking_news()
            new_items  = [n for n in news_items if n.url not in _PROCESSED_URLS]
            print(f"  {len(news_items)} recent article(s), {len(new_items)} unprocessed")

            if not new_items:
                print(f"  Nothing new — sleeping {POLL_INTERVAL}s\n")
                if run_once:
                    break
                time.sleep(POLL_INTERVAL)
                continue

            for n in new_items:
                _PROCESSED_URLS.add(n.url)

            # 5. Match each headline against markets
            bets_this_cycle = 0
            for item in new_items:
                print(f"  Checking: \"{item.headline[:80]}\"")
                match = _match_news_to_market(client, item, markets)
                if match is None:
                    print(f"    -> No confident match")
                    continue
                conf = float(match.get("confidence", 0))
                print(f"    -> MATCH  {match['side']} on '{match.get('question','')[:55]}'  conf={conf:.0%}")
                print(f"       {match.get('reasoning','')[:100]}")
                market = next((m for m in markets if m["market_id"] == match.get("market_id")), None)
                if market:
                    _place_sniper_bet(executor, market, match, item)
                    bets_this_cycle += 1

            if bets_this_cycle == 0:
                print(f"  No confident matches this cycle")

        except SnifferAPIError as e:
            print(f"\n  [PAUSED] News feeds unavailable: {e}")

        except KeyboardInterrupt:
            print("\nSniper bot stopped by user.")
            executor.print_stats()
            sys.exit(0)

        except Exception as e:
            print(f"\n  Unexpected error: {e}")

        if run_once:
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
