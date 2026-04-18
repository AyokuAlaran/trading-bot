"""
bot.py — Analytical Bot (Claude AI)

Loads live or mock prediction markets, asks Claude to find edges > 5%,
and places trades via the Supabase-backed executor.

Position sizing : 25% Kelly, max 5% of analytical bankroll per trade.
Scheduling      : loops every ANALYTICAL_BOT_INTERVAL_MINUTES (default 30).
Data source     : USE_LIVE_MARKETS=true  → Polymarket Gamma API
                  USE_LIVE_MARKETS=false → mock_markets.json  (default)
"""

import json
import os
import sys
import time
from datetime import datetime

# Force UTF-8 output so Unicode box/dash chars print on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import anthropic
from dotenv import load_dotenv

from executor import SupabaseExecutor
from market_fetcher import fetch_live_markets

load_dotenv(override=True)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL            = "claude-sonnet-4-6"
MARKETS_FILE     = "mock_markets.json"
MIN_EDGE         = 0.03
MAX_TOKENS       = 16000
STRATEGY         = "analytical"
MAX_BET_PCT      = float(os.environ.get("ANALYTICAL_MAX_BET_PCT", "0.05"))
LOOP_INTERVAL_M  = int(os.environ.get("ANALYTICAL_BOT_INTERVAL_MINUTES", "30"))


# ── Market loading ─────────────────────────────────────────────────────────────

def load_mock_markets(filepath: str = MARKETS_FILE) -> list:
    if not os.path.exists(filepath):
        print(f"ERROR: '{filepath}' not found.")
        sys.exit(1)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            markets = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Could not load '{filepath}': {e}")
        sys.exit(1)
    if not isinstance(markets, list) or not markets:
        print(f"ERROR: '{filepath}' must be a non-empty JSON array.")
        sys.exit(1)
    required = {"market_id", "question", "current_yes_price"}
    for i, m in enumerate(markets):
        missing = required - m.keys()
        if missing:
            print(f"ERROR: Market {i} missing fields: {missing}")
            sys.exit(1)
    print(f"Loaded {len(markets)} mock markets from {filepath}")
    return markets


def get_markets() -> list:
    """Return markets from live API or mock file based on USE_LIVE_MARKETS."""
    use_live = os.environ.get("USE_LIVE_MARKETS", "false").strip().lower() == "true"
    if use_live:
        markets = fetch_live_markets()
        if not markets:
            print("WARNING: Live fetch empty — falling back to mock_markets.json")
            return load_mock_markets()
        return markets
    return load_mock_markets()


# ── Claude analysis ────────────────────────────────────────────────────────────

def analyze_markets(client: anthropic.Anthropic, markets: list, bankroll: float) -> dict:
    """
    Send markets to Claude and receive JSON trading recommendations.
    Raises ValueError on parse failure; anthropic.APIError on API failure.
    """
    system_prompt = f"""You are a professional prediction market analyst and quantitative trader.

Your task: analyze open prediction markets and identify profitable trading opportunities.

For each market:
1. Estimate the TRUE probability of the event resolving YES
2. Compare to the current YES token price (market probability)
3. Calculate edge = |true_probability - market_price|
4. Recommend YES when true_probability > market_price + {MIN_EDGE}
5. Recommend NO  when true_probability < market_price - {MIN_EDGE}
6. Suggest bet_fraction using 25% Kelly: edge * confidence * 0.25, capped at {MAX_BET_PCT}
7. SKIP markets with edge <= {MIN_EDGE}

Return ONLY valid JSON — no prose before or after:

{{
  "opportunities": [
    {{
      "market_id": "string",
      "question": "string",
      "current_market_price": <float 0-1>,
      "estimated_true_probability": <float 0-1>,
      "edge": <float>,
      "recommendation": "YES" | "NO" | "SKIP",
      "confidence": <float 0-1>,
      "reasoning": "1 sentence explanation",
      "suggested_bet_fraction": <float 0-{MAX_BET_PCT}>
    }}
  ],
  "summary": "1-2 sentence overall market commentary"
}}

Include ALL {len(markets)} markets even as SKIPs."""

    market_list = [
        {
            "market_id":         m["market_id"],
            "question":          m["question"],
            "category":          m.get("category", ""),
            "current_yes_price": m["current_yes_price"],
            "volume_24h_usd":    m.get("volume_24h", 0),
            "liquidity_usd":     m.get("liquidity", 0),
            "end_date":          m.get("end_date", ""),
            "description":       m.get("description", ""),
        }
        for m in markets
    ]

    user_msg = (
        f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Analytical bankroll: ${bankroll:.2f}\n\n"
        f"Analyze these {len(markets)} markets and return JSON:\n\n"
        f"{json.dumps(market_list, indent=2)}"
    )

    print(f"\nSending {len(markets)} markets to Claude ({MODEL})...")
    print("=" * 60)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text
    if "```json" in raw:
        raw = raw[raw.find("```json") + 7 : raw.find("```", raw.find("```json") + 7)].strip()
    elif "```" in raw:
        raw = raw[raw.find("```") + 3 : raw.find("```", raw.find("```") + 3)].strip()

    # If Claude prefixed prose before the JSON object, strip it
    if not raw.lstrip().startswith("{"):
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude response is not valid JSON.\nError: {e}\nRaw (500 chars): {raw[:500]}"
        ) from e

    if not isinstance(result.get("opportunities"), list):
        raise ValueError(f"Missing 'opportunities' list. Keys: {list(result.keys())}")

    u = response.usage
    print(f"Claude usage — in: {u.input_tokens} tokens | out: {u.output_tokens} tokens")
    return result


# ── Single analysis cycle ──────────────────────────────────────────────────────

def run_cycle(executor: SupabaseExecutor, client: anthropic.Anthropic):
    print(f"\n{'='*60}")
    print(f"  ANALYTICAL BOT  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    bankroll = executor.get_bankroll()
    print(f"\nBankroll: ${bankroll:.2f}  "
          f"(initial: ${executor.get_initial_bankroll():.2f})")

    print("\n── Checking open positions ──")
    executor.simulate_resolutions()

    use_live = os.environ.get("USE_LIVE_MARKETS", "false").strip().lower() == "true"
    print(f"\n── Loading markets ({'live API' if use_live else 'mock file'}) ──")
    markets = get_markets()
    if not markets:
        print("No markets available — skipping analysis this cycle.")
        return

    try:
        analysis = analyze_markets(client, markets, executor.get_bankroll())
    except anthropic.APIError as e:
        print(f"\nERROR: Anthropic API failed: {e}")
        return
    except ValueError as e:
        print(f"\nERROR: {e}")
        return

    summary = analysis.get("summary", "")
    if summary:
        print(f"\nClaude: {summary}\n")

    opportunities = analysis.get("opportunities", [])
    tradeable = [o for o in opportunities if o.get("recommendation") in ("YES", "NO")]
    skipped   = [o for o in opportunities if o.get("recommendation") == "SKIP"]

    print(f"── {len(tradeable)} trade(s) | {len(skipped)} skip(s) out of {len(opportunities)} ──")

    if not tradeable:
        print("No edges above threshold — holding cash.")
    else:
        for opp in tradeable:
            market = next((m for m in markets if m["market_id"] == opp["market_id"]), None)
            if market:
                opp["current_price"] = market["current_yes_price"]
                opp["yes_token_id"]  = market.get("yes_token_id")
                opp["no_token_id"]   = market.get("no_token_id")
            executor.execute_trade(opp, source="claude")

    if skipped:
        print(f"\n── Skipped (edge < {MIN_EDGE:.0%}) ──")
        for s in skipped:
            print(f"  SKIP  {s.get('market_id','?'):<12}  "
                  f"mkt={s.get('current_market_price',0):.2f}  "
                  f"est={s.get('estimated_true_probability',0):.2f}  "
                  f"edge={s.get('edge',0):.2f}")

    executor.print_stats()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  ANALYTICAL BOT  —  Powered by Claude AI")
    print("=" * 60)
    print(f"  Model    : {MODEL}")
    print(f"  Strategy : {STRATEGY}")
    print(f"  Max bet  : {MAX_BET_PCT:.0%} of bankroll  (25% Kelly)")
    print(f"  Interval : {LOOP_INTERVAL_M} minutes")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "\nERROR: ANTHROPIC_API_KEY not set.\n"
            "  Copy .env.example to .env and add your key.\n"
            "  Get one at: https://console.anthropic.com/\n"
        )
        sys.exit(1)

    executor = SupabaseExecutor(
        strategy=STRATEGY,
        max_bet_fraction=MAX_BET_PCT,
        kelly_fraction=0.25,
    )
    executor.sync_bankroll_from_polymarket(num_strategies=2)
    client = anthropic.Anthropic(api_key=api_key)

    run_once = "--once" in sys.argv
    cycle = 0

    while True:
        cycle += 1
        print(f"\n[Cycle {cycle}]")
        try:
            run_cycle(executor, client)
        except Exception as e:
            print(f"\nUnexpected error in cycle {cycle}: {e}")

        if run_once:
            print("\nRan with --once flag. Exiting.")
            break

        print(f"\nSleeping {LOOP_INTERVAL_M} minutes until next cycle...")
        print("(Ctrl+C to stop)\n")
        try:
            time.sleep(LOOP_INTERVAL_M * 60)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break


if __name__ == "__main__":
    main()
