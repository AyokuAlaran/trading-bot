"""
executor.py — Supabase-backed Trade Executor
Shared by both the analytical bot (bot.py) and the sniper bot (sniper.py).
All state (bankroll, trades) lives in Supabase — no local file writes.
Real orders are placed via Polymarket CLOB when credentials are present;
falls back to simulation-only mode if CLOB is unavailable.
"""

import json
import os
import random
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(override=True)

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
INITIAL_BANKROLL = 1000.0  # Expected to be pre-seeded in Supabase


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_list(value) -> list:
    """Safely coerce a JSON-string or list to a plain Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


class SupabaseExecutor:
    """
    Manages trades and bankroll for one strategy in Supabase.

    Args:
        strategy:         'analytical' | 'sniper'
        max_bet_fraction: hard cap per trade (0.05 for analytical, 0.20 for sniper)
        kelly_fraction:   Kelly multiplier (0.25 for analytical, 0.50 for sniper)
    """

    MIN_BET_SIZE = 1.0   # Polymarket minimum order size

    def __init__(
        self,
        strategy: str,
        max_bet_fraction: float = 0.05,
        kelly_fraction: float = 0.25,
    ):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"
            )
        self.strategy         = strategy
        self.max_bet_fraction = max_bet_fraction
        self.kelly_fraction   = kelly_fraction
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── Bankroll ───────────────────────────────────────────────────────────────

    def get_bankroll(self) -> float:
        """Fetch current balance from Supabase; seeds row if missing."""
        try:
            res = (
                self.supabase.table("bankroll")
                .select("balance")
                .eq("strategy", self.strategy)
                .execute()
            )
            if res.data:
                return float(res.data[0]["balance"])
        except Exception as e:
            print(f"  WARNING: Could not read bankroll — {e}. Using {INITIAL_BANKROLL}.")
            return INITIAL_BANKROLL

        # Seed a fresh row if not pre-existing
        try:
            self.supabase.table("bankroll").insert({
                "strategy":   self.strategy,
                "balance":    INITIAL_BANKROLL,
                "updated_at": _now_iso(),
            }).execute()
        except Exception as e:
            print(f"  WARNING: Could not seed bankroll row — {e}")
        return INITIAL_BANKROLL

    def get_initial_bankroll(self) -> float:
        """Returns current balance as a proxy for initial (initial_balance col may not exist)."""
        return self.get_bankroll()

    def _set_bankroll(self, new_balance: float):
        try:
            self.supabase.table("bankroll").update({
                "balance":    round(new_balance, 2),
                "updated_at": _now_iso(),
            }).eq("strategy", self.strategy).execute()
        except Exception as e:
            print(f"  WARNING: Could not update bankroll — {e}")

    # ── Polymarket balance sync ────────────────────────────────────────────────

    def sync_bankroll_from_polymarket(self, num_strategies: int = 2):
        """
        Fetch real USDC balance from Polymarket CLOB and update this strategy's
        bankroll to (total_balance / num_strategies). Safe to call on startup.
        """
        try:
            clob = self._build_clob_client()
            if clob is None:
                print(f"  [sync] No CLOB credentials — skipping balance sync.")
                return
            raw = clob.get_balance()
            # get_balance may return a dict or a numeric string depending on version
            if isinstance(raw, dict):
                total = float(raw.get("balance", raw.get("USDC", 0)))
            else:
                total = float(raw)
            share = round(total / num_strategies, 2)
            self._set_bankroll(share)
            # Also update initial_balance so ROI is calculated correctly
            try:
                self.supabase.table("bankroll").update({
                    "initial_balance": share,
                    "updated_at": _now_iso(),
                }).eq("strategy", self.strategy).execute()
            except Exception:
                pass
            print(f"  [sync] Polymarket USDC ${total:.2f} → {self.strategy}: ${share:.2f}")
        except AttributeError:
            print(f"  [sync] get_balance() not available in this py-clob-client version — keeping existing bankroll.")
        except Exception as e:
            print(f"  [sync] Could not fetch Polymarket balance ({e}) — keeping existing bankroll.")

    # ── Schema-safe DB helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_bad_column(e: Exception) -> Optional[str]:
        """Pull the offending column name from a postgrest schema error."""
        import re
        # postgrest APIError has a .message attribute with the clean string
        msg = getattr(e, "message", None) or str(e)
        for pat in [
            r"Could not find the '(\w+)' column",
            r"column '(\w+)' of",
            r"column (\w+) does not exist",
            r"'(\w+)' does not exist",
        ]:
            m = re.search(pat, msg)
            if m:
                return m.group(1)
        return None

    def _safe_insert(self, table: str, row: dict):
        """Insert row, auto-stripping any columns the schema doesn't have (up to 12)."""
        payload = dict(row)
        for _ in range(12):
            try:
                return self.supabase.table(table).insert(payload).execute()
            except Exception as e:
                col = self._extract_bad_column(e)
                if col and col in payload:
                    payload.pop(col)
                else:
                    print(f"  [{self.strategy}] ERROR inserting into {table}: {e}")
                    return None
        print(f"  [{self.strategy}] ERROR: too many schema mismatches on {table}")
        return None

    def _safe_update(self, table: str, data: dict, match_col: str, match_val):
        """Update row, stripping columns that don't exist in the schema."""
        payload = dict(data)
        for _ in range(12):
            try:
                self.supabase.table(table).update(payload).eq(match_col, match_val).execute()
                return
            except Exception as e:
                col = self._extract_bad_column(e)
                if col and col in payload:
                    payload.pop(col)
                else:
                    print(f"  WARNING: Could not update {table}: {e}")
                    return

    # ── CLOB helpers ──────────────────────────────────────────────────────────

    def _build_clob_client(self):
        """Build a py-clob-client ClobClient. Returns None if credentials missing."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            pk         = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
            api_key    = os.environ.get("POLYMARKET_API_KEY", "")
            secret     = os.environ.get("POLYMARKET_SECRET", "")
            passphrase = os.environ.get("POLYMARKET_PASSPHRASE", "")
            funder     = os.environ.get("POLYMARKET_FUNDER", "")

            if not all([pk, api_key, secret, passphrase, funder]):
                return None

            creds = ApiCreds(
                api_key=api_key,
                api_secret=secret,
                api_passphrase=passphrase,
            )
            return ClobClient(
                CLOB_HOST,
                key=pk,
                chain_id=137,
                creds=creds,
                signature_type=1,
                funder=funder,
            )
        except Exception as e:
            print(f"  WARNING: Could not build CLOB client — {e}")
            return None

    def _resolve_token_id(self, market_id: str, recommendation: str) -> Optional[str]:
        """Fetch the YES or NO clobTokenId for a market from the Gamma API."""
        try:
            resp = requests.get(
                f"{GAMMA_HOST}/markets",
                params={"id": market_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            market = data[0] if isinstance(data, list) and data else data
            raw_ids = market.get("clobTokenIds", "[]")
            token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            # index 0 = YES, index 1 = NO
            idx = 0 if recommendation == "YES" else 1
            return str(token_ids[idx]) if len(token_ids) > idx else None
        except Exception as e:
            print(f"  WARNING: Could not resolve token_id for {market_id} — {e}")
            return None

    def _place_clob_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
    ) -> Optional[str]:
        """
        Place a market (FOK) BUY order on the CLOB. Returns order id on success.
        size_usd is the dollar amount (minimum $1 per Polymarket rules).
        """
        try:
            from py_clob_client.clob_types import MarketOrderArgs
            from py_clob_client.order_builder.constants import BUY

            clob = self._build_clob_client()
            if clob is None:
                return None

            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=round(size_usd, 4),
                side=BUY,
            )
            from py_clob_client.clob_types import OrderType
            signed = clob.create_market_order(order_args)
            resp = clob.post_order(signed, OrderType.FOK)
            print(f"  [CLOB] Raw response: {resp!r}")
            if isinstance(resp, dict):
                order_id = resp.get("orderID") or resp.get("id") or resp.get("orderId")
                status   = resp.get("status") or resp.get("success")
                error    = resp.get("errorMsg") or resp.get("error")
                if error:
                    print(f"  [CLOB] Order rejected — {error}")
                    return None
                print(f"  [CLOB] Order placed — id={order_id}  status={status}  "
                      f"token={token_id[:16]}...  amount=${size_usd:.2f}")
                return order_id
            return None
        except Exception as e:
            print(f"  WARNING: CLOB order failed — {e}  (trade still logged in Supabase)")
            return None

    # ── Real settlement checker ────────────────────────────────────────────────

    def settle_open_positions(self):
        """
        Query Polymarket for each open position and settle any that have resolved.
        Called at the top of every bot cycle in place of the old simulate_resolutions().
        Reinvestment policy: sniper = 100% of profit, analytical = 50%.
        """
        try:
            res = (
                self.supabase.table("trades")
                .select("*")
                .eq("strategy", self.strategy)
                .eq("status", "open")
                .execute()
            )
            open_trades = res.data or []
        except Exception as e:
            print(f"  WARNING: Could not fetch open positions: {e}")
            return

        if not open_trades:
            print("  No open positions to check.")
            return

        print(f"  Checking {len(open_trades)} open position(s) against Polymarket...")
        bankroll = self.get_bankroll()
        settled  = 0

        for trade in open_trades:
            market_id = trade.get("market_id", "")
            try:
                resp = requests.get(
                    f"{GAMMA_HOST}/markets",
                    params={"id": market_id},
                    timeout=10,
                )
                resp.raise_for_status()
                data   = resp.json()
                market = data[0] if isinstance(data, list) and data else data
            except Exception as e:
                print(f"    WARNING: Could not fetch market {market_id}: {e}")
                continue

            if not market.get("resolved"):
                continue

            outcomes = _parse_json_list(market.get("outcomes"))
            prices   = _parse_json_list(market.get("outcomePrices"))

            if not outcomes or not prices or len(outcomes) != len(prices):
                continue

            yes_idx = next(
                (i for i, o in enumerate(outcomes)
                 if str(o).strip().lower() in ("yes", "true", "1")),
                None,
            )
            if yes_idx is None:
                continue

            market_resolved_yes = float(prices[yes_idx]) >= 0.99
            rec      = trade.get("recommendation", "YES")
            bet_wins = (rec == "YES" and market_resolved_yes) or \
                       (rec == "NO"  and not market_resolved_yes)

            amount      = float(trade.get("amount", 0))
            entry_price = float(trade.get("entry_price", 0.5))

            if bet_wins:
                payout = round(amount / max(entry_price, 0.001), 4)
                pnl    = round(payout - amount, 4)
                if self.strategy == "sniper":
                    bankroll = round(bankroll + payout, 2)          # 100% reinvest
                else:
                    bankroll = round(bankroll + amount + pnl * 0.5, 2)  # 50% reinvest
                outcome    = "WIN"
                exit_price = 1.0
            else:
                pnl        = round(-amount, 4)
                outcome    = "LOSS"
                exit_price = 0.0

            self._safe_update("trades", {
                "status":         "closed",
                "pnl":            pnl,
                "outcome":        outcome,
                "exit_price":     exit_price,
                "exit_timestamp": _now_iso(),
            }, "id", trade["id"])

            tag = "[WIN] " if bet_wins else "[LOSS]"
            q   = trade.get("question", market_id)[:50]
            print(f"  {tag} '{q}...'  P&L=${pnl:+.2f}")
            settled += 1

        if settled:
            self._set_bankroll(bankroll)
            print(f"  Settled {settled} position(s). New bankroll: ${bankroll:.2f}")

    # ── Liquidity-weighted sizing ──────────────────────────────────────────────

    @staticmethod
    def _liquidity_multiplier(volume_24h: float) -> float:
        """Scale bet size by 24h market volume to favour liquid markets."""
        if volume_24h >= 100_000:
            return 1.2
        if volume_24h >= 50_000:
            return 1.0
        if volume_24h >= 10_000:
            return 0.8
        return 0.5

    # ── Trade execution ────────────────────────────────────────────────────────

    def execute_trade(
        self,
        opportunity: dict,
        source: str = "claude",
    ) -> Optional[dict]:
        """
        Record a new trade in Supabase and deduct the bet from the bankroll.

        Args:
            opportunity: analysis dict (market_id, question, recommendation, edge,
                         confidence, estimated_true_probability, current_price,
                         suggested_bet_fraction, reasoning)
            source:      'claude' for analytical, 'sniper' for live-result bets
        Returns:
            Inserted trade dict (with 'id' from Supabase), or None if skipped.
        """
        recommendation = opportunity.get("recommendation", "SKIP")
        if recommendation not in ("YES", "NO"):
            return None

        bankroll = self.get_bankroll()
        if bankroll <= 0:
            print(f"  [{self.strategy}] Bankroll ${bankroll:.2f} — no funds.")
            return None

        # ── Bet sizing with liquidity multiplier ─────────────────────────────
        edge       = float(opportunity.get("edge", 0.0))
        confidence = float(opportunity.get("confidence", 1.0))
        volume_24h = float(opportunity.get("volume_24h", 0))
        raw_frac   = edge * confidence * self.kelly_fraction
        bet_frac   = min(max(raw_frac, 0.005), self.max_bet_fraction)
        liq_mult   = self._liquidity_multiplier(volume_24h)
        bet_size   = round(bankroll * bet_frac * liq_mult, 2)
        # Hard cap: never exceed max_bet_fraction × bankroll regardless of multiplier
        bet_size   = min(bet_size, round(bankroll * self.max_bet_fraction, 2))

        if bet_size < self.MIN_BET_SIZE:
            if bankroll >= self.MIN_BET_SIZE:
                bet_size = self.MIN_BET_SIZE
                print(f"  [{self.strategy}] Bet floored to minimum ${bet_size:.2f}")
            else:
                print(f"  [{self.strategy}] Bankroll ${bankroll:.2f} below ${self.MIN_BET_SIZE:.2f} min — skipped")
                return None

        # ── Cross-bot duplicate guard (checks all strategies) ─────────────────
        market_id = str(opportunity.get("market_id", ""))
        try:
            dup = (
                self.supabase.table("trades")
                .select("id")
                .eq("market_id", market_id)
                .eq("status", "open")
                .execute()
            )
            if dup.data:
                print(f"  [{self.strategy}] Already open in {market_id} — skipped")
                return None
        except Exception:
            pass

        # ── Entry price ───────────────────────────────────────────────────────
        yes_price = float(
            opportunity.get("current_price",
            opportunity.get("current_market_price", 0.5))
        )
        yes_price   = max(0.001, min(0.999, yes_price))
        entry_price = yes_price if recommendation == "YES" else (1.0 - yes_price)

        trade_row = {
            "strategy":                   self.strategy,
            "source":                     source,
            "market_id":                  market_id,
            "question":                   str(opportunity.get("question", "")),
            "recommendation":             recommendation,
            "amount":                     bet_size,
            "entry_price":                round(entry_price, 6),
            "estimated_true_probability": float(opportunity.get("estimated_true_probability", 0.5)),
            "edge":                       round(edge, 6),
            "confidence":                 round(confidence, 6),
            "reasoning":                  str(opportunity.get("reasoning", "")),
            "status":                     "open",
            "pnl":                        0.0,
            "outcome":                    None,
            "exit_price":                 None,
            "created_at":                 _now_iso(),
            "exit_timestamp":             None,
        }

        res = self._safe_insert("trades", trade_row)
        if res is None:
            return None
        if res.data:
            trade_row["id"] = res.data[0]["id"]

        self._set_bankroll(bankroll - bet_size)

        q = trade_row["question"]
        print(f"\n  [{self.strategy.upper()}] {recommendation} "
              f"'{q[:60]}{'...' if len(q)>60 else ''}'")
        print(f"    Bet: ${bet_size:.2f}  Entry: {entry_price:.3f}  "
              f"Edge: {edge:.1%}  Conf: {confidence:.0%}  Source: {source}")

        # ── Real CLOB order (best-effort, never crashes the bot) ──────────────
        # Prefer token IDs forwarded from market data; fall back to API re-fetch.
        if recommendation == "YES":
            token_id = opportunity.get("yes_token_id") or self._resolve_token_id(market_id, "YES")
        else:
            token_id = opportunity.get("no_token_id") or self._resolve_token_id(market_id, "NO")

        if token_id:
            self._place_clob_order(token_id, entry_price, bet_size)
        else:
            print("  [CLOB] No token_id in market data — skipping real order.")

        return trade_row

    # ── Analytical resolution (probabilistic) ──────────────────────────────────

    def simulate_resolutions(self):
        """
        Randomly resolve a subset of open analytical trades using Claude's
        estimated true probability. Mirrors realistic market settlement rates.
        """
        try:
            res = (
                self.supabase.table("trades")
                .select("*")
                .eq("strategy", self.strategy)
                .eq("status", "open")
                .execute()
            )
            open_trades = res.data or []
        except Exception as e:
            print(f"  WARNING: Could not fetch open positions: {e}")
            return

        if not open_trades:
            print("  No open positions to resolve.")
            return

        candidates = [t for t in open_trades if random.random() < 0.40]
        if not candidates:
            print(f"  No positions resolved this run ({len(open_trades)} still open).")
            return

        print(f"\nResolving {len(candidates)} position(s):")
        bankroll = self.get_bankroll()
        for trade in candidates:
            bankroll = self._close_trade_probabilistic(trade, bankroll)
        self._set_bankroll(bankroll)

    def _close_trade_probabilistic(self, trade: dict, bankroll: float) -> float:
        """Probabilistically settle one open trade. Returns updated bankroll."""
        true_prob   = float(trade.get("estimated_true_probability", 0.5))
        entry_price = float(trade.get("entry_price", 0.5))
        amount      = float(trade.get("amount", 0))
        rec         = trade.get("recommendation", "YES")

        resolved_yes = random.random() < true_prob
        bet_wins = (rec == "YES" and resolved_yes) or (rec == "NO" and not resolved_yes)

        if bet_wins:
            payout       = round(amount / max(entry_price, 0.001), 4)
            pnl          = round(payout - amount, 4)
            exit_price   = 1.0
            outcome      = "WIN"
            # Compound 50% of profit back; stake always returns
            compounded   = round(amount + pnl * 0.50, 4)
            bankroll     = round(bankroll + compounded, 2)
        else:
            pnl        = round(-amount, 4)
            exit_price = 0.0
            outcome    = "LOSS"

        self._safe_update("trades", {
            "status":         "closed",
            "pnl":            pnl,
            "outcome":        outcome,
            "exit_price":     exit_price,
            "exit_timestamp": _now_iso(),
        }, "id", trade["id"])

        tag = "[WIN] " if outcome == "WIN" else "[LOSS]"
        q   = trade.get("question", "")
        print(f"  {tag} {outcome}: '{q[:50]}{'...' if len(q)>50 else ''}'  "
              f"P&L=${pnl:+.2f}  Bankroll→${bankroll:.2f}")
        return bankroll

    # ── Sniper settlement (known outcome) ─────────────────────────────────────

    def settle_sniper_trade(
        self,
        trade_id: str,
        outcome: str,
        entry_price: float,
        amount: float,
    ):
        """
        Immediately settle a sniper trade with the confirmed real-world outcome.
        Called right after execute_trade() when the live result is known.
        """
        if outcome == "WIN":
            payout     = round(amount / max(entry_price, 0.001), 4)
            pnl        = round(payout - amount, 4)
            exit_price = 1.0
        else:
            payout     = 0.0
            pnl        = round(-amount, 4)
            exit_price = 0.0

        self._safe_update("trades", {
            "status":         "closed",
            "pnl":            pnl,
            "outcome":        outcome,
            "exit_price":     exit_price,
            "exit_timestamp": _now_iso(),
        }, "id", trade_id)

        new_bankroll = self.get_bankroll() + payout
        self._set_bankroll(new_bankroll)
        print(f"  [SNIPER SETTLED] {outcome}: P&L=${pnl:+.2f}  "
              f"Bankroll→${new_bankroll:.2f}")

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        try:
            res    = (
                self.supabase.table("trades")
                .select("*")
                .eq("strategy", self.strategy)
                .execute()
            )
            trades = res.data or []
        except Exception:
            trades = []

        closed  = [t for t in trades if t.get("status") == "closed"]
        wins    = [t for t in closed if t.get("outcome") == "WIN"]
        losses  = [t for t in closed if t.get("outcome") == "LOSS"]
        open_   = [t for t in trades if t.get("status") == "open"]

        bankroll         = self.get_bankroll()
        initial_bankroll = self.get_initial_bankroll()
        total_pnl        = sum(float(t.get("pnl") or 0) for t in closed)
        win_rate         = (len(wins) / len(closed) * 100) if closed else 0.0
        avg_win          = (sum(float(t.get("pnl") or 0) for t in wins)   / len(wins))   if wins   else 0.0
        avg_loss         = (sum(float(t.get("pnl") or 0) for t in losses) / len(losses)) if losses else 0.0
        in_play          = sum(float(t.get("amount") or 0) for t in open_)
        roi              = ((bankroll - initial_bankroll) / initial_bankroll * 100) if initial_bankroll > 0 else 0.0

        return {
            "strategy":         self.strategy,
            "bankroll":         round(bankroll, 2),
            "initial_bankroll": initial_bankroll,
            "total_pnl":        round(total_pnl, 2),
            "roi":              round(roi, 2),
            "total_trades":     len(trades),
            "closed_trades":    len(closed),
            "open_trades":      len(open_),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(win_rate, 1),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "amount_in_play":   round(in_play, 2),
        }

    def print_stats(self):
        s = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  [{s['strategy'].upper()}] SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"  Bankroll      : ${s['bankroll']:.2f}")
        print(f"  Total P&L     : ${s['total_pnl']:+.2f}  (ROI {s['roi']:+.1f}%)")
        print(f"  Win Rate      : {s['win_rate']:.1f}%  ({s['wins']}W / {s['losses']}L)")
        print(f"  Open Positions: {s['open_trades']}")
        print(f"  In Play       : ${s['amount_in_play']:.2f}")
        print(f"{'='*60}")
