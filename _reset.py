"""Delete all fake simulation trades and reset bankroll to real balance."""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)
import os
from datetime import datetime, timezone
from supabase import create_client

db  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
now = datetime.now(timezone.utc).isoformat()

# ── 1. Delete all trades ──────────────────────────────────────────────────────
r = db.table("trades").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
print(f"Deleted {len(r.data or [])} trades")

# ── 2. Try real balance from py-clob-client ───────────────────────────────────
real_balance = None
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds
    creds = ApiCreds(
        api_key=os.environ["POLYMARKET_API_KEY"],
        api_secret=os.environ["POLYMARKET_SECRET"],
        api_passphrase=os.environ["POLYMARKET_PASSPHRASE"],
    )
    clob = ClobClient(
        "https://clob.polymarket.com",
        key=os.environ["POLYMARKET_PRIVATE_KEY"],
        chain_id=137,
        creds=creds,
        signature_type=1,
        funder=os.environ["POLYMARKET_FUNDER"],
    )
    # Try various balance methods across py-clob-client versions
    for method in ("get_balance", "get_balances"):
        fn = getattr(clob, method, None)
        if fn:
            raw = fn()
            print(f"  {method}() returned: {raw!r}")
            if isinstance(raw, dict):
                real_balance = float(raw.get("USDC", raw.get("balance", raw.get("cash", 0))))
            elif isinstance(raw, (int, float, str)):
                real_balance = float(raw)
            if real_balance is not None:
                break
except Exception as e:
    print(f"  CLOB balance fetch failed: {e}")

# Fall back to known wallet amount if API didn't return a value
if real_balance is None or real_balance == 0:
    real_balance = 10.12
    print(f"  Using known wallet amount: ${real_balance}")
else:
    print(f"  Real Polymarket balance: ${real_balance:.4f}")

share = round(real_balance / 2, 2)
print(f"  Setting each strategy to ${share:.2f}")

# ── 3. Reset bankroll rows ────────────────────────────────────────────────────
for strat in ["analytical", "sniper"]:
    r = db.table("bankroll").update({"balance": share, "updated_at": now}).eq("strategy", strat).execute()
    bal = r.data[0]['balance'] if r.data else 'ERROR'
    print(f"  {strat} bankroll -> ${bal}")
