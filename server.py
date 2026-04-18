"""
server.py — Flask health server + bot supervisor

Render's free web-service tier requires a live HTTP server.
This file starts Flask on $PORT, then launches bot.py and sniper.py
in supervised daemon threads that auto-restart on any crash.
"""

import os
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(override=True)

from flask import Flask, jsonify

import bot as _bot
import sniper as _sniper

app = Flask(__name__)

RESTART_DELAY = 30  # seconds to wait before restarting a crashed bot


# ── Health endpoints ───────────────────────────────────────────────────────────

@app.route("/")
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Bot supervisor ─────────────────────────────────────────────────────────────

def _supervised(name: str, target_fn):
    """Run target_fn in a loop, restarting whenever it exits or crashes."""
    while True:
        print(f"[supervisor] Starting {name}...", flush=True)
        try:
            target_fn()
            # main() returned cleanly (shouldn't happen in loop mode)
            print(f"[supervisor] {name} exited cleanly — restarting in {RESTART_DELAY}s")
        except SystemExit:
            # sys.exit() in a thread raises SystemExit but only kills the thread
            print(f"[supervisor] {name} called sys.exit() — restarting in {RESTART_DELAY}s")
        except Exception as e:
            print(f"[supervisor] {name} crashed: {e} — restarting in {RESTART_DELAY}s")
        time.sleep(RESTART_DELAY)


def _start_bots():
    for name, fn in [("bot.py", _bot.main), ("sniper.py", _sniper.main)]:
        t = threading.Thread(
            target=_supervised,
            args=(name, fn),
            name=name,
            daemon=True,
        )
        t.start()
        print(f"[supervisor] {name} thread launched", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _start_bots()
    port = int(os.environ.get("PORT", 8080))
    print(f"[server] Listening on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port)
