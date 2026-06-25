#!/usr/bin/env python3
"""
fetch_real_trades.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetches actual fill history from AsterDEX for BTCUSDT from a
given start date, groups individual fills into trade sessions
(open → close), and writes the result to trade_log.json.

This replaces formula-estimated values with real exchange data:
  - entry  = weighted avg price of all opening fills
  - close  = weighted avg price of all closing fills
  - PnL    = sum of realizedPnl fields (gross) minus commission
  - qty    = actual position size including T2 and babies

Usage:
    python fetch_real_trades.py
"""

import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "asterdex_trade"))
from account_data import get_trade_history

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL    = "BTCUSDT"
ROOT_DIR  = Path(__file__).parent
TRADE_LOG = ROOT_DIR / "trade_log.json"

# May 12 2026 00:00:00 UTC  (user's format: DD-MM-YYYY)
START_DATE = datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc)
START_MS   = int(START_DATE.timestamp() * 1000)


# ── Step 1: Fetch all fills from AsterDEX ─────────────────────────────────────
def fetch_all_fills(symbol: str, start_ms: int) -> list:
    """
    Pull every userTrade from start_ms to now.
    Pages through the API using fromId when needed (max 1000 per call).
    Raises RuntimeError on API auth errors.
    """
    all_fills = []
    from_id   = None

    while True:
        resp = get_trade_history(
            symbol     = symbol,
            start_time = start_ms if from_id is None else None,
            from_id    = from_id,
            limit      = 1000,
        )

        # Detect API error responses (dict with "code" key)
        if isinstance(resp, dict) and "code" in resp:
            code = resp["code"]
            msg  = resp.get("msg", "")
            if "No agent found" in msg:
                raise RuntimeError(
                    f"AsterDEX: No agent found (-1000)\n"
                    f"\n"
                    f"  The SIGNER address is not approved as an agent for USER.\n"
                    f"\n"
                    f"  USER   (account) : 0x2C6cC1f1C27Add18b01064ae4ad9C173E46faCC8\n"
                    f"  SIGNER (trading) : 0x1E8e98a16CF6cdF040C206aF97D7cFd3C18E8f0F\n"
                    f"\n"
                    f"  FIX: Go to AsterDEX web app > Account > API / Agents\n"
                    f"       Add SIGNER as an authorized agent for USER.\n"
                    f"       Then run this script again."
                )
            raise RuntimeError(f"AsterDEX API error {code}: {msg}")

        fills = resp if isinstance(resp, list) else resp.get("data", [])
        if not fills:
            break

        # When paginating via fromId, filter out anything before start
        if from_id is not None:
            fills = [f for f in fills if int(f.get("time", 0)) >= start_ms]

        all_fills.extend(fills)
        print(f"  Fetched {len(all_fills)} fills so far ...")

        if len(fills) < 1000:
            break   # last page

        # Next page: start from the fill after the highest id in this batch
        from_id = max(int(f.get("id", 0)) for f in fills) + 1

    all_fills.sort(key=lambda f: int(f.get("time", 0)))
    return all_fills


# ── Step 2: Group fills into trade sessions ────────────────────────────────────
def group_sessions(fills: list) -> list:
    """
    Group individual exchange fills into logical trade sessions.

    SHORT session:
      Opening fills = SELL  side, positionSide=SHORT
      Closing fills = BUY   side, positionSide=SHORT
    LONG session:
      Opening fills = BUY   side, positionSide=LONG
      Closing fills = SELL  side, positionSide=LONG

    A session is complete when cumulative close qty >= cumulative open qty
    (i.e. the position has been fully closed).
    """
    sessions = []

    for pos_side, direction, open_side, close_side in [
        ("SHORT", "SHORT", "SELL", "BUY"),
        ("LONG",  "LONG",  "BUY",  "SELL"),
    ]:
        pos_fills = [
            f for f in fills
            if f.get("positionSide", "").upper() == pos_side
        ]
        if not pos_fills:
            continue

        open_buf   = []
        close_buf  = []
        open_qty   = 0.0
        close_qty  = 0.0

        for f in pos_fills:
            side = f.get("side", "").upper()
            qty  = float(f.get("qty", 0))

            if side == open_side:
                open_buf.append(f)
                open_qty += qty

            elif side == close_side:
                close_buf.append(f)
                close_qty += qty

                # Session fully closed?
                if open_buf and close_qty >= open_qty - 0.0005:
                    sessions.append({
                        "direction":   direction,
                        "open_fills":  list(open_buf),
                        "close_fills": list(close_buf),
                    })
                    open_buf  = []
                    close_buf = []
                    open_qty  = 0.0
                    close_qty = 0.0

        # Partially closed / still open session — skip (no realized PnL yet)

    # Sort by when each session opened
    sessions.sort(key=lambda s: int(s["open_fills"][0].get("time", 0)))
    return sessions


# ── Step 3: Convert a session to a trade_log entry ────────────────────────────
def session_to_entry(session: dict, trade_id: int, running_pnl: float) -> tuple:
    """
    Build a trade_log.json dict from a grouped session.
    Returns (entry_dict, updated_running_pnl).
    """
    direction   = session["direction"]
    open_fills  = session["open_fills"]
    close_fills = session["close_fills"]
    all_fills   = open_fills + close_fills

    def wavg_price(fills):
        total_notional = sum(float(f["qty"]) * float(f["price"]) for f in fills)
        total_qty      = sum(float(f["qty"]) for f in fills)
        return total_notional / total_qty if total_qty else 0.0

    avg_entry  = wavg_price(open_fills)
    avg_close  = wavg_price(close_fills)
    open_qty   = sum(float(f["qty"]) for f in open_fills)
    close_qty  = sum(float(f["qty"]) for f in close_fills)

    # Realized PnL: sum of realizedPnl across all fills
    # AsterDEX posts realizedPnl on closing fills (non-zero) and opening fills (zero)
    gross_pnl  = sum(float(f.get("realizedPnl", 0)) for f in all_fills)

    # Commission: paid on every fill
    commission = sum(float(f.get("commission", 0)) for f in all_fills)

    # Net PnL after commissions
    net_pnl = gross_pnl - commission

    result = "WIN" if net_pnl >= 0 else "LOSS"
    running_pnl += net_pnl

    # Timestamps
    open_ts  = int(open_fills[0].get("time", 0)) / 1000
    close_ts = int(close_fills[-1].get("time", 0)) / 1000 if close_fills else open_ts

    entry = {
        "id":           trade_id,
        "time":         datetime.fromtimestamp(open_ts).isoformat(),
        "close_time":   datetime.fromtimestamp(close_ts).isoformat(),
        "symbol":       SYMBOL,
        "direction":    direction,
        "entry":        round(avg_entry, 1),
        "actual_close": round(avg_close, 1),
        "actual_qty":   round(close_qty, 4),
        "open_fills":   len(open_fills),
        "close_fills":  len(close_fills),
        "gross_pnl":    round(gross_pnl, 6),
        "commission":   round(commission, 6),
        "pnl_usd":      round(net_pnl, 6),
        "total_pnl":    round(running_pnl, 6),
        "result":       result,
        "source":       "asterdex_actual",
    }

    return entry, running_pnl


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AsterDEX Trade History Fetcher")
    print(f"  Symbol  : {SYMBOL}")
    print(f"  From    : {START_DATE.strftime('%Y-%m-%d')}  (May 12 2026)")
    print(f"  Output  : {TRADE_LOG.name}")
    print("=" * 60)
    print()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    print("Fetching fills from AsterDEX ...")
    try:
        fills = fetch_all_fills(SYMBOL, START_MS)
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR (unexpected): {e}")
        sys.exit(1)

    # Check for API auth errors in the response
    if isinstance(fills, dict):
        code = fills.get("code")
        msg  = fills.get("msg", "")
        if code == -1000 and "No agent found" in msg:
            print()
            print("  ══════════════════════════════════════════════════════")
            print("  ERROR: AsterDEX returned 'No agent found'")
            print()
            print("  The SIGNER address is not approved as an agent for USER.")
            print()
            print(f"  USER   (account) : {__import__('account_data').USER}")
            print(f"  SIGNER (trading) : {__import__('account_data').SIGNER}")
            print()
            print("  FIX: Go to AsterDEX web app → Account → API / Agents")
            print("       Add SIGNER as an authorized agent for USER.")
            print("       Then run this script again.")
            print("  ══════════════════════════════════════════════════════")
            sys.exit(1)
        elif code is not None:
            print(f"  API error {code}: {msg}")
            sys.exit(1)

    print(f"Total fills fetched: {len(fills)}")
    if not fills:
        print("No fills found after the start date. Nothing written.")
        return

    # ── Print raw fills summary ───────────────────────────────────────────────
    print()
    print("Raw fills:")
    for f in fills:
        ts  = datetime.fromtimestamp(int(f.get("time", 0)) / 1000).strftime("%m-%d %H:%M:%S")
        print(f"  [{ts}]  {f.get('positionSide','?'):5s}  {f.get('side','?'):4s}"
              f"  qty={f.get('qty','?'):>8}  price={float(f.get('price',0)):>10,.1f}"
              f"  realizedPnl={float(f.get('realizedPnl',0)):>+10.4f}"
              f"  commission={float(f.get('commission',0)):>8.4f}")

    # ── Group into sessions ───────────────────────────────────────────────────
    print()
    sessions = group_sessions(fills)
    print(f"Grouped into {len(sessions)} complete trade sessions")

    if not sessions:
        print("No completed sessions found (positions may still be open). Nothing written.")
        return

    # ── Build trade log ───────────────────────────────────────────────────────
    log         = []
    running_pnl = 0.0

    print()
    print("Sessions:")
    for i, session in enumerate(sessions, 1):
        entry, running_pnl = session_to_entry(session, i, running_pnl)
        log.append(entry)

        wins_so_far   = sum(1 for e in log if e["result"] == "WIN")
        losses_so_far = sum(1 for e in log if e["result"] == "LOSS")

        print(f"  #{i:2d}  {entry['direction']:5s}  "
              f"entry={entry['entry']:>10,.1f}  "
              f"close={entry['actual_close']:>10,.1f}  "
              f"qty={entry['actual_qty']:>7}  "
              f"gross={entry['gross_pnl']:>+8.4f}  "
              f"comm={entry['commission']:>7.4f}  "
              f"net={entry['pnl_usd']:>+8.4f}  "
              f"[{entry['result']}]")

    print()
    total_wins   = sum(1 for e in log if e["result"] == "WIN")
    total_losses = sum(1 for e in log if e["result"] == "LOSS")
    win_rate     = total_wins / len(log) * 100 if log else 0
    print(f"  Total trades : {len(log)}")
    print(f"  Win / Loss   : {total_wins} / {total_losses}  ({win_rate:.0f}% win rate)")
    print(f"  Total PnL    : ${running_pnl:+.4f}")

    # ── Write ─────────────────────────────────────────────────────────────────
    print()
    print(f"Writing to {TRADE_LOG} ...")
    with open(TRADE_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
