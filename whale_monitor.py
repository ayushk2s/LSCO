#!/usr/bin/env python3
"""
Multi-Exchange Whale Order Monitor
Exchanges : Binance (spot + perp), OKX (spot + perp), Coinbase (spot)
Threshold : $500k per fill (default)
Output    : live print on each whale fill + full breakdown at every 1-min candle close
"""

import asyncio
import json
import sys
import time
import argparse
import requests
import websockets
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ── ANSI colours ──────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; M = "\033[95m"; B = "\033[1m"
D = "\033[2m";  X = "\033[0m"

EX_COL = {"BINANCE": Y, "OKX": C, "COINBASE": M}

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Exchange Whale Order Monitor")
    p.add_argument("--symbol",    default="BTCUSDT",
                   help="Binance-style pair, e.g. BTCUSDT, ETHUSDT (default: BTCUSDT)")
    p.add_argument("--threshold", type=float, default=500_000,
                   help="Min USD per fill to count as whale (default: 500000)")
    p.add_argument("--wall",      type=float, default=1_000_000,
                   help="Min USD at a price level to show as OB wall (default: 1000000)")
    p.add_argument("--quiet",     action="store_true",
                   help="Suppress live whale prints; only show candle summaries")
    return p.parse_args()

# ── Symbol mapping ────────────────────────────────────────────────────────────
def derive_symbols(sym: str) -> dict:
    sym = sym.upper()
    if   sym.endswith("USDT"): base, quote = sym[:-4], "USDT"
    elif sym.endswith("BUSD"): base, quote = sym[:-4], "BUSD"
    elif sym.endswith("USD"):  base, quote = sym[:-3],  "USD"
    else:                      base, quote = sym[:3],   sym[3:]
    return {
        "base":          base,
        "quote":         quote,
        "binance_spot":  sym,
        "binance_perp":  sym,           # USDT-M futures uses same symbol
        "okx_spot":      f"{base}-{quote}",
        "okx_perp":      f"{base}-{quote}-SWAP",
        "coinbase_spot": f"{base}-USD",  # Coinbase always settles in USD
    }

def get_okx_ct_val(inst_id: str, base: str) -> float:
    """Fetch OKX contract size (base currency per contract) for SWAP instruments."""
    defaults = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0, "BNB": 0.1,
                "XRP": 100.0, "DOGE": 1000.0, "MATIC": 10.0}
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/instruments",
            params={"instType": "SWAP", "instId": inst_id},
            timeout=5,
        )
        return float(r.json()["data"][0]["ctVal"])
    except Exception:
        return defaults.get(base, 1.0)

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Trade:
    price:    float
    qty:      float     # base-currency units
    usd:      float     # notional value in USD
    is_buy:   bool      # True = aggressive buy (taker hit the ask)
    ts:       float     # Unix seconds
    exchange: str       # BINANCE | OKX | COINBASE
    market:   str       # SPOT | PERP


class SharedBuffer:
    """Thread-safe-by-design (asyncio single-thread) trade buffer for the current minute."""

    def __init__(self):
        self._trades: List[Trade] = []
        self._open_ts: int = int(time.time() / 60) * 60

    def add(self, t: Trade):
        self._trades.append(t)

    def close(self) -> Tuple[int, List[Trade]]:
        """Snapshot current candle and reset. Returns (candle_open_ts, trades)."""
        trades, open_ts = self._trades, self._open_ts
        self._trades  = []
        self._open_ts = int(time.time() / 60) * 60
        return open_ts, trades

# ── Formatters ────────────────────────────────────────────────────────────────
def fmt(v: float) -> str:
    return f"${v/1e6:.3f}M" if v >= 1e6 else f"${v/1e3:.1f}k"

def side_s(is_buy: bool) -> str:
    return f"{G}BUY {X}" if is_buy else f"{R}SELL{X}"

def pbucket(price: float, base: str) -> float:
    bk = {"BTC": 50, "ETH": 5, "SOL": 0.5, "BNB": 1,
          "XRP": 0.01, "DOGE": 0.001}.get(base, 0.1)
    return round(price / bk) * bk

# ── Order-book wall fetchers ──────────────────────────────────────────────────
def _parse_ob(levels, threshold: float) -> List[Tuple[float, float, float]]:
    out = []
    for row in levels:
        p, q = float(row[0]), float(row[1])
        usd = p * q
        if usd >= threshold:
            out.append((p, q, usd))
    return out

def walls_binance(sym: str, thr: float):
    try:
        ob = requests.get(
            f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=500",
            timeout=5
        ).json()
        bids = sorted(_parse_ob(ob["bids"], thr), reverse=True)[:5]
        asks = sorted(_parse_ob(ob["asks"], thr))[:5]
        return bids, asks, None
    except Exception as e:
        return [], [], str(e)

def walls_binance_perp(sym: str, thr: float):
    try:
        ob = requests.get(
            f"https://fapi.binance.com/fapi/v1/depth?symbol={sym}&limit=500",
            timeout=5
        ).json()
        bids = sorted(_parse_ob(ob["bids"], thr), reverse=True)[:5]
        asks = sorted(_parse_ob(ob["asks"], thr))[:5]
        return bids, asks, None
    except Exception as e:
        return [], [], str(e)

# ── Candle report ─────────────────────────────────────────────────────────────
SOURCES = [
    ("BINANCE", "SPOT"), ("BINANCE", "PERP"),
    ("OKX",     "SPOT"), ("OKX",     "PERP"),
    ("COINBASE","SPOT"),
]

def print_candle(
    open_ts: int,
    trades: List[Trade],
    syms: dict,
    threshold: float,
    wall_thr: float,
):
    if not trades:
        return

    ts_str = datetime.fromtimestamp(open_ts).strftime("%Y-%m-%d %H:%M")
    base   = syms["base"]
    sym    = syms["binance_spot"]

    # Sort by exchange timestamp for correct OHLC
    tr = sorted(trades, key=lambda t: t.ts)
    op, cl = tr[0].price, tr[-1].price
    hi = max(t.price for t in tr)
    lo = min(t.price for t in tr)
    pct = (cl - op) / op * 100
    cc  = G if pct >= 0 else R

    total_usd = sum(t.usd for t in tr)
    buy_usd   = sum(t.usd for t in tr if t.is_buy)
    sell_usd  = total_usd - buy_usd
    whales    = [t for t in tr if t.usd >= threshold]

    sep = "═" * 74
    print(f"\n{B}{sep}{X}")
    print(f"{B}  {sym}  │  1-min  {ts_str}  │  {len(tr):,} trades  │  {len(whales)} whale fills{X}")
    print(
        f"  O {cc}{op:>10,.2f}{X}  "
        f"H {G}{hi:>10,.2f}{X}  "
        f"L {R}{lo:>10,.2f}{X}  "
        f"C {cc}{cl:>10,.2f}  {cc}{pct:+.2f}%{X}"
    )
    print(
        f"  All vol: {fmt(total_usd)}   "
        f"{G}Buys {fmt(buy_usd)}{X}   "
        f"{R}Sells {fmt(sell_usd)}{X}"
    )

    # ── Whale fills by exchange / market ──────────────────────────────────────
    print(f"\n{B}{Y}  WHALE FILLS  ≥ {fmt(threshold)}{X}")

    if not whales:
        print(f"  {D}No whale fills this candle.{X}")
    else:
        for ex, mkt in SOURCES:
            src = [t for t in whales if t.exchange == ex and t.market == mkt]
            if not src:
                continue

            col = EX_COL.get(ex, X)
            sb  = sum(t.usd for t in src if t.is_buy)
            ss  = sum(t.usd for t in src if not t.is_buy)

            print(
                f"\n  {B}{col}▸ {ex}  {mkt}{X}   "
                f"{len(src)} fills   "
                f"{G}B {fmt(sb)}{X}   {R}S {fmt(ss)}{X}"
            )

            # Group by price bucket
            grps: Dict[float, Dict] = defaultdict(lambda: {"b": 0.0, "s": 0.0, "n": 0})
            for t in src:
                k = pbucket(t.price, base)
                if t.is_buy: grps[k]["b"] += t.usd
                else:        grps[k]["s"] += t.usd
                grps[k]["n"] += 1

            print(f"    {'Price':>10}  {'Buys':>12}  {'Sells':>12}  {'#':>3}")
            print("    " + "─" * 42)
            for price in sorted(grps.keys(), reverse=True):
                g   = grps[price]
                bs  = f"{G}{fmt(g['b'])}{X}" if g["b"] else f"{D}{'—':>8}{X}"
                ss2 = f"{R}{fmt(g['s'])}{X}" if g["s"] else f"{D}{'—':>8}{X}"
                print(f"    {price:>10,.2f}  {bs:>20}  {ss2:>19}  {g['n']:>3}")

        # Net delta across all exchanges
        wb = sum(t.usd for t in whales if t.is_buy)
        ws = sum(t.usd for t in whales if not t.is_buy)
        d  = wb - ws
        nc, nl = (G, "▲ BULLISH") if d >= 0 else (R, "▼ BEARISH")
        print(f"\n  {B}Total Whale Delta (all exchanges): {nc}{fmt(abs(d))}  {nl}{X}")

    # ── Order-book walls ──────────────────────────────────────────────────────
    print(f"\n{B}{C}  ORDER BOOK WALLS  ≥ {fmt(wall_thr)}{X}")
    any_wall = False
    for label, bids, asks, err in [
        (f"{Y}Binance Spot{X}", *walls_binance(syms["binance_spot"], wall_thr)),
        (f"{Y}Binance Perp{X}", *walls_binance_perp(syms["binance_perp"], wall_thr)),
    ]:
        if err:
            print(f"  {label}  {D}{err}{X}")
            continue
        if not bids and not asks:
            continue
        any_wall = True
        print(f"  {label}")
        for p, q, u in asks[:4]:
            print(f"    {R}{p:>10,.2f}{X}  {q:>9.4f}  {fmt(u):>10}  {D}ask  +{(p-cl)/cl*100:.2f}%{X}")
        for p, q, u in bids[:4]:
            print(f"    {G}{p:>10,.2f}{X}  {q:>9.4f}  {fmt(u):>10}  {D}bid  -{(cl-p)/cl*100:.2f}%{X}")

    if not any_wall:
        print(f"  {D}No significant walls found.{X}")

    print(f"{B}{sep}{X}\n")

    # ── Save candle summary to JSON for liq_algo to read ─────────────────────
    _save_whale_candle(open_ts, cl, whales, threshold, syms["binance_spot"])


def _save_whale_candle(open_ts: int, close_price: float,
                       whales: List[Trade], threshold: float, symbol: str = "BTCUSDT"):
    """Write last completed candle's whale summary to whale_<SYMBOL>.json.
    liq_algo.py reads the matching file every minute for whale confirmation."""
    import pathlib

    whale_buys  = [t for t in whales if t.is_buy]
    whale_sells = [t for t in whales if not t.is_buy]

    def weighted_avg_price(trades):
        total_usd = sum(t.usd for t in trades)
        if total_usd == 0:
            return 0.0
        return sum(t.price * t.usd for t in trades) / total_usd

    data = {
        "candle_ts":        open_ts,
        "candle_time":      datetime.fromtimestamp(open_ts).strftime("%Y-%m-%d %H:%M"),
        "close_price":      close_price,
        "threshold_usd":    threshold,
        "whale_buy_usd":    sum(t.usd for t in whale_buys),
        "whale_sell_usd":   sum(t.usd for t in whale_sells),
        "whale_buy_count":  len(whale_buys),
        "whale_sell_count": len(whale_sells),
        # weighted-average price where whales were buying / selling
        "whale_buy_avg_price":  weighted_avg_price(whale_buys),
        "whale_sell_avg_price": weighted_avg_price(whale_sells),
    }

    out = pathlib.Path(__file__).parent / f"whale_{symbol}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {D}[whale] candle data saved → {out.name}{X}")

# ── Live print ────────────────────────────────────────────────────────────────
def live_print(t: Trade):
    col = EX_COL.get(t.exchange, X)
    ts  = datetime.fromtimestamp(t.ts).strftime("%H:%M:%S")
    print(
        f"  {Y}[WHALE]{X} {ts}  "
        f"{col}{t.exchange:8s}{X} {t.market:4s}  "
        f"{side_s(t.is_buy)}  "
        f"{t.qty:>9.4f}  @ {t.price:>10,.2f}  = {fmt(t.usd)}"
    )

# ── Reconnecting WebSocket helper ─────────────────────────────────────────────
async def _connect_loop(label: str, uri: str, on_connect, on_message):
    delay = 1
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                delay = 1
                print(f"  {D}[{label}] connected{X}")
                await on_connect(ws)
                async for raw in ws:
                    await on_message(ws, raw)
        except Exception as e:
            print(f"  {D}[{label}] {type(e).__name__}: {e} — retry {delay}s{X}")
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)

# ── Binance streams ───────────────────────────────────────────────────────────
async def stream_binance_spot(syms: dict, buf: SharedBuffer, thr: float, quiet: bool):
    sym = syms["binance_spot"].lower()
    uri = f"wss://stream.binance.com:9443/ws/{sym}@aggTrade"

    async def on_connect(_): pass

    async def on_message(_, raw):
        d   = json.loads(raw)
        p   = float(d["p"]); q = float(d["q"]); usd = p * q
        t   = Trade(p, q, usd, not d["m"], d["T"] / 1000, "BINANCE", "SPOT")
        buf.add(t)
        if not quiet and usd >= thr:
            live_print(t)

    await _connect_loop(f"Binance Spot {sym.upper()}", uri, on_connect, on_message)


async def stream_binance_perp(syms: dict, buf: SharedBuffer, thr: float, quiet: bool):
    sym = syms["binance_perp"].lower()
    uri = f"wss://fstream.binance.com/ws/{sym}@aggTrade"

    async def on_connect(_): pass

    async def on_message(_, raw):
        d   = json.loads(raw)
        p   = float(d["p"]); q = float(d["q"]); usd = p * q
        t   = Trade(p, q, usd, not d["m"], d["T"] / 1000, "BINANCE", "PERP")
        buf.add(t)
        if not quiet and usd >= thr:
            live_print(t)

    await _connect_loop(f"Binance Perp {sym.upper()}", uri, on_connect, on_message)

# ── OKX stream (spot or perp) ─────────────────────────────────────────────────
async def stream_okx(
    syms: dict, market: str, ct_val: float,
    buf: SharedBuffer, thr: float, quiet: bool
):
    inst_id = syms["okx_spot"] if market == "SPOT" else syms["okx_perp"]
    uri     = "wss://ws.okx.com:8443/ws/v5/public"

    async def on_connect(ws):
        await ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": inst_id}]
        }))

    async def on_message(ws, raw):
        if raw == "ping":           # OKX heartbeat
            await ws.send("pong")
            return
        msg = json.loads(raw)
        if msg.get("event"):        # subscribe ack / error
            return
        for item in msg.get("data", []):
            p   = float(item["px"])
            sz  = float(item["sz"])
            # For SWAP, sz is in contracts; multiply by contract value to get base qty
            qty = sz * ct_val if market == "PERP" else sz
            usd = p * qty
            t   = Trade(p, qty, usd, item["side"] == "buy",
                        int(item["ts"]) / 1000, "OKX", market)
            buf.add(t)
            if not quiet and usd >= thr:
                live_print(t)

    await _connect_loop(f"OKX {market} {inst_id}", uri, on_connect, on_message)

# ── Coinbase stream ───────────────────────────────────────────────────────────
async def stream_coinbase(syms: dict, buf: SharedBuffer, thr: float, quiet: bool):
    product_id = syms["coinbase_spot"]
    uri        = "wss://advanced-trade-ws.coinbase.com"

    async def on_connect(ws):
        await ws.send(json.dumps({
            "type":        "subscribe",
            "channel":     "market_trades",
            "product_ids": [product_id],
        }))

    async def on_message(ws, raw):
        msg = json.loads(raw)
        if msg.get("channel") != "market_trades":
            return
        for event in msg.get("events", []):
            for item in event.get("trades", []):
                p   = float(item["price"])
                q   = float(item["size"])
                usd = p * q
                t   = Trade(p, q, usd, item["side"].upper() == "BUY",
                            time.time(), "COINBASE", "SPOT")
                buf.add(t)
                if not quiet and usd >= thr:
                    live_print(t)

    await _connect_loop(f"Coinbase Spot {product_id}", uri, on_connect, on_message)

# ── Candle ticker — fires once per minute boundary ────────────────────────────
async def candle_ticker(
    buf: SharedBuffer, syms: dict, thr: float, wall_thr: float
):
    while True:
        now  = time.time()
        wait = (int(now / 60) + 1) * 60 - now + 0.15   # 150 ms grace
        await asyncio.sleep(wait)
        open_ts, candle_trades = buf.close()
        # Run synchronous HTTP OB fetches in thread pool so we don't stall the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, print_candle, open_ts, candle_trades, syms, thr, wall_thr
        )

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    syms = derive_symbols(args.symbol)
    buf  = SharedBuffer()

    print(f"\n{B}{'─'*74}{X}")
    print(f"{B}  Multi-Exchange Whale Order Monitor{X}")
    print(f"  Exchanges  : {Y}Binance spot+perp{X}  │  {C}OKX spot+perp{X}  │  {M}Coinbase spot{X}")
    print(f"  Symbol     : {syms['base']}/{syms['quote']}")
    print(f"  Threshold  : {fmt(args.threshold)} per fill")
    print(f"  OB Walls   : {fmt(args.wall)}")
    print(f"  Mode       : {'quiet (summaries only)' if args.quiet else 'verbose (live + summaries)'}")

    # Fetch OKX contract value once at startup
    print(f"\n{D}  Fetching OKX contract spec for {syms['okx_perp']}…{X}", end="", flush=True)
    ct_val = get_okx_ct_val(syms["okx_perp"], syms["base"])
    print(f"\r  OKX ct_val : {ct_val} {syms['base']} per contract           ")

    secs_left = 60 - (time.time() % 60)
    print(f"  Next candle: {secs_left:.0f}s away")
    print(f"{B}{'─'*74}{X}\n")

    async def run():
        await asyncio.gather(
            stream_binance_spot(syms, buf, args.threshold, args.quiet),
            stream_binance_perp(syms, buf, args.threshold, args.quiet),
            stream_okx(syms, "SPOT", 1.0,    buf, args.threshold, args.quiet),
            stream_okx(syms, "PERP", ct_val, buf, args.threshold, args.quiet),
            stream_coinbase(syms, buf, args.threshold, args.quiet),
            candle_ticker(buf, syms, args.threshold, args.wall),
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
