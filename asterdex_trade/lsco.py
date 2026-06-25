"""
lsco.py  —  LSCO Trading Engine v2  (Multi-Asset Liquidation Scalp)
====================================================================
Runs three parallel engines: BTCUSDT, ETHUSDT, XAUUSDT

Key differences from liq_algo.py (do NOT modify that file):
  1. Touch trigger  — fires on zone touch + rejection wick
  2. Re-entry       — same zone retradable after short cooldown
  3. IOC close      — SL / emergency closes use IOC for guaranteed fill
  4. Tighter exits  — TP 1.2x ATR, SL 0.60x ATR  (2:1 R:R)
  5. Multi-asset    — BTCUSDT + ETHUSDT + XAUUSDT parallel engines

Run: python lsco.py
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from account_data   import (place_order, cancel_order, cancel_all_orders,
                             get_position_risk, get_balances, set_leverage,
                             get_order_history)
from market_data    import get_price, get_klines, calc_atr, get_last_1m_candle, get_ob
from order_executor import SmartLimitOrder


# ==============================================================================
# CONFIG
# ==============================================================================

# Per-symbol configuration
SYMBOL_CONFIG = {
    "BTCUSDT": {
        "leverage":     20,
        "min_qty":      0.001,
        "qty_round":    3,
        "price_round":  1,
        "min_zone_usd": 10_000_000,
        "q_min_usd":    20_000_000,
        "approach_pct": 1.0,
        "whale_file":   "whale_BTCUSDT.json",
    },
    "ETHUSDT": {
        "leverage":     20,
        "min_qty":      0.001,
        "qty_round":    3,
        "price_round":  2,
        "min_zone_usd": 2_000_000,
        "q_min_usd":    5_000_000,
        "approach_pct": 1.0,
        "whale_file":   "whale_ETHUSDT.json",
    },
    "XAUUSDT": {
        "leverage":     10,
        "min_qty":      0.01,
        "qty_round":    2,
        "price_round":  2,
        "min_zone_usd": 500_000,
        "q_min_usd":    1_000_000,
        "approach_pct": 1.0,
        "whale_file":   "whale_XAUUSDT.json",
    },
}

SYMBOLS = list(SYMBOL_CONFIG.keys())

# Sizing — minimum for testing phase
BALANCE_UTILIZATION = 0.01    # 1% of free balance per engine
MIN_RISK_USDT       = 1.0
MAX_RISK_USDT       = 2.0     # hard cap — forces minimum lot size

# Touch trigger thresholds
TOUCH_BUF  = 0.30    # wick within 30% of ATR from zone = "touched"
MIN_WICK   = 0.20    # rejection wick must be >= 20% of ATR
MAX_OB_ATR = 0.25    # close <= 25% ATR beyond zone = not a breakout

# Exit
ATR_PERIOD   = 14
ATR_INTERVAL = "5m"
TP_MULT      = 1.2
SL_MULT      = 0.60

# Confidence → Trade2 size multiplier
CONF_MULT = {
    (0,  65): 1.5,
    (65, 80): 1.8,
    (80, 101): 2.0,
}

# Re-entry cooldown
REENTRY_WIN_COOL  = 180
REENTRY_LOSS_COOL = 600

# File paths
ROOT_DIR        = Path(__file__).parent.parent
TRADE_LOG_JSON  = ROOT_DIR / "trade_log.json"
HEATMAP_SCRIPT  = ROOT_DIR / "data_fetching" / "binance_liq_heatmap_2.py"
HEATMAP_REFRESH = 5 * 60


def heatmap_json(symbol: str) -> Path:
    return ROOT_DIR / "data_fetching" / f"binance_liq_heatmap_{symbol}.json"

def whale_json(symbol: str) -> Path:
    return ROOT_DIR / SYMBOL_CONFIG[symbol]["whale_file"]

def state_json(symbol: str) -> Path:
    return ROOT_DIR / f"algo_state_v2_{symbol}.json"


# ==============================================================================
# ZONE MATH & SIZING
# ==============================================================================

def load_heatmap(symbol: str):
    with open(heatmap_json(symbol)) as f:
        return json.load(f)


def load_whale(symbol: str):
    try:
        with open(whale_json(symbol)) as f:
            d = json.load(f)
        return d if time.time() - d["candle_ts"] < 180 else None
    except Exception:
        return None


def scan_zones(price, hm, min_zone_usd, approach_pct):
    shorts = [z for z in hm["short_liquidations"] if z["usd"] >= min_zone_usd]
    longs  = [z for z in hm["long_liquidations"]  if z["usd"] >= min_zone_usd]

    above = [z for z in shorts if z["price"] > price]
    below = [z for z in longs  if z["price"] <= price]

    nearest_above = min(above, key=lambda z: z["price"]) if above else None
    nearest_below = max(below, key=lambda z: z["price"]) if below else None

    dist_above = ((nearest_above["price"] - price) / price * 100) if nearest_above else None
    dist_below = ((price - nearest_below["price"]) / price * 100) if nearest_below else None

    return {
        "price":            price,
        "nearest_above":    nearest_above,
        "nearest_below":    nearest_below,
        "dist_above_pct":   dist_above,
        "dist_below_pct":   dist_below,
        "approaching_up":   dist_above is not None and dist_above <= approach_pct,
        "approaching_down": dist_below is not None and dist_below <= approach_pct,
    }


def calc_q(zones, nearest, direction, q_min_usd, min_zone_usd):
    threshold = max(2.0 * nearest["usd"], q_min_usd)
    if direction == "above":
        cands = sorted([z for z in zones if z["price"] > nearest["price"]],
                       key=lambda z: z["price"])
    else:
        cands = sorted([z for z in zones if z["price"] < nearest["price"]],
                       key=lambda z: -z["price"])

    q_zones = [nearest]
    total   = nearest["usd"]
    for z in cands:
        q_zones.append(z)
        total += z["usd"]
        if total >= threshold:
            break

    avg = sum(z["price"] * z["usd"] for z in q_zones) / total
    return q_zones, avg, total


def calc_confidence(scan, q_zones, q_total, whale, direction):
    score = 50
    nearest = scan["nearest_above"] if direction == "short" else scan["nearest_below"]
    if nearest:
        if nearest["usd"] >= 30_000_000: score += 10
        if nearest["usd"] >= 60_000_000: score += 5
    if q_total >= 60_000_000: score += 10
    if len(q_zones) >= 3:     score += 10
    if whale:
        if direction == "short" and whale["whale_sell_usd"] > whale["whale_buy_usd"]:
            score += 15
        elif direction == "long" and whale["whale_buy_usd"] > whale["whale_sell_usd"]:
            score += 15
        if direction == "short" and whale["whale_buy_usd"] > whale["whale_sell_usd"] * 1.5:
            score -= 20
        elif direction == "long" and whale["whale_sell_usd"] > whale["whale_buy_usd"] * 1.5:
            score -= 20
    return max(0, min(100, score))


def get_conf_mult(score):
    for (lo, hi), m in CONF_MULT.items():
        if lo <= score < hi:
            return m
    return 1.5


def calc_risk(symbol: str):
    cfg = SYMBOL_CONFIG[symbol]
    try:
        bal  = get_balances()
        bals = bal if isinstance(bal, list) else bal.get("data", [])
        avail = 0.0
        for b in bals:
            if b.get("asset", "").upper() in ("USDT", "BUSD"):
                av = float(b.get("availableBalance") or b.get("walletBalance") or b.get("balance") or 0)
                avail = max(avail, av)
        if avail <= 0:
            return MIN_RISK_USDT
        factor  = 1.0 + max(m for m in CONF_MULT.values())
        clamped = max(MIN_RISK_USDT, min(MAX_RISK_USDT, avail * BALANCE_UTILIZATION / factor))
        print(f"  [{symbol}][sizing] Balance=${avail:.2f}  RISK=${clamped:.4f}")
        return round(clamped, 4)
    except Exception as e:
        print(f"  [{symbol}][sizing] failed ({e}) -> ${MIN_RISK_USDT}")
        return MIN_RISK_USDT


def qty_from_risk(risk_usdt, price, leverage, min_qty, qty_round):
    return max(round(risk_usdt * leverage / price, qty_round), min_qty)


# ==============================================================================
# TRADING ENGINE
# ==============================================================================

MIN_CONFIDENCE = 50

class TradingEngine:

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cfg    = SYMBOL_CONFIG[symbol]

        self.state           = "WATCHING"
        self.direction       = None
        self.q_zones         = []
        self.q_avg           = 0.0
        self.q_total_usd     = 0.0
        self.confidence      = 0
        self.entry_price     = 0.0
        self.atr             = 0.0
        self.tp_price        = 0.0
        self.sl_price        = 0.0
        self.trail_stage     = "none"
        self.best_seen       = None
        self.risk_usdt       = MIN_RISK_USDT
        self._heatmap        = None
        self._last_candle_ts      = 0
        self._current_zone_price  = 0.0
        self._zone_cooldowns      = {}
        self._consecutive_losses  = 0

        self.trade1    = None
        self.trade2    = None
        self.tp_order  = None
        self.t1_filled = False
        self.t1_qty    = 0.0

        self.trade_count = 0
        self.wins        = 0
        self.losses      = 0
        self.total_pnl   = 0.0

    def _tag(self, msg):
        """Prefix log lines with symbol tag."""
        print(f"  [{self.symbol}] {msg}")

    # ── Startup position recovery ──────────────────────────────────────────────

    async def recover_open_position(self):
        try:
            pos_data  = await asyncio.to_thread(get_position_risk, self.symbol)
            positions = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
            active    = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
        except Exception as e:
            self._tag(f"[recover] position check failed: {e}")
            return

        if not active:
            return

        for pos in active:
            side = pos.get("positionSide", "")
            amt  = abs(float(pos.get("positionAmt", 0)))
            ep   = float(pos.get("entryPrice", 0))
            self._tag(f"[recover] Found open {side} {amt} @ ${ep:,.2f}")

        recovered = False
        sf = state_json(self.symbol)
        if sf.exists():
            try:
                st = json.loads(sf.read_text())
                if (st.get("state") in ("IN_TRADE", "WATCHING") and
                        st.get("tp", 0) > 0 and st.get("sl", 0) > 0 and
                        st.get("direction") in ("LONG", "SHORT")):
                    matching = any(p.get("positionSide") == st["direction"] for p in active)
                    if matching:
                        pos = next(p for p in active if p.get("positionSide") == st["direction"])
                        amt = abs(float(pos.get("positionAmt", 0)))
                        ep  = float(pos.get("entryPrice", 0))
                        self.direction   = st["direction"]
                        self.entry_price = st.get("entry", ep)
                        self.tp_price    = st["tp"]
                        self.sl_price    = st["sl"]
                        self.atr         = st.get("atr", 100.0)
                        self.trail_stage = st.get("trail_stage", "none")
                        self.risk_usdt   = st.get("risk_usdt", MIN_RISK_USDT)
                        self.t1_filled   = True
                        self.t1_qty      = amt
                        self.state       = "IN_TRADE"
                        self._write_state()
                        self._tag(f"[recover] Resumed IN_TRADE {self.direction}"
                                  f"  entry=${self.entry_price:,.1f}"
                                  f"  TP=${self.tp_price:,.1f}  SL=${self.sl_price:,.1f}")
                        try:
                            await asyncio.to_thread(cancel_all_orders, self.symbol)
                            self._tag("[recover] Cancelled stale exchange orders")
                        except Exception:
                            pass
                        recovered = True
            except Exception as e:
                self._tag(f"[recover] state file error: {e}")

        if not recovered:
            self._tag("[recover] No valid state — closing orphan position")
            try:
                await asyncio.to_thread(cancel_all_orders, self.symbol)
            except Exception:
                pass
            for pos in active:
                side       = pos.get("positionSide", "")
                amt        = abs(float(pos.get("positionAmt", 0)))
                close_side = "BUY" if side == "SHORT" else "SELL"
                for attempt in range(3):
                    try:
                        ob    = await asyncio.to_thread(get_ob, self.symbol)
                        price = (round(ob["best_ask"] * 1.001,
                                       self.cfg["price_round"]) if close_side == "BUY"
                                 else round(ob["best_bid"] * 0.999,
                                            self.cfg["price_round"]))
                        resp  = await asyncio.to_thread(
                            place_order, self.symbol, close_side, "LIMIT",
                            amt, price, "IOC", side,
                        )
                        filled = float(resp.get("executedQty") or 0)
                        status = resp.get("status", "")
                        self._tag(f"[recover] Close attempt {attempt+1}: {status}  filled={filled}")
                        if filled > 0:
                            break
                    except Exception as e:
                        self._tag(f"[recover] close error: {e}")
                    await asyncio.sleep(0.5)

    # ── Loops ──────────────────────────────────────────────────────────────────

    async def minute_loop(self):
        self._tag("v2 engine started. State: WATCHING")
        while True:
            now  = time.time()
            wait = (int(now / 60) + 1) * 60 - now + 0.2
            await asyncio.sleep(wait)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'─'*55}")
            self._tag(f"[{ts}] State={self.state}")
            try:
                await self._tick()
            except Exception as e:
                import traceback; traceback.print_exc()
            self._write_state()

    async def position_monitor_loop(self):
        while True:
            await asyncio.sleep(3)
            if self.state == "IN_TRADE":
                try:
                    await self._monitor_position()
                except Exception as e:
                    self._tag(f"[monitor] error: {e}")

    # ── Tick ───────────────────────────────────────────────────────────────────

    async def _tick(self):
        if   self.state == "WATCHING":    await self._tick_watching()
        elif self.state == "APPROACHING": await self._tick_approaching()
        elif self.state == "IN_TRADE":    await self._print_status()

    async def _tick_watching(self):
        try:
            self._heatmap = await asyncio.to_thread(load_heatmap, self.symbol)
        except Exception as e:
            self._tag(f"[scan] heatmap error: {e}"); return

        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(
            scan_zones, price, self._heatmap,
            self.cfg["min_zone_usd"], self.cfg["approach_pct"]
        )

        if scan["nearest_above"]:
            z    = scan["nearest_above"]
            flag = "  <- APPROACHING" if scan["approaching_up"] else ""
            self._tag(f"[scan] Above: ${z['price']:>10,.1f}  ${z['usd']/1e6:.1f}M"
                      f"  ({scan['dist_above_pct']:.2f}% away){flag}")
        if scan["nearest_below"]:
            z    = scan["nearest_below"]
            flag = "  <- APPROACHING" if scan["approaching_down"] else ""
            self._tag(f"[scan] Below: ${z['price']:>10,.1f}  ${z['usd']/1e6:.1f}M"
                      f"  ({scan['dist_below_pct']:.2f}% away){flag}")

        if not (scan["approaching_up"] or scan["approaching_down"]):
            return

        if scan["approaching_up"] and scan["approaching_down"]:
            du = scan["dist_above_pct"] or 999
            dd = scan["dist_below_pct"] or 999
            direction = "SHORT" if du < dd else "LONG"
        elif scan["approaching_up"]:
            direction = "SHORT"
        else:
            direction = "LONG"

        zone = scan["nearest_above"] if direction == "SHORT" else scan["nearest_below"]
        cool = self._zone_cooldowns.get(zone["price"], 0)
        if time.time() < cool:
            self._tag(f"[cooldown] Zone ${zone['price']:,.0f}  {cool - time.time():.0f}s remaining")
            return

        self.direction = direction
        self.state     = "APPROACHING"
        self._tag(f"[engine] -> APPROACHING  dir={direction}")
        await self._tick_approaching()

    async def _tick_approaching(self):
        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(
            scan_zones, price, self._heatmap,
            self.cfg["min_zone_usd"], self.cfg["approach_pct"]
        )

        if self.direction == "SHORT" and not scan["approaching_up"]:
            self._tag("[engine] Zone passed -> WATCHING")
            self.state = "WATCHING"; return
        if self.direction == "LONG" and not scan["approaching_down"]:
            self._tag("[engine] Zone passed -> WATCHING")
            self.state = "WATCHING"; return

        if self.direction == "SHORT":
            nearest = scan["nearest_above"]
            zones   = [z for z in self._heatmap["short_liquidations"]
                       if z["usd"] >= self.cfg["min_zone_usd"]]
            dir_str = "above"
        else:
            nearest = scan["nearest_below"]
            zones   = [z for z in self._heatmap["long_liquidations"]
                       if z["usd"] >= self.cfg["min_zone_usd"]]
            dir_str = "below"

        q_zones, q_avg, q_total = await asyncio.to_thread(
            calc_q, zones, nearest, dir_str,
            self.cfg["q_min_usd"], self.cfg["min_zone_usd"]
        )
        self.q_zones     = q_zones
        self.q_avg       = q_avg
        self.q_total_usd = q_total

        whale = await asyncio.to_thread(load_whale, self.symbol)
        conf  = calc_confidence(scan, q_zones, q_total, whale, self.direction.lower())
        self.confidence = conf

        klines   = await asyncio.to_thread(get_klines, self.symbol, ATR_INTERVAL, 50)
        self.atr = calc_atr(klines, ATR_PERIOD)

        whale_avg = 0.0
        if whale:
            if self.direction == "SHORT" and whale.get("whale_sell_usd", 0) > 0:
                whale_avg = whale["whale_sell_avg_price"]
            elif self.direction == "LONG" and whale.get("whale_buy_usd", 0) > 0:
                whale_avg = whale["whale_buy_avg_price"]
        trade2_price = (q_avg + whale_avg) / 2 if whale_avg > 0 else q_avg

        mult = get_conf_mult(conf)
        self._tag(f"[engine] Q=${q_total/1e6:.1f}M  zones={len(q_zones)}"
                  f"  conf={conf}  mult={mult:.1f}x  ATR={self.atr:.1f}")

        last_candle = await asyncio.to_thread(get_last_1m_candle, self.symbol)
        zone_price  = nearest["price"]

        if last_candle["ts"] == self._last_candle_ts:
            self._tag("[engine] Same candle already fired -> WATCHING")
            self.state = "WATCHING"; return

        if conf < MIN_CONFIDENCE:
            self._tag(f"[engine] conf={conf} < {MIN_CONFIDENCE} -> skip")
            self.state = "WATCHING"; return

        # ── Trend filter: don't trade counter-trend ───────────────────────────
        if len(klines) >= 4:
            c1, c2, c3 = (float(klines[-4][4]), float(klines[-3][4]), float(klines[-2][4]))
            trending_up   = c3 > c2 > c1
            trending_down = c3 < c2 < c1
            if self.direction == "SHORT" and trending_up:
                self._tag(f"[trend] Uptrend detected (closes {c1:.2f}<{c2:.2f}<{c3:.2f})"
                          f" — skipping SHORT")
                self.state = "WATCHING"; return
            if self.direction == "LONG" and trending_down:
                self._tag(f"[trend] Downtrend detected (closes {c1:.2f}>{c2:.2f}>{c3:.2f})"
                          f" — skipping LONG")
                self.state = "WATCHING"; return

        cool = self._zone_cooldowns.get(zone_price, 0)
        if time.time() < cool:
            self._tag(f"[cooldown] Zone ${zone_price:,.0f}  {cool-time.time():.0f}s remaining -> WATCHING")
            self.state = "WATCHING"; return

        if not self._is_triggered(last_candle, zone_price):
            return

        # ── FIRE ──────────────────────────────────────────────────────────────
        self._tag(f"\n  * LSCO v2 TRIGGER — dir={self.direction}  conf={conf}  mult={mult:.1f}x")
        self._last_candle_ts     = last_candle["ts"]
        self._current_zone_price = zone_price

        self.risk_usdt   = await asyncio.to_thread(calc_risk, self.symbol)
        current          = last_candle["close"]
        self.entry_price = current

        if self.direction == "SHORT":
            self.tp_price = current - TP_MULT * self.atr
            self.sl_price = max(zone_price + self.atr * 0.30,
                                current    + self.atr * SL_MULT)
        else:
            self.tp_price = current + TP_MULT * self.atr
            self.sl_price = min(zone_price - self.atr * 0.30,
                                current    - self.atr * SL_MULT)

        lev        = self.cfg["leverage"]
        min_qty    = self.cfg["min_qty"]
        qty_round  = self.cfg["qty_round"]
        base_qty   = qty_from_risk(self.risk_usdt,        current, lev, min_qty, qty_round)
        trade2_qty = qty_from_risk(self.risk_usdt * mult, current, lev, min_qty, qty_round)

        self._tag(f"[trade] Entry={current:,.2f}  TP={self.tp_price:,.2f}"
                  f"  SL={self.sl_price:,.2f}  ATR={self.atr:.2f}")
        self._tag(f"[trade] RISK=${self.risk_usdt:.4f}  T1={base_qty}"
                  f"  T2={trade2_qty} @ {trade2_price:,.2f}")

        entry_side = "SELL" if self.direction == "SHORT" else "BUY"
        pos_side   = self.direction
        pr         = self.cfg["price_round"]

        # ── T1 entry: chase best bid (LONG) / best ask (SHORT) ───────────────
        t1_fill, t1_filled = await self._chase_limit(
            entry_side, base_qty, pos_side, timeout=20.0, label="T1_entry"
        )
        if t1_filled <= 0:
            self._tag("[executor] T1 entry chase timed out — aborting")
            try:
                await asyncio.to_thread(cancel_all_orders, self.symbol)
            except Exception:
                pass
            self.state = "WATCHING"
            self._write_state()
            return

        self.entry_price = t1_fill
        self.t1_filled   = True
        self.t1_qty      = t1_filled

        if self.direction == "SHORT":
            self.tp_price = self.entry_price - TP_MULT * self.atr
            self.sl_price = max(zone_price + self.atr * 0.30,
                                self.entry_price + self.atr * SL_MULT)
        else:
            self.tp_price = self.entry_price + TP_MULT * self.atr
            self.sl_price = min(zone_price - self.atr * 0.30,
                                self.entry_price - self.atr * SL_MULT)
        self._tag(f"[trade] Actual entry={self.entry_price:,.{pr}f}"
                  f"  TP={self.tp_price:,.{pr}f}  SL={self.sl_price:,.{pr}f}")

        close_side = "BUY" if self.direction == "SHORT" else "SELL"
        self.tp_order = SmartLimitOrder(
            symbol=self.symbol, side=close_side, qty=self.t1_qty,
            pos_side=pos_side, label="T1_TP",
            track_ob=False, fixed_price=self.tp_price,
            price_round=pr,
        )
        await self.tp_order.place()

        self.trade2 = SmartLimitOrder(
            symbol=self.symbol, side=entry_side, qty=trade2_qty,
            pos_side=pos_side, label="T2_limit",
            track_ob=False, fixed_price=trade2_price,
            price_round=pr,
        )
        await self.trade2.place()

        self.state = "IN_TRADE"
        self._tag("[engine] -> IN_TRADE")

    def _is_triggered(self, candle, zone_price):
        atr = self.atr or 100.0
        if self.direction == "SHORT":
            touched     = candle["high"] >= zone_price - atr * TOUCH_BUF
            wick        = candle["high"] - candle["close"]
            has_wick    = wick >= atr * MIN_WICK
            no_breakout = candle["close"] <= zone_price + atr * MAX_OB_ATR
            self._tag(f"[trigger?] SHORT: H={candle['high']:,.2f}  zone={zone_price:,.2f}"
                      f"  touched={touched}  wick={wick:.2f}(need={atr*MIN_WICK:.2f})={has_wick}"
                      f"  no_breakout={no_breakout}")
            return touched and has_wick and no_breakout
        else:
            touched     = candle["low"] <= zone_price + atr * TOUCH_BUF
            wick        = candle["close"] - candle["low"]
            has_wick    = wick >= atr * MIN_WICK
            no_breakout = candle["close"] >= zone_price - atr * MAX_OB_ATR
            self._tag(f"[trigger?] LONG:  L={candle['low']:,.2f}  zone={zone_price:,.2f}"
                      f"  touched={touched}  wick={wick:.2f}(need={atr*MIN_WICK:.2f})={has_wick}"
                      f"  no_breakout={no_breakout}")
            return touched and has_wick and no_breakout

    # ── Position monitor ───────────────────────────────────────────────────────

    async def _monitor_position(self):
        price = await asyncio.to_thread(get_price, self.symbol)

        if self.t1_filled:
            try:
                def _pos_exists():
                    pos_data  = get_position_risk(self.symbol)
                    positions = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
                    return any(
                        p.get("positionSide") == self.direction
                        and abs(float(p.get("positionAmt", 0))) > 0
                        for p in positions
                    )
                exists1 = await asyncio.to_thread(_pos_exists)
                if not exists1:
                    await asyncio.sleep(1.5)
                    exists2 = await asyncio.to_thread(_pos_exists)
                    if not exists2:
                        self._tag("[monitor] Position closed externally (confirmed ×2)")
                        t1q = self.t1_qty
                        t2q = self.trade2.qty if (self.trade2 and self.trade2.is_filled) else 0.0
                        qty = round(t1q + t2q, 3)
                        if qty > 0 and self.entry_price > 0:
                            pnl = ((self.entry_price - price) if self.direction == "SHORT"
                                   else (price - self.entry_price)) * qty
                            result = "WIN" if pnl >= 0 else "LOSS"
                            if pnl >= 0: self.wins   += 1
                            else:        self.losses += 1
                            self.total_pnl += pnl
                            self._set_cooldown(result)
                            self._log_trade(result, pnl, price, qty)
                        self._cleanup(); return
            except Exception:
                pass

        await self._update_trail(price)

        if not self.t1_filled:
            return

        if self.direction == "SHORT" and price <= self.tp_price:
            self._tag(f"[monitor] TP reached @ {price:,.2f}")
            await self._on_tp_hit(); return
        if self.direction == "LONG"  and price >= self.tp_price:
            self._tag(f"[monitor] TP reached @ {price:,.2f}")
            await self._on_tp_hit(); return
        if self.direction == "SHORT" and price >= self.sl_price:
            self._tag(f"[monitor] SL hit @ {price:,.2f}  (sl={self.sl_price:,.2f})")
            await self._on_sl_hit(); return
        if self.direction == "LONG"  and price <= self.sl_price:
            self._tag(f"[monitor] SL hit @ {price:,.2f}  (sl={self.sl_price:,.2f})")
            await self._on_sl_hit(); return

    async def _update_trail(self, price):
        if not self.t1_filled:
            return
        pts = ((self.entry_price - price) if self.direction == "SHORT"
               else (price - self.entry_price))
        if pts <= 0:
            return

        self.best_seen = (min(self.best_seen, price) if self.best_seen and self.direction == "SHORT"
                          else max(self.best_seen or price, price) if self.direction == "LONG"
                          else price)

        new_sl = None
        if pts >= 1.0 * self.atr:
            if self.direction == "SHORT":
                c = round(self.best_seen + 0.5 * self.atr,
                          self.cfg["price_round"])
                if c < self.sl_price: new_sl = c
            else:
                c = round(self.best_seen - 0.5 * self.atr,
                          self.cfg["price_round"])
                if c > self.sl_price: new_sl = c
            if new_sl: self.trail_stage = "trailing"
        elif pts >= 0.5 * self.atr and self.trail_stage == "none":
            be = round(self.entry_price + (5.0 if self.direction == "SHORT" else -5.0),
                       self.cfg["price_round"])
            if ((self.direction == "SHORT" and be < self.sl_price) or
                    (self.direction == "LONG" and be > self.sl_price)):
                new_sl = be
                self.trail_stage = "breakeven"

        if new_sl:
            self.sl_price = new_sl
            self._tag(f"[trail] SL -> ${new_sl:,.{self.cfg['price_round']}f}  ({self.trail_stage})")
            self._write_state()

    async def _on_tp_hit(self):
        close_price, close_qty = await self._chase_close("TP")
        approx_price = close_price if close_price > 0 else self.tp_price
        approx_qty   = close_qty   if close_qty   > 0 else self.t1_qty
        profit = max(((self.entry_price - approx_price) if self.direction == "SHORT"
                      else (approx_price - self.entry_price)) * approx_qty, 0.0)
        self.wins      += 1
        self.total_pnl += profit
        self._set_cooldown("WIN")
        if close_qty > 0:
            self._log_trade("WIN", profit, approx_price, approx_qty)
        else:
            self._tag("[engine] TP: close detect failed — cooldown set, not logged")
        self._cleanup()
        self._tag(f"[engine] WIN  +${profit:.4f}  session_total=${self.total_pnl:.4f}")

    async def _on_sl_hit(self):
        close_price, close_qty = await self._chase_close("SL")
        approx_price = close_price if close_price > 0 else self.sl_price
        approx_qty   = close_qty   if close_qty   > 0 else self.t1_qty
        loss = max(((approx_price - self.entry_price) if self.direction == "SHORT"
                    else (self.entry_price - approx_price)) * approx_qty, 0.0)
        self.losses    += 1
        self.total_pnl -= loss
        self._set_cooldown("LOSS")
        if close_qty > 0:
            self._log_trade("LOSS", -loss, approx_price, approx_qty)
        else:
            self._tag("[engine] SL: close detect failed — cooldown set, not logged")
        self._cleanup()
        self._tag(f"[engine] LOSS  -${loss:.4f}  session_total=${self.total_pnl:.4f}")

    async def _chase_limit(self, side: str, qty: float, pos_side: str,
                            timeout: float = 20.0, label: str = "order") -> tuple:
        """
        Place a limit at best bid (SELL) or best ask (BUY).
        Re-place every time best price moves by ≥1 tick.
        Polls every 500ms until filled or timeout.
        Returns (avg_price, filled_qty).
        """
        pr          = self.cfg["price_round"]
        tick        = 10 ** (-pr)
        order_id    = None
        order_price = None
        deadline    = time.time() + timeout

        while time.time() < deadline:
            ob   = await asyncio.to_thread(get_ob, self.symbol)
            best = round(ob["best_bid"] if side == "SELL" else ob["best_ask"], pr)

            need_replace = (order_id is None) or (abs(best - order_price) >= tick)

            if need_replace:
                if order_id:
                    try:
                        await asyncio.to_thread(cancel_order, self.symbol, order_id)
                    except Exception:
                        pass
                    order_id = None

                resp        = await asyncio.to_thread(
                    place_order, self.symbol, side, "LIMIT", qty, best, "GTC", pos_side
                )
                order_id    = resp.get("orderId")
                order_price = best
                status      = resp.get("status", "")
                filled      = float(resp.get("executedQty") or 0)
                avg_px      = float(resp.get("avgPrice") or resp.get("price") or best)
                self._tag(f"[{label}] LIMIT {side} {qty} @ {best:.{pr}f}"
                          f"  status={status}  id={order_id}")

                if status == "FILLED" and filled > 0:
                    return avg_px, filled
                if not order_id:
                    self._tag(f"[{label}] place failed: {resp}")
                    await asyncio.sleep(0.5)
                    continue
            else:
                try:
                    hist    = await asyncio.to_thread(
                        get_order_history, self.symbol, int(order_id), None, None, 1
                    )
                    orders  = hist if isinstance(hist, list) else hist.get("data", [hist])
                    matched = next((o for o in orders
                                    if str(o.get("orderId")) == str(order_id)), None)
                    if matched:
                        status = matched.get("status", "")
                        filled = float(matched.get("executedQty") or 0)
                        avg_px = float(matched.get("avgPrice") or
                                       matched.get("price") or order_price)
                        if status == "FILLED" and filled > 0:
                            self._tag(f"[{label}] filled @ {avg_px:.{pr}f}  qty={filled}")
                            return avg_px, filled
                        if status in ("CANCELED", "EXPIRED", "REJECTED"):
                            order_id = None
                except Exception:
                    pass

            await asyncio.sleep(0.5)

        if order_id:
            try:
                await asyncio.to_thread(cancel_order, self.symbol, order_id)
            except Exception:
                pass
        self._tag(f"[{label}] chase timeout after {timeout:.0f}s — no fill")
        return 0.0, 0.0

    async def _chase_close(self, reason: str) -> tuple:
        """
        Cancel all open orders, then chase-limit close the full position.
        SELL close (LONG): chase best_bid.
        BUY  close (SHORT): chase best_ask.
        Returns (avg_price, filled_qty).
        """
        for o in [self.trade1, self.trade2, self.tp_order]:
            if o and not o.is_filled and not o.is_cancelled:
                o.cancel()
        try:
            await asyncio.to_thread(cancel_all_orders, self.symbol)
        except Exception:
            pass

        close_side = "BUY" if self.direction == "SHORT" else "SELL"
        qty = 0.0
        try:
            pos_data  = await asyncio.to_thread(get_position_risk, self.symbol)
            positions = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
            for pos in positions:
                if pos.get("positionSide") == self.direction:
                    qty = abs(float(pos.get("positionAmt", 0)))
                    break
        except Exception as e:
            self._tag(f"[close] position fetch error: {e}")

        if qty <= 0:
            self._tag(f"[close] {reason}: no open position found")
            return 0.0, 0.0

        self._tag(f"[close] {reason}: chase {close_side} {qty}")
        return await self._chase_limit(
            close_side, qty, self.direction, timeout=15.0, label=f"close_{reason}"
        )

    def _set_cooldown(self, result: str):
        if not self._current_zone_price:
            return
        if result == "LOSS":
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3:
                cd = 1800
                self._tag(f"[cooldown] {self._consecutive_losses} consecutive losses"
                          f" — 30min pause on zone")
            else:
                cd = REENTRY_LOSS_COOL
        else:
            self._consecutive_losses = 0
            cd = REENTRY_WIN_COOL
        self._zone_cooldowns[self._current_zone_price] = time.time() + cd
        self._tag(f"[cooldown] Zone ${self._current_zone_price:,.0f}"
                  f"  cooldown={cd}s  ({result})")

    def _cleanup(self):
        for o in [self.trade2, self.tp_order]:
            if o and not o.is_filled and not o.is_cancelled:
                o.cancel()
        try:
            import asyncio as _aio
            loop = _aio.get_event_loop()
            if loop.is_running():
                loop.create_task(asyncio.to_thread(cancel_all_orders, self.symbol))
        except Exception:
            pass
        self.trade1 = self.trade2 = self.tp_order = None
        self.t1_filled   = False
        self.t1_qty      = 0.0
        self.trail_stage = "none"
        self.best_seen   = None
        self.state       = "WATCHING"
        self._write_state()

    async def _print_status(self):
        try:
            price = await asyncio.to_thread(get_price, self.symbol)
            pr    = self.cfg["price_round"]
            self._tag(f"[status] {self.direction}  price=${price:,.{pr}f}"
                      f"  tp=${self.tp_price:,.{pr}f} ({abs(price-self.tp_price):.{pr}f} away)"
                      f"  sl=${self.sl_price:,.{pr}f} ({abs(price-self.sl_price):.{pr}f} away)"
                      f"  trail={self.trail_stage}")
        except Exception:
            pass
        self._write_state()

    def _log_trade(self, result, pnl, close_price, close_qty):
        self.trade_count += 1
        pr = self.cfg["price_round"]
        new_entry = {
            "id":           self.trade_count,
            "time":         datetime.now().isoformat(),
            "symbol":       self.symbol,
            "direction":    self.direction,
            "entry":        round(self.entry_price, pr),
            "actual_close": round(close_price, pr) if close_price else None,
            "actual_qty":   close_qty or None,
            "tp":           self.tp_price,
            "sl":           self.sl_price,
            "atr":          self.atr,
            "confidence":   self.confidence,
            "risk_usdt":    self.risk_usdt,
            "mode":         "LSCO_v2",
            "result":       result,
            "pnl_usd":      round(pnl, 6),
            "pnl_pct":      round(pnl / self.risk_usdt * 100, 4) if self.risk_usdt else 0,
            "total_pnl":    round(self.total_pnl, 6),
        }
        log = []
        if TRADE_LOG_JSON.exists():
            try:
                with open(TRADE_LOG_JSON) as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append(new_entry)
        with open(TRADE_LOG_JSON, "w") as f:
            json.dump(log, f, indent=2)
        self._tag(f"[log] #{self.trade_count} logged  result={result}  pnl=${pnl:+.4f}")

    def _write_state(self):
        try:
            with open(state_json(self.symbol), "w") as f:
                json.dump({
                    "state":       self.state,
                    "symbol":      self.symbol,
                    "direction":   self.direction or "",
                    "entry":       self.entry_price,
                    "tp":          self.tp_price,
                    "sl":          self.sl_price,
                    "atr":         self.atr,
                    "confidence":  self.confidence,
                    "trail_stage": self.trail_stage,
                    "risk_usdt":   self.risk_usdt,
                    "mode":        "LSCO_v2",
                    "updated_at":  datetime.now().isoformat(),
                }, f, indent=2)
        except Exception:
            pass


# ==============================================================================
# STARTUP & MAIN
# ==============================================================================

async def heatmap_refresh_loop(symbol: str):
    first = True
    while True:
        label = "startup fetch" if first else "5-min refresh"
        print(f"  [{symbol}][heatmap] {label} ...")
        first = False
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(HEATMAP_SCRIPT), "--symbol", symbol,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            for line in stdout.decode("utf-8", errors="replace").splitlines():
                if any(kw in line for kw in ["Current Price", "Data saved", "Chart saved",
                                              "Total modeled", "Sources", "Heatmap v2",
                                              "failed", "Error", "Traceback"]):
                    print(f"  [{symbol}][heatmap] {line.strip()}")
            print(f"  [{symbol}][heatmap] {'OK' if proc.returncode == 0 else f'exit {proc.returncode}'}")
        except Exception as e:
            print(f"  [{symbol}][heatmap] error: {e}")
        await asyncio.sleep(HEATMAP_REFRESH)


async def startup():
    for sym, cfg in SYMBOL_CONFIG.items():
        try:
            await asyncio.to_thread(set_leverage, sym, cfg["leverage"])
            print(f"  [startup] Leverage = {cfg['leverage']}x on {sym}")
        except Exception as e:
            print(f"  [startup] {sym} leverage: {e}")
    try:
        bal  = await asyncio.to_thread(get_balances)
        bals = bal if isinstance(bal, list) else bal.get("data", [])
        for b in bals:
            w = float(b.get("balance") or b.get("walletBalance") or 0)
            if w > 0:
                print(f"  [startup] Balance: {w:.4f} {b.get('asset', '?')}")
    except Exception as e:
        print(f"  [startup] balance: {e}")


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  LSCO v2  --  Multi-Asset Liquidation Scalp Engine")
    print(f"  Symbols  : {', '.join(SYMBOLS)}")
    print(f"  Sizing   : {BALANCE_UTILIZATION:.0%} balance  cap=${MAX_RISK_USDT}  (min lots)")
    print(f"  Trigger  : touch+wick (wick>={MIN_WICK:.0%}xATR)")
    print(f"  TP / SL  : {TP_MULT}x / {SL_MULT}x ATR  (2:1 R:R)")
    print(f"  Re-entry : {REENTRY_WIN_COOL}s after WIN  /  {REENTRY_LOSS_COOL}s after LOSS")
    print("=" * 60)
    print()

    await startup()

    engines = [TradingEngine(sym) for sym in SYMBOLS]
    for eng in engines:
        await eng.recover_open_position()

    # Stagger heatmap startup by 30s to avoid simultaneous API hammering
    tasks = []
    for i, (eng, sym) in enumerate(zip(engines, SYMBOLS)):
        tasks.append(asyncio.create_task(_delayed_heatmap(sym, delay=i * 30)))
        tasks.append(asyncio.create_task(eng.minute_loop()))
        tasks.append(asyncio.create_task(eng.position_monitor_loop()))

    await asyncio.gather(*tasks)


async def _delayed_heatmap(symbol: str, delay: int):
    if delay:
        await asyncio.sleep(delay)
    await heatmap_refresh_loop(symbol)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  [lsco] Stopped. Ctrl+C")
        sys.exit(0)
