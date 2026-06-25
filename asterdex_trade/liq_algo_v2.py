"""
liq_algo_v2.py  —  Multi-Asset Liquidation Zone Reversal Engine
================================================================
Runs BTCUSDT, ETHUSDT, and XAUUSDT in parallel.

Same algorithm as liq_algo.py (close-back-inside trigger + baby layer),
but parameterized by symbol. Original liq_algo.py is NOT modified.

HOW IT WORKS:
  1. Every minute: read per-symbol heatmap and find zones near price.
  2. Price within APPROACH_PCT% of zone → calculate Q and confidence.
  3. ENTRY TRIGGER: 1m candle closes back inside zone boundary.
  4. Trade 1: SmartLimitOrder at best bid/ask (immediate entry).
  5. Trade 2: resting limit at Q-avg (deep retracement entry).
  6. Baby layer: 3 auto-resetting scalp orders around entry.
  7. Exits: TP at 1.5×ATR, SL at 0.75×ATR.

Run: python liq_algo_v2.py
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from account_data   import (place_order, cancel_all_orders,
                             get_position_risk, get_balances, set_leverage)
from market_data    import get_price, get_klines, calc_atr, get_last_1m_candle
from order_executor import SmartLimitOrder


# ==============================================================================
# PER-SYMBOL CONFIG
# ==============================================================================

SYMBOL_CONFIG = {
    "BTCUSDT": {
        "leverage":     20,
        "min_qty":      0.001,
        "qty_round":    3,
        "price_round":  1,
        "min_zone_usd": 10_000_000,
        "q_min_usd":    20_000_000,
        "approach_pct": 0.8,
        "whale_file":   "whale_BTCUSDT.json",
        "state_file":   "algo_state_v1_BTCUSDT.json",
    },
    "ETHUSDT": {
        "leverage":     20,
        "min_qty":      0.001,
        "qty_round":    3,
        "price_round":  2,
        "min_zone_usd": 2_000_000,
        "q_min_usd":    5_000_000,
        "approach_pct": 0.8,
        "whale_file":   "whale_ETHUSDT.json",
        "state_file":   "algo_state_v1_ETHUSDT.json",
    },
    "XAUUSDT": {
        "leverage":     10,
        "min_qty":      0.01,
        "qty_round":    2,
        "price_round":  2,
        "min_zone_usd": 500_000,
        "q_min_usd":    1_000_000,
        "approach_pct": 0.8,
        "whale_file":   "whale_XAUUSDT.json",
        "state_file":   "algo_state_v1_XAUUSDT.json",
    },
}

SYMBOLS = list(SYMBOL_CONFIG.keys())

# Shared risk settings
BALANCE_UTILIZATION = 0.01
MIN_RISK_USDT       = 1.0
MAX_RISK_USDT       = 2.0

# Confidence gate
MIN_CONFIDENCE = 65

# Exit parameters
ATR_PERIOD   = 14
ATR_INTERVAL = "5m"
TP_ATR_MULT  = 1.5
SL_ATR_MULT  = 0.75

# Baby layer
BABY_OFFSETS  = [0.001, 0.002, 0.004]
BABY_SIZES    = [0.10,  0.20,  0.40]
BABY_TP_PCT   = 0.0015

# Confidence → Trade2 size multiplier
CONF_MULTIPLIERS = {
    (0,  60): 1.6,
    (60, 75): 1.8,
    (75, 90): 2.0,
    (90, 101): 2.2,
}

ROOT_DIR       = Path(__file__).parent.parent
TRADE_LOG_JSON = ROOT_DIR / "trade_log.json"
HEATMAP_SCRIPT = ROOT_DIR / "data_fetching" / "binance_liq_heatmap_2.py"
HEATMAP_REFRESH = 5 * 60


def heatmap_json(symbol: str) -> Path:
    return ROOT_DIR / "data_fetching" / f"binance_liq_heatmap_{symbol}.json"

def whale_json(symbol: str) -> Path:
    return ROOT_DIR / SYMBOL_CONFIG[symbol]["whale_file"]

def state_json(symbol: str) -> Path:
    return ROOT_DIR / SYMBOL_CONFIG[symbol]["state_file"]


# ==============================================================================
# ZONE MATH
# ==============================================================================

def calc_risk_usdt(symbol: str) -> float:
    cfg        = SYMBOL_CONFIG[symbol]
    max_mult   = max(CONF_MULTIPLIERS.values())
    tot_factor = 1.0 + max_mult + sum(BABY_SIZES)
    try:
        bal_data  = get_balances()
        balances  = bal_data if isinstance(bal_data, list) else bal_data.get("data", [])
        available = 0.0
        for b in balances:
            if b.get("asset", "").upper() in ("USDT", "BUSD"):
                av_str = (b.get("availableBalance") or b.get("walletBalance")
                          or b.get("balance") or "0")
                available = max(available, float(av_str))
        if available <= 0:
            return MIN_RISK_USDT
        raw     = available * BALANCE_UTILIZATION / tot_factor
        clamped = max(MIN_RISK_USDT, min(MAX_RISK_USDT, raw))
        print(f"  [{symbol}][sizing] Balance=${available:.2f}  → RISK_USDT=${clamped:.4f}")
        return round(clamped, 4)
    except Exception as e:
        print(f"  [{symbol}][sizing] failed ({e}) → fallback ${MIN_RISK_USDT}")
        return MIN_RISK_USDT


def load_heatmap(symbol: str) -> dict:
    path = heatmap_json(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Heatmap not found: {path}  (run heatmap script first)")
    with open(path) as f:
        data = json.load(f)
    generated = datetime.fromisoformat(data["generated_at"])
    age_min = (datetime.now() - generated).total_seconds() / 60
    if age_min > 60:
        print(f"  [{symbol}][scanner] Heatmap is {age_min:.0f} min old")
    return data


def load_whale_candle(symbol: str) -> dict | None:
    try:
        with open(whale_json(symbol)) as f:
            data = json.load(f)
        return data if time.time() - data["candle_ts"] < 180 else None
    except Exception:
        return None


def scan_zones(current_price: float, heatmap: dict, cfg: dict) -> dict:
    min_z = cfg["min_zone_usd"]
    shorts = [z for z in heatmap["short_liquidations"] if z["usd"] >= min_z]
    longs  = [z for z in heatmap["long_liquidations"]  if z["usd"] >= min_z]

    above = [z for z in shorts if z["price"] > current_price]
    below = [z for z in longs  if z["price"] <= current_price]
    nearest_above = min(above, key=lambda z: z["price"]) if above else None
    nearest_below = max(below, key=lambda z: z["price"]) if below else None

    result = {
        "price": current_price,
        "nearest_above":    nearest_above,
        "nearest_below":    nearest_below,
        "approaching_up":   False,
        "approaching_down": False,
        "dist_above_pct":   None,
        "dist_below_pct":   None,
    }
    app = cfg["approach_pct"]
    if nearest_above:
        dist = (nearest_above["price"] - current_price) / current_price * 100
        result["dist_above_pct"] = dist
        result["approaching_up"] = dist <= app
    if nearest_below:
        dist = (current_price - nearest_below["price"]) / current_price * 100
        result["dist_below_pct"] = dist
        result["approaching_down"] = dist <= app
    return result


def calc_q(zones: list, nearest_zone: dict, direction: str, cfg: dict) -> tuple:
    threshold = max(2.0 * nearest_zone["usd"], cfg["q_min_usd"])
    if direction == "above":
        candidates = sorted([z for z in zones if z["price"] > nearest_zone["price"]],
                            key=lambda z: z["price"])
    else:
        candidates = sorted([z for z in zones if z["price"] < nearest_zone["price"]],
                            key=lambda z: -z["price"])

    q_zones, running_sum = [nearest_zone], nearest_zone["usd"]
    for z in candidates:
        if z["usd"] < cfg["min_zone_usd"]:
            continue
        q_zones.append(z)
        running_sum += z["usd"]
        if running_sum >= threshold:
            break

    total_usd = sum(z["usd"] for z in q_zones)
    q_avg     = sum(z["price"] * z["usd"] for z in q_zones) / total_usd
    return q_zones, q_avg, total_usd


def calc_confidence(scan: dict, q_zones: list, q_total: float,
                    whale: dict | None, direction: str) -> int:
    score   = 50
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


def confidence_multiplier(score: int) -> float:
    for (lo, hi), mult in CONF_MULTIPLIERS.items():
        if lo <= score < hi:
            return mult
    return 1.6


def qty_from_risk(risk_usdt: float, price: float, cfg: dict) -> float:
    notional = risk_usdt * cfg["leverage"]
    qty      = round(notional / price, cfg["qty_round"])
    return max(qty, cfg["min_qty"])


# ==============================================================================
# BABY LAYER
# ==============================================================================

class BabyLayer:

    def __init__(self, symbol: str, direction: str,
                 entry_price: float, parent_notional: float,
                 pos_side: str, sl_price: float, price_round: int):
        self.symbol          = symbol
        self.direction       = direction.upper()
        self.entry_price     = entry_price
        self.parent_notional = parent_notional
        self.pos_side        = pos_side.upper()
        self.sl_price        = sl_price
        self.price_round     = price_round
        self.alive           = True
        self._active_orders  = []

    async def start(self):
        print(f"  [{self.symbol}][baby] Spawning baby layer ({self.direction})"
              f" around entry ${self.entry_price:,.{self.price_round}f}")
        for idx, (offset, size_frac) in enumerate(zip(BABY_OFFSETS, BABY_SIZES), 1):
            asyncio.create_task(self._baby_lifecycle(offset, size_frac, idx))

    async def _baby_lifecycle(self, offset_pct: float, size_frac: float, idx: int):
        while self.alive:
            if self.direction == "SHORT":
                baby_entry_price = round(self.entry_price * (1 + offset_pct), self.price_round)
                baby_side        = "SELL"
                tp_price         = round(baby_entry_price * (1 - BABY_TP_PCT), self.price_round)
                if baby_entry_price >= self.sl_price:
                    break
            else:
                baby_entry_price = round(self.entry_price * (1 - offset_pct), self.price_round)
                baby_side        = "BUY"
                tp_price         = round(baby_entry_price * (1 + BABY_TP_PCT), self.price_round)
                if baby_entry_price <= self.sl_price:
                    break

            qty   = max(round((self.parent_notional * size_frac) / baby_entry_price, 3), 0.001)
            label = f"baby{idx}({offset_pct*100:.1f}%)"

            entry_order = SmartLimitOrder(
                symbol=self.symbol, side=baby_side, qty=qty,
                pos_side=self.pos_side, label=label,
                track_ob=False, fixed_price=baby_entry_price,
                price_round=self.price_round,
            )
            self._active_orders.append(entry_order)
            await entry_order.place()

            filled = await entry_order.wait_fill(timeout=3600)
            if not self.alive or not filled:
                entry_order.cancel()
                break

            fill_px    = entry_order.fill_price
            close_side = "BUY" if baby_side == "SELL" else "SELL"
            tp_order   = SmartLimitOrder(
                symbol=self.symbol, side=close_side, qty=qty,
                pos_side=self.pos_side, label=f"{label}_tp",
                track_ob=False, fixed_price=tp_price,
                price_round=self.price_round,
            )
            self._active_orders.append(tp_order)
            await tp_order.place()

            tp_filled = await tp_order.wait_fill(timeout=1800)
            if not self.alive:
                tp_order.cancel(); break

            if tp_filled:
                profit = abs(tp_price - fill_px) * qty
                print(f"  [{self.symbol}][baby] {label} TP hit  profit≈${profit:.4f}")
            else:
                tp_order.cancel(); break

    def kill_all(self):
        self.alive = False
        for order in self._active_orders:
            if not order.is_filled and not order.is_cancelled:
                order.cancel()
        self._active_orders.clear()


# ==============================================================================
# TRADING ENGINE
# ==============================================================================

class TradingEngine:

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cfg    = SYMBOL_CONFIG[symbol]

        self.state            = "WATCHING"
        self.direction        = None
        self.setup            = {}
        self.q_zones          = []
        self.q_avg            = 0.0
        self.q_total_usd      = 0.0
        self.confidence       = 0
        self.entry_price      = 0.0
        self.atr              = 0.0
        self.tp_price         = 0.0
        self.sl_price         = 0.0
        self.trail_stage      = "none"
        self.best_price_seen  = None
        self.risk_usdt        = MIN_RISK_USDT
        self._last_trigger_ts = 0
        self._heatmap         = None

        self.trade1   = None
        self.trade2   = None
        self.tp_order = None
        self.babies   = None

        self.trade_count = 0
        self.wins        = 0
        self.losses      = 0
        self.total_pnl   = 0.0

    def _tag(self, msg):
        print(f"  [{self.symbol}] {msg}")

    # ── Loops ──────────────────────────────────────────────────────────────────

    async def minute_loop(self):
        self._tag("v1 engine started. State: WATCHING")
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
                print(f"  [{self.symbol}] ERROR: {e}")
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
        if self.state == "WATCHING":
            await self._tick_watching()
        elif self.state == "APPROACHING":
            await self._tick_approaching()
        elif self.state == "IN_TRADE":
            await self._print_position_status()

    async def _tick_watching(self):
        try:
            self._heatmap = await asyncio.to_thread(load_heatmap, self.symbol)
        except FileNotFoundError as e:
            self._tag(f"[scanner] {e}"); return

        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(scan_zones, price, self._heatmap, self.cfg)

        if scan["nearest_above"]:
            z = scan["nearest_above"]
            flag = " ← APPROACHING" if scan["approaching_up"] else ""
            self._tag(f"[scan] Above: ${z['price']:>10,.{self.cfg['price_round']}f}"
                      f"  ${z['usd']/1e6:.1f}M  ({scan['dist_above_pct']:.2f}% away){flag}")
        if scan["nearest_below"]:
            z = scan["nearest_below"]
            flag = " ← APPROACHING" if scan["approaching_down"] else ""
            self._tag(f"[scan] Below: ${z['price']:>10,.{self.cfg['price_round']}f}"
                      f"  ${z['usd']/1e6:.1f}M  ({scan['dist_below_pct']:.2f}% away){flag}")

        if scan["approaching_up"] or scan["approaching_down"]:
            if scan["approaching_up"] and scan["approaching_down"]:
                du = scan["dist_above_pct"] or 999
                dd = scan["dist_below_pct"] or 999
                direction = "SHORT" if du < dd else "LONG"
            elif scan["approaching_up"]:
                direction = "SHORT"
            else:
                direction = "LONG"
            self.direction = direction
            self.setup     = scan
            self.state     = "APPROACHING"
            self._tag(f"[engine] → APPROACHING  direction={direction}")
            await self._tick_approaching()

    async def _tick_approaching(self):
        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(scan_zones, price, self._heatmap, self.cfg)

        if self.direction == "SHORT" and not scan["approaching_up"]:
            self._tag("[engine] Zone passed → WATCHING")
            self.state = "WATCHING"; return
        if self.direction == "LONG" and not scan["approaching_down"]:
            self._tag("[engine] Zone passed → WATCHING")
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
            calc_q, zones, nearest, dir_str, self.cfg
        )
        self.q_zones     = q_zones
        self.q_avg       = q_avg
        self.q_total_usd = q_total

        whale     = await asyncio.to_thread(load_whale_candle, self.symbol)
        conf      = calc_confidence(scan, q_zones, q_total, whale, self.direction.lower())
        self.confidence = conf

        whale_avg = 0.0
        if whale:
            if self.direction == "SHORT" and whale["whale_sell_usd"] > 0:
                whale_avg = whale["whale_sell_avg_price"]
            elif self.direction == "LONG" and whale["whale_buy_usd"] > 0:
                whale_avg = whale["whale_buy_avg_price"]
        trade2_price = (q_avg + whale_avg) / 2 if whale_avg > 0 else q_avg

        mult = confidence_multiplier(conf)
        self._tag(f"[engine] Q=${q_total/1e6:.1f}M  {len(q_zones)} zones"
                  f"  conf={conf}  mult={mult:.1f}x")

        last_candle = await asyncio.to_thread(get_last_1m_candle, self.symbol)
        zone_price  = nearest["price"]

        if last_candle["ts"] == self._last_trigger_ts:
            self._tag("[engine] Same candle — skipping")
            self.state = "WATCHING"; return

        if conf < MIN_CONFIDENCE:
            self._tag(f"[engine] conf={conf} < {MIN_CONFIDENCE} — skipping")
            self.state = "WATCHING"; return

        # ── Trigger: closed back inside zone ─────────────────────────────────
        triggered = False
        pr = self.cfg["price_round"]
        if self.direction == "SHORT":
            triggered = (last_candle["high"] > zone_price and
                         last_candle["close"] < zone_price)
            self._tag(f"[trigger?] H={last_candle['high']:,.{pr}f}"
                      f"  zone={zone_price:,.{pr}f}"
                      f"  close={last_candle['close']:,.{pr}f}  triggered={triggered}")
        else:
            triggered = (last_candle["low"] < zone_price and
                         last_candle["close"] > zone_price)
            self._tag(f"[trigger?] L={last_candle['low']:,.{pr}f}"
                      f"  zone={zone_price:,.{pr}f}"
                      f"  close={last_candle['close']:,.{pr}f}  triggered={triggered}")

        if not triggered:
            return

        # ── FIRE ──────────────────────────────────────────────────────────────
        self._tag(f"\n  ★ TRIGGER — dir={self.direction}  conf={conf}  mult={mult:.1f}×")
        self._last_trigger_ts = last_candle["ts"]
        self.risk_usdt        = await asyncio.to_thread(calc_risk_usdt, self.symbol)

        klines   = await asyncio.to_thread(get_klines, self.symbol, ATR_INTERVAL, 50)
        atr      = calc_atr(klines, ATR_PERIOD)
        self.atr = atr

        current          = last_candle["close"]
        self.entry_price = current

        if self.direction == "SHORT":
            self.tp_price = current - TP_ATR_MULT * atr
            self.sl_price = max(zone_price + atr, current + 0.5 * atr)
        else:
            self.tp_price = current + TP_ATR_MULT * atr
            self.sl_price = min(zone_price - atr, current - 0.5 * atr)

        base_qty   = qty_from_risk(self.risk_usdt,        current, self.cfg)
        trade2_qty = qty_from_risk(self.risk_usdt * mult, current, self.cfg)

        self._tag(f"[trade] Entry={current:,.{pr}f}  TP={self.tp_price:,.{pr}f}"
                  f"  SL={self.sl_price:,.{pr}f}  ATR={atr:.{pr}f}")
        self._tag(f"[trade] T1={base_qty}  T2={trade2_qty} @ {trade2_price:,.{pr}f}")

        entry_side, pos_side = (("SELL", "SHORT") if self.direction == "SHORT"
                                 else ("BUY", "LONG"))

        self.trade1 = SmartLimitOrder(
            symbol=self.symbol, side=entry_side, qty=base_qty,
            pos_side=pos_side, label="Trade1_entry", track_ob=True,
            price_round=pr,
        )
        await self.trade1.place()

        self.trade2 = SmartLimitOrder(
            symbol=self.symbol, side=entry_side, qty=trade2_qty,
            pos_side=pos_side, label="Trade2_limit",
            track_ob=False, fixed_price=trade2_price,
            price_round=pr,
        )
        await self.trade2.place()

        self.state = "IN_TRADE"
        self._tag("[engine] → IN_TRADE")

    # ── Position monitor ───────────────────────────────────────────────────────

    async def _update_trail(self, price: float):
        if not self.trade1 or not self.trade1.is_filled:
            return
        profit_pts = (self.entry_price - price if self.direction == "SHORT"
                      else price - self.entry_price)
        if profit_pts <= 0:
            return

        if self.best_price_seen is None:
            self.best_price_seen = price
        if self.direction == "SHORT":
            self.best_price_seen = min(self.best_price_seen, price)
        else:
            self.best_price_seen = max(self.best_price_seen, price)

        pr     = self.cfg["price_round"]
        new_sl = None
        if profit_pts >= 1.0 * self.atr:
            if self.direction == "SHORT":
                c = round(self.best_price_seen + 0.6 * self.atr, pr)
                if c < self.sl_price: new_sl = c
            else:
                c = round(self.best_price_seen - 0.6 * self.atr, pr)
                if c > self.sl_price: new_sl = c
            if new_sl:
                prev = self.trail_stage; self.trail_stage = "trailing"
                tag  = "TRAILING started" if prev != "trailing" else "TRAIL updated"
                self._tag(f"[trail] {tag}  SL -> ${new_sl:,.{pr}f}")
        elif profit_pts >= 0.5 * self.atr and self.trail_stage == "none":
            be = round(self.entry_price + (5.0 if self.direction == "SHORT" else -5.0), pr)
            if ((self.direction == "SHORT" and be < self.sl_price) or
                    (self.direction == "LONG" and be > self.sl_price)):
                new_sl = be
                self.trail_stage = "breakeven"
                self._tag(f"[trail] BREAKEVEN  SL -> ${new_sl:,.{pr}f}")

        if new_sl:
            self.sl_price = new_sl
            self._write_state()

    async def _monitor_position(self):
        price = await asyncio.to_thread(get_price, self.symbol)

        # External close detection
        if self.trade1 and self.trade1.is_filled:
            try:
                pos_data   = await asyncio.to_thread(get_position_risk, self.symbol)
                positions  = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
                pos_exists = any(
                    p.get("positionSide") == self.direction
                    and abs(float(p.get("positionAmt", 0))) > 0
                    for p in positions
                )
                if not pos_exists:
                    self._tag("[monitor] Position closed externally — resetting")
                    t1q  = self.trade1.qty if self.trade1 else 0.0
                    t2q  = (self.trade2.qty if (self.trade2 and self.trade2.is_filled) else 0.0)
                    qty  = round(t1q + t2q, 3)
                    if qty > 0 and self.entry_price > 0:
                        pnl = ((self.entry_price - price) if self.direction == "SHORT"
                               else (price - self.entry_price)) * qty
                        result = "WIN" if pnl >= 0 else "LOSS"
                        if pnl >= 0: self.wins   += 1
                        else:        self.losses += 1
                        self.total_pnl += pnl
                        self._log_trade(result, pnl, price, qty)
                    self._reset_trade_state()
                    return
            except Exception:
                pass

        # Check Trade1 fill → spawn babies
        if self.trade1 and not self.trade1.is_filled and not self.babies:
            await self.trade1._check_fill()
            if self.trade1.is_filled and not self.babies:
                fill_px          = self.trade1.fill_price or self.entry_price
                self.entry_price = fill_px
                notional         = self.risk_usdt * self.cfg["leverage"]
                pos_side         = self.direction
                self.babies      = BabyLayer(
                    symbol=self.symbol, direction=self.direction,
                    entry_price=fill_px, parent_notional=notional,
                    pos_side=pos_side, sl_price=self.sl_price,
                    price_round=self.cfg["price_round"],
                )
                await self.babies.start()

                close_side    = "BUY" if self.direction == "SHORT" else "SELL"
                self.tp_order = SmartLimitOrder(
                    symbol=self.symbol, side=close_side,
                    qty=self.trade1.qty, pos_side=pos_side, label="Trade1_TP",
                    track_ob=False, fixed_price=self.tp_price,
                    price_round=self.cfg["price_round"],
                )
                await self.tp_order.place()

        await self._update_trail(price)

        if not (self.trade1 and self.trade1.is_filled):
            return

        if self.direction == "SHORT" and price <= self.tp_price:
            self._tag(f"[monitor] TP reached @ {price:,.{self.cfg['price_round']}f}")
            await self._on_tp_hit(); return
        if self.direction == "LONG"  and price >= self.tp_price:
            self._tag(f"[monitor] TP reached @ {price:,.{self.cfg['price_round']}f}")
            await self._on_tp_hit(); return
        if self.direction == "SHORT" and price >= self.sl_price:
            self._tag(f"[monitor] SL hit @ {price:,.{self.cfg['price_round']}f}")
            await self._on_sl_hit(); return
        if self.direction == "LONG"  and price <= self.sl_price:
            self._tag(f"[monitor] SL hit @ {price:,.{self.cfg['price_round']}f}")
            await self._on_sl_hit(); return

    async def _on_tp_hit(self):
        close_price, close_qty = await self._emergency_close("TP")
        if close_qty == 0:
            self._tag("[engine] TP: no position — not logged.")
            self._reset_trade_state(); return
        self.wins += 1
        profit = (((self.entry_price - close_price) if self.direction == "SHORT"
                   else (close_price - self.entry_price)) * close_qty)
        profit = max(profit, 0.0)
        self.total_pnl += profit
        self._log_trade("WIN", profit, close_price, close_qty)
        self._reset_trade_state()
        self._tag(f"[engine] WIN  +${profit:.4f}  total=${self.total_pnl:.4f}")

    async def _on_sl_hit(self):
        close_price, close_qty = await self._emergency_close("SL")
        if close_qty == 0:
            self._tag("[engine] SL: no position — not logged.")
            self._reset_trade_state(); return
        self.losses += 1
        loss = (((close_price - self.entry_price) if self.direction == "SHORT"
                 else (self.entry_price - close_price)) * close_qty)
        loss = max(loss, 0.0)
        self.total_pnl -= loss
        self._log_trade("LOSS", -loss, close_price, close_qty)
        self._reset_trade_state()
        self._tag(f"[engine] LOSS  -${loss:.4f}  total=${self.total_pnl:.4f}")

    async def _emergency_close(self, reason: str) -> tuple:
        if self.babies:   self.babies.kill_all()
        for o in [self.trade1, self.trade2, self.tp_order]:
            if o and not o.is_filled and not o.is_cancelled:
                o.cancel()
        try:
            await asyncio.to_thread(cancel_all_orders, self.symbol)
        except Exception:
            pass

        actual_close_price = 0.0
        actual_close_qty   = 0.0
        pr                 = self.cfg["price_round"]
        try:
            pos_data     = await asyncio.to_thread(get_position_risk, self.symbol)
            positions    = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
            close_side   = "BUY" if self.direction == "SHORT" else "SELL"
            for pos in positions:
                if pos.get("positionSide") == self.direction:
                    total_qty = abs(float(pos.get("positionAmt", 0)))
                    if total_qty > 0:
                        close_order = SmartLimitOrder(
                            symbol=self.symbol, side=close_side, qty=total_qty,
                            pos_side=self.direction, label=f"close_{reason}",
                            track_ob=True, price_round=pr,
                        )
                        await close_order.place()
                        await close_order.wait_fill(timeout=120)
                        actual_close_price = close_order.fill_price or 0.0
                        actual_close_qty   = total_qty
                    break
        except Exception as e:
            self._tag(f"[close] error: {e}")

        return actual_close_price, actual_close_qty

    def _reset_trade_state(self):
        self.trade1          = None
        self.trade2          = None
        self.tp_order        = None
        self.babies          = None
        self.trail_stage     = "none"
        self.best_price_seen = None
        self.state           = "WATCHING"
        self._write_state()

    async def _print_position_status(self):
        try:
            price = await asyncio.to_thread(get_price, self.symbol)
            pr    = self.cfg["price_round"]
            self._tag(f"[status] {self.direction}  price={price:,.{pr}f}"
                      f"  tp={self.tp_price:,.{pr}f} ({abs(price-self.tp_price):.{pr}f} away)"
                      f"  sl={self.sl_price:,.{pr}f} ({abs(price-self.sl_price):.{pr}f} away)")
        except Exception:
            pass
        self._write_state()

    def _write_state(self):
        try:
            pr = self.cfg["price_round"]
            with open(state_json(self.symbol), "w") as f:
                json.dump({
                    "state":       self.state,
                    "symbol":      self.symbol,
                    "direction":   self.direction or "",
                    "entry":       round(self.entry_price, pr),
                    "tp":          round(self.tp_price, pr),
                    "sl":          round(self.sl_price, pr),
                    "atr":         round(self.atr, pr),
                    "confidence":  self.confidence,
                    "trail_stage": self.trail_stage,
                    "risk_usdt":   self.risk_usdt,
                    "mode":        "LIQ_v2",
                    "updated_at":  datetime.now().isoformat(),
                }, f, indent=2)
        except Exception:
            pass

    def _log_trade(self, result: str, pnl: float,
                   actual_close_price: float = 0.0, actual_close_qty: float = 0.0):
        self.trade_count += 1
        pr    = self.cfg["price_round"]
        entry = {
            "id":          self.trade_count,
            "time":        datetime.now().isoformat(),
            "symbol":      self.symbol,
            "direction":   self.direction,
            "entry":       round(self.entry_price, pr),
            "actual_close": round(actual_close_price, pr) if actual_close_price else None,
            "actual_qty":  actual_close_qty or None,
            "tp":          self.tp_price,
            "sl":          self.sl_price,
            "atr":         self.atr,
            "confidence":  self.confidence,
            "risk_usdt":   self.risk_usdt,
            "q_zones":     len(self.q_zones),
            "q_total_usd": self.q_total_usd,
            "mode":        "LIQ_v2",
            "result":      result,
            "pnl_usd":     round(pnl, 6),
            "pnl_pct":     round(pnl / self.risk_usdt * 100, 4) if self.risk_usdt else 0,
            "total_pnl":   round(self.total_pnl, 6),
        }
        log = []
        if TRADE_LOG_JSON.exists():
            try:
                with open(TRADE_LOG_JSON) as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append(entry)
        with open(TRADE_LOG_JSON, "w") as f:
            json.dump(log, f, indent=2)
        self._tag(f"[log] #{self.trade_count} {result}  pnl=${pnl:+.4f}")


# ==============================================================================
# STARTUP & MAIN
# ==============================================================================

async def heatmap_refresh_loop(symbol: str):
    first = True
    while True:
        label = "startup" if first else "5-min refresh"
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
                                              "Total modeled", "failed", "Error", "Traceback"]):
                    print(f"  [{symbol}][heatmap] {line.strip()}")
            print(f"  [{symbol}][heatmap] "
                  f"{'OK' if proc.returncode == 0 else f'exit {proc.returncode}'}")
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


async def _delayed_heatmap(symbol: str, delay: int):
    if delay:
        await asyncio.sleep(delay)
    await heatmap_refresh_loop(symbol)


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  LIQ ALGO v2  --  Multi-Asset Liquidation Reversal Engine")
    print(f"  Symbols  : {', '.join(SYMBOLS)}")
    print(f"  Sizing   : {BALANCE_UTILIZATION:.0%} balance  cap=${MAX_RISK_USDT}")
    print(f"  TP / SL  : {TP_ATR_MULT}x / {SL_ATR_MULT}x ATR")
    print(f"  Approach : within {SYMBOL_CONFIG['BTCUSDT']['approach_pct']}% of zone")
    print("=" * 60)
    print()

    await startup()

    engines = [TradingEngine(sym) for sym in SYMBOLS]

    tasks = []
    for i, (eng, sym) in enumerate(zip(engines, SYMBOLS)):
        tasks.append(asyncio.create_task(_delayed_heatmap(sym, delay=i * 30)))
        tasks.append(asyncio.create_task(eng.minute_loop()))
        tasks.append(asyncio.create_task(eng.position_monitor_loop()))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  [liq_algo_v2] Stopped. Ctrl+C")
        sys.exit(0)
