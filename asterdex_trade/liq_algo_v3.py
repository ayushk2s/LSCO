"""
liq_algo_v3.py  —  Multi-Asset Liquidation Zone Reversal Engine  v3
====================================================================
BTC + ETH + XAU in parallel.

KEY CHANGES FROM v2:
  - Baby layer removed: no more static resting limits that pile into a losing trade
  - New: FlashScalpLayer — ONLY fires when sudden 0.1–0.25% flash dip/spike
    detected within a 15-second window (caused by liquidation cascade or OB
    imbalance). Places 1 baby at best bid/ask, TP at snap-back recovery level.
    If price keeps trending (not a flash) → baby auto-cancels in 20s, no fill.

HOW THE REVERSAL SYSTEM WORKS:
  1. Liquidation heatmap zone detected → price approaching
  2. Entry candle: closes back inside the zone (reversal signal)
  3. Trade1: SmartLimitOrder at best bid/ask, chases until filled
  4. Trade2: resting limit at Q-avg (deep retracement entry)
  5. FlashScalpLayer: monitors price every 1.5s after Trade1 fills
       - Flash dip detected (LONG): price drops 0.10–0.25% in <15s
         → liquidation cascade / OB imbalance = snap-back opportunity
         → 1 baby BUY at best bid, TP at +0.15% recovery
         → auto-cancel if not filled in 20s (not a flash, it's a trend)
       - Flash spike (SHORT): same logic inverted
       - Max 2 babies active, 30s cooldown between spawns
  6. TP at 1.5×ATR, SL at 0.75×ATR (trailing stop)
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
                             get_position_risk, get_balances, set_leverage)
from market_data    import get_price, get_klines, calc_atr, get_last_1m_candle, get_ob
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
        "state_file":   "algo_state_v3_BTCUSDT.json",
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
        "state_file":   "algo_state_v3_ETHUSDT.json",
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
        "state_file":   "algo_state_v3_XAUUSDT.json",
    },
}

SYMBOLS = list(SYMBOL_CONFIG.keys())

BALANCE_UTILIZATION = 0.01
MIN_RISK_USDT       = 1.0
MAX_RISK_USDT       = 2.0
MIN_CONFIDENCE      = 65

ATR_PERIOD   = 14
ATR_INTERVAL = "5m"
TP_ATR_MULT  = 1.5
SL_ATR_MULT  = 0.75

CONF_MULTIPLIERS = {
    (0,  60): 1.6,
    (60, 75): 1.8,
    (75, 90): 2.0,
    (90, 101): 2.2,
}

# Flash scalp settings
FLASH_DIP_MIN_PCT  = 0.10   # minimum % drop to count as flash dip
FLASH_DIP_MAX_PCT  = 0.25   # above this = trend, not flash → skip
FLASH_WINDOW_SEC   = 15     # price must drop this fast to be a flash
BABY_TP_PCT        = 0.0015 # 0.15% snap-back TP per baby
BABY_FILL_TIMEOUT  = 20     # seconds to wait for baby fill, else cancel
BABY_MAX_ACTIVE    = 2      # max concurrent baby orders
BABY_COOLDOWN_SEC  = 30     # seconds between spawning babies
FLASH_POLL_SEC     = 1.5    # how often flash monitor checks price

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
    try:
        bal_data  = get_balances()
        balances  = bal_data if isinstance(bal_data, list) else bal_data.get("data", [])
        available = 0.0
        for b in balances:
            if b.get("asset", "").upper() in ("USDT", "BUSD"):
                av = (b.get("availableBalance") or b.get("walletBalance")
                      or b.get("balance") or "0")
                available = max(available, float(av))
        if available <= 0:
            return MIN_RISK_USDT
        raw = available * BALANCE_UTILIZATION
        clamped = max(MIN_RISK_USDT, min(MAX_RISK_USDT, raw))
        print(f"  [{symbol}][sizing] Balance=${available:.2f}  → RISK=${clamped:.4f}")
        return round(clamped, 4)
    except Exception as e:
        print(f"  [{symbol}][sizing] failed ({e}) → ${MIN_RISK_USDT}")
        return MIN_RISK_USDT


def load_heatmap(symbol: str) -> dict:
    path = heatmap_json(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Heatmap not found: {path}")
    with open(path) as f:
        data = json.load(f)
    age = (datetime.now() - datetime.fromisoformat(data["generated_at"])).total_seconds() / 60
    if age > 60:
        print(f"  [{symbol}][scanner] heatmap is {age:.0f} min old")
    return data


def load_whale_candle(symbol: str) -> dict | None:
    try:
        with open(whale_json(symbol)) as f:
            data = json.load(f)
        return data if time.time() - data["candle_ts"] < 180 else None
    except Exception:
        return None


def scan_zones(current_price: float, heatmap: dict, cfg: dict) -> dict:
    min_z  = cfg["min_zone_usd"]
    shorts = [z for z in heatmap["short_liquidations"] if z["usd"] >= min_z]
    longs  = [z for z in heatmap["long_liquidations"]  if z["usd"] >= min_z]

    above = [z for z in shorts if z["price"] > current_price]
    below = [z for z in longs  if z["price"] <= current_price]
    nearest_above = min(above, key=lambda z: z["price"]) if above else None
    nearest_below = max(below, key=lambda z: z["price"]) if below else None

    result = {
        "price":            current_price,
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
    threshold   = max(2.0 * nearest_zone["usd"], cfg["q_min_usd"])
    candidates  = sorted(
        [z for z in zones if (z["price"] > nearest_zone["price"] if direction == "above"
                               else z["price"] < nearest_zone["price"])],
        key=lambda z: z["price"] if direction == "above" else -z["price"]
    )
    q_zones, running = [nearest_zone], nearest_zone["usd"]
    for z in candidates:
        if z["usd"] < cfg["min_zone_usd"]:
            continue
        q_zones.append(z)
        running += z["usd"]
        if running >= threshold:
            break
    total = sum(z["usd"] for z in q_zones)
    q_avg = sum(z["price"] * z["usd"] for z in q_zones) / total
    return q_zones, q_avg, total


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
    qty = round(risk_usdt * cfg["leverage"] / price, cfg["qty_round"])
    return max(qty, cfg["min_qty"])


# ==============================================================================
# FLASH SCALP LAYER
# Replaces the static BabyLayer. Only fires on sudden OB-imbalance flash events.
# ==============================================================================

class FlashScalpLayer:
    """
    Monitors price every FLASH_POLL_SEC seconds after the parent trade fills.
    Detects sudden flash dips (LONG) or flash spikes (SHORT) caused by:
      - Liquidation cascades (someone's liq order hits the book)
      - Order book imbalance (large market order clears thin liquidity)

    Qualifying flash event:
      - Price moves FLASH_DIP_MIN_PCT–FLASH_DIP_MAX_PCT in one direction
      - Within FLASH_WINDOW_SEC seconds
      - Beyond this range = real trend, skip it

    On detection: place 1 baby at best bid/ask, TP = BABY_TP_PCT snap-back.
    Auto-cancel if not filled in BABY_FILL_TIMEOUT seconds.
    Max BABY_MAX_ACTIVE concurrent, BABY_COOLDOWN_SEC between spawns.
    """

    def __init__(self, symbol: str, direction: str, entry_price: float,
                 sl_price: float, baby_qty: float, pos_side: str, price_round: int):
        self.symbol       = symbol
        self.direction    = direction.upper()   # "LONG" or "SHORT"
        self.entry_price  = entry_price
        self.sl_price     = sl_price
        self.baby_qty     = baby_qty
        self.pos_side     = pos_side.upper()
        self.price_round  = price_round
        self.alive        = True

        self._active      = []          # list of active baby SmartLimitOrders
        self._last_spawn  = 0.0         # timestamp of last spawn
        self._price_hist  = []          # [(timestamp, price), ...]
        self._task        = None

    async def start(self):
        self._task = asyncio.create_task(self._monitor())
        print(f"  [{self.symbol}][flash] Flash scalp monitor started"
              f"  ({self.direction}  entry={self.entry_price:,.{self.price_round}f})")

    async def _monitor(self):
        try:
            while self.alive:
                await asyncio.sleep(FLASH_POLL_SEC)
                if not self.alive:
                    break

                try:
                    price = await asyncio.to_thread(get_price, self.symbol)
                except Exception:
                    continue

                now = time.time()
                self._price_hist.append((now, price))
                # keep only last FLASH_WINDOW_SEC seconds
                self._price_hist = [(t, p) for t, p in self._price_hist
                                    if now - t <= FLASH_WINDOW_SEC]

                # Clean up filled/cancelled babies
                self._active = [b for b in self._active
                                 if not b.is_filled and not b.is_cancelled]

                # Don't spawn if max babies active or cooldown not elapsed
                if len(self._active) >= BABY_MAX_ACTIVE:
                    continue
                if now - self._last_spawn < BABY_COOLDOWN_SEC:
                    continue

                # Detect flash event
                if len(self._price_hist) < 2:
                    continue

                recent_high = max(p for _, p in self._price_hist)
                recent_low  = min(p for _, p in self._price_hist)

                if self.direction == "LONG":
                    # Flash dip: price dropped quickly from recent high
                    drop_pct = (recent_high - price) / recent_high * 100
                    if FLASH_DIP_MIN_PCT <= drop_pct <= FLASH_DIP_MAX_PCT:
                        # Confirm: recent high was close to entry (not already a bad trade)
                        if price > self.sl_price * 1.002:  # still safely above SL
                            await self._spawn_baby(price, "BUY", drop_pct)

                else:  # SHORT
                    # Flash spike: price rose quickly from recent low
                    spike_pct = (price - recent_low) / recent_low * 100
                    if FLASH_DIP_MIN_PCT <= spike_pct <= FLASH_DIP_MAX_PCT:
                        if price < self.sl_price * 0.998:  # still safely below SL
                            await self._spawn_baby(price, "SELL", spike_pct)

        except asyncio.CancelledError:
            pass

    async def _spawn_baby(self, price: float, side: str, move_pct: float):
        pr = self.price_round
        try:
            ob   = await asyncio.to_thread(get_ob, self.symbol)
            best = round(ob["best_bid"] if side == "BUY" else ob["best_ask"], pr)
        except Exception:
            best = round(price, pr)

        # TP = snap back to near entry price
        if side == "BUY":
            tp = round(best * (1 + BABY_TP_PCT), pr)
        else:
            tp = round(best * (1 - BABY_TP_PCT), pr)

        print(f"  [{self.symbol}][flash] FLASH detected! {self.direction}"
              f"  move={move_pct:.2f}%  spawning baby @ {best:,.{pr}f}"
              f"  TP={tp:,.{pr}f}")

        baby = SmartLimitOrder(
            symbol=self.symbol, side=side, qty=self.baby_qty,
            pos_side=self.pos_side, label=f"flash_baby",
            track_ob=True, price_round=pr,
        )
        self._active.append(baby)
        self._last_spawn = time.time()
        await baby.place()

        # Wait for fill with timeout — if no fill, cancel (not a flash, it's a trend)
        asyncio.create_task(self._manage_baby(baby, tp))

    async def _manage_baby(self, baby: SmartLimitOrder, tp_price: float):
        pr = self.price_round
        filled = await baby.wait_fill(timeout=BABY_FILL_TIMEOUT)
        if not self.alive:
            baby.cancel()
            return
        if not filled:
            baby.cancel()
            print(f"  [{self.symbol}][flash] Baby not filled in {BABY_FILL_TIMEOUT}s"
                  f" — cancelled (not a flash, trending)")
            return

        fill_px = baby.fill_price
        print(f"  [{self.symbol}][flash] Baby filled @ {fill_px:,.{pr}f}"
              f"  TP={tp_price:,.{pr}f}")

        # Place TP at snap-back level (resting limit, no OB tracking)
        close_side = "SELL" if baby.side == "BUY" else "BUY"
        tp_order = SmartLimitOrder(
            symbol=self.symbol, side=close_side, qty=self.baby_qty,
            pos_side=self.pos_side, label="flash_tp",
            track_ob=False, fixed_price=tp_price, price_round=pr,
        )
        await tp_order.place()

        # Wait for TP (reasonable time — if parent closes first, kill_all cancels this)
        tp_filled = await tp_order.wait_fill(timeout=300)
        if tp_filled:
            profit = abs(tp_price - fill_px) * self.baby_qty
            print(f"  [{self.symbol}][flash] Baby TP hit  profit≈${profit:.4f}")
        else:
            tp_order.cancel()

    def kill_all(self):
        self.alive = False
        if self._task and not self._task.done():
            self._task.cancel()
        for b in self._active:
            if not b.is_filled and not b.is_cancelled:
                b.cancel()
        self._active.clear()
        print(f"  [{self.symbol}][flash] Flash layer stopped")


# ==============================================================================
# TRADING ENGINE
# ==============================================================================

class TradingEngine:

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cfg    = SYMBOL_CONFIG[symbol]

        self.state            = "WATCHING"
        self.direction        = None
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

        self.trade1      = None
        self.trade2      = None
        self.tp_order    = None
        self.flash_layer = None

        self.trade_count = 0
        self.wins        = 0
        self.losses      = 0
        self.total_pnl   = 0.0

    def _tag(self, msg):
        print(f"  [{self.symbol}] {msg}")

    # ── Loops ──────────────────────────────────────────────────────────────────

    async def minute_loop(self):
        self._tag("v3 engine started. State: WATCHING")
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
                self._tag(f"ERROR: {e}")
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
        except FileNotFoundError as e:
            self._tag(f"[scanner] {e}"); return

        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(scan_zones, price, self._heatmap, self.cfg)

        if scan["nearest_above"]:
            z    = scan["nearest_above"]
            flag = " ← APPROACHING" if scan["approaching_up"] else ""
            self._tag(f"[scan] Above: ${z['price']:>10,.{self.cfg['price_round']}f}"
                      f"  ${z['usd']/1e6:.1f}M  ({scan['dist_above_pct']:.2f}% away){flag}")
        if scan["nearest_below"]:
            z    = scan["nearest_below"]
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
            self.state     = "APPROACHING"
            self._tag(f"[engine] → APPROACHING  direction={direction}")
            await self._tick_approaching()

    async def _tick_approaching(self):
        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(scan_zones, price, self._heatmap, self.cfg)

        if self.direction == "SHORT" and not scan["approaching_up"]:
            self._tag("[engine] Zone passed → WATCHING"); self.state = "WATCHING"; return
        if self.direction == "LONG"  and not scan["approaching_down"]:
            self._tag("[engine] Zone passed → WATCHING"); self.state = "WATCHING"; return

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

        whale           = await asyncio.to_thread(load_whale_candle, self.symbol)
        conf            = calc_confidence(scan, q_zones, q_total, whale, self.direction.lower())
        self.confidence = conf

        whale_avg    = 0.0
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
        pr          = self.cfg["price_round"]

        if last_candle["ts"] == self._last_trigger_ts:
            self._tag("[engine] Same candle — skipping"); self.state = "WATCHING"; return

        if conf < MIN_CONFIDENCE:
            self._tag(f"[engine] conf={conf} < {MIN_CONFIDENCE} — skipping")
            self.state = "WATCHING"; return

        # ── Trigger ───────────────────────────────────────────────────────────
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
        baby_qty   = self.cfg["min_qty"]  # always minimum lot for flash babies

        self._tag(f"[trade] Entry={current:,.{pr}f}  TP={self.tp_price:,.{pr}f}"
                  f"  SL={self.sl_price:,.{pr}f}  ATR={atr:.{pr}f}")
        self._tag(f"[trade] T1={base_qty}  T2={trade2_qty} @ {trade2_price:,.{pr}f}"
                  f"  baby_qty={baby_qty}")

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
            track_ob=False, fixed_price=trade2_price, price_round=pr,
        )
        await self.trade2.place()

        self.flash_layer = FlashScalpLayer(
            symbol=self.symbol, direction=self.direction,
            entry_price=current, sl_price=self.sl_price,
            baby_qty=baby_qty, pos_side=pos_side,
            price_round=pr,
        )

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

        pr = self.cfg["price_round"]
        if self.best_price_seen is None:
            self.best_price_seen = price
        if self.direction == "SHORT":
            self.best_price_seen = min(self.best_price_seen, price)
        else:
            self.best_price_seen = max(self.best_price_seen, price)

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
                self._tag(f"[trail] {'TRAILING started' if prev != 'trailing' else 'TRAIL updated'}"
                          f"  SL → ${new_sl:,.{pr}f}")
        elif profit_pts >= 0.5 * self.atr and self.trail_stage == "none":
            be = round(self.entry_price + (5.0 if self.direction == "SHORT" else -5.0), pr)
            if ((self.direction == "SHORT" and be < self.sl_price) or
                    (self.direction == "LONG"  and be > self.sl_price)):
                new_sl = be
                self.trail_stage = "breakeven"
                self._tag(f"[trail] BREAKEVEN  SL → ${new_sl:,.{pr}f}")

        if new_sl:
            self.sl_price = new_sl
            # Also update flash layer's SL reference
            if self.flash_layer:
                self.flash_layer.sl_price = new_sl
            self._write_state()

    async def _monitor_position(self):
        price = await asyncio.to_thread(get_price, self.symbol)

        # Start flash monitor once Trade1 fills
        if (self.trade1 and self.trade1.is_filled
                and self.flash_layer and self.flash_layer._task is None):
            await self.flash_layer.start()

        # External close detection
        if self.trade1 and self.trade1.is_filled:
            try:
                pos_data  = await asyncio.to_thread(get_position_risk, self.symbol)
                positions = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
                exists    = any(p.get("positionSide") == self.direction
                                and abs(float(p.get("positionAmt", 0))) > 0
                                for p in positions)
                if not exists:
                    self._tag("[monitor] Position closed externally — resetting")
                    t1q = self.trade1.qty if self.trade1 else 0.0
                    t2q = self.trade2.qty if (self.trade2 and self.trade2.is_filled) else 0.0
                    qty = round(t1q + t2q, 3)
                    if qty > 0 and self.entry_price > 0:
                        pnl    = ((self.entry_price - price) if self.direction == "SHORT"
                                  else (price - self.entry_price)) * qty
                        result = "WIN" if pnl >= 0 else "LOSS"
                        if pnl >= 0: self.wins += 1
                        else:        self.losses += 1
                        self.total_pnl += pnl
                        self._log_trade(result, pnl, price, qty)
                    self._reset(); return
            except Exception:
                pass

        # Check Trade1 fill
        if self.trade1 and not self.trade1.is_filled:
            await self.trade1._check_fill()
            if self.trade1.is_filled:
                self.entry_price = self.trade1.fill_price or self.entry_price
                # Place TP order after fill
                close_side    = "BUY" if self.direction == "SHORT" else "SELL"
                pr            = self.cfg["price_round"]
                self.tp_order = SmartLimitOrder(
                    symbol=self.symbol, side=close_side,
                    qty=self.trade1.qty, pos_side=self.direction,
                    label="Trade1_TP", track_ob=False,
                    fixed_price=self.tp_price, price_round=pr,
                )
                await self.tp_order.place()

        await self._update_trail(price)

        if not (self.trade1 and self.trade1.is_filled):
            return

        pr = self.cfg["price_round"]
        if self.direction == "SHORT" and price <= self.tp_price:
            self._tag(f"[monitor] TP reached @ {price:,.{pr}f}")
            await self._on_tp_hit(); return
        if self.direction == "LONG"  and price >= self.tp_price:
            self._tag(f"[monitor] TP reached @ {price:,.{pr}f}")
            await self._on_tp_hit(); return
        if self.direction == "SHORT" and price >= self.sl_price:
            self._tag(f"[monitor] SL hit @ {price:,.{pr}f}")
            await self._on_sl_hit(); return
        if self.direction == "LONG"  and price <= self.sl_price:
            self._tag(f"[monitor] SL hit @ {price:,.{pr}f}")
            await self._on_sl_hit(); return

    async def _on_tp_hit(self):
        close_price, close_qty = await self._emergency_close("TP")
        approx = close_price if close_price > 0 else self.tp_price
        qty    = close_qty   if close_qty   > 0 else self.trade1.qty
        profit = max(((self.entry_price - approx) if self.direction == "SHORT"
                      else (approx - self.entry_price)) * qty, 0.0)
        self.wins      += 1
        self.total_pnl += profit
        self._log_trade("WIN", profit, approx, qty)
        self._reset()
        self._tag(f"[engine] WIN  +${profit:.4f}  total=${self.total_pnl:.4f}")

    async def _on_sl_hit(self):
        close_price, close_qty = await self._emergency_close("SL")
        approx = close_price if close_price > 0 else self.sl_price
        qty    = close_qty   if close_qty   > 0 else self.trade1.qty
        loss   = max(((approx - self.entry_price) if self.direction == "SHORT"
                      else (self.entry_price - approx)) * qty, 0.0)
        self.losses    += 1
        self.total_pnl -= loss
        self._log_trade("LOSS", -loss, approx, qty)
        self._reset()
        self._tag(f"[engine] LOSS  -${loss:.4f}  total=${self.total_pnl:.4f}")

    async def _emergency_close(self, reason: str) -> tuple:
        if self.flash_layer:
            self.flash_layer.kill_all()
        for o in [self.trade1, self.trade2, self.tp_order]:
            if o and not o.is_filled and not o.is_cancelled:
                o.cancel()
        try:
            await asyncio.to_thread(cancel_all_orders, self.symbol)
        except Exception:
            pass

        actual_price = 0.0
        actual_qty   = 0.0
        pr           = self.cfg["price_round"]
        try:
            pos_data   = await asyncio.to_thread(get_position_risk, self.symbol)
            positions  = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
            close_side = "BUY" if self.direction == "SHORT" else "SELL"
            for pos in positions:
                if pos.get("positionSide") == self.direction:
                    qty = abs(float(pos.get("positionAmt", 0)))
                    if qty > 0:
                        close_order = SmartLimitOrder(
                            symbol=self.symbol, side=close_side, qty=qty,
                            pos_side=self.direction, label=f"close_{reason}",
                            track_ob=True, price_round=pr,
                        )
                        await close_order.place()
                        await close_order.wait_fill(timeout=120)
                        actual_price = close_order.fill_price or 0.0
                        actual_qty   = qty
                    break
        except Exception as e:
            self._tag(f"[close] error: {e}")

        return actual_price, actual_qty

    def _reset(self):
        if self.flash_layer:
            self.flash_layer.kill_all()
        self.trade1          = None
        self.trade2          = None
        self.tp_order        = None
        self.flash_layer     = None
        self.trail_stage     = "none"
        self.best_price_seen = None
        self.state           = "WATCHING"
        self._write_state()

    async def _print_status(self):
        try:
            price = await asyncio.to_thread(get_price, self.symbol)
            pr    = self.cfg["price_round"]
            self._tag(f"[status] {self.direction}  price={price:,.{pr}f}"
                      f"  tp={self.tp_price:,.{pr}f} ({abs(price-self.tp_price):.{pr}f} away)"
                      f"  sl={self.sl_price:,.{pr}f} ({abs(price-self.sl_price):.{pr}f} away)"
                      f"  trail={self.trail_stage}")
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
                    "mode":        "LIQ_v3",
                    "updated_at":  datetime.now().isoformat(),
                }, f, indent=2)
        except Exception:
            pass

    def _log_trade(self, result: str, pnl: float,
                   close_price: float = 0.0, close_qty: float = 0.0):
        self.trade_count += 1
        pr    = self.cfg["price_round"]
        entry = {
            "id":          self.trade_count,
            "time":        datetime.now().isoformat(),
            "symbol":      self.symbol,
            "direction":   self.direction,
            "entry":       round(self.entry_price, pr),
            "actual_close": round(close_price, pr) if close_price else None,
            "actual_qty":  close_qty or None,
            "tp":          self.tp_price,
            "sl":          self.sl_price,
            "atr":         self.atr,
            "confidence":  self.confidence,
            "risk_usdt":   self.risk_usdt,
            "q_zones":     len(self.q_zones),
            "q_total_usd": self.q_total_usd,
            "mode":        "LIQ_v3",
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
        first = False
        print(f"  [{symbol}][heatmap] {label} ...")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(HEATMAP_SCRIPT), "--symbol", symbol,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            for line in stdout.decode("utf-8", errors="replace").splitlines():
                if any(kw in line for kw in ["Current Price", "Data saved",
                                              "Total modeled", "Error", "Traceback"]):
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
                print(f"  [startup] Balance: {w:.4f} {b.get('asset','?')}")
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
    print("  LIQ ALGO v3  --  Multi-Asset Liquidation + Flash Scalp")
    print(f"  Symbols    : {', '.join(SYMBOLS)}")
    print(f"  Entry      : close-back-inside liq zone + SmartLimit")
    print(f"  Flash baby : {FLASH_DIP_MIN_PCT}–{FLASH_DIP_MAX_PCT}% in <{FLASH_WINDOW_SEC}s")
    print(f"  Baby TP    : {BABY_TP_PCT*100:.2f}% snap-back")
    print(f"  TP / SL    : {TP_ATR_MULT}x / {SL_ATR_MULT}x ATR")
    print("=" * 60)

    await startup()

    engines = [TradingEngine(sym) for sym in SYMBOLS]
    tasks   = []
    for i, (eng, sym) in enumerate(zip(engines, SYMBOLS)):
        tasks.append(asyncio.create_task(_delayed_heatmap(sym, delay=i * 30)))
        tasks.append(asyncio.create_task(eng.minute_loop()))
        tasks.append(asyncio.create_task(eng.position_monitor_loop()))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  [liq_algo_v3] Stopped. Ctrl+C")
        sys.exit(0)
