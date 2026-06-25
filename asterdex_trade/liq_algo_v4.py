"""
liq_algo_v4.py  —  Multi-Asset Liquidation Zone Reversal Engine  v4
====================================================================
BTC + ETH + XAU in parallel.

FIXES FROM v3:
  1. Double-fill bug: entry fill confirmed via POSITION SIZE on exchange
     (not order history — eliminates AsterDEX async reporting delay).
     GTC limit replaces GTX post-only so orders never EXPIRE unexpectedly.
  2. Regime filter: skip LONG if last 3 hourly closes descending.
     Skip SHORT if last 3 hourly closes ascending. No counter-trend trades.
  3. Zone cooldown: 600s after LOSS, 180s after WIN. No immediate re-entry
     on same zone after a loss.
  4. T2 only after breakeven: Trade2 resting limit at Q_avg is placed ONLY
     after trail_stage reaches "breakeven" (SL moved to entry). Prevents
     building size into a losing position.
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
        "state_file":   "algo_state_v4_BTCUSDT.json",
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
        "state_file":   "algo_state_v4_ETHUSDT.json",
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
        "state_file":   "algo_state_v4_XAUUSDT.json",
    },
}

SYMBOLS = list(SYMBOL_CONFIG.keys())

BALANCE_UTILIZATION = 0.01
MIN_RISK_USDT       = 1.0
MAX_RISK_USDT       = 2.0
MIN_CONFIDENCE      = {
    "BTCUSDT": 65,   # large OI — require whale/zone confirmation
    "ETHUSDT": 50,   # zones rarely exceed $5M — same floor as XAU
    "XAUUSDT": 50,   # tiny OI — zones physically can't be bigger
}

TOUCH_BUFFER_PCT    = 0.003   # 0.3% — liquidation cascades start slightly before zone

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

# Cooldowns (Fix #3)
COOLDOWN_LOSS = 600
COOLDOWN_WIN  = 180

# Flash scalp (kept from v3)
FLASH_DIP_MIN_PCT = 0.10
FLASH_DIP_MAX_PCT = 0.25
FLASH_WINDOW_SEC  = 15
BABY_TP_PCT       = 0.0015
BABY_FILL_TIMEOUT = 20
BABY_MAX_ACTIVE   = 2
BABY_COOLDOWN_SEC = 30
FLASH_POLL_SEC    = 1.5

# Entry chase
ENTRY_TIMEOUT = 30.0
ENTRY_POLL    = 0.5

ROOT_DIR        = Path(__file__).parent.parent
TRADE_LOG_JSON  = ROOT_DIR / "trade_log.json"
HEATMAP_SCRIPT  = ROOT_DIR / "data_fetching" / "binance_liq_heatmap_2.py"
HEATMAP_REFRESH = 5 * 60


def heatmap_json(sym):
    return ROOT_DIR / "data_fetching" / f"binance_liq_heatmap_{sym}.json"
def whale_json(sym):
    return ROOT_DIR / SYMBOL_CONFIG[sym]["whale_file"]
def state_json(sym):
    return ROOT_DIR / SYMBOL_CONFIG[sym]["state_file"]


# ==============================================================================
# ZONE MATH
# ==============================================================================

def calc_risk_usdt(symbol: str) -> float:
    try:
        bal_data  = get_balances()
        bals      = bal_data if isinstance(bal_data, list) else bal_data.get("data", [])
        available = 0.0
        for b in bals:
            if b.get("asset", "").upper() in ("USDT", "BUSD"):
                av = (b.get("availableBalance") or b.get("walletBalance")
                      or b.get("balance") or "0")
                available = max(available, float(av))
        raw = available * BALANCE_UTILIZATION
        clamped = max(MIN_RISK_USDT, min(MAX_RISK_USDT, raw))
        print(f"  [{symbol}][sizing] Balance=${available:.2f}  → RISK=${clamped:.4f}")
        return round(clamped, 4)
    except Exception as e:
        print(f"  [{symbol}][sizing] failed ({e}) → ${MIN_RISK_USDT}")
        return MIN_RISK_USDT


def load_heatmap(symbol: str) -> dict:
    p = heatmap_json(symbol)
    if not p.exists():
        raise FileNotFoundError(f"Heatmap not found: {p}")
    with open(p) as f:
        data = json.load(f)
    age = (datetime.now() - datetime.fromisoformat(data["generated_at"])).total_seconds() / 60
    if age > 60:
        print(f"  [{symbol}][scanner] heatmap is {age:.0f} min old")
    return data


def load_whale(symbol: str) -> dict | None:
    try:
        with open(whale_json(symbol)) as f:
            d = json.load(f)
        return d if time.time() - d["candle_ts"] < 180 else None
    except Exception:
        return None


def scan_zones(price: float, hm: dict, cfg: dict) -> dict:
    mn = cfg["min_zone_usd"]
    shorts = [z for z in hm["short_liquidations"] if z["usd"] >= mn]
    longs  = [z for z in hm["long_liquidations"]  if z["usd"] >= mn]
    above  = [z for z in shorts if z["price"] > price]
    below  = [z for z in longs  if z["price"] <= price]
    na = min(above, key=lambda z: z["price"]) if above else None
    nb = max(below, key=lambda z: z["price"]) if below else None
    r  = {"price": price, "nearest_above": na, "nearest_below": nb,
          "approaching_up": False, "approaching_down": False,
          "dist_above_pct": None, "dist_below_pct": None}
    app = cfg["approach_pct"]
    if na:
        d = (na["price"] - price) / price * 100
        r["dist_above_pct"] = d; r["approaching_up"] = d <= app
    if nb:
        d = (price - nb["price"]) / price * 100
        r["dist_below_pct"] = d; r["approaching_down"] = d <= app
    return r


def calc_q(zones, nearest, direction, cfg):
    thr = max(2.0 * nearest["usd"], cfg["q_min_usd"])
    cands = sorted(
        [z for z in zones if (z["price"] > nearest["price"] if direction == "above"
                               else z["price"] < nearest["price"])],
        key=lambda z: z["price"] if direction == "above" else -z["price"]
    )
    qz, run = [nearest], nearest["usd"]
    for z in cands:
        if z["usd"] < cfg["min_zone_usd"]: continue
        qz.append(z); run += z["usd"]
        if run >= thr: break
    tot  = sum(z["usd"] for z in qz)
    qavg = sum(z["price"] * z["usd"] for z in qz) / tot
    return qz, qavg, tot


def calc_confidence(scan, qz, qtot, whale, direction):
    s = 50
    near = scan["nearest_above"] if direction == "short" else scan["nearest_below"]
    if near:
        if near["usd"] >= 30_000_000: s += 10
        if near["usd"] >= 60_000_000: s += 5
    if qtot >= 60_000_000: s += 10
    if len(qz) >= 3:       s += 10
    if whale:
        if direction == "short" and whale["whale_sell_usd"] > whale["whale_buy_usd"]:
            s += 15
        elif direction == "long" and whale["whale_buy_usd"] > whale["whale_sell_usd"]:
            s += 15
        if direction == "short" and whale["whale_buy_usd"] > whale["whale_sell_usd"] * 1.5:
            s -= 20
        elif direction == "long" and whale["whale_sell_usd"] > whale["whale_buy_usd"] * 1.5:
            s -= 20
    return max(0, min(100, s))


def conf_mult(score):
    for (lo, hi), m in CONF_MULTIPLIERS.items():
        if lo <= score < hi: return m
    return 1.6


def qty_from_risk(risk, price, cfg):
    q = round(risk * cfg["leverage"] / price, cfg["qty_round"])
    return max(q, cfg["min_qty"])


# ==============================================================================
# FLASH SCALP LAYER (from v3, unchanged)
# ==============================================================================

class FlashScalpLayer:

    def __init__(self, symbol, direction, entry_price, sl_price,
                 baby_qty, pos_side, price_round):
        self.symbol      = symbol
        self.direction   = direction.upper()
        self.entry_price = entry_price
        self.sl_price    = sl_price
        self.baby_qty    = baby_qty
        self.pos_side    = pos_side.upper()
        self.pr          = price_round
        self.alive       = True
        self._active     = []
        self._last_spawn = 0.0
        self._hist       = []
        self._task       = None

    async def start(self):
        self._task = asyncio.create_task(self._monitor())
        print(f"  [{self.symbol}][flash] Monitor started"
              f"  entry={self.entry_price:,.{self.pr}f}")

    async def _monitor(self):
        try:
            while self.alive:
                await asyncio.sleep(FLASH_POLL_SEC)
                if not self.alive: break
                try:
                    price = await asyncio.to_thread(get_price, self.symbol)
                except Exception:
                    continue
                now = time.time()
                self._hist.append((now, price))
                self._hist = [(t, p) for t, p in self._hist
                              if now - t <= FLASH_WINDOW_SEC]
                self._active = [b for b in self._active
                                if not b.is_filled and not b.is_cancelled]
                if len(self._active) >= BABY_MAX_ACTIVE: continue
                if now - self._last_spawn < BABY_COOLDOWN_SEC: continue
                if len(self._hist) < 2: continue
                hi = max(p for _, p in self._hist)
                lo = min(p for _, p in self._hist)
                if self.direction == "LONG":
                    drop = (hi - price) / hi * 100
                    if FLASH_DIP_MIN_PCT <= drop <= FLASH_DIP_MAX_PCT:
                        if price > self.sl_price * 1.002:
                            await self._spawn("BUY", price, drop)
                else:
                    spike = (price - lo) / lo * 100
                    if FLASH_DIP_MIN_PCT <= spike <= FLASH_DIP_MAX_PCT:
                        if price < self.sl_price * 0.998:
                            await self._spawn("SELL", price, spike)
        except asyncio.CancelledError:
            pass

    async def _spawn(self, side, price, move_pct):
        try:
            ob   = await asyncio.to_thread(get_ob, self.symbol)
            best = round(ob["best_bid"] if side=="BUY" else ob["best_ask"], self.pr)
        except Exception:
            best = round(price, self.pr)
        tp = round(best * (1 + BABY_TP_PCT) if side=="BUY"
                   else best * (1 - BABY_TP_PCT), self.pr)
        print(f"  [{self.symbol}][flash] Flash {self.direction} {move_pct:.2f}%"
              f"  baby @ {best:,.{self.pr}f}  TP={tp:,.{self.pr}f}")
        baby = SmartLimitOrder(
            symbol=self.symbol, side=side, qty=self.baby_qty,
            pos_side=self.pos_side, label="flash_baby",
            track_ob=True, price_round=self.pr,
        )
        self._active.append(baby)
        self._last_spawn = time.time()
        await baby.place()
        asyncio.create_task(self._manage(baby, tp))

    async def _manage(self, baby, tp):
        filled = await baby.wait_fill(timeout=BABY_FILL_TIMEOUT)
        if not self.alive: baby.cancel(); return
        if not filled:
            baby.cancel()
            print(f"  [{self.symbol}][flash] Baby not filled in {BABY_FILL_TIMEOUT}s — cancelled")
            return
        fill_px = baby.fill_price
        close   = "SELL" if baby.side == "BUY" else "BUY"
        tp_ord  = SmartLimitOrder(
            symbol=self.symbol, side=close, qty=self.baby_qty,
            pos_side=self.pos_side, label="flash_tp",
            track_ob=False, fixed_price=tp, price_round=self.pr,
        )
        await tp_ord.place()
        ok = await tp_ord.wait_fill(timeout=300)
        if ok:
            profit = abs(tp - fill_px) * self.baby_qty
            print(f"  [{self.symbol}][flash] Baby TP  profit≈${profit:.4f}")
        else:
            tp_ord.cancel()

    def kill_all(self):
        self.alive = False
        if self._task and not self._task.done():
            self._task.cancel()
        for b in self._active:
            if not b.is_filled and not b.is_cancelled:
                b.cancel()
        self._active.clear()


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
        self.t1_qty           = 0.0
        self.atr              = 0.0
        self.tp_price         = 0.0
        self.sl_price         = 0.0
        self.trail_stage      = "none"
        self.best_seen        = None
        self.risk_usdt        = MIN_RISK_USDT
        self._last_trigger_ts = 0
        self._heatmap         = None
        self._zone_cooldowns  = {}          # Fix #3: zone cooldowns
        self._t2_placed       = False       # Fix #4: T2 after breakeven
        self._t2_order        = None
        self._t2_price        = 0.0
        self._t2_qty          = 0.0
        self._t2_pos_side     = ""

        self.tp_order    = None
        self.flash_layer = None

        self.trade_count = 0
        self.wins        = 0
        self.losses      = 0
        self.total_pnl   = 0.0

    def _tag(self, msg):
        print(f"  [{self.symbol}] {msg}")

    # ── Position helpers ───────────────────────────────────────────────────────

    async def _get_position_qty(self) -> float:
        """Fix #1: Read actual position size from exchange."""
        try:
            pd = await asyncio.to_thread(get_position_risk, self.symbol)
            ps = pd if isinstance(pd, list) else pd.get("data", [])
            for p in ps:
                if p.get("positionSide") == self.direction:
                    return abs(float(p.get("positionAmt", 0)))
        except Exception:
            pass
        return 0.0

    async def _get_entry_price_from_exchange(self) -> float:
        try:
            pd = await asyncio.to_thread(get_position_risk, self.symbol)
            ps = pd if isinstance(pd, list) else pd.get("data", [])
            for p in ps:
                if p.get("positionSide") == self.direction:
                    return float(p.get("entryPrice", 0))
        except Exception:
            pass
        return self.entry_price

    # ── Fix #1: Chase entry using position size for fill confirmation ──────────

    async def _chase_entry(self, side: str, qty: float, pos_side: str) -> tuple:
        """
        Place GTC limit at best bid/ask. Chase if price moves ≥1 tick.
        Confirm fill by checking POSITION SIZE on exchange — not order history.
        This eliminates the double-fill bug from AsterDEX async reporting.
        Returns (fill_price, filled_qty) or (0.0, 0.0) on timeout.
        """
        pr       = self.cfg["price_round"]
        tick     = 10 ** (-pr)
        deadline = time.time() + ENTRY_TIMEOUT
        order_id = None
        order_px = None
        pre_qty  = await self._get_position_qty()

        while time.time() < deadline:
            ob   = await asyncio.to_thread(get_ob, self.symbol)
            best = round(ob["best_bid"] if side == "BUY" else ob["best_ask"], pr)

            need = (order_id is None) or (abs(best - order_px) >= tick)

            if need:
                if order_id:
                    try:
                        await asyncio.to_thread(cancel_order, self.symbol, order_id)
                    except Exception:
                        pass
                    order_id = None
                    await asyncio.sleep(0.3)

                resp     = await asyncio.to_thread(
                    place_order, self.symbol, side, "LIMIT", qty, best, "GTC", pos_side
                )
                order_id = resp.get("orderId")
                order_px = best
                self._tag(f"[entry] {side} {qty} @ {best:,.{pr}f}  id={order_id}")

                if not order_id:
                    self._tag(f"[entry] place rejected: {resp}")
                    await asyncio.sleep(1.0)
                    continue

            await asyncio.sleep(ENTRY_POLL)

            # Check fill via position size delta
            cur_qty = await self._get_position_qty()
            filled  = round(cur_qty - pre_qty, self.cfg["qty_round"])

            if filled >= qty * 0.99:
                if order_id:
                    try:
                        await asyncio.to_thread(cancel_order, self.symbol, order_id)
                    except Exception:
                        pass
                fill_px = await self._get_entry_price_from_exchange()
                self._tag(f"[entry] FILLED  qty={filled}  price≈{fill_px:,.{pr}f}")
                return fill_px, filled

        if order_id:
            try:
                await asyncio.to_thread(cancel_order, self.symbol, order_id)
            except Exception:
                pass
        self._tag(f"[entry] timeout after {ENTRY_TIMEOUT:.0f}s — no fill")
        return 0.0, 0.0

    # ── Fix #2: Regime filter ──────────────────────────────────────────────────

    async def _trending_against(self, direction: str) -> bool:
        """
        Returns True if market hourly trend opposes the trade direction.
        Skip LONG in downtrend (3 descending closes).
        Skip SHORT in uptrend (3 ascending closes).
        """
        try:
            klines = await asyncio.to_thread(get_klines, self.symbol, "1h", 5)
            if len(klines) < 4:
                return False
            c1 = float(klines[-4][4])
            c2 = float(klines[-3][4])
            c3 = float(klines[-2][4])
            if direction == "LONG"  and c3 < c2 < c1:
                self._tag(f"[regime] Downtrend ({c1:.2f}>{c2:.2f}>{c3:.2f}) — skip LONG")
                return True
            if direction == "SHORT" and c3 > c2 > c1:
                self._tag(f"[regime] Uptrend ({c1:.2f}<{c2:.2f}<{c3:.2f}) — skip SHORT")
                return True
        except Exception:
            pass
        return False

    # ── Fix #3: Zone cooldown ──────────────────────────────────────────────────

    def _set_cooldown(self, zone_price: float, result: str):
        cd = COOLDOWN_LOSS if result == "LOSS" else COOLDOWN_WIN
        self._zone_cooldowns[zone_price] = time.time() + cd
        self._tag(f"[cooldown] Zone ${zone_price:,.0f}  {cd}s  ({result})")

    # ── Loops ──────────────────────────────────────────────────────────────────

    async def minute_loop(self):
        self._tag("v4 engine started. State: WATCHING")
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

        for label, zone, flag_key in [
            ("Above", scan["nearest_above"], "approaching_up"),
            ("Below", scan["nearest_below"], "approaching_down"),
        ]:
            if zone:
                dist = scan["dist_above_pct"] if label=="Above" else scan["dist_below_pct"]
                flag = " ← APPROACHING" if scan[flag_key] else ""
                self._tag(f"[scan] {label}: ${zone['price']:>10,.{self.cfg['price_round']}f}"
                          f"  ${zone['usd']/1e6:.1f}M  ({dist:.2f}% away){flag}")

        if scan["approaching_up"] or scan["approaching_down"]:
            if scan["approaching_up"] and scan["approaching_down"]:
                direction = "SHORT" if (scan["dist_above_pct"] or 999) < (scan["dist_below_pct"] or 999) else "LONG"
            elif scan["approaching_up"]:
                direction = "SHORT"
            else:
                direction = "LONG"
            self.direction = direction
            self.state     = "APPROACHING"
            self._tag(f"[engine] → APPROACHING  dir={direction}")
            await self._tick_approaching()

    async def _tick_approaching(self):
        price = await asyncio.to_thread(get_price, self.symbol)
        scan  = await asyncio.to_thread(scan_zones, price, self._heatmap, self.cfg)

        if self.direction == "SHORT" and not scan["approaching_up"]:
            self._tag("[engine] Zone passed → WATCHING"); self.state = "WATCHING"; return
        if self.direction == "LONG"  and not scan["approaching_down"]:
            self._tag("[engine] Zone passed → WATCHING"); self.state = "WATCHING"; return

        nearest = scan["nearest_above"] if self.direction == "SHORT" else scan["nearest_below"]
        dir_str = "above" if self.direction == "SHORT" else "below"
        zones   = [z for z in (self._heatmap["short_liquidations"] if self.direction=="SHORT"
                                else self._heatmap["long_liquidations"])
                   if z["usd"] >= self.cfg["min_zone_usd"]]

        qz, qavg, qtot = await asyncio.to_thread(calc_q, zones, nearest, dir_str, self.cfg)
        self.q_zones = qz; self.q_avg = qavg; self.q_total_usd = qtot

        whale           = await asyncio.to_thread(load_whale, self.symbol)
        conf            = calc_confidence(scan, qz, qtot, whale, self.direction.lower())
        self.confidence = conf
        mult            = conf_mult(conf)

        whale_avg    = 0.0
        if whale:
            if self.direction == "SHORT" and whale["whale_sell_usd"] > 0:
                whale_avg = whale["whale_sell_avg_price"]
            elif self.direction == "LONG" and whale["whale_buy_usd"] > 0:
                whale_avg = whale["whale_buy_avg_price"]
        t2_price = (qavg + whale_avg) / 2 if whale_avg > 0 else qavg

        self._tag(f"[engine] Q=${qtot/1e6:.1f}M  {len(qz)} zones  conf={conf}  mult={mult:.1f}x")

        last_candle = await asyncio.to_thread(get_last_1m_candle, self.symbol)
        zone_price  = nearest["price"]
        pr          = self.cfg["price_round"]

        if last_candle["ts"] == self._last_trigger_ts:
            self._tag("[engine] Same candle — skip"); self.state = "WATCHING"; return

        min_conf = MIN_CONFIDENCE[self.symbol]
        if conf < min_conf:
            self._tag(f"[engine] conf={conf} < {min_conf} — skip")
            self.state = "WATCHING"; return

        # Fix #2: Regime filter
        if await self._trending_against(self.direction):
            self.state = "WATCHING"; return

        # Fix #3: Zone cooldown check
        cool = self._zone_cooldowns.get(zone_price, 0)
        if time.time() < cool:
            self._tag(f"[cooldown] {cool - time.time():.0f}s remaining — skip")
            self.state = "WATCHING"; return

        # Trigger — allow 0.5% buffer since cascades begin slightly before zone
        if self.direction == "SHORT":
            touched  = last_candle["high"] >= zone_price * (1 - TOUCH_BUFFER_PCT)
            triggered = touched and last_candle["close"] < zone_price
            self._tag(f"[trigger?] H={last_candle['high']:,.{pr}f}"
                      f"  zone={zone_price:,.{pr}f}  close={last_candle['close']:,.{pr}f}"
                      f"  triggered={triggered}")
        else:
            touched   = last_candle["low"] <= zone_price * (1 + TOUCH_BUFFER_PCT)
            triggered = touched and last_candle["close"] > zone_price
            self._tag(f"[trigger?] L={last_candle['low']:,.{pr}f}"
                      f"  zone={zone_price:,.{pr}f}  close={last_candle['close']:,.{pr}f}"
                      f"  triggered={triggered}")

        if not triggered:
            return

        # ── FIRE ──────────────────────────────────────────────────────────────
        self._tag(f"\n  ★ TRIGGER — dir={self.direction}  conf={conf}  mult={mult:.1f}×")
        self._last_trigger_ts = last_candle["ts"]
        self.risk_usdt        = await asyncio.to_thread(calc_risk_usdt, self.symbol)

        klines   = await asyncio.to_thread(get_klines, self.symbol, ATR_INTERVAL, 50)
        atr      = calc_atr(klines, ATR_PERIOD)
        self.atr = atr

        current = last_candle["close"]
        if self.direction == "SHORT":
            self.tp_price = current - TP_ATR_MULT * atr
            self.sl_price = max(zone_price + atr, current + 0.5 * atr)
        else:
            self.tp_price = current + TP_ATR_MULT * atr
            self.sl_price = min(zone_price - atr, current - 0.5 * atr)

        base_qty  = qty_from_risk(self.risk_usdt, current, self.cfg)
        t2_qty    = qty_from_risk(self.risk_usdt * mult, current, self.cfg)
        baby_qty  = self.cfg["min_qty"]

        self._tag(f"[trade] Entry≈{current:,.{pr}f}  TP={self.tp_price:,.{pr}f}"
                  f"  SL={self.sl_price:,.{pr}f}  ATR={atr:.{pr}f}")
        self._tag(f"[trade] T1={base_qty}  T2={t2_qty} @ {t2_price:,.{pr}f} (after BE only)")

        entry_side, pos_side = ("SELL","SHORT") if self.direction == "SHORT" else ("BUY","LONG")

        # Fix #1: Use _chase_entry (position-size confirmed, GTC, no double-fill)
        self.state = "IN_TRADE"
        fill_px, fill_qty = await self._chase_entry(entry_side, base_qty, pos_side)

        if fill_qty <= 0:
            self._tag("[entry] No fill — aborting")
            try:
                await asyncio.to_thread(cancel_all_orders, self.symbol)
            except Exception:
                pass
            self.state = "WATCHING"
            self._write_state()
            return

        self.entry_price = fill_px
        self.t1_qty      = fill_qty
        self._t2_placed  = False
        self._t2_price   = t2_price
        self._t2_qty     = t2_qty
        self._t2_pos_side = pos_side
        self.trail_stage = "none"
        self.best_seen   = None

        # Recalculate TP/SL from actual fill price
        if self.direction == "SHORT":
            self.tp_price = fill_px - TP_ATR_MULT * atr
            self.sl_price = max(zone_price + atr, fill_px + 0.5 * atr)
        else:
            self.tp_price = fill_px + TP_ATR_MULT * atr
            self.sl_price = min(zone_price - atr, fill_px - 0.5 * atr)

        # Place T1 TP order
        close_side    = "BUY" if self.direction == "SHORT" else "SELL"
        self.tp_order = SmartLimitOrder(
            symbol=self.symbol, side=close_side, qty=self.t1_qty,
            pos_side=pos_side, label="T1_TP",
            track_ob=False, fixed_price=self.tp_price, price_round=pr,
        )
        await self.tp_order.place()

        # Start flash scalp monitor
        self.flash_layer = FlashScalpLayer(
            symbol=self.symbol, direction=self.direction,
            entry_price=fill_px, sl_price=self.sl_price,
            baby_qty=baby_qty, pos_side=pos_side, price_round=pr,
        )
        await self.flash_layer.start()

        self._tag(f"[engine] → IN_TRADE  entry={fill_px:,.{pr}f}"
                  f"  TP={self.tp_price:,.{pr}f}  SL={self.sl_price:,.{pr}f}")
        self._write_state()

    # ── Position monitor ───────────────────────────────────────────────────────

    async def _update_trail(self, price: float):
        profit = (self.entry_price - price if self.direction == "SHORT"
                  else price - self.entry_price)
        if profit <= 0:
            return

        pr = self.cfg["price_round"]
        if self.best_seen is None:
            self.best_seen = price
        if self.direction == "SHORT":
            self.best_seen = min(self.best_seen, price)
        else:
            self.best_seen = max(self.best_seen, price)

        new_sl = None
        if profit >= 1.0 * self.atr:
            if self.direction == "SHORT":
                c = round(self.best_seen + 0.6 * self.atr, pr)
                if c < self.sl_price: new_sl = c
            else:
                c = round(self.best_seen - 0.6 * self.atr, pr)
                if c > self.sl_price: new_sl = c
            if new_sl:
                prev = self.trail_stage; self.trail_stage = "trailing"
                self._tag(f"[trail] {'TRAILING' if prev!='trailing' else 'TRAIL updated'}"
                          f"  SL → ${new_sl:,.{pr}f}")

        elif profit >= 0.5 * self.atr and self.trail_stage == "none":
            be = round(self.entry_price + (5 if self.direction=="SHORT" else -5), pr)
            if ((self.direction=="SHORT" and be < self.sl_price) or
                    (self.direction=="LONG" and be > self.sl_price)):
                new_sl = be
                self.trail_stage = "breakeven"
                self._tag(f"[trail] BREAKEVEN  SL → ${new_sl:,.{pr}f}")

        if new_sl:
            self.sl_price = new_sl
            if self.flash_layer:
                self.flash_layer.sl_price = new_sl
            self._write_state()

        # Fix #4: Place T2 only after breakeven confirmed
        if (self.trail_stage in ("breakeven","trailing")
                and not self._t2_placed and self._t2_qty > 0):
            self._tag(f"[T2] Breakeven reached — placing T2 @ {self._t2_price:,.{pr}f}")
            entry_side = "SELL" if self.direction == "SHORT" else "BUY"
            t2 = SmartLimitOrder(
                symbol=self.symbol, side=entry_side, qty=self._t2_qty,
                pos_side=self._t2_pos_side, label="T2_limit",
                track_ob=False, fixed_price=self._t2_price,
                price_round=pr,
            )
            await t2.place()
            self._t2_placed = True
            self._t2_order  = t2

    async def _monitor_position(self):
        price = await asyncio.to_thread(get_price, self.symbol)

        # External close detection
        try:
            cur_qty = await self._get_position_qty()
            if cur_qty == 0 and self.t1_qty > 0:
                await asyncio.sleep(1.5)
                cur_qty = await self._get_position_qty()
                if cur_qty == 0:
                    self._tag("[monitor] Position closed externally")
                    pnl = ((self.entry_price - price) if self.direction=="SHORT"
                           else (price - self.entry_price)) * self.t1_qty
                    result = "WIN" if pnl >= 0 else "LOSS"
                    if pnl >= 0: self.wins += 1
                    else:        self.losses += 1
                    self.total_pnl += pnl
                    self._set_cooldown(self.q_avg, result)  # Fix #3
                    self._log_trade(result, pnl, price, self.t1_qty)
                    self._reset()
                    return
        except Exception:
            pass

        await self._update_trail(price)

        pr = self.cfg["price_round"]
        if self.direction == "SHORT" and price <= self.tp_price:
            self._tag(f"[monitor] TP @ {price:,.{pr}f}")
            await self._on_tp(); return
        if self.direction == "LONG"  and price >= self.tp_price:
            self._tag(f"[monitor] TP @ {price:,.{pr}f}")
            await self._on_tp(); return
        if self.direction == "SHORT" and price >= self.sl_price:
            self._tag(f"[monitor] SL @ {price:,.{pr}f}")
            await self._on_sl(); return
        if self.direction == "LONG"  and price <= self.sl_price:
            self._tag(f"[monitor] SL @ {price:,.{pr}f}")
            await self._on_sl(); return

    async def _on_tp(self):
        cp, cq = await self._close_all("TP")
        approx = cp if cp > 0 else self.tp_price
        qty    = cq if cq > 0 else self.t1_qty
        profit = max((self.entry_price - approx) if self.direction=="SHORT"
                     else (approx - self.entry_price), 0.0) * qty
        self.wins += 1; self.total_pnl += profit
        self._set_cooldown(self.q_avg, "WIN")     # Fix #3
        self._log_trade("WIN", profit, approx, qty)
        self._reset()
        self._tag(f"[engine] WIN  +${profit:.4f}  total=${self.total_pnl:.4f}")

    async def _on_sl(self):
        cp, cq = await self._close_all("SL")
        approx = cp if cp > 0 else self.sl_price
        qty    = cq if cq > 0 else self.t1_qty
        loss   = max((approx - self.entry_price) if self.direction=="SHORT"
                     else (self.entry_price - approx), 0.0) * qty
        self.losses += 1; self.total_pnl -= loss
        self._set_cooldown(self.q_avg, "LOSS")    # Fix #3
        self._log_trade("LOSS", -loss, approx, qty)
        self._reset()
        self._tag(f"[engine] LOSS  -${loss:.4f}  total=${self.total_pnl:.4f}")

    async def _close_all(self, reason: str) -> tuple:
        if self.flash_layer: self.flash_layer.kill_all()
        if self.tp_order and not self.tp_order.is_filled:
            self.tp_order.cancel()
        if self._t2_order and not self._t2_order.is_filled:
            self._t2_order.cancel()
        try:
            await asyncio.to_thread(cancel_all_orders, self.symbol)
        except Exception:
            pass

        actual_price = 0.0
        actual_qty   = 0.0
        pr           = self.cfg["price_round"]
        try:
            cur_qty = await self._get_position_qty()
            if cur_qty > 0:
                close_side = "BUY" if self.direction=="SHORT" else "SELL"
                close_ord  = SmartLimitOrder(
                    symbol=self.symbol, side=close_side, qty=cur_qty,
                    pos_side=self.direction, label=f"close_{reason}",
                    track_ob=True, price_round=pr,
                )
                await close_ord.place()
                await close_ord.wait_fill(timeout=60)
                actual_price = close_ord.fill_price or 0.0
                actual_qty   = cur_qty
        except Exception as e:
            self._tag(f"[close] error: {e}")

        return actual_price, actual_qty

    def _reset(self):
        if self.flash_layer: self.flash_layer.kill_all()
        if self.tp_order and not self.tp_order.is_filled:
            self.tp_order.cancel()
        if self._t2_order and not self._t2_order.is_filled:
            self._t2_order.cancel()
        self.tp_order    = None
        self._t2_order   = None
        self._t2_placed  = False
        self.flash_layer = None
        self.trail_stage = "none"
        self.best_seen   = None
        self.t1_qty      = 0.0
        self.state       = "WATCHING"
        self._write_state()

    async def _print_status(self):
        try:
            price = await asyncio.to_thread(get_price, self.symbol)
            pr    = self.cfg["price_round"]
            self._tag(f"[status] {self.direction}  price={price:,.{pr}f}"
                      f"  tp={self.tp_price:,.{pr}f} ({abs(price-self.tp_price):.{pr}f} away)"
                      f"  sl={self.sl_price:,.{pr}f} ({abs(price-self.sl_price):.{pr}f} away)"
                      f"  trail={self.trail_stage}  T2={'placed' if self._t2_placed else 'waiting_BE'}")
        except Exception:
            pass
        self._write_state()

    def _write_state(self):
        try:
            pr = self.cfg["price_round"]
            with open(state_json(self.symbol), "w") as f:
                json.dump({
                    "state":      self.state,
                    "symbol":     self.symbol,
                    "direction":  self.direction or "",
                    "entry":      round(self.entry_price, pr),
                    "tp":         round(self.tp_price, pr),
                    "sl":         round(self.sl_price, pr),
                    "atr":        round(self.atr, pr),
                    "confidence": self.confidence,
                    "trail_stage":self.trail_stage,
                    "t2_placed":  self._t2_placed,
                    "risk_usdt":  self.risk_usdt,
                    "mode":       "LIQ_v4",
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception:
            pass

    def _log_trade(self, result, pnl, close_px=0.0, close_qty=0.0):
        self.trade_count += 1
        pr = self.cfg["price_round"]
        e  = {
            "id": self.trade_count, "time": datetime.now().isoformat(),
            "symbol": self.symbol, "direction": self.direction,
            "entry": round(self.entry_price, pr),
            "actual_close": round(close_px, pr) if close_px else None,
            "actual_qty":   close_qty or None,
            "tp": self.tp_price, "sl": self.sl_price, "atr": self.atr,
            "confidence": self.confidence, "risk_usdt": self.risk_usdt,
            "q_zones": len(self.q_zones), "q_total_usd": self.q_total_usd,
            "mode": "LIQ_v4", "result": result,
            "pnl_usd": round(pnl, 6),
            "pnl_pct": round(pnl / self.risk_usdt * 100, 4) if self.risk_usdt else 0,
            "total_pnl": round(self.total_pnl, 6),
        }
        log = []
        if TRADE_LOG_JSON.exists():
            try:
                with open(TRADE_LOG_JSON) as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append(e)
        with open(TRADE_LOG_JSON, "w") as f:
            json.dump(log, f, indent=2)
        self._tag(f"[log] #{self.trade_count} {result}  pnl=${pnl:+.4f}")


# ==============================================================================
# STARTUP & MAIN
# ==============================================================================

async def heatmap_loop(symbol: str):
    first = True
    while True:
        print(f"  [{symbol}][heatmap] {'startup' if first else '5-min refresh'} ...")
        first = False
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(HEATMAP_SCRIPT), "--symbol", symbol,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            for line in out.decode("utf-8", errors="replace").splitlines():
                if any(k in line for k in ["Current Price","Data saved","Total modeled","Error","Traceback"]):
                    print(f"  [{symbol}][heatmap] {line.strip()}")
            print(f"  [{symbol}][heatmap] {'OK' if proc.returncode==0 else f'exit {proc.returncode}'}")
        except Exception as e:
            print(f"  [{symbol}][heatmap] error: {e}")
        await asyncio.sleep(HEATMAP_REFRESH)


async def startup():
    for sym, cfg in SYMBOL_CONFIG.items():
        try:
            await asyncio.to_thread(set_leverage, sym, cfg["leverage"])
            print(f"  [startup] Leverage = {cfg['leverage']}x on {sym}")
        except Exception as e:
            print(f"  [startup] {sym}: {e}")
    try:
        bal  = await asyncio.to_thread(get_balances)
        bals = bal if isinstance(bal, list) else bal.get("data", [])
        for b in bals:
            w = float(b.get("balance") or b.get("walletBalance") or 0)
            if w > 0:
                print(f"  [startup] Balance: {w:.4f} {b.get('asset','?')}")
    except Exception as e:
        print(f"  [startup] balance: {e}")


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  LIQ ALGO v4  --  Multi-Asset Liquidation Reversal Engine")
    print(f"  Symbols    : {', '.join(SYMBOLS)}")
    print(f"  Fix 1      : GTC entry + position-size fill confirmation")
    print(f"  Fix 2      : Regime filter (hourly trend check)")
    print(f"  Fix 3      : Zone cooldown {COOLDOWN_LOSS}s LOSS / {COOLDOWN_WIN}s WIN")
    print(f"  Fix 4      : T2 placed only after breakeven")
    print(f"  Flash baby : {FLASH_DIP_MIN_PCT}–{FLASH_DIP_MAX_PCT}% in <{FLASH_WINDOW_SEC}s")
    print(f"  TP / SL    : {TP_ATR_MULT}x / {SL_ATR_MULT}x ATR")
    print("=" * 60)

    await startup()

    engines = [TradingEngine(sym) for sym in SYMBOLS]
    tasks   = []
    for i, (eng, sym) in enumerate(zip(engines, SYMBOLS)):
        tasks.append(asyncio.create_task(
            (lambda s=sym, d=i*30: _delayed(d, heatmap_loop(s)))()
        ))
        tasks.append(asyncio.create_task(eng.minute_loop()))
        tasks.append(asyncio.create_task(eng.position_monitor_loop()))

    await asyncio.gather(*tasks)


async def _delayed(delay, coro):
    if delay: await asyncio.sleep(delay)
    await coro


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  [liq_algo_v4] Stopped. Ctrl+C")
        sys.exit(0)
