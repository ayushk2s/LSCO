"""
liq_algo.py  —  Liquidation Zone Reversal Trading Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW IT WORKS (plain english):
  1. Every minute: read the liquidation heatmap JSON and find zones near price.
  2. If price is within APPROACH_PCT% of a zone → calculate Q (how much
     liquidation stacks up from there outward) and confidence score.
  3. Wait for the ENTRY TRIGGER: a 1-minute candle that closes BACK inside
     the zone boundary after being outside it. That means the cascade ended
     and reversal is starting.
  4. Place PARENT TRADE 1 immediately (SmartLimitOrder at best bid/ask).
  5. Place PARENT TRADE 2 as limit at Q_avg (weighted average of the Q zone).
     Size of Trade 2 = Trade 1 × confidence multiplier.
  6. Spawn BABY LAYER: 3 small limit orders at 0.1%, 0.2%, 0.4% away from
     parent entry — each targeting 0.15% profit. Auto-reset after each TP.
  7. Manage exits: TP at 1.5×ATR, SL at 0.75×ATR from entry.
     If SL hit → cancel Trade 2 + kill all babies + close full position.
     If both TPs hit → look for Trade 3 if confidence ≥ 75.

HOW TO RUN:
  1. Make sure whale_monitor.py is running in a separate terminal.
  2. Make sure binance_liq_heatmap_2.py ran recently (updates every 30 min).
  3. Run: python liq_algo.py
  4. Press Ctrl+C to stop cleanly.

REQUIRES:
  - account_data.py (AsterDEX API client) in same folder
  - market_data.py  (public market data)
  - order_executor.py (SmartLimitOrder)
  - ../binance_liq_heatmap_2.json  (updated by heatmap script)
  - ../whale_last_candle.json      (updated by whale_monitor.py)
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# ── Import our own modules ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from account_data   import (place_order, cancel_order, cancel_all_orders,
                             get_position_risk, get_balances, set_leverage)
from market_data    import get_price, get_klines, calc_atr, get_last_1m_candle
from order_executor import SmartLimitOrder


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 1: CONFIGURATION  (change these to tune the algorithm) ────────────
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL       = "BTCUSDT"
LEVERAGE     = 20                 # leverage on AsterDEX (set once at start)

# ── Risk per trade ────────────────────────────────────────────────────────────
RISK_USDT         = 1.0           # fallback; real value set by calc_risk_usdt() each trade
BALANCE_UTILIZATION = 0.05        # use only 5% of balance — minimal capital for live demo
MIN_RISK_USDT     = 1.0           # never risk less than $1
MAX_RISK_USDT     = 5.0           # hard cap at $5 per trade during trial/demo period

# ── Zone detection ────────────────────────────────────────────────────────────
APPROACH_PCT = 0.8                # enter "approaching" state when within 0.8% of zone
MIN_ZONE_USD    = 10_000_000      # ignore zones below $10M (raised from $5M — filters noise)
Q_MIN_USD       = 20_000_000     # minimum Q threshold
MIN_CONFIDENCE  = 65             # skip trades with confidence below this (both losses were 60)

# ── Exit parameters ───────────────────────────────────────────────────────────
ATR_PERIOD   = 14
ATR_INTERVAL = "5m"
TP_ATR_MULT  = 1.5
SL_ATR_MULT  = 0.75

# ── Baby layer ────────────────────────────────────────────────────────────────
BABY_OFFSETS  = [0.001, 0.002, 0.004]   # 0.1%, 0.2%, 0.4% from parent entry
BABY_SIZES    = [0.10,  0.20,  0.40]    # 10%, 20%, 40% of parent notional
BABY_TP_PCT   = 0.0015                  # 0.15% take-profit per baby

# ── Confidence score → position size multiplier ───────────────────────────────
CONF_MULTIPLIERS = {
    (0,  60): 1.6,
    (60, 75): 1.8,
    (75, 90): 2.0,
    (90,101): 2.2,
}

# ── File paths ────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
HEATMAP_JSON   = ROOT_DIR / "data_fetching" / "binance_liq_heatmap.json"
WHALE_JSON     = ROOT_DIR / "whale_BTCUSDT.json"
TRADE_LOG_JSON = ROOT_DIR / "trade_log.json"
ALGO_STATE_JSON = ROOT_DIR / "algo_state.json"


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 2: ZONE SCANNER  (reads heatmap + whale JSON, finds Q & confidence)
# ══════════════════════════════════════════════════════════════════════════════

def calc_risk_usdt() -> float:
    """
    Dynamically size the base risk amount from actual wallet balance.

    Total margin used per trade at MAX confidence:
      Trade 1  = RISK_USDT × 1
      Trade 2  = RISK_USDT × max_mult   (2.2 at conf ≥ 90)
      Babies   = RISK_USDT × sum(BABY_SIZES)  (= 0.70)
      ─────────────────────────────────────────
      Total    = RISK_USDT × (1 + 2.2 + 0.70)  =  RISK_USDT × 3.9

    So: RISK_USDT = available_balance × BALANCE_UTILIZATION / 3.9
    Clamped to [MIN_RISK_USDT, MAX_RISK_USDT].
    Falls back to the static RISK_USDT constant if the API call fails.
    """
    max_mult     = max(v for v in CONF_MULTIPLIERS.values())
    total_factor = 1.0 + max_mult + sum(BABY_SIZES)   # = 3.9 at default settings

    try:
        bal_data  = get_balances()
        balances  = bal_data if isinstance(bal_data, list) else bal_data.get("data", [])
        available = 0.0
        for b in balances:
            if b.get("asset", "").upper() in ("USDT", "BUSD"):
                # availableBalance = free margin (after existing positions)
                # fall back to walletBalance or balance if field is missing
                av_str = (b.get("availableBalance")
                          or b.get("walletBalance")
                          or b.get("balance")
                          or "0")
                available = max(available, float(av_str))

        if available <= 0:
            print(f"  [sizing] Balance returned 0 — using fallback ${RISK_USDT}")
            return RISK_USDT

        raw     = available * BALANCE_UTILIZATION / total_factor
        clamped = max(MIN_RISK_USDT, min(MAX_RISK_USDT, raw))
        print(f"  [sizing] Balance=${available:.2f}  factor={total_factor:.1f}"
              f"  → RISK_USDT=${clamped:.4f}")
        return round(clamped, 4)

    except Exception as e:
        print(f"  [sizing] Balance fetch failed ({e}) — using fallback ${RISK_USDT}")
        return RISK_USDT


def load_heatmap() -> dict:
    """Read the heatmap JSON that binance_liq_heatmap.py produces."""
    if not HEATMAP_JSON.exists():
        raise FileNotFoundError(
            f"Heatmap JSON not found: {HEATMAP_JSON}\n"
            "Run binance_liq_heatmap.py first."
        )
    with open(HEATMAP_JSON) as f:
        data = json.load(f)

    # Warn if heatmap is stale (older than 1 hour)
    from datetime import datetime as dt
    generated = dt.fromisoformat(data["generated_at"])
    age_min = (dt.now() - generated).total_seconds() / 60
    if age_min > 60:
        print(f"  [scanner] ⚠ Heatmap is {age_min:.0f} min old — re-run "
              f"binance_liq_heatmap.py for fresh data")
    return data


def load_whale_candle() -> dict | None:
    """Read the last 1-minute candle whale summary written by whale_monitor.py."""
    if not WHALE_JSON.exists():
        return None
    try:
        with open(WHALE_JSON) as f:
            data = json.load(f)
        # Ignore if older than 3 minutes (stale) — candle_ts is seconds
        age = time.time() - data["candle_ts"]
        if age > 180:
            return None
        return data
    except Exception:
        return None


def scan_zones(current_price: float, heatmap: dict) -> dict:
    """
    Find the nearest significant liq zones above and below current price.
    Returns a dict describing what's near and whether we're approaching.
    """
    short_zones = [z for z in heatmap["short_liquidations"] if z["usd"] >= MIN_ZONE_USD]
    long_zones  = [z for z in heatmap["long_liquidations"]  if z["usd"] >= MIN_ZONE_USD]

    # Nearest short-liq zone ABOVE price (price going up hits these → shorts liquidated)
    above = [z for z in short_zones if z["price"] > current_price]
    nearest_above = min(above, key=lambda z: z["price"]) if above else None

    # Nearest long-liq zone BELOW price (price going down hits these → longs liquidated)
    below = [z for z in long_zones if z["price"] <= current_price]
    nearest_below = max(below, key=lambda z: z["price"]) if below else None

    result = {
        "price":           current_price,
        "nearest_above":   nearest_above,
        "nearest_below":   nearest_below,
        "approaching_up":  False,    # price nearing SHORT liq zone (consider SHORT trade)
        "approaching_down":False,    # price nearing LONG  liq zone (consider LONG  trade)
        "dist_above_pct":  None,
        "dist_below_pct":  None,
    }

    if nearest_above:
        dist = (nearest_above["price"] - current_price) / current_price * 100
        result["dist_above_pct"]  = dist
        result["approaching_up"]  = dist <= APPROACH_PCT

    if nearest_below:
        dist = (current_price - nearest_below["price"]) / current_price * 100
        result["dist_below_pct"]  = dist
        result["approaching_down"] = dist <= APPROACH_PCT

    return result


def calc_q(zones: list, nearest_zone: dict, direction: str) -> tuple:
    """
    Walk zones outward from nearest_zone, accumulate USD until
    running_sum >= max(2 × nearest_zone_usd, Q_MIN_USD).

    direction: "above" (short setup) or "below" (long setup)

    Returns: (q_zones, q_avg_price, total_usd)
      q_avg_price = weighted average price across all Q zones (weighted by USD)
      This is where Trade 2 limit order is placed.
    """
    threshold = max(2.0 * nearest_zone["usd"], Q_MIN_USD)

    # Sort outward: for "above" direction → ascending price; "below" → descending
    if direction == "above":
        candidates = sorted(zones, key=lambda z: z["price"])
        candidates = [z for z in candidates if z["price"] > nearest_zone["price"]]
    else:
        candidates = sorted(zones, key=lambda z: -z["price"])
        candidates = [z for z in candidates if z["price"] < nearest_zone["price"]]

    q_zones     = [nearest_zone]   # always include nearest zone
    running_sum = nearest_zone["usd"]

    for z in candidates:
        if z["usd"] < MIN_ZONE_USD:
            continue
        q_zones.append(z)
        running_sum += z["usd"]
        if running_sum >= threshold:
            break   # Q condition reached

    # Weighted-average price (zones with more USD pull the average more)
    total_usd = sum(z["usd"] for z in q_zones)
    q_avg     = sum(z["price"] * z["usd"] for z in q_zones) / total_usd

    return q_zones, q_avg, total_usd


def calc_confidence(scan: dict, q_zones: list, q_total: float,
                    whale: dict | None, direction: str) -> int:
    """
    Score 0–100 representing how strong this reversal setup is.
    Higher score → larger Trade 2 position size.

    direction: "short" or "long"
    """
    score = 50   # always start at 50 (base)

    nearest = scan["nearest_above"] if direction == "short" else scan["nearest_below"]
    if nearest:
        if nearest["usd"] >= 30_000_000:
            score += 10   # big zone (≥$30M) = stronger magnet for price
        if nearest["usd"] >= 60_000_000:
            score += 5    # extra bonus for very large zone

    if q_total >= 60_000_000:
        score += 10       # Q has lots of total liquidation = strong cluster

    if len(q_zones) >= 3:
        score += 10       # 3+ consecutive zones = dense liquidation region

    # Whale confirmation: buys confirm LONG, sells confirm SHORT
    if whale:
        if direction == "short" and whale["whale_sell_usd"] > whale["whale_buy_usd"]:
            score += 15   # whales were selling into the zone (confirms short)
        elif direction == "long" and whale["whale_buy_usd"] > whale["whale_sell_usd"]:
            score += 15   # whales were buying the dip (confirms long)
        # Opposing whale activity reduces confidence
        if direction == "short" and whale["whale_buy_usd"] > whale["whale_sell_usd"] * 1.5:
            score -= 20   # big whale buying = don't short!
        elif direction == "long" and whale["whale_sell_usd"] > whale["whale_buy_usd"] * 1.5:
            score -= 20   # big whale selling = don't long!

    return max(0, min(100, score))   # clamp to [0, 100]


def confidence_multiplier(score: int) -> float:
    """Map confidence score to Trade 2 size multiplier."""
    for (lo, hi), mult in CONF_MULTIPLIERS.items():
        if lo <= score < hi:
            return mult
    return 1.6   # default


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 3: POSITION SIZING  (converts $ risk to BTC quantity)
# ══════════════════════════════════════════════════════════════════════════════

def qty_from_risk(risk_usdt: float, price: float) -> float:
    """
    How many BTC to trade given a margin amount and leverage.
    Example: $5 margin × 20 leverage = $100 notional → 0.00123 BTC at $81,400.
    Rounds to 3 decimal places (minimum AsterDEX order size is usually 0.001 BTC).
    """
    notional = risk_usdt * LEVERAGE
    qty      = notional / price
    qty      = round(qty, 3)
    return max(qty, 0.001)   # enforce minimum order size


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 4: BABY LAYER  (auto-resetting scalp orders around parent entry)
# ══════════════════════════════════════════════════════════════════════════════

class BabyLayer:
    """
    After the parent trade fills, BabyLayer spawns 3 small limit orders
    placed INTO the spike (same direction as parent) at increasing distances.

    SHORT setup example (parent entry = $81,400):
      Baby 1: Limit SELL @ $81,481  (+0.1%)  size = 10% of parent notional
      Baby 2: Limit SELL @ $81,562  (+0.2%)  size = 20% of parent notional
      Baby 3: Limit SELL @ $81,724  (+0.4%)  size = 40% of parent notional

    Each baby has a Take Profit at 0.15% below its fill price.
    When a baby's TP fills, it immediately places a new baby at the same
    level (auto-reset) — catching the next oscillation.

    kill_all() → call when parent trade closes (win or lose).
    It cancels every open baby entry and TP order immediately.
    """

    def __init__(self, symbol: str, direction: str,
                 entry_price: float, parent_notional: float,
                 pos_side: str, sl_price: float):
        self.symbol           = symbol
        self.direction        = direction.upper()
        self.entry_price      = entry_price
        self.parent_notional  = parent_notional
        self.pos_side         = pos_side.upper()
        self.sl_price         = sl_price         # babies must not be placed beyond SL
        self.alive            = True
        self._active_orders   = []

    async def start(self):
        """Spawn all 3 babies concurrently."""
        print(f"\n  [baby] Spawning baby layer ({self.direction}) "
              f"around entry ${self.entry_price:,.1f}")
        # Create all 3 babies at once (they run independently)
        tasks = [
            asyncio.create_task(self._baby_lifecycle(offset, size_frac, idx))
            for idx, (offset, size_frac)
            in enumerate(zip(BABY_OFFSETS, BABY_SIZES), 1)
        ]
        # Don't await here — babies run in background while parent monitors

    async def _baby_lifecycle(self, offset_pct: float, size_frac: float, idx: int):
        """
        Full lifecycle of one baby slot: place entry → wait fill →
        place TP → wait TP fill → auto-reset → repeat until killed.
        """
        while self.alive:
            # ── Calculate baby entry price ────────────────────────────────────
            # SHORT: place limit SELL above parent entry (sell into the spike)
            # LONG:  place limit BUY  below parent entry (buy the dip deeper)
            if self.direction == "SHORT":
                baby_entry_price = self.entry_price * (1 + offset_pct)
                baby_side        = "SELL"
                tp_price         = baby_entry_price * (1 - BABY_TP_PCT)
                # Don't place baby above SL — would add to a losing position
                if baby_entry_price >= self.sl_price:
                    print(f"  [baby] baby{idx} skipped — level {baby_entry_price:,.1f}"
                          f" is at/above SL {self.sl_price:,.1f}")
                    break
            else:
                baby_entry_price = self.entry_price * (1 - offset_pct)
                baby_side        = "BUY"
                tp_price         = baby_entry_price * (1 + BABY_TP_PCT)
                if baby_entry_price <= self.sl_price:
                    print(f"  [baby] baby{idx} skipped — level {baby_entry_price:,.1f}"
                          f" is at/below SL {self.sl_price:,.1f}")
                    break

            qty = round((self.parent_notional * size_frac) / baby_entry_price, 3)
            qty = max(qty, 0.001)

            label = f"baby{idx}({offset_pct*100:.1f}%)"

            # ── Place baby entry at fixed computed level ──────────────────────
            # Resting limit exactly at baby_entry_price — waits for price to
            # reach that level. OB-tracking here would give a current-market fill
            # instead of catching the spike at the intended distance.
            entry_order = SmartLimitOrder(
                symbol=self.symbol, side=baby_side, qty=qty,
                pos_side=self.pos_side, label=label,
                track_ob=False, fixed_price=baby_entry_price,
            )
            self._active_orders.append(entry_order)
            await entry_order.place()

            # ── Wait for baby entry to fill ───────────────────────────────────
            filled = await entry_order.wait_fill(timeout=3600)   # 1h max wait
            if not self.alive or not filled:
                entry_order.cancel()
                break

            fill_px = entry_order.fill_price
            print(f"  [baby] {label} entry filled @ {fill_px:,.1f} "
                  f"| TP target: {tp_price:,.1f}")

            # ── Place baby TP (fixed price, no OB tracking) ───────────────────
            close_side = "BUY" if baby_side == "SELL" else "SELL"
            tp_order = SmartLimitOrder(
                symbol=self.symbol, side=close_side, qty=qty,
                pos_side=self.pos_side, label=f"{label}_tp",
                track_ob=False, fixed_price=tp_price,
            )
            self._active_orders.append(tp_order)
            await tp_order.place()

            # ── Wait for TP to fill ───────────────────────────────────────────
            tp_filled = await tp_order.wait_fill(timeout=1800)   # 30 min max
            if not self.alive:
                tp_order.cancel()
                break

            if tp_filled:
                profit = abs(tp_price - fill_px) * qty
                print(f"  [baby] {label} TP hit ✓  profit ≈ ${profit:.4f}")
                # Auto-reset: loop continues and places a new baby at same level

            else:
                # TP didn't fill within 30 min — parent likely closed. Stop.
                tp_order.cancel()
                break

    def kill_all(self):
        """
        Cancel every open baby order immediately.
        Called when parent trade closes (TP or SL).
        """
        self.alive = False
        print(f"  [baby] Killing all baby orders ({len(self._active_orders)} tracked)")
        for order in self._active_orders:
            if not order.is_filled and not order.is_cancelled:
                order.cancel()
        self._active_orders.clear()


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 5: STATE MACHINE  (main trading logic)
# ══════════════════════════════════════════════════════════════════════════════

class TradingEngine:
    """
    State machine with 5 states:
      WATCHING    → scanning for approaching zones (idle)
      APPROACHING → within APPROACH_PCT% of a zone, calculating Q
      TRIGGERED   → entry condition met, orders being placed
      IN_TRADE    → parent orders active, managing exits
      CLOSED      → trade done, logging, back to WATCHING
    """

    def __init__(self):
        self.state        = "WATCHING"
        self.direction    = None    # "SHORT" or "LONG"
        self.setup        = {}      # zone scan result
        self.q_zones      = []
        self.q_avg        = 0.0
        self.q_total_usd  = 0.0
        self.confidence   = 0
        self.entry_price  = 0.0
        self.atr          = 0.0
        self.tp_price     = 0.0
        self.sl_price     = 0.0
        self.trail_stage      = "none"   # "none" | "breakeven" | "trailing"
        self.best_price_seen  = None     # most favourable price since entry
        self.risk_usdt        = RISK_USDT  # dynamic per-trade, set by calc_risk_usdt()
        self._last_trigger_ts = 0        # candle timestamp of last fired trigger (prevents double-fire)

        # Active orders
        self.trade1       = None    # SmartLimitOrder
        self.trade2       = None    # SmartLimitOrder
        self.tp_order     = None    # SmartLimitOrder
        self.babies       = None    # BabyLayer

        # Trade stats
        self.trade_count  = 0
        self.wins         = 0
        self.losses       = 0
        self.total_pnl    = 0.0

        # Heatmap data (refreshed every scan)
        self._heatmap     = None

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────────────

    async def minute_loop(self):
        """
        Runs every minute (aligned to clock boundaries).
        Handles zone detection and state transitions.
        """
        print(f"  [engine] Starting. State: WATCHING")
        while True:
            # Sleep until the start of the next minute
            now  = time.time()
            wait = (int(now / 60) + 1) * 60 - now + 0.2   # 200ms grace
            await asyncio.sleep(wait)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'─'*60}")
            print(f"  [{ts}] State={self.state}")

            try:
                await self._tick()
            except Exception as e:
                print(f"  [engine] ERROR in tick: {e}")
                import traceback; traceback.print_exc()

            self._write_state()

    async def position_monitor_loop(self):
        """
        Runs every 3 seconds when in IN_TRADE state.
        Checks if TP or SL was hit and triggers exit.
        """
        while True:
            await asyncio.sleep(3)
            if self.state == "IN_TRADE":
                try:
                    await self._monitor_position()
                except Exception as e:
                    print(f"  [engine] monitor error: {e}")

    # ── TICK — called every minute ────────────────────────────────────────────

    async def _tick(self):
        if self.state == "WATCHING":
            await self._tick_watching()

        elif self.state == "APPROACHING":
            await self._tick_approaching()

        elif self.state == "IN_TRADE":
            # Position monitor loop handles this; minute tick just prints status
            await self._print_position_status()

    # ── WATCHING STATE ────────────────────────────────────────────────────────

    async def _tick_watching(self):
        """Scan heatmap, detect if price is approaching a liq zone."""
        # Refresh heatmap (reload JSON from disk — re-run heatmap script periodically)
        try:
            self._heatmap = await asyncio.to_thread(load_heatmap)
        except FileNotFoundError as e:
            print(f"  [scanner] {e}")
            return

        price = await asyncio.to_thread(get_price, SYMBOL)
        scan  = await asyncio.to_thread(scan_zones, price, self._heatmap)

        # Print what we see
        if scan["nearest_above"]:
            z = scan["nearest_above"]
            flag = " ← APPROACHING" if scan["approaching_up"] else ""
            print(f"  [scan] Above: ${z['price']:>10,.1f}  ${z['usd']/1e6:.1f}M"
                  f"  ({scan['dist_above_pct']:.2f}% away){flag}")
        if scan["nearest_below"]:
            z = scan["nearest_below"]
            flag = " ← APPROACHING" if scan["approaching_down"] else ""
            print(f"  [scan] Below: ${z['price']:>10,.1f}  ${z['usd']/1e6:.1f}M"
                  f"  ({scan['dist_below_pct']:.2f}% away){flag}")

        if scan["approaching_up"] or scan["approaching_down"]:
            # Determine direction
            if scan["approaching_up"] and scan["approaching_down"]:
                # Both sides approaching — pick the closer one
                du = scan["dist_above_pct"] or 999
                dd = scan["dist_below_pct"] or 999
                direction = "SHORT" if du < dd else "LONG"
            elif scan["approaching_up"]:
                direction = "SHORT"   # liq zone above → short reversal trade
            else:
                direction = "LONG"

            self.direction = direction
            self.setup     = scan
            self.state     = "APPROACHING"
            print(f"  [engine] → APPROACHING  direction={direction}")
            # Run the approaching tick immediately in the same minute
            await self._tick_approaching()

    # ── APPROACHING STATE ─────────────────────────────────────────────────────

    async def _tick_approaching(self):
        """
        Calculate Q, confidence, whale confirmation.
        Check for the entry trigger (1m candle closes back inside zone).
        """
        price = await asyncio.to_thread(get_price, SYMBOL)

        # Refresh scan to make sure we're still approaching
        scan = await asyncio.to_thread(scan_zones, price, self._heatmap)

        # If we've moved away from the zone, go back to watching
        if self.direction == "SHORT" and not scan["approaching_up"]:
            print(f"  [engine] Zone passed/missed → back to WATCHING")
            self.state = "WATCHING"
            return
        if self.direction == "LONG" and not scan["approaching_down"]:
            print(f"  [engine] Zone passed/missed → back to WATCHING")
            self.state = "WATCHING"
            return

        # Determine which zone list to use for Q calculation
        if self.direction == "SHORT":
            nearest  = scan["nearest_above"]
            zones    = [z for z in self._heatmap["short_liquidations"]
                        if z["usd"] >= MIN_ZONE_USD]
            dir_str  = "above"
        else:
            nearest  = scan["nearest_below"]
            zones    = [z for z in self._heatmap["long_liquidations"]
                        if z["usd"] >= MIN_ZONE_USD]
            dir_str  = "below"

        # Calculate Q range
        q_zones, q_avg, q_total = await asyncio.to_thread(
            calc_q, zones, nearest, dir_str
        )
        self.q_zones     = q_zones
        self.q_avg       = q_avg
        self.q_total_usd = q_total

        # Load whale data from last 1m candle
        whale = await asyncio.to_thread(load_whale_candle)

        # Calculate confidence
        conf = calc_confidence(scan, q_zones, q_total, whale,
                               self.direction.lower())
        self.confidence = conf

        # Weighted average price from whale fills (if available and confirming)
        whale_avg = 0.0
        if whale:
            if self.direction == "SHORT" and whale["whale_sell_usd"] > 0:
                whale_avg = whale["whale_sell_avg_price"]
            elif self.direction == "LONG" and whale["whale_buy_usd"] > 0:
                whale_avg = whale["whale_buy_avg_price"]

        # If whale data confirms, blend Q_avg with whale_avg price for Trade 2
        if whale_avg > 0:
            trade2_price = (q_avg + whale_avg) / 2
            print(f"  [engine] Q_avg={q_avg:,.1f}  whale_avg={whale_avg:,.1f}"
                  f"  Trade2_price={trade2_price:,.1f}")
        else:
            trade2_price = q_avg
            print(f"  [engine] Q_avg={q_avg:,.1f} (no whale data)")

        mult = confidence_multiplier(conf)
        print(f"  [engine] Q total=${q_total/1e6:.1f}M  {len(q_zones)} zones"
              f"  confidence={conf}  multiplier={mult:.1f}x")

        # ── CHECK ENTRY TRIGGER ───────────────────────────────────────────────
        # Trigger = last completed 1m candle closed BACK INSIDE the zone boundary
        # (price entered the zone and came back = cascade done = reversal starting)
        last_candle = await asyncio.to_thread(get_last_1m_candle, SYMBOL)
        zone_price  = nearest["price"]
        triggered   = False

        if self.direction == "SHORT":
            # Price went above zone, closed back below zone → short trigger
            triggered = (last_candle["high"] > zone_price and
                         last_candle["close"] < zone_price)
            print(f"  [trigger?] Candle high={last_candle['high']:,.1f} "
                  f"zone={zone_price:,.1f} close={last_candle['close']:,.1f} "
                  f"| triggered={triggered}")
        else:
            # Price went below zone, closed back above zone → long trigger
            triggered = (last_candle["low"] < zone_price and
                         last_candle["close"] > zone_price)
            print(f"  [trigger?] Candle low={last_candle['low']:,.1f} "
                  f"zone={zone_price:,.1f} close={last_candle['close']:,.1f} "
                  f"| triggered={triggered}")

        if not triggered:
            return   # still waiting for the reversal candle

        # ── Same-candle guard — don't fire twice on the same 1m candle ────────
        # This prevents a false "externally closed" reset from re-triggering
        # the same candle on the very next minute tick.
        if last_candle["ts"] == self._last_trigger_ts:
            print(f"  [engine] Trigger already fired on this candle (ts={last_candle['ts']}) — skipping")
            self.state = "WATCHING"
            return

        # ── Confidence gate — skip low-quality setups ─────────────────────────
        if conf < MIN_CONFIDENCE:
            print(f"  [engine] Trigger fired but confidence={conf} < {MIN_CONFIDENCE} — skipping")
            self.state = "WATCHING"
            return

        # ── PLACE TRADES ─────────────────────────────────────────────────────
        print(f"\n  ★ ENTRY TRIGGER FIRED — direction={self.direction}  "
              f"conf={conf}/100  mult={mult:.1f}×")

        # Record candle timestamp so the same candle can't trigger twice
        self._last_trigger_ts = last_candle["ts"]

        # Dynamic position sizing from wallet balance
        self.risk_usdt = await asyncio.to_thread(calc_risk_usdt)

        # Get ATR for TP/SL levels
        klines   = await asyncio.to_thread(get_klines, SYMBOL, ATR_INTERVAL, 50)
        atr      = calc_atr(klines, ATR_PERIOD)
        self.atr = atr

        # Calculate TP and SL
        # SL is placed 1 ATR BEYOND the zone edge — not just ATR from entry.
        # Rationale: if price breaks clearly past the zone that caused the signal,
        # the reversal thesis is wrong. Entry-based SL was too tight (inside the zone).
        current  = last_candle["close"]
        self.entry_price = current

        if self.direction == "SHORT":
            self.tp_price = current - TP_ATR_MULT * atr
            self.sl_price = zone_price + atr          # 1 ATR above the zone edge
            # Safety: SL must always be above entry for SHORT
            self.sl_price = max(self.sl_price, current + 0.5 * atr)
        else:
            self.tp_price = current + TP_ATR_MULT * atr
            self.sl_price = zone_price - atr          # 1 ATR below the zone edge
            # Safety: SL must always be below entry for LONG
            self.sl_price = min(self.sl_price, current - 0.5 * atr)

        # Risk amounts (dynamic, based on current wallet balance)
        base_qty   = qty_from_risk(self.risk_usdt, current)
        trade2_qty = qty_from_risk(self.risk_usdt * mult, current)

        print(f"  [trade] Entry={current:,.1f}  TP={self.tp_price:,.1f}"
              f"  SL={self.sl_price:,.1f}  ATR={atr:.1f}")
        print(f"  [trade] RISK_USDT=${self.risk_usdt:.4f}  T1 qty={base_qty}"
              f"  T2 qty={trade2_qty} @ {trade2_price:,.1f}")

        # Determine order sides
        if self.direction == "SHORT":
            entry_side = "SELL"; pos_side = "SHORT"
        else:
            entry_side = "BUY";  pos_side = "LONG"

        # ── Trade 1: immediate SmartLimitOrder at best bid/ask ────────────────
        self.trade1 = SmartLimitOrder(
            symbol=SYMBOL, side=entry_side, qty=base_qty,
            pos_side=pos_side, label="Trade1_entry", track_ob=True,
        )
        await self.trade1.place()

        # ── Trade 2: resting limit at Q-average price ────────────────────────
        # Fixed price — waits at trade2_price until price retraces that deep.
        # Do NOT OB-track: that would slap it at current market, not the Q zone.
        self.trade2 = SmartLimitOrder(
            symbol=SYMBOL, side=entry_side, qty=trade2_qty,
            pos_side=pos_side, label="Trade2_limit",
            track_ob=False, fixed_price=trade2_price,
        )
        await self.trade2.place()

        self.state = "IN_TRADE"
        print(f"  [engine] → IN_TRADE")

    # ── IN_TRADE STATE (position monitor, runs every 3s) ─────────────────────

    async def _update_trail(self, price: float):
        """Two-stage trailing stop.
        Stage 1 (breakeven): move SL to entry+$5 once profit >= 0.5*ATR
        Stage 2 (trailing):  trail SL 0.6*ATR behind best price once profit >= 1.0*ATR
        """
        if not self.trade1 or not self.trade1.is_filled:
            return

        profit_pts = (self.entry_price - price
                      if self.direction == "SHORT"
                      else price - self.entry_price)
        if profit_pts <= 0:
            return

        # Track the best price seen
        if self.best_price_seen is None:
            self.best_price_seen = price
        if self.direction == "SHORT":
            self.best_price_seen = min(self.best_price_seen, price)
        else:
            self.best_price_seen = max(self.best_price_seen, price)

        new_sl = None

        # Stage 2: ATR trailing (kicks in at 1.0×ATR profit)
        if profit_pts >= 1.0 * self.atr:
            if self.direction == "SHORT":
                candidate = round(self.best_price_seen + 0.6 * self.atr, 1)
                if candidate < self.sl_price:   # only tighten, never loosen
                    new_sl = candidate
            else:
                candidate = round(self.best_price_seen - 0.6 * self.atr, 1)
                if candidate > self.sl_price:
                    new_sl = candidate
            if new_sl is not None:
                prev = self.trail_stage
                self.trail_stage = "trailing"
                tag = "TRAILING started" if prev != "trailing" else "TRAIL updated"
                print(f"  [trail] {tag}  best={self.best_price_seen:,.1f}"
                      f"  SL -> ${new_sl:,.1f}")

        # Stage 1: breakeven (kicks in at 0.5×ATR profit, only if not already trailing)
        elif profit_pts >= 0.5 * self.atr and self.trail_stage == "none":
            be = round(self.entry_price + (5.0 if self.direction == "SHORT" else -5.0), 1)
            tightens = (
                (self.direction == "SHORT" and be < self.sl_price) or
                (self.direction == "LONG"  and be > self.sl_price)
            )
            if tightens:
                new_sl = be
                self.trail_stage = "breakeven"
                print(f"  [trail] BREAKEVEN  SL -> ${new_sl:,.1f}")

        if new_sl is not None:
            self.sl_price = new_sl
            self._write_state()

    async def _monitor_position(self):
        """
        Called every 3 seconds while in IN_TRADE state.
        Checks fill status of Trade1, spawns babies on fill,
        checks if SL or TP price was breached.
        """
        price = await asyncio.to_thread(get_price, SYMBOL)

        # ── Verify actual position still exists on exchange ───────────────────
        # Only check AFTER Trade1 has confirmed filled — before fill there is no
        # position yet and the check would wrongly detect an "external close".
        if self.trade1 and self.trade1.is_filled:
            try:
                pos_data     = await asyncio.to_thread(get_position_risk, SYMBOL)
                positions    = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
                pos_side_str = "SHORT" if self.direction == "SHORT" else "LONG"
                pos_exists   = any(
                    p.get("positionSide", "") == pos_side_str
                    and abs(float(p.get("positionAmt", 0))) > 0
                    for p in positions
                )
                if not pos_exists:
                    print(f"  [monitor] Position closed externally — logging PnL and resetting")

                    # ── Log the trade using actual exchange data ──────────────
                    # entry_price is the real T1 fill (updated when T1 fills).
                    # Use current market price as best estimate of close price.
                    # Qty = T1 + T2 if T2 filled (babies excluded — they close
                    # themselves and their PnL is not tracked in parent log).
                    t1_qty  = self.trade1.qty if self.trade1 else 0.0
                    t2_qty  = (self.trade2.qty
                               if (self.trade2 and self.trade2.is_filled) else 0.0)
                    est_qty = round(t1_qty + t2_qty, 3)

                    if est_qty > 0 and self.entry_price > 0:
                        if self.direction == "SHORT":
                            pnl = (self.entry_price - price) * est_qty
                        else:
                            pnl = (price - self.entry_price) * est_qty
                        result = "WIN" if pnl >= 0 else "LOSS"
                        if pnl >= 0:
                            self.wins += 1
                        else:
                            self.losses += 1
                        self.total_pnl += pnl
                        self._log_trade(result, pnl, price, est_qty)
                        print(f"  [monitor] External close logged:"
                              f" entry={self.entry_price:,.1f}"
                              f"  close~={price:,.1f}  qty={est_qty}"
                              f"  PnL=${pnl:.4f}")

                    # ── Cleanup ───────────────────────────────────────────────
                    if self.babies:
                        self.babies.kill_all()
                    if self.trade1 and not self.trade1.is_filled:
                        self.trade1.cancel()
                    if self.trade2 and not self.trade2.is_filled:
                        self.trade2.cancel()
                    if self.tp_order and not self.tp_order.is_filled:
                        self.tp_order.cancel()
                    try:
                        await asyncio.to_thread(cancel_all_orders, SYMBOL)
                    except Exception:
                        pass
                    self.trade1 = self.trade2 = self.tp_order = self.babies = None
                    self.trail_stage = "none"
                    self.best_price_seen = None
                    self.state = "WATCHING"
                    self._write_state()
                    return
            except Exception:
                pass   # if check fails, continue normal monitoring

        # ── Check Trade 1 fill → spawn babies ────────────────────────────────
        if self.trade1 and not self.trade1.is_filled and not self.babies:
            await self.trade1._check_fill()
            if self.trade1.is_filled and not self.babies:
                fill_px  = self.trade1.fill_price or self.entry_price
                self.entry_price = fill_px   # actual exchange fill, not candle close
                notional = self.risk_usdt * LEVERAGE
                pos_side = "SHORT" if self.direction == "SHORT" else "LONG"
                self.babies = BabyLayer(
                    symbol=SYMBOL, direction=self.direction,
                    entry_price=fill_px, parent_notional=notional,
                    pos_side=pos_side, sl_price=self.sl_price,
                )
                await self.babies.start()

                # Place parent TP order (fixed price, no OB tracking)
                close_side = "BUY" if self.direction == "SHORT" else "SELL"
                self.tp_order = SmartLimitOrder(
                    symbol=SYMBOL, side=close_side,
                    qty=self.trade1.qty,
                    pos_side=pos_side, label="Trade1_TP",
                    track_ob=False, fixed_price=self.tp_price,
                )
                await self.tp_order.place()

        # ── Update trailing stop ─────────────────────────────────────────────
        await self._update_trail(price)

        # ── Check if TP / SL hit — only after Trade1 has confirmed filled ────────
        # If Trade1 was never filled (order rejected / never executed), there is
        # no open position. Triggering an exit here would log a phantom trade.
        if not (self.trade1 and self.trade1.is_filled):
            return

        if self.direction == "SHORT" and price <= self.tp_price:
            print(f"  [monitor] TP reached @ {price:,.1f} ✓")
            await self._on_tp_hit()
            return

        if self.direction == "LONG" and price >= self.tp_price:
            print(f"  [monitor] TP reached @ {price:,.1f} ✓")
            await self._on_tp_hit()
            return

        if self.direction == "SHORT" and price >= self.sl_price:
            print(f"  [monitor] SL hit @ {price:,.1f}  ✗  (sl={self.sl_price:,.1f})")
            await self._on_sl_hit()
            return

        if self.direction == "LONG" and price <= self.sl_price:
            print(f"  [monitor] SL hit @ {price:,.1f}  ✗  (sl={self.sl_price:,.1f})")
            await self._on_sl_hit()
            return

    async def _on_tp_hit(self):
        """TP reached: close position at best price, kill babies, log trade."""
        close_price, close_qty = await self._emergency_close("TP")

        if close_qty == 0:
            # No position found on exchange — Trade1 was never actually filled.
            # Do not log a phantom trade; just return to WATCHING.
            print("  [engine] TP: no position on exchange — trade was not opened, nothing logged.")
            self.state = "WATCHING"
            return

        self.wins += 1

        if close_price > 0 and close_qty > 0:
            # Actual PnL from real exchange fill and full position size
            if self.direction == "SHORT":
                profit = (self.entry_price - close_price) * close_qty
            else:
                profit = (close_price - self.entry_price) * close_qty
            profit = max(profit, 0.0)
            print(f"  [engine] Actual TP PnL: entry={self.entry_price:,.1f}"
                  f"  close={close_price:,.1f}  qty={close_qty}  → ${profit:.4f}")
        else:
            # Fallback estimate (T1 only, candle-close entry) — less accurate
            profit = abs(self.tp_price - self.entry_price) * (self.risk_usdt * LEVERAGE / self.entry_price)
            print(f"  [engine] Fallback TP estimate: ${profit:.4f}")

        self.total_pnl += profit
        self._log_trade("WIN", profit, close_price, close_qty)
        self.state = "WATCHING"
        print(f"  [engine] → WATCHING  |  PnL this trade = ${profit:.4f}"
              f"  |  Total wins={self.wins}  losses={self.losses}")

    async def _on_sl_hit(self):
        """SL hit: close full position at best price, cancel everything, log."""
        close_price, close_qty = await self._emergency_close("SL")

        if close_qty == 0:
            # No position found on exchange — Trade1 was never actually filled.
            print("  [engine] SL: no position on exchange — trade was not opened, nothing logged.")
            self.state = "WATCHING"
            return

        self.losses += 1

        if close_price > 0 and close_qty > 0:
            # Actual PnL from real exchange fill and full position size
            if self.direction == "SHORT":
                loss = (close_price - self.entry_price) * close_qty
            else:
                loss = (self.entry_price - close_price) * close_qty
            loss = max(loss, 0.0)
            print(f"  [engine] Actual SL PnL: entry={self.entry_price:,.1f}"
                  f"  close={close_price:,.1f}  qty={close_qty}  → -${loss:.4f}")
        else:
            # Fallback estimate (T1 only, candle-close entry) — less accurate
            loss = abs(self.sl_price - self.entry_price) * (self.risk_usdt * LEVERAGE / self.entry_price)
            print(f"  [engine] Fallback SL estimate: ${loss:.4f}")

        self.total_pnl -= loss
        self._log_trade("LOSS", -loss, close_price, close_qty)
        self.state = "WATCHING"
        print(f"  [engine] → WATCHING  |  Loss = ${loss:.4f}"
              f"  |  Total wins={self.wins}  losses={self.losses}")

    async def _emergency_close(self, reason: str) -> tuple:
        """
        Close the full position using a SmartLimitOrder (maker, chases best price).
        Kill babies first, cancel all pending orders.
        Returns (actual_close_price, actual_qty) for accurate PnL calculation.
        Falls back to (0.0, 0.0) if the close order can't be confirmed.
        """
        print(f"  [engine] Emergency close — reason={reason}")

        # 1. Kill all baby orders and all unfilled entry/TP orders immediately
        if self.babies:
            self.babies.kill_all()
        if self.trade1 and not self.trade1.is_filled:
            self.trade1.cancel()   # stop OB-chasing loop before it places more orders
        if self.trade2 and not self.trade2.is_filled:
            self.trade2.cancel()
        if self.tp_order and not self.tp_order.is_filled:
            self.tp_order.cancel()

        # 2. Cancel all open orders on the symbol (safety net)
        try:
            await asyncio.to_thread(cancel_all_orders, SYMBOL)
        except Exception:
            pass

        # 3. Check actual position size (babies may have added to it) and close it
        actual_close_price = 0.0
        actual_close_qty   = 0.0
        try:
            pos_data = await asyncio.to_thread(get_position_risk, SYMBOL)
            positions = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
            pos_side_str = "SHORT" if self.direction == "SHORT" else "LONG"
            close_side   = "BUY"  if self.direction == "SHORT" else "SELL"

            for pos in positions:
                if pos.get("positionSide", "") == pos_side_str:
                    total_qty = abs(float(pos.get("positionAmt", 0)))
                    if total_qty > 0:
                        close_order = SmartLimitOrder(
                            symbol=SYMBOL, side=close_side, qty=total_qty,
                            pos_side=pos_side_str,
                            label=f"close_{reason}", track_ob=True,
                        )
                        await close_order.place()
                        await close_order.wait_fill(timeout=120)
                        actual_close_price = close_order.fill_price or 0.0
                        actual_close_qty   = total_qty
                        print(f"  [engine] Position closed {total_qty} BTC"
                              f" @ ~{actual_close_price:,.1f}")
                    break
        except Exception as e:
            print(f"  [engine] close error: {e}")

        # Reset trade state
        self.trade1          = None
        self.trade2          = None
        self.tp_order        = None
        self.babies          = None
        self.trail_stage     = "none"
        self.best_price_seen = None

        return actual_close_price, actual_close_qty

    async def _print_position_status(self):
        """Print a short status line while in IN_TRADE state."""
        try:
            price = await asyncio.to_thread(get_price, SYMBOL)
            dist_to_tp = abs(price - self.tp_price)
            dist_to_sl = abs(price - self.sl_price)
            print(f"  [status] {self.direction}  price={price:,.1f}"
                  f"  tp={self.tp_price:,.1f} ({dist_to_tp:.1f} away)"
                  f"  sl={self.sl_price:,.1f} ({dist_to_sl:.1f} away)")
        except Exception:
            pass
        self._write_state()

    def _write_state(self):
        """Write current engine state to algo_state.json for the dashboard."""
        try:
            data = {
                "state":       self.state,
                "direction":   self.direction or "",
                "entry":       self.entry_price,
                "tp":          self.tp_price,
                "sl":          self.sl_price,
                "atr":         self.atr,
                "confidence":  self.confidence,
                "trail_stage": self.trail_stage,
                "best_price":  self.best_price_seen,
                "risk_usdt":   self.risk_usdt,
                "mode":        "TRIAL",
                "updated_at":  datetime.now().isoformat(),
            }
            with open(ALGO_STATE_JSON, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ── TRADE LOG ─────────────────────────────────────────────────────────────

    def _log_trade(self, result: str, pnl: float,
                   actual_close_price: float = 0.0, actual_close_qty: float = 0.0):
        """Append trade result to trade_log.json for review."""
        self.trade_count += 1
        entry = {
            "id":                self.trade_count,
            "time":              datetime.now().isoformat(),
            "symbol":            SYMBOL,
            "direction":         self.direction,
            "entry":             self.entry_price,
            "actual_close":      round(actual_close_price, 1) if actual_close_price else None,
            "actual_qty":        actual_close_qty or None,
            "tp":                self.tp_price,
            "sl":                self.sl_price,
            "atr":               self.atr,
            "confidence":        self.confidence,
            "risk_usdt":         self.risk_usdt,
            "q_zones":           len(self.q_zones),
            "q_total_usd":       self.q_total_usd,
            "mode":              "TRIAL",
            "result":            result,
            "pnl_usd":           round(pnl, 6),
            "pnl_pct":           round(pnl / self.risk_usdt * 100, 4) if self.risk_usdt else 0,
            "total_pnl":         round(self.total_pnl, 6),
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
        print(f"  [log] Trade #{self.trade_count} logged → {TRADE_LOG_JSON.name}")


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 6: STARTUP & MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

HEATMAP_SCRIPT  = ROOT_DIR / "data_fetching" / "binance_liq_heatmap.py"
HEATMAP_REFRESH = 5 * 60    # refresh interval in seconds (5 minutes)


async def heatmap_refresh_loop():
    """
    Keeps the liquidation heatmap up-to-date automatically.
    Always runs once at startup (zones are anchored to current price),
    then repeats every HEATMAP_REFRESH seconds (default 30 min).

    Why 30 min is fine: zones are built from 30 days of OI history —
    they don't shift significantly in minutes. Current price is always
    fetched live each tick regardless of heatmap age.
    """
    first_run = True
    while True:
        if first_run:
            print(f"  [heatmap] Fetching fresh liquidation zones at startup ...")
            first_run = False
        else:
            print(f"  [heatmap] 5-min refresh — updating liquidation zones ...")

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(HEATMAP_SCRIPT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace")
            # Print only the key summary lines (skip the per-zone table)
            for line in output.splitlines():
                if any(kw in line for kw in
                       ["Current Price", "Total modeled", "Data saved",
                        "Chart saved", "failed", "Error", "Traceback"]):
                    print(f"  [heatmap] {line.strip()}")
            if proc.returncode == 0:
                print(f"  [heatmap] Zones refreshed OK")
            else:
                print(f"  [heatmap] Script exited with code {proc.returncode}")
        except Exception as e:
            print(f"  [heatmap] Refresh error: {e}")

        await asyncio.sleep(HEATMAP_REFRESH)


async def startup():
    """Run once at startup: set leverage, print account balance."""
    print("  [startup] Setting leverage ...")
    try:
        await asyncio.to_thread(set_leverage, SYMBOL, LEVERAGE)
        print(f"  [startup] Leverage = {LEVERAGE}x  on {SYMBOL}")
    except Exception as e:
        print(f"  [startup] Leverage set failed: {e}  (may already be set)")

    try:
        bal = await asyncio.to_thread(get_balances)
        balances = bal if isinstance(bal, list) else bal.get("data", [])
        for b in balances:
            asset  = b.get("asset", "")
            wallet = float(b.get("balance", 0) or b.get("walletBalance", 0))
            if wallet > 0:
                print(f"  [startup] Balance: {wallet:.4f} {asset}")
    except Exception as e:
        print(f"  [startup] Balance fetch failed: {e}")


async def main():
    """Entry point. Runs all async loops concurrently."""
    print("=" * 60)
    print("  LIQ ALGO -- Liquidation Zone Reversal Engine")
    print(f"  Symbol     : {SYMBOL}")
    print(f"  Leverage   : {LEVERAGE}x")
    print(f"  Mode       : TRIAL  (min capital, full logging)")
    print(f"  Risk/trade : wallet × {BALANCE_UTILIZATION:.0%}  capped at ${MIN_RISK_USDT}–${MAX_RISK_USDT}")
    print(f"  Approach   : within {APPROACH_PCT}% of zone")
    print(f"  TP / SL    : {TP_ATR_MULT}x / {SL_ATR_MULT}x ATR")
    print("=" * 60)
    print()
    print("  NOTE: whale_monitor.py should be running in another terminal.")
    print("        Heatmap auto-refreshes every 5 min internally.")
    print()

    await startup()

    engine = TradingEngine()

    # Run all loops concurrently:
    #   - heatmap_refresh_loop: keeps zone data fresh (every 30 min)
    #   - minute_loop:          scans zones, fires entry triggers
    #   - position_monitor_loop: checks SL/TP every 3s while in trade
    await asyncio.gather(
        heatmap_refresh_loop(),
        engine.minute_loop(),
        engine.position_monitor_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  [liq_algo] Stopped by user. Ctrl+C")
        sys.exit(0)
