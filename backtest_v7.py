"""
backtest_v7.py  --  Liquidation Zone Reversal  v7  (1m EXECUTION MODE)
=======================================================================
Builds directly on v6 (STRICT REALISTIC MODE) with one architectural upgrade:

  v6  evaluated SL / TP / trailing stops on 1h candles.
  v7  evaluates them on the underlying 1m candles.

WHY THIS MATTERS
════════════════
On 1h candles, if both the stop and a target fall within the same hour
the intrabar order is unknowable from OHLC data.  v6 applied the
conservative "worst case first" rule — but that is still an approximation.

On 1m candles:
  • Each candle is 60× smaller.  The probability that SL AND TP are
    both breached inside a single 1m bar is ≈ negligible in practice.
  • The trailing stop is updated every minute — far more accurate.
  • Partial-TP exits fire at the exact 1m bar where the target is crossed.
  • Consecutive-candle intrabar order uncertainty is essentially eliminated.

SIGNAL GENERATION  (unchanged from v6)
  • All entry signals detected on 1h candles (ATR, EMA, zones, volume).
  • Entry price = open of the very next 1h bar (same as v6).
  • After signal, 1h loop jumps forward to the bar containing the close.

EXECUTION  (new in v7)
  • SL / partial-TP / trail / hard-TP evaluated bar-by-bar on 1m data.
  • The conservative tie-break rule (trail > hard_tp) is kept: even within
    a single 1m bar, if BOTH trail and hard_tp fire we take the trail (worst).
  • Forex pairs fall back to 1h execution — yfinance only gives 1h data.

ALL v6 FIXES INHERITED
  [FIX 1] Zone snapshot  : < not <=  (no partial-bar look-ahead)
  [FIX 2] Conservative   : trail wins over hard_tp when both triggered
  [FIX 3] Exit slippage  : charged on entry + partial exit + final exit
  [FIX 4] Notional cap   : $500K per trade / $200K balance ceiling
  [FIX 5] WIN/LOSS by    : net P&L, not gross
  [FIX 6] Profit Factor  : gross PF + net PF both reported
  [FIX 7] Balance floor  : halt entries below 20% of initial
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Strategy parameters ──────────────────────────────────────────────────────
DATA_DIR        = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR      = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10

SL_MULT         = 0.75
PARTIAL_TP_MULT = 1.0
TRAIL_DIST_MULT = 0.5
HARD_TP_MULT    = 3.0

APPROACH_PCT    = 0.008
TOUCH_BUF       = 0.003
ATR_PERIOD      = 14
SWING_LOOKBACK  = 20
MIN_ZONE_GAP    = 0.005

VOL_MULT        = 1.8
VOL_LOOKBACK    = 20
EMA4H_PERIOD    = 20
EMA4H_LOOKBACK  = 3
ZONE_MAX_TOUCH  = 2
ZONE_WINDOW     = 168

COOLDOWN_LOSS   = 10
COOLDOWN_WIN    = 3

# ─── Costs ────────────────────────────────────────────────────────────────────
CRYPTO_FEE_RT   = 0.0004
CRYPTO_SLIP_PCT = 0.0003

FOREX_SPREAD = {
    "EURUSD": 0.00010, "GBPUSD": 0.00010, "USDJPY": 0.00010,
    "AUDUSD": 0.00015, "USDCAD": 0.00020, "EURJPY": 0.00015, "GBPJPY": 0.00025,
}
FOREX_SLIP_PCT = 0.0001

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FOREX_PAIRS    = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}

# ─── STRICT MODE ──────────────────────────────────────────────────────────────
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20
WARMUP_BARS           = ATR_PERIOD * 4   # 56 bars


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_crypto_raw(symbol):
    """Return (df_1m, df_1h) from raw 1m CSV.  Both share the same source."""
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    df = df.set_index("ts").sort_index()
    df_1m = df.copy()
    df_1h = df_1m.resample("1h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),  close=("close", "last"), vol=("vol", "sum")
    ).dropna()
    return df_1m, df_1h


def download_forex_1h(ticker, name):
    import yfinance as yf
    print(f"    Downloading {name} ({ticker})...")
    raw = yf.download(ticker, period="2y", interval="1h",
                      auto_adjust=True, progress=False, actions=False)
    if raw.empty:
        raw = yf.Ticker(ticker).history(period="2y", interval="1h", auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns={"Open": "open", "High": "high",
                              "Low": "low", "Close": "close"})
    df.index.name = "ts"
    df = df[["open", "high", "low", "close"]].copy()
    df["vol"] = 0.0
    return df.sort_index().dropna()


def resample_4h(df):
    return df.resample("4h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),  close=("close", "last")
    ).dropna()


def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_4h):
    n, lb = len(df_4h), SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_4h["high"].iloc[i] == df_4h["high"].iloc[i - lb:i + lb + 1].max():
            highs.append(df_4h["high"].iloc[i])
        if df_4h["low"].iloc[i] == df_4h["low"].iloc[i - lb:i + lb + 1].min():
            lows.append(df_4h["low"].iloc[i])
    levels, merged = sorted(set(highs + lows)), []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── 1m trade executor ────────────────────────────────────────────────────────

def _exec_1m(df_1m, m_start, entry_px, direction,
              sl_px, partial_tp_px, hard_tp_px,
              trade_atr, qty, fee_side, slip_pct):
    """
    Run the full trade state machine on 1m candles.

    Conservative rule (inherited from v6): when BOTH trail and hard_tp
    trigger inside the same 1m bar, assume trail hit first (worst case).

    Returns (close_ts, gross, exit_px, total_fee, total_slip, final_state)
    or None if the data ends before the trade closes.
    """
    half    = qty / 2
    state   = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct

    n = len(df_1m)
    for m_idx in range(m_start, n):
        mr     = df_1m.iloc[m_idx]
        gross  = None
        closed = False
        exit_px = 0.0

        if state == "full":
            if direction == "LONG":
                sl_hit  = mr["low"]  <= sl_px
                ptp_hit = mr["high"] >= partial_tp_px
                if sl_hit:
                    gross   = (sl_px - entry_px) * qty
                    exit_px = sl_px
                    closed  = True
                elif ptp_hit:
                    partial_locked  = (partial_tp_px - entry_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["high"]
                    trail_sl        = max(entry_px,
                                          running_ext - TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"
            else:  # SHORT
                sl_hit  = mr["high"] >= sl_px
                ptp_hit = mr["low"]  <= partial_tp_px
                if sl_hit:
                    gross   = (entry_px - sl_px) * qty    # SHORT PnL: entry - exit
                    exit_px = sl_px
                    closed  = True
                elif ptp_hit:
                    partial_locked  = (entry_px - partial_tp_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["low"]
                    trail_sl        = min(entry_px,
                                          running_ext + TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"

        elif state == "partial":
            if direction == "LONG":
                old_trail   = trail_sl
                running_ext = max(running_ext, mr["high"])
                trail_sl    = max(entry_px,
                                   running_ext - TRAIL_DIST_MULT * trade_atr)

                trail_hit = mr["low"]  <= old_trail
                htp_hit   = mr["high"] >= hard_tp_px

                # Conservative: trail (worst case) wins when both fire same bar
                if trail_hit:
                    gross   = partial_locked + (old_trail   - entry_px) * half
                    exit_px = old_trail
                    closed  = True
                elif htp_hit:
                    gross   = partial_locked + (hard_tp_px  - entry_px) * half
                    exit_px = hard_tp_px
                    closed  = True

            else:  # SHORT partial
                old_trail   = trail_sl
                running_ext = min(running_ext, mr["low"])
                trail_sl    = min(entry_px,
                                   running_ext + TRAIL_DIST_MULT * trade_atr)

                trail_hit = mr["high"] >= old_trail
                htp_hit   = mr["low"]  <= hard_tp_px

                # Conservative: trail wins when both fire same bar
                if trail_hit:
                    gross   = partial_locked + (entry_px - old_trail)  * half
                    exit_px = old_trail
                    closed  = True
                elif htp_hit:
                    gross   = partial_locked + (entry_px - hard_tp_px) * half
                    exit_px = hard_tp_px
                    closed  = True

        if closed:
            exit_qty   = qty if state == "full" else half
            exit_fee   = exit_px * exit_qty * fee_side
            exit_slip  = exit_px * exit_qty * slip_pct
            total_fee  = fee_acc  + exit_fee
            total_slip = slip_acc + exit_slip
            return (df_1m.index[m_idx], gross, exit_px,
                    total_fee, total_slip, state)

    return None   # data ended before trade closed


# ─── 1h fallback executor (Forex — no 1m data) ───────────────────────────────

def _exec_1h(df_1h, start_idx, entry_px, direction,
              sl_px, partial_tp_px, hard_tp_px,
              trade_atr, qty, fee_side, slip_pct):
    """
    Identical logic to _exec_1m but iterates over 1h bars.
    Used for Forex where only 1h OHLCV is available.
    Returns (close_1h_idx, gross, exit_px, total_fee, total_slip, final_state)
    or None if data ends before close.
    """
    half    = qty / 2
    state   = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct

    for idx in range(start_idx, len(df_1h)):
        row    = df_1h.iloc[idx]
        gross  = None
        closed = False
        exit_px = 0.0

        if state == "full":
            if direction == "LONG":
                if row["low"] <= sl_px:
                    gross   = (sl_px - entry_px) * qty
                    exit_px = sl_px
                    closed  = True
                elif row["high"] >= partial_tp_px:
                    partial_locked  = (partial_tp_px - entry_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = row["high"]
                    trail_sl        = max(entry_px,
                                          running_ext - TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"
            else:
                if row["high"] >= sl_px:
                    gross   = (entry_px - sl_px) * qty    # SHORT PnL: entry - exit
                    exit_px = sl_px
                    closed  = True
                elif row["low"] <= partial_tp_px:
                    partial_locked  = (entry_px - partial_tp_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = row["low"]
                    trail_sl        = min(entry_px,
                                          running_ext + TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"

        elif state == "partial":
            if direction == "LONG":
                old_trail   = trail_sl
                running_ext = max(running_ext, row["high"])
                trail_sl    = max(entry_px,
                                   running_ext - TRAIL_DIST_MULT * trade_atr)
                trail_hit = row["low"]  <= old_trail
                htp_hit   = row["high"] >= hard_tp_px
                if trail_hit:
                    gross   = partial_locked + (old_trail   - entry_px) * half
                    exit_px = old_trail
                    closed  = True
                elif htp_hit:
                    gross   = partial_locked + (hard_tp_px  - entry_px) * half
                    exit_px = hard_tp_px
                    closed  = True
            else:
                old_trail   = trail_sl
                running_ext = min(running_ext, row["low"])
                trail_sl    = min(entry_px,
                                   running_ext + TRAIL_DIST_MULT * trade_atr)
                trail_hit = row["high"] >= old_trail
                htp_hit   = row["low"]  <= hard_tp_px
                if trail_hit:
                    gross   = partial_locked + (entry_px - old_trail)  * half
                    exit_px = old_trail
                    closed  = True
                elif htp_hit:
                    gross   = partial_locked + (entry_px - hard_tp_px) * half
                    exit_px = hard_tp_px
                    closed  = True

        if closed:
            exit_qty   = qty if state == "full" else half
            exit_fee   = exit_px * exit_qty * fee_side
            exit_slip  = exit_px * exit_qty * slip_pct
            total_fee  = fee_acc  + exit_fee
            total_slip = slip_acc + exit_slip
            return (idx, gross, exit_px, total_fee, total_slip, state)

    return None


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(symbol, df_1h, fee_rt, slip_pct, category,
                 use_vol_filter, df_1m=None):
    """
    df_1m : raw 1m DataFrame for crypto (enables minute-resolution execution).
             Pass None for Forex — falls back to 1h execution.
    """
    FEE_SIDE = fee_rt / 2
    use_1m   = df_1m is not None and len(df_1m) > 0

    df_4h = resample_4h(df_1h)
    atr_s = calc_atr(df_1h)

    df_4h["ema"]      = df_4h["close"].ewm(span=EMA4H_PERIOD, adjust=False).mean()
    df_4h["ema_lag"]  = df_4h["ema"].shift(1)
    df_4h["ema_prev"] = df_4h["ema"].shift(1 + EMA4H_LOOKBACK)
    ema_now_1h  = df_4h["ema_lag"].reindex(df_1h.index,  method="ffill")
    ema_prev_1h = df_4h["ema_prev"].reindex(df_1h.index, method="ffill")

    # Zone snapshots — strict < (FIX 1, inherited)
    n_days    = len(df_1h) // 24 + 2
    zone_snap = []
    for d in range(n_days):
        end_ts  = df_1h.index[min(d * 24, len(df_1h) - 1)]
        past_4h = df_4h[df_4h.index < end_ts]
        if len(past_4h) < SWING_LOOKBACK * 2 + 5:
            zone_snap.append([])
        else:
            zone_snap.append(find_zones(past_4h.iloc[-200:]))

    balance          = INITIAL_BALANCE
    min_balance      = INITIAL_BALANCE * MIN_BAL_RATIO
    equity           = [balance]
    trades           = []
    zone_cooldown    = {}
    zone_touches     = defaultdict(list)
    last_trigger_bar = -999

    # Pre-build 1m index lookup array for fast searchsorted (crypto only)
    if use_1m:
        m_timestamps = df_1m.index

    # ── Main loop: WHILE (allows index jumps after trade close) ───────────────
    i = 0
    n_1h = len(df_1h)

    while i < n_1h:

        # Warmup: no signals, just fill equity
        if i < WARMUP_BARS:
            equity.append(balance)
            i += 1
            continue

        ts    = df_1h.index[i]
        row   = df_1h.iloc[i]
        atr   = atr_s.iloc[i]
        price = row["close"]

        if atr <= 0 or np.isnan(atr):
            equity.append(balance)
            i += 1
            continue

        # ── Zone lookup ───────────────────────────────────────────────────────
        day_idx = i // 24
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            equity.append(balance)
            i += 1
            continue
        zones = zone_snap[day_idx]

        zones_below = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
        zones_above = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]
        if zones_below:
            near_zone = max(zones_below)
            near_dir  = "LONG"
        elif zones_above:
            near_zone = min(zones_above)
            near_dir  = "SHORT"
        else:
            equity.append(balance)
            i += 1
            continue

        # Guards
        if zone_cooldown.get(near_zone, 0) > i:
            equity.append(balance)
            i += 1
            continue
        if i == last_trigger_bar:
            equity.append(balance)
            i += 1
            continue
        if balance < min_balance:          # FIX 7
            equity.append(balance)
            i += 1
            continue

        # ── Trigger check ─────────────────────────────────────────────────────
        if near_dir == "LONG":
            triggered = (row["low"] <= near_zone * (1 + TOUCH_BUF)
                         and row["close"] > near_zone)
        else:
            triggered = (row["high"] >= near_zone * (1 - TOUCH_BUF)
                         and row["close"] < near_zone)
        if not triggered:
            equity.append(balance)
            i += 1
            continue

        # Filter 1: Volume spike
        if use_vol_filter:
            vol_avg = df_1h["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
            if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
                equity.append(balance)
                i += 1
                continue

        # Filter 2: 4h EMA trend
        ema_now  = ema_now_1h.iloc[i]
        ema_prev = ema_prev_1h.iloc[i]
        if not (pd.isna(ema_now) or pd.isna(ema_prev)):
            if near_dir == "LONG"  and ema_now <= ema_prev:
                equity.append(balance)
                i += 1
                continue
            if near_dir == "SHORT" and ema_now >= ema_prev:
                equity.append(balance)
                i += 1
                continue

        # Filter 3: Zone freshness
        recent = [b for b in zone_touches[near_zone] if i - b <= ZONE_WINDOW]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance)
            i += 1
            continue

        # Need at least one more bar
        if i + 1 >= n_1h:
            equity.append(balance)
            i += 1
            continue

        # ── Position sizing (FIX 4) ───────────────────────────────────────────
        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance)
            i += 1
            continue

        eff_bal  = min(balance, MAX_BALANCE) if MAX_BALANCE else balance
        risk_usd = eff_bal * RISK_PCT
        qty      = risk_usd / sl_dist

        entry_px  = df_1h["open"].iloc[i + 1]
        trade_atr = atr

        if MAX_POSITION_NOTIONAL:
            qty = min(qty, MAX_POSITION_NOTIONAL / entry_px)

        if near_dir == "LONG":
            sl_px         = entry_px - SL_MULT         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT         * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        # ── Execute trade on 1m (crypto) or 1h (forex) ───────────────────────
        entry_ts = df_1h.index[i + 1]

        if use_1m:
            m_start    = int(m_timestamps.searchsorted(entry_ts))
            close_info = _exec_1m(df_1m, m_start, entry_px, near_dir,
                                   sl_px, partial_tp_px, hard_tp_px,
                                   trade_atr, qty, FEE_SIDE, slip_pct)
            if close_info is None:
                equity.append(balance)
                i += 1
                continue
            close_ts, gross, exit_px, total_fee, total_slip, final_state = close_info
            # Map 1m close timestamp → 1h bar index
            close_1h_idx = int(df_1h.index.searchsorted(close_ts, side="right")) - 1
        else:
            close_info = _exec_1h(df_1h, i + 1, entry_px, near_dir,
                                   sl_px, partial_tp_px, hard_tp_px,
                                   trade_atr, qty, FEE_SIDE, slip_pct)
            if close_info is None:
                equity.append(balance)
                i += 1
                continue
            close_1h_idx, gross, exit_px, total_fee, total_slip, final_state = close_info
            close_ts = df_1h.index[close_1h_idx]

        # Clamp to valid range
        close_1h_idx = max(close_1h_idx, i + 1)
        close_1h_idx = min(close_1h_idx, n_1h - 1)

        # ── Settle trade ──────────────────────────────────────────────────────
        net      = gross - total_fee - total_slip
        result   = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        zone_cooldown[near_zone]  = close_1h_idx + (COOLDOWN_LOSS if result == "LOSS"
                                                      else COOLDOWN_WIN)
        zone_touches[near_zone].append(i)
        last_trigger_bar = i

        trades.append({
            "ts":           ts,
            "close_ts":     close_ts,
            "dir":          near_dir,
            "entry":        round(entry_px,        6),
            "exit":         round(exit_px,         6),
            "qty":          round(qty,              8),
            "notional":     round(entry_px * qty,   2),
            "gross":        round(gross,             4),
            "fee":          round(total_fee,         4),
            "slip":         round(total_slip,        4),
            "net":          round(net,               4),
            "result":       result,
            "balance_open": round(bal_open,          4),
            "balance":      round(balance,           4),
        })

        # ── Fill equity array for the trade's 1h span ─────────────────────────
        # Bar i (signal bar)            : old balance
        equity.append(bal_open)
        # Bars i+1 .. close_1h_idx-1   : trade running, balance unchanged
        for _ in range(close_1h_idx - i - 1):
            equity.append(bal_open)
        # Bar close_1h_idx              : updated balance after close
        equity.append(balance)

        # Jump the 1h index past the trade
        i = close_1h_idx + 1

    # ─── Statistics ───────────────────────────────────────────────────────────
    if not trades:
        return _empty(symbol, category, equity, df_1h)

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]

    gross_pnl   = df_t["gross"].sum()
    total_fees  = df_t["fee"].sum()
    total_slip  = df_t["slip"].sum()
    total_costs = total_fees + total_slip
    net_pnl     = df_t["net"].sum()

    gross_wins = wins["gross"].sum()
    gross_loss = abs(loses["gross"].sum())
    net_wins   = wins["net"].sum()
    net_loss   = abs(loses["net"].sum())

    gross_pf = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    net_pf   = round(net_wins   / net_loss,   3) if net_loss   > 0 else float("inf")
    win_rate = round(len(wins) / len(df_t) * 100, 1)

    final_bal = balance
    years     = (df_1h.index[-1] - df_1h.index[0]).days / 365.25
    total_ret = (final_bal / INITIAL_BALANCE - 1) * 100
    cagr      = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0

    eq       = pd.Series(equity)
    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    max_dd   = round(dd_pct.min(), 2)
    max_dd_usd = round((eq - roll_max).min(), 2)

    in_dd      = (dd_pct < 0).astype(int)
    dd_dur_max = 0
    dd_cur     = 0
    for v in in_dd:
        dd_cur     = dd_cur + 1 if v else 0
        dd_dur_max = max(dd_dur_max, dd_cur)

    tpy      = len(df_t) / max(years, 0.1)
    pct_rets = (df_t["net"] / df_t["balance_open"]).values
    sharpe   = round((pct_rets.mean() / pct_rets.std() * np.sqrt(max(tpy, 1)))
                     if pct_rets.std() > 0 else 0, 2)
    calmar   = round(cagr / abs(max_dd) if max_dd != 0 else 0, 2)

    max_ws = max_ls = cur_ws = cur_ls = 0
    for r in df_t["result"]:
        if r == "WIN":
            cur_ws += 1; cur_ls = 0
        else:
            cur_ls += 1; cur_ws = 0
        max_ws = max(max_ws, cur_ws)
        max_ls = max(max_ls, cur_ls)

    return {
        "symbol":       symbol,
        "category":     category,
        "date_from":    df_1h.index[0].date(),
        "date_to":      df_1h.index[-1].date(),
        "years":        round(years, 2),
        "trades":       len(df_t),
        "wins":         len(wins),
        "losses":       len(loses),
        "win_rate":     win_rate,
        "gross_pf":     gross_pf,
        "net_pf":       net_pf,
        "gross_pnl":    round(gross_pnl,    2),
        "total_fees":   round(total_fees,   2),
        "total_slip":   round(total_slip,   2),
        "total_costs":  round(total_costs,  2),
        "net_pnl":      round(net_pnl,      2),
        "initial_bal":  INITIAL_BALANCE,
        "final_bal":    round(final_bal,    2),
        "total_ret":    round(total_ret,    2),
        "cagr":         round(cagr,         2),
        "max_dd":       max_dd,
        "max_dd_usd":   max_dd_usd,
        "dd_dur_bars":  dd_dur_max,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "avg_win":      round(wins["net"].mean(),       4) if len(wins)  else 0,
        "avg_loss":     round(loses["net"].mean(),      4) if len(loses) else 0,
        "best_trade":   round(df_t["net"].max(),        4),
        "worst_trade":  round(df_t["net"].min(),        4),
        "avg_fee_pt":   round(total_fees  / len(df_t),  4),
        "avg_slip_pt":  round(total_slip  / len(df_t),  4),
        "avg_cost_pt":  round(total_costs / len(df_t),  4),
        "avg_notional": round(df_t["notional"].mean(),  2),
        "max_win_str":  max_ws,
        "max_los_str":  max_ls,
        "equity":       equity,
        "trades_df":    df_t,
    }


def _empty(symbol, category, equity, df_1h):
    years = (df_1h.index[-1] - df_1h.index[0]).days / 365.25 if len(df_1h) > 1 else 0
    return {
        "symbol": symbol, "category": category, "trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "gross_pf": 0, "net_pf": 0, "gross_pnl": 0, "total_fees": 0,
        "total_slip": 0, "total_costs": 0, "net_pnl": 0, "initial_bal": INITIAL_BALANCE,
        "final_bal": INITIAL_BALANCE, "total_ret": 0, "cagr": 0,
        "max_dd": 0, "max_dd_usd": 0, "dd_dur_bars": 0, "sharpe": 0, "calmar": 0,
        "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
        "avg_fee_pt": 0, "avg_slip_pt": 0, "avg_cost_pt": 0, "avg_notional": 0,
        "max_win_str": 0, "max_los_str": 0, "years": round(years, 2),
        "date_from": df_1h.index[0].date() if len(df_1h) else "",
        "date_to":   df_1h.index[-1].date() if len(df_1h) else "",
        "equity": equity, "trades_df": pd.DataFrame(),
    }


# ─── Report helpers ───────────────────────────────────────────────────────────

W = 72
def div(c="="): print(c * W)
def hdiv():     print("-" * W)
def row2(a, av, b, bv, w=28): print(f"  {a:<{w}} {av!s:<18}  {b:<{w}} {bv!s}")
def row1(a, av, w=28):         print(f"  {a:<{w}} {av}")


def print_block(r):
    yr  = r["years"]
    tpy = r["trades"] / yr if yr > 0 else 0
    div()
    print(f"  {r['symbol']}  [{r['category']}]  "
          f"{r['date_from']} → {r['date_to']}  ({yr:.2f} yr)")
    div()
    print("\n  PERFORMANCE")
    hdiv()
    row2("Total Trades",         f"{r['trades']}  ({tpy:.0f}/yr)",
         "Win / Loss",           f"{r['wins']} / {r['losses']}")
    row2("Win Rate",             f"{r['win_rate']}%",
         "Gross PF",             f"{r['gross_pf']}")
    row2("Net PF (after costs)", f"{r['net_pf']}",
         "",                     "")
    row2("Avg Win (net)",        f"${r['avg_win']:+.2f}",
         "Avg Loss (net)",       f"${r['avg_loss']:+.2f}")
    row2("Best Trade",           f"${r['best_trade']:+.2f}",
         "Worst Trade",          f"${r['worst_trade']:+.2f}")
    row2("Max Win Streak",       f"{r['max_win_str']}",
         "Max Loss Streak",      f"{r['max_los_str']}")
    print(f"\n  PnL BREAKDOWN  (compounding {RISK_PCT*100:.0f}% risk | "
          f"notional cap ${MAX_POSITION_NOTIONAL:,.0f} | "
          f"balance cap ${MAX_BALANCE:,.0f})")
    hdiv()
    row1("Gross PnL (before all costs)", f"${r['gross_pnl']:>+14,.2f}")
    row1("  Exchange fees",              f"${-r['total_fees']:>+14,.2f}")
    row1("  Slippage (ALL fills)",       f"${-r['total_slip']:>+14,.2f}")
    row1("  Total Costs",                f"${-r['total_costs']:>+14,.2f}")
    hdiv()
    row1("NET PnL (total)",              f"${r['net_pnl']:>+14,.2f}")
    row1("Initial Balance",              f"${r['initial_bal']:>14,.2f}")
    row1("Final   Balance",              f"${r['final_bal']:>14,.2f}")
    row1("Total Return",                 f"{r['total_ret']:>+13.2f}%")
    row1(f"CAGR ({yr:.1f} yrs)",         f"{r['cagr']:>+13.2f}%/yr")
    print("\n  RISK METRICS")
    hdiv()
    row2("Max Drawdown %",  f"{r['max_dd']:.2f}%",
         "Max DD ($)",      f"${r['max_dd_usd']:,.2f}")
    row2("DD Duration",     f"{r['dd_dur_bars']} bars (~{r['dd_dur_bars']//24}d)",
         "Calmar",          f"{r['calmar']}")
    row2("Sharpe",          f"{r['sharpe']}",
         "",                "")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_crypto, results_forex = [], []

    div("="); div(" ")
    print("  LIQUIDATION ZONE REVERSAL  v7  —  1m EXECUTION MODE")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    div(" "); div("=")
    print(f"""
  v7 UPGRADE  (all v6 strict-mode fixes inherited)
    Signal detection   : 1h candles  (unchanged from v6)
    Trade execution    : 1m candles  (NEW — eliminates intrabar ambiguity)
    Forex execution    : 1h fallback (no 1m data available for FX pairs)

  STRATEGY
    Risk/trade         : {RISK_PCT*100:.0f}% of balance  (compounding)
    Max balance cap    : ${MAX_BALANCE:,.0f}
    Max notional/trade : ${MAX_POSITION_NOTIONAL:,.0f}
    SL / Partial TP    : {SL_MULT}× / {PARTIAL_TP_MULT}× ATR
    Trail / Hard Cap   : {TRAIL_DIST_MULT}× / {HARD_TP_MULT}× ATR
    Warmup             : {WARMUP_BARS} bars

  STRICT MODE FIXES ACTIVE (inherited from v6)
    [FIX 1] Zone snapshot  : < not <=
    [FIX 2] Conservative   : trail wins over hard_tp when both triggered
    [FIX 3] Slippage       : {CRYPTO_SLIP_PCT*100:.2f}% on entry + partial + final exit
    [FIX 4] Position caps  : ${MAX_POSITION_NOTIONAL:,.0f} notional / ${MAX_BALANCE:,.0f} balance
    [FIX 5] WIN/LOSS by    : net P&L
    [FIX 6] Both PFs       : gross and net reported
    [FIX 7] Balance floor  : ${INITIAL_BALANCE*MIN_BAL_RATIO:.0f}
""")

    div("=")
    print("  CRYPTO  (1m execution)")
    div("=")
    for sym in CRYPTO_SYMBOLS:
        try:
            t0 = datetime.now()
            print(f"\n  [{sym}]  loading 1m data...")
            df_1m, df_1h = load_crypto_raw(sym)
            print(f"  [{sym}]  {len(df_1m):,} 1m bars  →  {len(df_1h):,} 1h bars  "
                  f"{df_1h.index[0].date()} → {df_1h.index[-1].date()}  running...")
            r = run_backtest(sym, df_1h, CRYPTO_FEE_RT, CRYPTO_SLIP_PCT,
                             "Crypto", use_vol_filter=True, df_1m=df_1m)
            elapsed = (datetime.now() - t0).seconds
            results_crypto.append(r)
            print(f"  → {r['trades']} trades  WR {r['win_rate']}%  "
                  f"GrossPF {r['gross_pf']}  NetPF {r['net_pf']}  "
                  f"CAGR {r['cagr']:+.1f}%  Final ${r['final_bal']:,.2f}  "
                  f"({elapsed}s)")
        except Exception as e:
            print(f"  ERROR {sym}: {e}")

    div("=")
    print("  FOREX  (1h fallback — no 1m data)")
    div("=")
    for name, ticker in FOREX_PAIRS.items():
        try:
            df = download_forex_1h(ticker, name)
            spread = FOREX_SPREAD.get(name, 0.0002)
            r = run_backtest(name, df, fee_rt=spread * 2,
                             slip_pct=FOREX_SLIP_PCT,
                             category="Forex", use_vol_filter=False,
                             df_1m=None)
            results_forex.append(r)
            print(f"  [{name}]  {r['trades']} trades  WR {r['win_rate']}%  "
                  f"CAGR {r['cagr']:+.1f}%  Final ${r['final_bal']:,.2f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    all_results = results_crypto + results_forex
    if not all_results:
        print("No results."); return

    print("\n\n")
    div("="); div(" ")
    print("  DETAILED RESULTS — CRYPTO")
    div(" "); div("=")
    for r in results_crypto:
        print_block(r)

    div("="); div(" ")
    print("  DETAILED RESULTS — FOREX")
    div(" "); div("=")
    for r in results_forex:
        print_block(r)

    # ─── Summary table ────────────────────────────────────────────────────────
    div("="); div(" ")
    print("  SUMMARY  (sorted by CAGR)")
    div(" "); div("=")
    hdr = (f"  {'Symbol':<10} {'Tr/yr':>6} {'WR%':>6} {'GrossPF':>8} {'NetPF':>7}"
           f" {'Net$':>10} {'Final$':>10} {'CAGR%':>7} {'MaxDD%':>7} {'Sharpe':>7}")
    print(hdr); hdiv()
    for r in sorted(all_results, key=lambda x: x["cagr"], reverse=True):
        yr  = r["years"]
        tpy = r["trades"] / yr if yr > 0 else 0
        print(f"  {r['symbol']:<10} {tpy:>6.0f} {r['win_rate']:>6.1f}"
              f" {r['gross_pf']:>8.3f} {r['net_pf']:>7.3f}"
              f" {r['net_pnl']:>+10,.2f} {r['final_bal']:>10,.2f}"
              f" {r['cagr']:>+7.1f} {r['max_dd']:>7.2f} {r['sharpe']:>7.2f}")
    hdiv()
    for label, rlist in [("CRYPTO", results_crypto), ("FOREX", results_forex)]:
        active = [r for r in rlist if r["trades"] > 0 and r["years"] > 0]
        if not active: continue
        avg_tpy  = np.mean([r["trades"] / r["years"] for r in active])
        avg_wr   = np.mean([r["win_rate"] for r in active])
        avg_gpf  = np.mean([r["gross_pf"] for r in active if r["gross_pf"] != float("inf")])
        avg_npf  = np.mean([r["net_pf"]   for r in active if r["net_pf"]   != float("inf")])
        avg_cagr = np.mean([r["cagr"]     for r in active])
        avg_dd   = np.mean([r["max_dd"]   for r in active])
        avg_sh   = np.mean([r["sharpe"]   for r in active])
        print(f"  {label:<10} {avg_tpy:>6.0f} {avg_wr:>6.1f}"
              f" {avg_gpf:>8.3f} {avg_npf:>7.3f}"
              f" {'':>10} {'':>10}"
              f" {avg_cagr:>+7.1f} {avg_dd:>7.2f} {avg_sh:>7.2f}")

    # ─── v6 vs v7 comparison ─────────────────────────────────────────────────
    div("="); div(" ")
    print("  v6 vs v7 — WHAT CHANGED")
    div(" "); div("=")
    print("""
  v6  evaluated SL/TP/trail on 1h candles.
      When BOTH stop and target fall in the same 1h bar, the engine
      used the conservative "worst case first" rule — but that is an
      approximation because the true intrabar path is unknowable.

  v7  evaluates SL/TP/trail on 1m candles.
      Each 1m bar is 60× smaller than a 1h bar.  The probability that
      both stop AND target are hit inside the same 1m bar is negligible.
      Trailing stop also updates every minute instead of every hour.

  Expected impact on results:
    • Wins: slightly lower — trail stop is tighter, catches earlier reversals
    • Losses: slightly lower — SL fires earlier in the hour (less gap risk)
    • Net effect: modest reduction in CAGR (more realistic), better Sharpe
    • The underlying edge (Net PF > 2) should be largely unchanged
""")

    csv = OUTPUT_DIR / "backtest_v7_full.csv"
    rows = [{k: r[k] for k in r if k not in ("equity", "trades_df")}
            for r in all_results]
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  Results saved → {csv}")
    div("=")


if __name__ == "__main__":
    main()
