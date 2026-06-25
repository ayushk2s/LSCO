"""
backtest_ict_v1.py  --  ICT / SMC Liquidity Sweep Reversal  v1
==============================================================
Pure price-action engine — NO EMA, NO volume filters.
Only ATR is used for position sizing (not signal generation).

STRATEGY (LONG-only reversal):
  1. Find swing lows on 4h bars (20-bar lookback)
  2. Detect a "Liquidity Sweep": bar.low dips BELOW swing low AND bar.close is BACK ABOVE it
     → this is a stop-hunt / Sellside Liquidity grab
  3. Detect the "Order Block": last bearish candle before the sweep move began
     → this becomes the re-entry zone if price pulls back
  4. Detect "Fair Value Gaps" (FVGs): 3-candle imbalances on the 4h
     → nearest bullish FVG above entry becomes the Partial TP target
  5. Enter LONG on the sweep bar's close (or next bar open)
  6. SL: below the sweep low by one small buffer
  7. Partial TP (50%): nearest bullish FVG or 2× ATR
  8. Trail remaining with 0.5× ATR lag
  9. Hard TP: 4× ATR (or nearest bearish OB / swing high)

EXECUTION: 1m bars, gap-aware fills (same as v8).
SHORT SL formula correct (if shorts ever added).
NEVER modifies account_data.py or liq_algo.py.
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10

# ICT-specific parameters
SWING_LOOKBACK   = 20     # bars each side for swing detection
SWEEP_MIN_PCT    = 0.0001 # price must go at least 0.01% through the level
SWEEP_MAX_PCT    = 0.006  # and no more than 0.6% through (avoid large gap moves)
SL_BUFFER_MULT   = 0.30   # SL = sweep_low - 0.30*ATR below the swept level
PARTIAL_TP_ATR   = 2.0    # partial TP at 2× ATR (wider than LZR, letting it breathe)
TRAIL_DIST_MULT  = 0.75   # trail lag: 0.75× ATR (slightly wider trail)
HARD_TP_ATR      = 4.0    # hard TP at 4× ATR (or nearest swing high)
ATR_PERIOD       = 14
FVG_MIN_PCT      = 0.0003 # minimum FVG size (0.03% of price)
OB_LOOKBACK      = 50     # bars to look back for order blocks
COOLDOWN_BARS    = 3      # bars of 4h cooldown after a loss (= 12h)
MAX_OPEN_TRADES  = 1      # only one position at a time
SWING_REUSE_MAX  = 2      # don't sweep the same swing low more than twice

# Execution
CRYPTO_FEE_RT   = 0.0004
CRYPTO_SLIP_PCT = 0.0003
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20
WARMUP_BARS           = SWING_LOOKBACK * 2 + 5

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_crypto(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def build_4h(df_1m):
    agg = dict(open=("open","first"), high=("high","max"),
               low=("low","min"), close=("close","last"), vol=("vol","sum"))
    return df_1m.resample("4h").agg(**agg).dropna()


# ─── ATR ──────────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


# ─── ICT structure detection ──────────────────────────────────────────────────

def find_swing_lows(df, lookback=SWING_LOOKBACK):
    """Return dict of {bar_index: price} for all swing lows."""
    swings = {}
    for i in range(lookback, len(df) - lookback):
        lo = df["low"].iloc[i]
        if lo == df["low"].iloc[i - lookback:i + lookback + 1].min():
            swings[i] = lo
    return swings


def find_swing_highs(df, lookback=SWING_LOOKBACK):
    swings = {}
    for i in range(lookback, len(df) - lookback):
        hi = df["high"].iloc[i]
        if hi == df["high"].iloc[i - lookback:i + lookback + 1].max():
            swings[i] = hi
    return swings


def detect_fvgs(df):
    """
    Fair Value Gaps (3-candle imbalances).
    Bullish FVG: df["high"][i-1] < df["low"][i+1]  → gap between candle i-1 top and candle i+1 bottom
    Bearish FVG: df["low"][i-1]  > df["high"][i+1]
    Returns list of dicts: {ts, type, low, high, filled}
    """
    fvgs = []
    for i in range(1, len(df) - 1):
        # Bullish FVG
        if df["high"].iloc[i - 1] < df["low"].iloc[i + 1]:
            gap_low  = df["high"].iloc[i - 1]
            gap_high = df["low"].iloc[i + 1]
            size_pct = (gap_high - gap_low) / df["close"].iloc[i]
            if size_pct >= FVG_MIN_PCT:
                fvgs.append({
                    "ts": df.index[i], "idx": i,
                    "type": "bullish",
                    "low": gap_low, "high": gap_high,
                    "size_pct": round(size_pct * 100, 4),
                })
        # Bearish FVG
        elif df["low"].iloc[i - 1] > df["high"].iloc[i + 1]:
            gap_low  = df["high"].iloc[i + 1]
            gap_high = df["low"].iloc[i - 1]
            size_pct = (gap_high - gap_low) / df["close"].iloc[i]
            if size_pct >= FVG_MIN_PCT:
                fvgs.append({
                    "ts": df.index[i], "idx": i,
                    "type": "bearish",
                    "low": gap_low, "high": gap_high,
                    "size_pct": round(size_pct * 100, 4),
                })
    return fvgs


def detect_order_blocks(df, lookback=OB_LOOKBACK):
    """
    Bullish OB: last bearish candle (close < open) before a strong bullish impulse.
    Bearish OB: last bullish candle before a strong bearish impulse.
    Returns list of dicts: {ts, idx, type, low, high}
    """
    obs = []
    atr = calc_atr(df)
    for i in range(2, len(df) - 2):
        # Bullish OB: bearish candle followed by 2+ bullish candles with cumulative range > 1.5 ATR
        if df["close"].iloc[i] < df["open"].iloc[i]:  # bearish candle
            move = df["high"].iloc[i + 1:i + 3].max() - df["low"].iloc[i]
            if move >= 1.5 * atr.iloc[i]:
                obs.append({
                    "ts": df.index[i], "idx": i,
                    "type": "bullish",
                    "low": min(df["close"].iloc[i], df["open"].iloc[i]),
                    "high": max(df["close"].iloc[i], df["open"].iloc[i]),
                })
        # Bearish OB: bullish candle followed by 2+ bearish candles
        elif df["close"].iloc[i] > df["open"].iloc[i]:
            move = df["high"].iloc[i] - df["low"].iloc[i + 1:i + 3].min()
            if move >= 1.5 * atr.iloc[i]:
                obs.append({
                    "ts": df.index[i], "idx": i,
                    "type": "bearish",
                    "low": min(df["close"].iloc[i], df["open"].iloc[i]),
                    "high": max(df["close"].iloc[i], df["open"].iloc[i]),
                })
    return obs


# ─── 1m executor — gap-aware (SHORT SL fixed) ─────────────────────────────────

def _exec_1m(df_1m, m_start, entry_px, direction,
              sl_px, partial_tp_px, hard_tp_px,
              trade_atr, qty, fee_side, slip_pct):
    half           = qty / 2
    state          = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct

    for m_idx in range(m_start, len(df_1m)):
        mr = df_1m.iloc[m_idx]
        closed, gross, exit_px = False, None, 0.0

        if state == "full":
            if direction == "LONG":
                if mr["open"] <= sl_px:
                    exit_px, gross, closed = mr["open"], (mr["open"] - entry_px) * qty, True
                elif mr["open"] >= partial_tp_px:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["open"]
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["high"] >= partial_tp_px:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["high"]
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["low"] <= sl_px:
                    exit_px, gross, closed = sl_px, (sl_px - entry_px) * qty, True
            else:  # SHORT — SL fixed
                if mr["open"] >= sl_px:
                    exit_px, gross, closed = mr["open"], (entry_px - mr["open"]) * qty, True
                elif mr["open"] <= partial_tp_px:
                    partial_locked = (entry_px - partial_tp_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["open"]
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["low"] <= partial_tp_px:
                    partial_locked = (entry_px - partial_tp_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["low"]
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["high"] >= sl_px:
                    exit_px, gross, closed = sl_px, (entry_px - sl_px) * qty, True

        elif state == "partial":
            if direction == "LONG":
                old_trail = trail_sl
                if mr["open"] <= old_trail:
                    exit_px, gross, closed = mr["open"], partial_locked + (mr["open"] - entry_px) * half, True
                elif mr["open"] >= hard_tp_px:
                    exit_px, gross, closed = hard_tp_px, partial_locked + (hard_tp_px - entry_px) * half, True
                else:
                    running_ext = max(running_ext, mr["high"])
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    if mr["low"] <= old_trail:
                        exit_px, gross, closed = old_trail, partial_locked + (old_trail - entry_px) * half, True
                    elif mr["high"] >= hard_tp_px:
                        exit_px, gross, closed = hard_tp_px, partial_locked + (hard_tp_px - entry_px) * half, True
            else:
                old_trail = trail_sl
                if mr["open"] >= old_trail:
                    exit_px, gross, closed = mr["open"], partial_locked + (entry_px - mr["open"]) * half, True
                elif mr["open"] <= hard_tp_px:
                    exit_px, gross, closed = hard_tp_px, partial_locked + (entry_px - hard_tp_px) * half, True
                else:
                    running_ext = min(running_ext, mr["low"])
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    if mr["high"] >= old_trail:
                        exit_px, gross, closed = old_trail, partial_locked + (entry_px - old_trail) * half, True
                    elif mr["low"] <= hard_tp_px:
                        exit_px, gross, closed = hard_tp_px, partial_locked + (entry_px - hard_tp_px) * half, True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * fee_side
            slip_acc += exit_px * exit_qty * slip_pct
            return (df_1m.index[m_idx], gross, exit_px, fee_acc, slip_acc, state)

    return None


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_ict_backtest(symbol, df_1m, df_4h):
    FEE_SIDE     = CRYPTO_FEE_RT / 2
    m_timestamps = df_1m.index
    atr_s        = calc_atr(df_4h)

    # Pre-compute structure
    print(f"    computing FVGs and OBs...", flush=True)
    fvgs = detect_fvgs(df_4h)
    obs  = detect_order_blocks(df_4h)

    # Build lookup: fvg by their bar index
    fvg_by_idx = defaultdict(list)
    for f in fvgs:
        fvg_by_idx[f["idx"]].append(f)

    ob_by_idx = defaultdict(list)
    for o in obs:
        ob_by_idx[o["idx"]].append(o)

    balance       = INITIAL_BALANCE
    min_balance   = INITIAL_BALANCE * MIN_BAL_RATIO
    equity        = [balance]
    trades        = []
    last_trigger  = -999
    cooldown_until = 0
    sweep_counts   = defaultdict(int)  # track how many times each swing has been swept

    i = 0
    n = len(df_4h)

    while i < n:
        if i < WARMUP_BARS:
            equity.append(balance); i += 1; continue
        if i <= cooldown_until:
            equity.append(balance); i += 1; continue
        if balance < INITIAL_BALANCE * MIN_BAL_RATIO:
            equity.append(balance); i += 1; continue

        row = df_4h.iloc[i]
        atr = atr_s.iloc[i]
        if atr <= 0 or np.isnan(atr):
            equity.append(balance); i += 1; continue

        # ── Find most recent swing lows in look-back window ───────────────────
        lb = max(0, i - SWING_LOOKBACK * 3)
        window_lows = []
        for j in range(lb, i - SWING_LOOKBACK):
            lo = df_4h["low"].iloc[j]
            # Is this a swing low? (lowest in ±SWING_LOOKBACK bars)
            left_lo  = df_4h["low"].iloc[max(0, j - SWING_LOOKBACK):j].min() if j > 0 else lo
            right_lo = df_4h["low"].iloc[j + 1:j + SWING_LOOKBACK + 1].min() if j < i else lo
            if lo <= left_lo and lo <= right_lo:
                window_lows.append((j, lo))

        # ── Find most recent swing highs ──────────────────────────────────────
        window_highs = []
        for j in range(lb, i - SWING_LOOKBACK):
            hi = df_4h["high"].iloc[j]
            left_hi  = df_4h["high"].iloc[max(0, j - SWING_LOOKBACK):j].max() if j > 0 else hi
            right_hi = df_4h["high"].iloc[j + 1:j + SWING_LOOKBACK + 1].max() if j < i else hi
            if hi >= left_hi and hi >= right_hi:
                window_highs.append((j, hi))

        # ── LONG setup: current bar sweeps a swing low then closes above it ───
        signal_long  = False
        signal_short = False
        swept_level  = None
        sweep_low_px = None
        sweep_idx_key = None

        if window_lows:
            # Check the 3 most recent swing lows
            recent_lows = sorted(window_lows, key=lambda x: x[0], reverse=True)[:3]
            for (sl_bar_idx, sl_price) in recent_lows:
                sweep_through = sl_price - row["low"]    # how far below swing low did we go
                sweep_pct     = sweep_through / sl_price

                already_swept = sweep_counts[(sl_bar_idx, sl_price)]
                if already_swept >= SWING_REUSE_MAX:
                    continue

                if (SWEEP_MIN_PCT <= sweep_pct <= SWEEP_MAX_PCT   # swept through the level
                        and row["close"] > sl_price                # but closed BACK ABOVE it
                        and row["open"]  > sl_price):              # opened above it too (not a gap down)
                    signal_long  = True
                    swept_level  = sl_price
                    sweep_low_px = row["low"]
                    sweep_idx_key = (sl_bar_idx, sl_price)
                    break

        # ── SHORT setup (buyide sweep) ────────────────────────────────────────
        # Uncomment to enable shorts (currently LONG-only per analysis)
        # if window_highs and not signal_long:
        #     recent_highs = sorted(window_highs, key=lambda x: x[0], reverse=True)[:3]
        #     for (sh_bar_idx, sh_price) in recent_highs:
        #         sweep_through = row["high"] - sh_price
        #         sweep_pct = sweep_through / sh_price
        #         if (SWEEP_MIN_PCT <= sweep_pct <= SWEEP_MAX_PCT
        #                 and row["close"] < sh_price and row["open"] < sh_price):
        #             signal_short = True
        #             swept_level = sh_price
        #             sweep_idx_key = (sh_bar_idx, sh_price)
        #             break

        if not signal_long and not signal_short:
            equity.append(balance); i += 1; continue

        if i == last_trigger:
            equity.append(balance); i += 1; continue
        if i + 1 >= n:
            equity.append(balance); i += 1; continue

        # ── Compute entry, SL, TPs ────────────────────────────────────────────
        direction = "LONG" if signal_long else "SHORT"
        entry_px  = df_4h["open"].iloc[i + 1]

        if direction == "LONG":
            sl_px     = sweep_low_px - SL_BUFFER_MULT * atr
            sl_dist   = entry_px - sl_px
            if sl_dist <= 0:
                equity.append(balance); i += 1; continue

            # Find nearest bullish FVG above entry as partial TP
            future_bullish_fvgs = [f for f in fvgs
                                   if f["type"] == "bullish"
                                   and f["idx"] < i
                                   and f["low"] > entry_px]
            if future_bullish_fvgs:
                nearest_fvg = min(future_bullish_fvgs, key=lambda f: f["low"])
                partial_tp_px = nearest_fvg["low"]  # bottom of the FVG = first target
                # Make sure partial TP is at least 1.5 ATR away
                if partial_tp_px < entry_px + 1.5 * atr:
                    partial_tp_px = entry_px + PARTIAL_TP_ATR * atr
            else:
                partial_tp_px = entry_px + PARTIAL_TP_ATR * atr

            # Find nearest swing high above entry as hard TP
            highs_above = [h for (_, h) in window_highs if h > entry_px + 0.5 * atr]
            if highs_above:
                hard_tp_px = min(highs_above)
                # Ensure hard TP is at least 3 ATR away
                if hard_tp_px < entry_px + 3.0 * atr:
                    hard_tp_px = entry_px + HARD_TP_ATR * atr
            else:
                hard_tp_px = entry_px + HARD_TP_ATR * atr

        else:  # SHORT
            sl_px = sweep_level + SL_BUFFER_MULT * atr
            sl_dist = sl_px - entry_px
            if sl_dist <= 0:
                equity.append(balance); i += 1; continue
            partial_tp_px = entry_px - PARTIAL_TP_ATR * atr
            hard_tp_px    = entry_px - HARD_TP_ATR    * atr

        # Position sizing: fixed fractional on SL distance
        eff_bal = min(balance, MAX_BALANCE)
        qty     = min((eff_bal * RISK_PCT) / sl_dist,
                      MAX_POSITION_NOTIONAL / entry_px)

        # ── Execute on 1m ─────────────────────────────────────────────────────
        entry_ts = df_4h.index[i + 1]
        m_start  = int(m_timestamps.searchsorted(entry_ts))

        info = _exec_1m(df_1m, m_start, entry_px, direction,
                        sl_px, partial_tp_px, hard_tp_px,
                        atr, qty, FEE_SIDE, CRYPTO_SLIP_PCT)
        if info is None:
            equity.append(balance); i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip, final_state = info
        j = max(int(df_4h.index.searchsorted(close_ts, side="right")) - 1, i + 1)
        j = min(j, n - 1)

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        # Update sweep tracking
        sweep_counts[sweep_idx_key] += 1

        # Cooldown after loss
        if result == "LOSS":
            cooldown_until = j + COOLDOWN_BARS

        last_trigger = i
        duration_h = (close_ts - entry_ts).total_seconds() / 3600

        # Identify what FVG / OB influenced this trade
        fvg_note = "none"
        if future_bullish_fvgs if direction == "LONG" else []:
            fvg_note = f"FVG@{partial_tp_px:.4f}"

        trades.append({
            "ts": df_4h.index[i], "close_ts": close_ts, "dir": direction,
            "entry": round(entry_px, 6), "exit": round(exit_px, 6),
            "sl": round(sl_px, 6), "partial_tp": round(partial_tp_px, 6),
            "hard_tp": round(hard_tp_px, 6),
            "swept_level": round(swept_level, 6) if swept_level else 0,
            "sweep_low": round(sweep_low_px, 6) if sweep_low_px else 0,
            "atr_pct": round(atr / entry_px * 100, 3),
            "gross": round(gross, 4), "fee": round(total_fee, 4),
            "net": round(net, 4), "result": result,
            "duration_h": round(duration_h, 1),
            "fvg_target": fvg_note,
            "balance": round(balance, 4),
        })

        equity.append(bal_open)
        for _ in range(j - i - 1):
            equity.append(bal_open)
        equity.append(balance)
        i = j + 1

    # ─── Stats ────────────────────────────────────────────────────────────────
    if not trades:
        return None, []

    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t["result"] == "WIN"]
    loses  = df_t[df_t["result"] == "LOSS"]
    nw     = wins["net"].sum()
    nl     = abs(loses["net"].sum())
    net_pf = round(nw / nl, 3) if nl > 0 else float("inf")
    wr     = round(len(wins) / len(df_t) * 100, 1)
    years  = (df_4h.index[-1] - df_4h.index[0]).days / 365.25
    cagr   = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    eq     = pd.Series(equity)
    max_dd = round(((eq - eq.cummax()) / eq.cummax() * 100).min(), 2)
    avg_dur = df_t["duration_h"].mean()

    stats = {
        "trades": len(df_t), "wins": len(wins), "losses": len(loses),
        "win_rate": wr, "net_pf": net_pf,
        "net_pnl": round(df_t["net"].sum(), 2),
        "final_bal": round(balance, 2),
        "cagr": round(cagr, 2), "max_dd": max_dd,
        "avg_duration_h": round(avg_dur, 1),
        "n_fvgs": len(fvgs), "n_obs": len(obs),
    }
    return stats, trades


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 72
    print("=" * W)
    print("  ICT / SMC  LIQUIDITY SWEEP REVERSAL  v1")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print("  Signals: Sellside Liquidity Sweep + close-back-above")
    print("  NO indicators — pure price structure (OB / FVG / Swings)")
    print("  Execution: 1m bars, gap-aware fills")
    print("  LONG-only  |  4h signal  |  ATR for sizing only")
    print()

    all_stats  = {}
    all_trades = []

    for sym in CRYPTO_SYMBOLS:
        print(f"  ── {sym} ──────────────────────────────────")
        try:
            df_1m = load_crypto(sym)
            df_4h = build_4h(df_1m)
            fvg_count = len(detect_fvgs(df_4h))
            ob_count  = len(detect_order_blocks(df_4h))
            print(f"    bars: {len(df_1m):,} 1m  |  {len(df_4h):,} 4h  "
                  f"|  FVGs: {fvg_count}  OBs: {ob_count}", flush=True)
            stats, trades = run_ict_backtest(sym, df_1m, df_4h)
            if stats:
                all_stats[sym] = stats
                for t in trades:
                    t["symbol"] = sym
                all_trades.extend(trades)
                sign = "+" if stats["cagr"] >= 0 else ""
                star = " ★" if stats["net_pf"] > 1.0 else ""
                print(f"    {stats['trades']} trades  WR {stats['win_rate']}%  "
                      f"NetPF {stats['net_pf']}  "
                      f"CAGR {sign}{stats['cagr']:.1f}%  "
                      f"MaxDD {stats['max_dd']:.1f}%  "
                      f"AvgDur {stats['avg_duration_h']:.0f}h{star}")
            else:
                print(f"    no trades generated")
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
        print()

    if not all_stats:
        print("  No results."); return

    print("=" * W)
    print("  SUMMARY")
    print("=" * W)
    print(f"  {'Symbol':<10} {'Trades':>6} {'WR%':>6} {'NetPF':>7} {'CAGR%':>7} {'MaxDD%':>8} {'AvgDur':>8}")
    print("-" * W)
    for sym in CRYPTO_SYMBOLS:
        if sym not in all_stats:
            continue
        s    = all_stats[sym]
        sign = "+" if s["cagr"] >= 0 else ""
        star = " ★" if s["net_pf"] > 1.0 else ""
        print(f"  {sym:<10} {s['trades']:>6} {s['win_rate']:>6.1f} "
              f"{s['net_pf']:>7.3f} {sign}{s['cagr']:>6.1f}% "
              f"{s['max_dd']:>8.2f}% {s['avg_duration_h']:>7.0f}h{star}")

    avg_cagr = np.mean([s["cagr"] for s in all_stats.values()])
    avg_npf  = np.mean([s["net_pf"] for s in all_stats.values()
                        if s["net_pf"] != float("inf")])
    pos      = sum(1 for s in all_stats.values() if s["cagr"] > 0)
    print("-" * W)
    print(f"  {'AVG':<10} {'':>6} {'':>6} {avg_npf:>7.3f} {avg_cagr:>+7.1f}% "
          f"{'':>8}  ({pos}/{len(all_stats)} profitable)")

    print()
    print("  WHAT THIS TELLS YOU vs LZR:")
    print("  ─────────────────────────────────────────────────────────────")
    print("  LZR v8 4h LONG-only:  BTC+1.2%  ETH+4.6%  SOL+20.9%")
    print("  ICT Sweep v1 (above): see results")
    print("  Key diff: ICT requires price to SWEEP through a level AND")
    print("  close back above — a much stronger reversal confirmation.")
    print("  Zones: swing lows (natural liquidity pools, not arbitrary levels).")

    # Comparison with LZR
    print()
    print("  STRUCTURE DETECTED (first symbol as example):")
    if CRYPTO_SYMBOLS[0] in all_stats:
        s = all_stats[CRYPTO_SYMBOLS[0]]
        print(f"  {CRYPTO_SYMBOLS[0]}: {s['n_fvgs']} Fair Value Gaps  |  {s['n_obs']} Order Blocks")

    # Save CSV
    if all_trades:
        df_out = pd.DataFrame(all_trades)
        out = OUTPUT_DIR / "backtest_ict_v1.csv"
        df_out.to_csv(out, index=False)
        print(f"\n  Saved → {out}")
    print("=" * W)


if __name__ == "__main__":
    main()
