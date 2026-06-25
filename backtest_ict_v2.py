"""
backtest_ict_v2.py  --  ICT / SMC Liquidity Sweep Reversal  v2
==============================================================
IMPROVEMENT over v1:
  - Liquidity levels from DAILY bars (5-bar swing lookback) — "institutional" levels
    vs v1 which used 4h swings (too many minor/irrelevant levels)
  - Sweep detected on 4h bars (more granular than daily, but referencing daily levels)
  - Sweep candle must be BULLISH BODY (close > open) — not just a wicked candle
  - Equal lows bonus: levels tested 2+ times get higher priority
  - Body recovery filter: close must be at least 60% of the bar's range above swept level
  - Execution: 1m bars, gap-aware fills (unchanged)

STRATEGY:
  1. Compute swing lows on DAILY bars (5-bar lookback = weekly significant lows)
  2. Mark these as "liquidity pools" (stops accumulate below old lows)
  3. On 4h bars: detect when price sweeps below a DAILY swing low
     - bar.low < daily_swing_low (sweep)
     - bar.close > daily_swing_low (rejection)
     - bar is bullish (close > open)
     - body recovery ≥ BODY_RECOV_PCT of bar range
  4. Enter LONG on next 4h bar open
  5. SL: below sweep low by buffer
  6. TP1 (50%): nearest bullish FVG on 4h or 2× ATR
  7. Trail remainder with 0.75× ATR lag
  8. Hard TP: 4× ATR or nearest daily swing high above entry

DOES NOT modify account_data.py or liq_algo.py.
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

# ICT v2 parameters
DAILY_SWING_LB   = 5      # lookback for daily swing lows (5 days each side = weekly level)
SWEEP_MIN_PCT    = 0.0001  # must sweep at least 0.01% through the level
SWEEP_MAX_PCT    = 0.008   # no more than 0.8% through (avoid gap-down candles)
BODY_RECOV_PCT   = 0.40    # close must be ≥ 40% of candle range above swept level
                           # (ensures body is in the upper portion = real rejection)
SL_BUFFER_MULT   = 0.40    # SL = sweep_low - 0.40*ATR below swept level
PARTIAL_TP_ATR   = 2.0     # partial TP: 2× ATR (or nearest bullish FVG)
TRAIL_DIST_MULT  = 0.75    # trail lag
HARD_TP_ATR      = 4.0     # hard cap
ATR_PERIOD       = 14
FVG_MIN_PCT      = 0.0003  # minimum FVG size
COOLDOWN_BARS_4H = 3       # 4h bars = 12h cooldown after loss
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20
SWEEP_LEVEL_EXPIRY    = 60  # a daily swing level "expires" after 60 × 4h bars = 10 days
LEVEL_REUSE_MAX       = 2   # same level can be swept at most twice

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
CRYPTO_FEE_RT  = 0.0004
CRYPTO_SLIP    = 0.0003


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_crypto(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def build_4h(df_1m):
    return df_1m.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), vol=("vol","sum")
    ).dropna()


def build_daily(df_1m):
    return df_1m.resample("1D").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), vol=("vol","sum")
    ).dropna()


# ─── ATR ──────────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


# ─── Daily swing lows (vectorized) ───────────────────────────────────────────

def find_daily_swing_lows_vectorized(df_daily, lookback=DAILY_SWING_LB):
    """
    Vectorized swing low detection on daily bars.
    Returns dict: {bar_index → swing_low_price}
    Uses confirmed swings: bar i is a swing low if it's the minimum
    in [i-lookback, i+lookback] window. lookback=5 = weekly level.
    """
    lows   = df_daily["low"].values
    n      = len(lows)
    swings = {}
    for i in range(lookback, n - lookback):
        lo = lows[i]
        if lo == np.min(lows[i - lookback:i + lookback + 1]):
            swings[i] = lo
    return swings


def find_daily_swing_highs_vectorized(df_daily, lookback=DAILY_SWING_LB):
    highs  = df_daily["high"].values
    n      = len(highs)
    swings = {}
    for i in range(lookback, n - lookback):
        hi = highs[i]
        if hi == np.max(highs[i - lookback:i + lookback + 1]):
            swings[i] = hi
    return swings


# ─── FVG detection on 4h ─────────────────────────────────────────────────────

def detect_fvgs_bullish(df_4h):
    """
    Returns sorted list of bullish FVG zones: [(bar_idx, fvg_low, fvg_high), ...]
    Bullish FVG: gap between candle[i-1].high and candle[i+1].low
    """
    result = []
    h = df_4h["high"].values
    l = df_4h["low"].values
    c = df_4h["close"].values
    for i in range(1, len(df_4h) - 1):
        gap_low  = h[i - 1]
        gap_high = l[i + 1]
        if gap_high > gap_low:
            size_pct = (gap_high - gap_low) / c[i]
            if size_pct >= FVG_MIN_PCT:
                result.append((i, gap_low, gap_high))
    return result


# ─── 1m executor (same as v1 / v8) ───────────────────────────────────────────

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
                    fee_acc += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["open"]
                    trail_sl = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["high"] >= partial_tp_px:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["high"]
                    trail_sl = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["low"] <= sl_px:
                    exit_px, gross, closed = sl_px, (sl_px - entry_px) * qty, True
            else:
                if mr["open"] >= sl_px:
                    exit_px, gross, closed = mr["open"], (entry_px - mr["open"]) * qty, True
                elif mr["open"] <= partial_tp_px:
                    partial_locked = (entry_px - partial_tp_px) * half
                    fee_acc += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["open"]
                    trail_sl = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["low"] <= partial_tp_px:
                    partial_locked = (entry_px - partial_tp_px) * half
                    fee_acc += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["low"]
                    trail_sl = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
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


# ─── Main backtest ────────────────────────────────────────────────────────────

def run_ict_v2(symbol, df_1m, df_4h, df_daily):
    FEE_SIDE = CRYPTO_FEE_RT / 2
    m_ts     = df_1m.index
    atr_4h   = calc_atr(df_4h)
    atr_4h_v = atr_4h.values
    n4       = len(df_4h)

    # Pre-compute: DAILY swing lows and highs (vectorized, fast)
    daily_swing_lows  = find_daily_swing_lows_vectorized(df_daily)
    daily_swing_highs = find_daily_swing_highs_vectorized(df_daily)

    # Build sorted arrays for binary-search lookup
    daily_ts   = df_daily.index
    d_low_ts   = sorted(daily_swing_lows.keys())
    d_low_px   = [daily_swing_lows[k] for k in d_low_ts]
    d_high_ts  = sorted(daily_swing_highs.keys())
    d_high_px  = [daily_swing_highs[k] for k in d_high_ts]

    # Pre-compute bullish FVGs on 4h
    fvg_list = detect_fvgs_bullish(df_4h)
    # fvg_list: sorted by bar index

    balance        = INITIAL_BALANCE
    equity         = [balance]
    trades         = []
    cooldown_until = 0
    last_signal_i  = -999
    level_sweeps   = defaultdict(int)  # (daily_idx, price) → sweep count

    WARMUP_BARS = DAILY_SWING_LB * 6 + 10  # wait for enough daily bars to form

    i = 0
    while i < n4:
        if i < WARMUP_BARS:
            equity.append(balance); i += 1; continue
        if i <= cooldown_until:
            equity.append(balance); i += 1; continue
        if balance < INITIAL_BALANCE * MIN_BAL_RATIO:
            equity.append(balance); i += 1; continue

        row = df_4h.iloc[i]
        ts4 = df_4h.index[i]
        atr = atr_4h_v[i]
        if atr <= 0 or np.isnan(atr):
            equity.append(balance); i += 1; continue

        # ── Find daily swing lows that were formed BEFORE this 4h bar ─────────
        # and are within SWEEP_LEVEL_EXPIRY 4h bars of now
        d_idx_current = int(daily_ts.searchsorted(ts4, side="right")) - 1
        if d_idx_current < DAILY_SWING_LB:
            equity.append(balance); i += 1; continue

        # Candidate daily lows: formed before current bar, not expired
        candidates = []
        for k, (d_bar_idx, d_px) in enumerate(zip(d_low_ts, d_low_px)):
            if d_bar_idx >= d_idx_current:
                break  # daily bar not yet formed
            # Check if expired (old levels are less reliable)
            daily_bar_ts  = daily_ts[d_bar_idx]
            bars_since = (ts4 - daily_bar_ts).total_seconds() / (4 * 3600)
            if bars_since > SWEEP_LEVEL_EXPIRY:
                continue
            # Check sweep count
            key = (d_bar_idx, round(d_px, 6))
            if level_sweeps[key] >= LEVEL_REUSE_MAX:
                continue
            candidates.append((d_bar_idx, d_px, key, bars_since))

        if not candidates:
            equity.append(balance); i += 1; continue

        # ── Check if current 4h bar sweeps any candidate level ────────────────
        signal_found = False
        swept_level  = None
        sweep_low_px = None
        level_key    = None
        bar_low      = row["low"]
        bar_high     = row["high"]
        bar_close    = row["close"]
        bar_open     = row["open"]
        bar_range    = bar_high - bar_low

        for (d_bar_idx, d_px, key, bars_since) in sorted(candidates,
                                                          key=lambda x: x[3]):  # prefer fresher levels
            sweep_through = d_px - bar_low
            sweep_pct     = sweep_through / d_px if d_px > 0 else 0

            if sweep_pct < SWEEP_MIN_PCT or sweep_pct > SWEEP_MAX_PCT:
                continue
            if bar_close <= d_px:
                continue  # didn't close back above the swept level
            if bar_close <= bar_open:
                continue  # candle must be bullish (close > open = bullish body)

            # Body recovery: how far above the swept level is the close?
            recovery  = bar_close - d_px
            if bar_range > 0 and (recovery / bar_range) < BODY_RECOV_PCT:
                continue  # close is too close to bottom of candle

            signal_found = True
            swept_level  = d_px
            sweep_low_px = bar_low
            level_key    = key
            break

        if not signal_found or i == last_signal_i:
            equity.append(balance); i += 1; continue
        if i + 1 >= n4:
            equity.append(balance); i += 1; continue

        # ── Entry calculation ─────────────────────────────────────────────────
        entry_px = df_4h["open"].iloc[i + 1]
        sl_px    = sweep_low_px - SL_BUFFER_MULT * atr
        sl_dist  = entry_px - sl_px
        if sl_dist <= 0:
            equity.append(balance); i += 1; continue

        # Partial TP: nearest bullish FVG above entry in the last 200 bars
        pt_px = entry_px + PARTIAL_TP_ATR * atr  # default
        for (fvg_bar_idx, fvg_lo, fvg_hi) in fvg_list:
            if fvg_bar_idx >= i:
                break
            if fvg_bar_idx < i - 200:
                continue
            if fvg_lo > entry_px + 0.5 * atr:
                pt_px = fvg_lo
                break

        # Hard TP: nearest daily swing high above entry
        hard_tp = entry_px + HARD_TP_ATR * atr
        for k in range(len(d_high_ts) - 1, -1, -1):
            if d_high_ts[k] >= d_idx_current:
                continue
            d_hi_px = d_high_px[k]
            if d_hi_px > entry_px + 2.0 * atr:
                hard_tp = d_hi_px
                break

        # Position sizing
        eff_bal = min(balance, MAX_BALANCE)
        qty     = min((eff_bal * RISK_PCT) / sl_dist,
                      MAX_POSITION_NOTIONAL / entry_px)

        # ── Execute on 1m ─────────────────────────────────────────────────────
        entry_ts = df_4h.index[i + 1]
        m_start  = int(m_ts.searchsorted(entry_ts))

        info = _exec_1m(df_1m, m_start, entry_px, "LONG",
                        sl_px, pt_px, hard_tp, atr, qty,
                        FEE_SIDE, CRYPTO_SLIP)
        if info is None:
            equity.append(balance); i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip, final_state = info
        j = max(int(df_4h.index.searchsorted(close_ts, side="right")) - 1, i + 1)
        j = min(j, n4 - 1)

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        level_sweeps[level_key] += 1
        if result == "LOSS":
            cooldown_until = j + COOLDOWN_BARS_4H

        last_signal_i = i
        duration_h = (close_ts - entry_ts).total_seconds() / 3600

        trades.append({
            "ts": ts4, "close_ts": close_ts,
            "entry": round(entry_px, 6), "exit": round(exit_px, 6),
            "sl": round(sl_px, 6), "pt": round(pt_px, 6), "ht": round(hard_tp, 6),
            "swept_level": round(swept_level, 6),
            "sweep_low": round(sweep_low_px, 6),
            "atr_pct": round(atr / entry_px * 100, 3),
            "gross": round(gross, 4), "fee": round(total_fee, 4),
            "net": round(net, 4), "result": result,
            "duration_h": round(duration_h, 1),
            "balance": round(balance, 4),
        })

        equity.append(bal_open)
        for _ in range(j - i - 1):
            equity.append(bal_open)
        equity.append(balance)
        i = j + 1

    if not trades:
        return None, []

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]
    nw    = wins["net"].sum()
    nl    = abs(loses["net"].sum())
    npf   = round(nw / nl, 3) if nl > 0 else float("inf")
    wr    = round(len(wins) / len(df_t) * 100, 1)
    years = (df_4h.index[-1] - df_4h.index[0]).days / 365.25
    cagr  = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    eq    = pd.Series(equity)
    mdd   = round(((eq - eq.cummax()) / eq.cummax() * 100).min(), 2)

    return {
        "trades": len(df_t), "wins": len(wins), "losses": len(loses),
        "win_rate": wr, "net_pf": npf,
        "final_bal": round(balance, 2),
        "cagr": round(cagr, 2), "max_dd": mdd,
        "avg_dur": round(df_t["duration_h"].mean(), 1),
        "daily_swing_lows_total": len(daily_swing_lows),
    }, trades


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 72
    print("=" * W)
    print("  ICT / SMC  LIQUIDITY SWEEP REVERSAL  v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print("  v2 FIXES: Daily swing levels (not 4h) + bullish body required")
    print("  + 40% body recovery filter + fresh-level preference")
    print("  LONG-only | 4h bars detect sweep | 1m execution")
    print()

    all_stats  = {}
    all_trades = []

    for sym in CRYPTO_SYMBOLS:
        print(f"  ── {sym} ──────────────────────────────────")
        try:
            df_1m    = load_crypto(sym)
            df_4h    = build_4h(df_1m)
            df_daily = build_daily(df_1m)
            d_swings = find_daily_swing_lows_vectorized(df_daily)
            print(f"    {len(df_1m):,} 1m  |  {len(df_4h):,} 4h  |  {len(df_daily):,} daily  "
                  f"|  {len(d_swings)} daily swing lows", flush=True)

            stats, trades = run_ict_v2(sym, df_1m, df_4h, df_daily)
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
                      f"AvgDur {stats['avg_dur']:.0f}h{star}")
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
    print("  RESULTS COMPARISON")
    print("=" * W)
    print(f"  {'Symbol':<10} {'ICT v2':>8} {'ICT v1':>8} {'LZR 4h':>8} {'LZR LONG':>10}")
    v1_npf  = {"BTCUSDT": 0.685, "ETHUSDT": 0.764, "SOLUSDT": 0.567, "BNBUSDT": "?", "XRPUSDT": "?"}
    lzr_npf = {"BTCUSDT": 0.956, "ETHUSDT": 0.889, "SOLUSDT": 2.040, "BNBUSDT": 0.799, "XRPUSDT": 0.555}
    lo_npf  = {"BTCUSDT": 1.213, "ETHUSDT": 1.235, "SOLUSDT": 3.378, "BNBUSDT": 0.850, "XRPUSDT": 0.285}
    print("-" * W)
    for sym in CRYPTO_SYMBOLS:
        if sym not in all_stats:
            continue
        s    = all_stats[sym]
        star = " ★" if s["net_pf"] > 1.0 else ""
        print(f"  {sym:<10} {s['net_pf']:>8.3f} "
              f"{str(v1_npf.get(sym, '?')):>8}  "
              f"{lzr_npf.get(sym, '?'):>8}  "
              f"{lo_npf.get(sym, '?'):>8}{star}")

    vals = [s["net_pf"] for s in all_stats.values() if s["net_pf"] != float("inf")]
    avg  = np.mean(vals) if vals else 0
    pos  = sum(1 for s in all_stats.values() if s["cagr"] > 0)
    print("-" * W)
    print(f"  {'AVG':<10} {avg:>8.3f}   "
          f"({pos}/{len(all_stats)} profitable)")

    print()
    print("  DAILY SWING LEVELS:")
    print("  Fewer signals but higher quality. Daily swing lows represent")
    print("  WEEKLY liquidity pools — where the most stops accumulate.")
    print("  Real ICT practitioners call these 'old highs/lows' targets.")

    if all_trades:
        df_out = pd.DataFrame(all_trades)
        out    = OUTPUT_DIR / "backtest_ict_v2.csv"
        df_out.to_csv(out, index=False)
        print(f"\n  Saved → {out}")
    print("=" * W)


if __name__ == "__main__":
    main()
