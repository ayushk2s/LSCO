"""
backtest_v11_portfolio.py  --  LZR v11  Portfolio + Let Winners Run
====================================================================
Three improvements over v8:

  1. PORTFOLIO: BTC + ETH + SOL share one capital account
     - True concurrent-position simulation with shared balance
     - One position per symbol at most (prevents double-exposure)
     - 7% risk per trade (up to 3 simultaneous = max 21% exposure)

  2. LET WINNERS RUN
     - Partial TP raised from 1.0 to 1.5 ATR
     - Hard TP raised from 3.0 to 6.0 ATR
     - Trail distance widened from 0.5 to 0.8 ATR
     - Locks in partial at 1.5 ATR, trails remainder to 6 ATR

  3. YEAR-BY-YEAR consistency (investor view)

Regime filter: weekly close > weekly EMA(20)  +  daily close > daily EMA(50)
Signal logic: same LZR v8 zone-touch engine (no changes)
Execution:    1m bars, gap-aware fills (same as v8)

DOES NOT modify account_data.py or liq_algo.py.
"""

import sys, warnings, heapq
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Timeframes ─────────────────────────────────────────────────────────────────
SIG_TF  = "4h"
ZONE_TF = "16h"

# ── Capital ────────────────────────────────────────────────────────────────────
INITIAL_BALANCE       = 1_000.0
RISK_PCT              = 0.07          # 7% per trade; max 21% if 3 fire together
MAX_BALANCE           = 200_000.0     # cap for position sizing
MAX_POSITION_NOTIONAL = 500_000.0
MIN_BAL_RATIO         = 0.20          # stop trading if balance < 20% of initial

# ── LZR core parameters (proven from v8) ───────────────────────────────────────
ZONE_WINDOW     = 6
CD_LOSS_BARS    = 42      # 42 x 4h = 7 days cooldown after loss
CD_WIN_BARS     = 3       # 3 x 4h = 12h cooldown after win
EMA_PERIOD      = 50
VOL_MA_PERIOD   = 20
VOL_MULT        = 1.8
SL_MULT         = 0.75
ZONE_TOUCH_MULT = 0.5
ATR_PERIOD      = 14

# ── v11 WIDER TP (key change from v8) ─────────────────────────────────────────
# FIX: keep partial at 1.0 ATR (preserves 93% WR — wider partial HURT WR)
# Let the SECOND half run much further: hard TP 6x ATR, wider trail 0.8x ATR
PARTIAL_TP_MULT = 1.0     # same as v8 -- first half exits early to lock profit
HARD_TP_MULT    = 6.0     # was 3.0 in v8 -- second half runs 2x as far
TRAIL_DIST_MULT = 0.8     # was 0.5 in v8 -- wider trail, survives noise

# ── Regime filter ──────────────────────────────────────────────────────────────
WEEKLY_EMA_PERIOD = 20    # weekly EMA(20) ~ 140-day EMA
DAILY_EMA_PERIOD  = 50

# ── Drawdown circuit breaker ───────────────────────────────────────────────────
PAUSE_RESUME_THRESHOLD = 0.30   # resume after recovering 30% of the drawdown gap
PAUSE_TIMEOUT_BARS     = 50     # also resume after 50 x 4h ~ 8 days

# ── Costs ──────────────────────────────────────────────────────────────────────
FEE_RT   = 0.0004
SLIP_PCT = 0.0003

# ── Portfolio configs to test ──────────────────────────────────────────────────
#  (label, symbols, use_regime, max_dd_pct)
PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

CONFIGS = [
    # Individual (for comparison)
    ("BTC  alone",  ["BTCUSDT"],                          True,  None),
    ("ETH  alone",  ["ETHUSDT"],                          True,  None),
    ("SOL  alone",  ["SOLUSDT"],                          True,  None),
    # 2-symbol portfolio
    ("BTC+ETH",     ["BTCUSDT", "ETHUSDT"],               True,  None),
    # 3-symbol portfolio (main target)
    ("BTC+ETH+SOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT"],   True,  None),
    # 3-symbol + circuit breaker
    ("Port3+CB15",  ["BTCUSDT", "ETHUSDT", "SOLUSDT"],   True,  0.15),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_1m(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def resample(df_1m, freq):
    return df_1m.resample(freq).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), vol=("vol", "sum")
    ).dropna()


# ── Indicators ─────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


# ── Zone detection (identical to v8) ──────────────────────────────────────────

def find_zones(df_zone):
    lo_v = df_zone["low"].values
    hi_v = df_zone["high"].values
    n    = len(df_zone)
    zl, zh = {}, {}
    for i in range(ZONE_WINDOW, n - ZONE_WINDOW):
        if lo_v[i] == np.min(lo_v[i - ZONE_WINDOW: i + ZONE_WINDOW + 1]):
            zl[i] = lo_v[i]
        if hi_v[i] == np.max(hi_v[i - ZONE_WINDOW: i + ZONE_WINDOW + 1]):
            zh[i] = hi_v[i]
    return zl, zh


# ── 1m execution (v11: wider TP targets via constants above) ───────────────────
# Gap-aware fills: if price gaps through SL/TP, fill at bar open.
# Partial TP: close half at PARTIAL_TP_MULT*ATR, trail remainder to HARD_TP_MULT*ATR.

def exec_1m(df_1m, m_start, entry_px, sl_px, partial_tp_px, hard_tp_px, atr, qty):
    """
    Execute a LONG trade on 1m bars starting at m_start.
    Returns (close_ts, gross_pnl, exit_px, total_fee, total_slip) or None.
    """
    half     = qty / 2.0
    state    = "full"           # "full" -> "partial" after first TP
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px

    FEE_SIDE = FEE_RT / 2.0
    fee_acc  = entry_px * qty * FEE_SIDE
    slip_acc = entry_px * qty * SLIP_PCT

    for m_idx in range(m_start, len(df_1m)):
        mr = df_1m.iloc[m_idx]
        closed = False
        gross  = 0.0
        exit_px = 0.0

        if state == "full":
            # Gap down through SL
            if mr["open"] <= sl_px:
                exit_px = mr["open"]
                gross   = (exit_px - entry_px) * qty
                closed  = True
            # Gap up through partial TP
            elif mr["open"] >= partial_tp_px:
                partial_locked = (partial_tp_px - entry_px) * half
                fee_acc  += partial_tp_px * half * FEE_SIDE
                slip_acc += partial_tp_px * half * SLIP_PCT
                running_ext = mr["open"]
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                state = "partial"
            # Price hits partial TP during bar
            elif mr["high"] >= partial_tp_px:
                partial_locked = (partial_tp_px - entry_px) * half
                fee_acc  += partial_tp_px * half * FEE_SIDE
                slip_acc += partial_tp_px * half * SLIP_PCT
                running_ext = mr["high"]
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                state = "partial"
            # Price hits SL during bar
            elif mr["low"] <= sl_px:
                exit_px = sl_px
                gross   = (exit_px - entry_px) * qty
                closed  = True

        elif state == "partial":
            old_trail = trail_sl
            # Gap down through trail stop
            if mr["open"] <= old_trail:
                exit_px = mr["open"]
                gross   = partial_locked + (exit_px - entry_px) * half
                closed  = True
            # Gap up through hard TP
            elif mr["open"] >= hard_tp_px:
                exit_px = hard_tp_px
                gross   = partial_locked + (exit_px - entry_px) * half
                closed  = True
            else:
                # Update trail
                running_ext = max(running_ext, mr["high"])
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                # Check if low hit old trail
                if mr["low"] <= old_trail:
                    exit_px = old_trail
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True
                # Check if high reached hard TP
                elif mr["high"] >= hard_tp_px:
                    exit_px = hard_tp_px
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * FEE_SIDE
            slip_acc += exit_px * exit_qty * SLIP_PCT
            return df_1m.index[m_idx], gross, exit_px, fee_acc, slip_acc

    return None  # trade still open at end of data


# ── Pre-compute per-symbol indicators ─────────────────────────────────────────

def prepare_symbol(df_1m, df_sig, df_zone, df_weekly, df_daily):
    """Compute all indicators and zone data for one symbol."""
    atr_sig    = calc_atr(df_sig)
    ema_sig    = df_sig["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    vol_ma     = df_sig["vol"].rolling(VOL_MA_PERIOD).mean()
    vol_ratio  = df_sig["vol"] / vol_ma

    atr_zone_s = calc_atr(df_zone)
    zl, _      = find_zones(df_zone)

    weekly_ema = df_weekly["close"].ewm(span=WEEKLY_EMA_PERIOD, adjust=False).mean()
    daily_ema  = df_daily["close"].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean()

    return dict(
        df_1m=df_1m, df_sig=df_sig, df_zone=df_zone,
        df_weekly=df_weekly, df_daily=df_daily,
        atr_sig=atr_sig, ema_sig=ema_sig, vol_ratio=vol_ratio,
        atr_zone_s=atr_zone_s, zl=zl,
        weekly_ema=weekly_ema, daily_ema=daily_ema,
        zone_ts=df_zone.index, weekly_ts=df_weekly.index,
        daily_ts=df_daily.index, m_ts=df_1m.index,
    )


# ── Portfolio simulation engine ────────────────────────────────────────────────

def run_portfolio(symbols, sym_data, use_regime, max_dd_pct):
    """
    True shared-capital portfolio simulation.

    Architecture:
    - Unified 4h timeline (all symbols share same bar count).
    - At each bar i: process closes that expired, then check for new signals.
    - Pending closes stored in a min-heap keyed by close bar index.
    - Each symbol can hold at most ONE open position (in_trade flag).
    - Shared balance updated only when a trade closes.
    """
    # Validate: all symbols must have identical 4h bar counts
    ref_sig = sym_data[symbols[0]]["df_sig"]
    n       = len(ref_sig)
    for sym in symbols:
        if len(sym_data[sym]["df_sig"]) != n:
            raise ValueError(f"{sym} has {len(sym_data[sym]['df_sig'])} bars vs {n}")

    WARMUP = max(EMA_PERIOD, VOL_MA_PERIOD, ZONE_WINDOW * 2, WEEKLY_EMA_PERIOD * 7) + 10

    # Shared state
    balance       = INITIAL_BALANCE
    peak_balance  = INITIAL_BALANCE
    paused_until  = -1
    pause_entry_bal = None

    # Per-symbol mutable state
    sym_state = {
        sym: {
            "cooldown_until": 0,
            "last_signal_i":  -999,
            "fired_zones":    set(),
            "in_trade":       False,
        }
        for sym in symbols
    }

    # Min-heap of pending closes: (close_bar, counter, sym, net, trade_dict)
    # counter breaks ties so dict-comparison is never reached
    pending   = []
    trade_ctr = 0

    all_trades   = []
    equity_curve = []   # one value per 4h bar (after processing closes)

    for i in range(n):
        sig_ts = ref_sig.index[i]

        # ── 1. Process all closes that happen at or before this bar ──────────
        while pending and pending[0][0] <= i:
            close_bar, _, sym, net, trade_dict = heapq.heappop(pending)
            balance += net
            peak_balance = max(peak_balance, balance)
            sym_state[sym]["in_trade"] = False
            all_trades.append({**trade_dict, "balance": round(balance, 4)})

        equity_curve.append(balance)

        # ── 2. Skip warmup / ruin guard ──────────────────────────────────────
        if i < WARMUP or balance < INITIAL_BALANCE * MIN_BAL_RATIO:
            continue

        # ── 3. Circuit breaker ───────────────────────────────────────────────
        if max_dd_pct is not None:
            current_dd = (balance / peak_balance) - 1.0
            if current_dd <= -max_dd_pct:
                if i > paused_until:
                    paused_until     = i + PAUSE_TIMEOUT_BARS
                    pause_entry_bal  = balance
            if i <= paused_until:
                if pause_entry_bal is not None:
                    recovery_target = (pause_entry_bal
                                       + (peak_balance - pause_entry_bal)
                                       * PAUSE_RESUME_THRESHOLD)
                    if balance >= recovery_target:
                        paused_until = -1   # resume early
                    else:
                        continue
                else:
                    continue

        # ── 4. Check each symbol for a new signal ────────────────────────────
        for sym in symbols:
            ss  = sym_state[sym]
            sd  = sym_data[sym]
            sig = sd["df_sig"]

            # One trade per symbol at a time; respect cooldown
            if ss["in_trade"] or i <= ss["cooldown_until"]:
                continue
            if i + 1 >= n:
                continue

            atr   = sd["atr_sig"].iloc[i]
            ema   = sd["ema_sig"].iloc[i]
            vol_r = sd["vol_ratio"].iloc[i]

            if atr <= 0 or np.isnan(atr) or np.isnan(ema) or np.isnan(vol_r):
                continue

            # ── Regime filter ────────────────────────────────────────────────
            if use_regime:
                w_idx = int(sd["weekly_ts"].searchsorted(sig_ts, side="right")) - 1
                if w_idx < 1:
                    continue
                if sd["df_weekly"]["close"].iloc[w_idx] < sd["weekly_ema"].iloc[w_idx]:
                    continue

                d_idx = int(sd["daily_ts"].searchsorted(sig_ts, side="right")) - 1
                if d_idx < 1:
                    continue
                if sd["df_daily"]["close"].iloc[d_idx] < sd["daily_ema"].iloc[d_idx]:
                    continue

            # ── Volume filter ────────────────────────────────────────────────
            if vol_r < VOL_MULT:
                continue

            # ── Zone lookup ──────────────────────────────────────────────────
            z_idx = int(sd["zone_ts"].searchsorted(sig_ts, side="right")) - 1
            if z_idx < ZONE_WINDOW:
                continue

            close   = sig.iloc[i]["close"]
            zl      = sd["zl"]
            fired   = ss["fired_zones"]
            atr_z_s = sd["atr_zone_s"]

            fired_zone_key = None
            zone_lo        = None

            past_lows = sorted(
                [k for k in zl if k < z_idx and k not in fired],
                reverse=True
            )[:10]

            for zb in past_lows:
                zp    = zl[zb]
                atr_z = atr_z_s.iloc[zb]
                z_lo  = zp - ZONE_TOUCH_MULT * atr_z
                z_hi  = zp + ZONE_TOUCH_MULT * atr_z
                if z_lo <= close <= z_hi:
                    fired_zone_key = zb
                    zone_lo        = z_lo
                    break

            if fired_zone_key is None:
                continue

            # ── EMA trend confirm ────────────────────────────────────────────
            if close < ema:
                continue

            # ── Prevent double-signal same bar ───────────────────────────────
            if i == ss["last_signal_i"]:
                continue

            # ── Entry price = next bar open ──────────────────────────────────
            entry_px = sig["open"].iloc[i + 1]
            sl_px    = zone_lo - SL_MULT * atr
            sl_dist  = entry_px - sl_px
            if sl_dist <= 0:
                continue

            partial_tp = entry_px + PARTIAL_TP_MULT * atr
            hard_tp    = entry_px + HARD_TP_MULT    * atr

            eff_bal = min(balance, MAX_BALANCE)
            qty     = min(
                (eff_bal * RISK_PCT) / sl_dist,
                MAX_POSITION_NOTIONAL / entry_px
            )
            if qty <= 0:
                continue

            # ── 1m execution ─────────────────────────────────────────────────
            entry_ts = sig.index[i + 1]
            m_start  = int(sd["m_ts"].searchsorted(entry_ts))

            info = exec_1m(
                sd["df_1m"], m_start,
                entry_px, sl_px, partial_tp, hard_tp, atr, qty
            )
            if info is None:
                continue

            close_ts, gross, exit_px, total_fee, total_slip = info

            net    = gross - total_fee - total_slip
            result = "WIN" if net > 0 else "LOSS"

            # Find the 4h bar that contains close_ts
            close_bar = int(sig.index.searchsorted(close_ts, side="right")) - 1
            close_bar = max(close_bar, i + 1)
            close_bar = min(close_bar, n - 1)

            dur_h = (close_ts - entry_ts).total_seconds() / 3600.0

            # Deplete the zone (never fires again for this symbol)
            ss["fired_zones"].add(fired_zone_key)

            # Cooldown after close
            if result == "LOSS":
                ss["cooldown_until"] = close_bar + CD_LOSS_BARS
            else:
                ss["cooldown_until"] = close_bar + CD_WIN_BARS

            ss["last_signal_i"] = i
            ss["in_trade"]      = True

            trade_dict = {
                "symbol":     sym,
                "ts":         sig_ts,
                "close_ts":   close_ts,
                "entry":      round(entry_px, 6),
                "exit":       round(exit_px,  6),
                "sl":         round(sl_px,    6),
                "gross":      round(gross,    4),
                "net":        round(net,      4),
                "result":     result,
                "duration_h": round(dur_h,    1),
            }

            heapq.heappush(pending, (close_bar, trade_ctr, sym, net, trade_dict))
            trade_ctr += 1

    # ── Flush any remaining open trades (use last 1m bar prices) ────────────
    while pending:
        close_bar, _, sym, net, trade_dict = heapq.heappop(pending)
        balance += net
        peak_balance = max(peak_balance, balance)
        sym_state[sym]["in_trade"] = False
        all_trades.append({**trade_dict, "balance": round(balance, 4)})

    return all_trades, equity_curve, balance


# ── Statistics helpers ─────────────────────────────────────────────────────────

def compute_stats(all_trades, equity_curve, final_balance, ref_sig_index):
    """Compute overall + year-by-year stats from trade list and equity curve."""
    if not all_trades:
        return None

    df_t  = pd.DataFrame(all_trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]

    nw  = wins["net"].sum()
    nl  = abs(loses["net"].sum()) if len(loses) else 0.0
    npf = round(nw / nl, 3) if nl > 0 else float("inf")
    wr  = round(len(wins) / len(df_t) * 100, 1)

    years = (ref_sig_index[-1] - ref_sig_index[0]).days / 365.25
    cagr  = ((final_balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    eq  = pd.Series(equity_curve)
    mdd = round(float(((eq - eq.cummax()) / eq.cummax() * 100).min()), 2)

    calmar = round(abs(cagr / mdd), 3) if mdd < 0 else float("inf")

    # ── Year-by-year breakdown ────────────────────────────────────────────────
    yearly = []
    all_years = sorted(pd.to_datetime(df_t["ts"]).dt.year.unique())
    for yr in all_years:
        # Find equity curve slice for this year
        ts_start = pd.Timestamp(f"{yr}-01-01", tz="UTC") if ref_sig_index.tz else pd.Timestamp(f"{yr}-01-01")
        ts_end   = pd.Timestamp(f"{yr+1}-01-01", tz="UTC") if ref_sig_index.tz else pd.Timestamp(f"{yr+1}-01-01")
        bar_start = int(ref_sig_index.searchsorted(ts_start, side="left"))
        bar_end   = int(ref_sig_index.searchsorted(ts_end,   side="left"))
        bar_start = max(bar_start, 0)
        bar_end   = min(bar_end, len(equity_curve))

        eq_yr = pd.Series(equity_curve[bar_start:bar_end])
        bal_start_yr = equity_curve[bar_start] if bar_start < len(equity_curve) else INITIAL_BALANCE
        bal_end_yr   = equity_curve[bar_end - 1] if bar_end > 0 and bar_end <= len(equity_curve) else bal_start_yr

        yr_cagr = round((bal_end_yr / bal_start_yr - 1) * 100, 1) if bal_start_yr > 0 else 0.0
        yr_mdd  = round(float(((eq_yr - eq_yr.cummax()) / eq_yr.cummax() * 100).min()), 1) if len(eq_yr) > 1 else 0.0

        yr_t    = df_t[pd.to_datetime(df_t["ts"]).dt.year == yr]
        yr_wins = yr_t[yr_t["result"] == "WIN"]
        yr_wr   = round(len(yr_wins) / len(yr_t) * 100, 0) if len(yr_t) > 0 else 0.0

        yearly.append({
            "year":   yr,
            "trades": len(yr_t),
            "wr":     yr_wr,
            "cagr":   yr_cagr,
            "max_dd": yr_mdd,
        })

    # ── Per-symbol contribution ───────────────────────────────────────────────
    per_sym_stats = {}
    for sym in df_t["symbol"].unique():
        s   = df_t[df_t["symbol"] == sym]
        sw  = s[s["result"] == "WIN"]
        per_sym_stats[sym] = {
            "trades": len(s),
            "wr":     round(len(sw) / len(s) * 100, 1),
            "net":    round(s["net"].sum(), 2),
        }

    return {
        "trades":    len(df_t),
        "wins":      len(wins),
        "win_rate":  wr,
        "net_pf":    npf,
        "final_bal": round(final_balance, 2),
        "cagr":      round(cagr, 2),
        "max_dd":    mdd,
        "calmar":    calmar,
        "yearly":    yearly,
        "per_sym":   per_sym_stats,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    W = 92
    print("=" * W)
    print("  LZR v11  PORTFOLIO + LET WINNERS RUN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print(f"  Target: CAGR >=20%  |  Calmar >=1.5  |  Consistent year-by-year")
    print(f"  LONG-only | Regime filter | 4h signal | 16h zone | 1m execution")
    print(f"  TP: partial {PARTIAL_TP_MULT}x ATR -> trail {TRAIL_DIST_MULT}x ATR -> hard {HARD_TP_MULT}x ATR")
    print(f"  Risk: {RISK_PCT*100:.0f}% per trade | Initial balance: ${INITIAL_BALANCE:,.0f}")
    print()

    # ── Load all required symbols ─────────────────────────────────────────────
    all_syms = sorted({s for _, syms, _, _ in CONFIGS for s in syms})
    print("  Loading data...", flush=True)
    raw_data  = {}
    sym_data  = {}
    for sym in all_syms:
        df_1m    = load_1m(sym)
        df_sig   = resample(df_1m, SIG_TF)
        df_zone  = resample(df_1m, ZONE_TF)
        df_weekly = resample(df_1m, "1W")
        df_daily  = resample(df_1m, "1D")
        raw_data[sym] = (df_1m, df_sig, df_zone, df_weekly, df_daily)
        sym_data[sym] = prepare_symbol(df_1m, df_sig, df_zone, df_weekly, df_daily)
        print(f"    {sym}: {len(df_1m):,} 1m bars | {len(df_sig):,} 4h bars", flush=True)
    print()

    ref_sig_index = sym_data[all_syms[0]]["df_sig"].index

    # ── Run each config ───────────────────────────────────────────────────────
    all_results = {}

    for cfg_label, cfg_syms, use_regime, max_dd in CONFIGS:
        print(f"  Running: {cfg_label:<18}  regime={'YES' if use_regime else 'NO '}  "
              f"CB={f'{int(max_dd*100)}%' if max_dd else 'none'} ...", flush=True)

        try:
            trades, equity, final_bal = run_portfolio(
                cfg_syms, sym_data, use_regime, max_dd
            )
            stats = compute_stats(trades, equity, final_bal, ref_sig_index)
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[cfg_label] = None
            continue

        if stats is None:
            print(f"    No trades generated.")
            all_results[cfg_label] = None
            continue

        all_results[cfg_label] = stats

        s = stats
        sign   = "+" if s["cagr"] >= 0 else ""
        star   = ("★★" if s["calmar"] >= 2.0 else
                  ("★ " if s["calmar"] >= 1.5 else
                   ("●  " if s["calmar"] >= 1.0 else "   ")))
        print(f"    {s['trades']:>3} trades | WR {s['win_rate']:>5.1f}% | "
              f"CAGR {sign}{s['cagr']:>5.1f}% | DD {s['max_dd']:>7.2f}% | "
              f"Calmar {s['calmar']:.2f} {star}| "
              f"Final ${s['final_bal']:,.0f}")
        print()

    # ── Detailed output ───────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  RESULTS SUMMARY")
    print("=" * W)
    hdr = f"  {'Config':<18} {'Trades':>7} {'WR':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'$Final':>10}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for cfg_label, _, _, _ in CONFIGS:
        s = all_results.get(cfg_label)
        if s is None:
            print(f"  {cfg_label:<18}  -- no result --")
            continue
        sign = "+" if s["cagr"] >= 0 else ""
        star = ("★★" if s["calmar"] >= 2.0 else
                ("★ " if s["calmar"] >= 1.5 else
                 ("● " if s["calmar"] >= 1.0 else "  ")))
        print(f"  {cfg_label:<18} {s['trades']:>7} {s['win_rate']:>6.1f}% "
              f"{sign}{s['cagr']:>6.1f}% {s['max_dd']:>7.1f}% "
              f"{s['calmar']:>7.2f}{star} ${s['final_bal']:>9,.0f}")
    print()

    # ── Year-by-year for main portfolio configs ───────────────────────────────
    main_configs = ["BTC+ETH+SOL", "Port3+CB15", "BTC+ETH"]
    for cfg_label in main_configs:
        s = all_results.get(cfg_label)
        if s is None:
            continue
        print("=" * W)
        print(f"  YEAR-BY-YEAR  [{cfg_label}]")
        print("=" * W)
        print(f"  {'Year':>6} {'Trades':>7} {'WR':>7} {'CAGR':>8} {'MaxDD':>8}")
        print("  " + "-" * 44)
        yr_signs_good = 0
        for row in s["yearly"]:
            sign = "+" if row["cagr"] >= 0 else ""
            flag = "" if row["cagr"] >= 0 else " <--"
            if row["cagr"] >= 0:
                yr_signs_good += 1
            print(f"  {row['year']:>6} {row['trades']:>7} {row['wr']:>6.0f}% "
                  f"{sign}{row['cagr']:>6.1f}% {row['max_dd']:>7.1f}%{flag}")
        total_yrs = len(s["yearly"])
        print(f"  {'':>6} {'':>7} {'':>7}  Profitable years: {yr_signs_good}/{total_yrs}")
        print()

        # Per-symbol contribution
        print(f"  PER-SYMBOL CONTRIBUTION  [{cfg_label}]")
        print(f"  {'Symbol':<12} {'Trades':>7} {'WR':>7} {'Net P&L':>10}")
        print("  " + "-" * 38)
        for sym, ps in sorted(s["per_sym"].items()):
            sign = "+" if ps["net"] >= 0 else ""
            print(f"  {sym:<12} {ps['trades']:>7} {ps['wr']:>6.1f}% "
                  f"  {sign}${ps['net']:>8,.2f}")
        print()

    # ── Best config summary ───────────────────────────────────────────────────
    print("=" * W)
    print("  BEST CALMAR CONFIG")
    print("=" * W)
    best_label = None
    best_calmar = -1.0
    for cfg_label, _, _, _ in CONFIGS:
        s = all_results.get(cfg_label)
        if s and s["cagr"] > 0:
            c = s["calmar"] if s["calmar"] != float("inf") else 99.0
            if c > best_calmar:
                best_calmar = c
                best_label  = cfg_label

    if best_label:
        s     = all_results[best_label]
        sign  = "+" if s["cagr"] >= 0 else ""
        star  = "★★" if s["calmar"] >= 2.0 else ("★" if s["calmar"] >= 1.5 else "")
        print(f"  {best_label}:")
        print(f"    Calmar   : {s['calmar']:.2f} {star}")
        print(f"    CAGR     : {sign}{s['cagr']:.1f}%")
        print(f"    MaxDD    : {s['max_dd']:.1f}%")
        print(f"    Win Rate : {s['win_rate']:.1f}%")
        print(f"    Trades   : {s['trades']}")
        print(f"    Final Bal: ${s['final_bal']:,.2f}  (started ${INITIAL_BALANCE:,.0f})")
        print()
        print(f"  TP: partial {PARTIAL_TP_MULT}x ATR  trail {TRAIL_DIST_MULT}x ATR  hard {HARD_TP_MULT}x ATR")
        print(f"  Risk per trade: {RISK_PCT*100:.0f}%")
        print(f"  Regime filter: weekly EMA({WEEKLY_EMA_PERIOD}) + daily EMA({DAILY_EMA_PERIOD})")

    # ── vs v8 comparison ──────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  v8 vs v11 IMPROVEMENT  (same regime filter)")
    print("=" * W)
    print(f"  {'':25} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Trades':>8}")
    print(f"  {'v8 BTC alone (10% risk)':25} {'  +9.3%':>8} {' -10.3%':>8} {'  0.90':>8} {'15':>8}")
    print(f"  {'v8 ETH alone (10% risk)':25} {'  +7.4%':>8} {' -10.2%':>8} {'  0.72':>8} {'11':>8}")

    btc_v11 = all_results.get("BTC  alone")
    eth_v11 = all_results.get("ETH  alone")
    sol_v11 = all_results.get("SOL  alone")
    port_v11 = all_results.get("BTC+ETH+SOL")

    for label, s in [("v11 BTC  alone (7% risk)", btc_v11),
                     ("v11 ETH  alone (7% risk)", eth_v11),
                     ("v11 SOL  alone (7% risk)", sol_v11),
                     ("v11 BTC+ETH+SOL (7%/trade)", port_v11)]:
        if s:
            sign = "+" if s["cagr"] >= 0 else ""
            star = " ★" if s["calmar"] >= 1.5 else (" ●" if s["calmar"] >= 1.0 else "")
            print(f"  {label:<25} {sign}{s['cagr']:>6.1f}%  {s['max_dd']:>6.1f}%  "
                  f"{s['calmar']:>6.2f}{star}  {s['trades']:>6}")

    # ── Save trade log ────────────────────────────────────────────────────────
    save_rows = []
    for cfg_label, cfg_syms, use_regime, max_dd in CONFIGS:
        s = all_results.get(cfg_label)
        if s:
            save_rows.append({
                "config": cfg_label,
                "symbols": "+".join(cfg_syms),
                "use_regime": use_regime,
                "max_dd_cb": max_dd or 0,
                "trades": s["trades"],
                "win_rate": s["win_rate"],
                "cagr": s["cagr"],
                "max_dd": s["max_dd"],
                "calmar": s["calmar"],
                "final_bal": s["final_bal"],
            })

    if save_rows:
        out = OUTPUT_DIR / "backtest_v11_portfolio.csv"
        pd.DataFrame(save_rows).to_csv(out, index=False)
        print()
        print(f"\n  Saved -> {out}")

    print("=" * W)


if __name__ == "__main__":
    main()
