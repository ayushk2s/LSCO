"""
backtest_v8_calmar.py  --  LZR v8 + Calmar Ratio Optimization
==============================================================
Problem: all previous versions have CAGR << MaxDD (Calmar < 1.0).
User target: MaxDD ≤ CAGR / 2  →  Calmar Ratio ≥ 2.0

Two additions to the proven LZR v8 LONG-only engine:

  1. REGIME FILTER (weekly trend)
     Only trade LONG when: weekly close > weekly EMA(20)
     Effect: stops trading in bear markets (2022 crash, 2018 crash)
     The biggest drawdowns in crypto happen in bear markets.
     Skipping them should halve MaxDD while keeping most wins.

  2. DRAWDOWN CIRCUIT BREAKER
     If equity drops more than MAX_DD_PCT below peak → pause for N bars
     This is a hard stop on MaxDD — regardless of market conditions.
     Resume when equity recovers past a % threshold OR after timeout.

Signal: same LZR v8 (zone touch on signal TF, EMA trend filter,
        volume filter), LONG-only, 4h bars, 1m execution.

Tests all combinations: with/without regime filter × DD limits.
Reports Calmar Ratio prominently alongside CAGR and MaxDD.

DOES NOT modify account_data.py or liq_algo.py.
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Signal / zone TF — 4h is the proven best (from all previous backtests)
SIG_TF  = "4h"
ZONE_TF = "16h"

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10

# LZR v8 core parameters (proven from backtest_v8_longonly.py)
ZONE_WINDOW      = 6
CD_LOSS_BARS     = 42     # 42 × 4h = 168h = 7 days cooldown after loss
EMA_PERIOD       = 50
VOL_MA_PERIOD    = 20
VOL_MULT         = 1.8
SL_MULT          = 0.75
ZONE_TOUCH_MULT  = 0.5
PARTIAL_TP_MULT  = 1.0
TRAIL_DIST_MULT  = 0.5
HARD_TP_MULT     = 3.0
ATR_PERIOD       = 14
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20

# Regime filter
WEEKLY_EMA_PERIOD = 20    # weeks (= ~140 days)
DAILY_EMA_PERIOD  = 50    # days

# Drawdown circuit breaker
PAUSE_RESUME_THRESHOLD = 0.30  # resume when equity recovers 30% of the drawdown
PAUSE_TIMEOUT_BARS     = 50    # also resume after 50 × 4h bars = ~8.3 days

FEE_RT   = 0.0004
SLIP_PCT = 0.0003

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# ─── Test configurations ──────────────────────────────────────────────────────
# (name, use_regime_filter, max_dd_pct)
#   use_regime_filter: True = only trade when weekly trend is bullish
#   max_dd_pct: None = no circuit breaker, else pause when equity drops this much from peak
CONFIGS = [
    ("Baseline",           False, None ),  # reference: no new additions
    ("Regime",             True,  None ),  # weekly regime filter only
    ("CB-20%",             False, 0.20 ),  # circuit breaker at 20% DD
    ("CB-15%",             False, 0.15 ),  # circuit breaker at 15% DD
    ("CB-10%",             False, 0.10 ),  # circuit breaker at 10% DD
    ("Regime+CB-20%",      True,  0.20 ),  # both filters: 20% DD cap
    ("Regime+CB-15%",      True,  0.15 ),  # both filters: 15% DD cap
    ("Regime+CB-10%",      True,  0.10 ),  # both filters: 10% DD cap
]


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_1m(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date":"ts","Open":"open","High":"high",
                             "Low":"low","Close":"close","Volume":"vol"})
    return df.set_index("ts").sort_index()


def resample(df_1m, freq):
    return df_1m.resample(freq).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), vol=("vol","sum")
    ).dropna()


# ─── Indicators ───────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


# ─── Zone detection (same logic as backtest_v8_longonly.py) ──────────────────

def find_zones(df_zone):
    lo_v = df_zone["low"].values
    hi_v = df_zone["high"].values
    n    = len(df_zone)
    zl, zh = {}, {}
    for i in range(ZONE_WINDOW, n - ZONE_WINDOW):
        if lo_v[i] == np.min(lo_v[i - ZONE_WINDOW:i + ZONE_WINDOW + 1]):
            zl[i] = lo_v[i]
        if hi_v[i] == np.max(hi_v[i - ZONE_WINDOW:i + ZONE_WINDOW + 1]):
            zh[i] = hi_v[i]
    return zl, zh


# ─── 1m executor (same as proven versions, SHORT SL fixed) ───────────────────

def exec_1m(df_1m, m_start, entry_px, direction,
             sl_px, partial_tp_px, hard_tp_px, atr, qty, fee_side, slip_pct):
    half = qty / 2
    state = "full"
    partial_locked = 0.0
    running_ext = entry_px
    trail_sl = sl_px
    fee_acc  = entry_px * qty * fee_side
    slip_acc = entry_px * qty * slip_pct

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
                    trail_sl = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                    state = "partial"
                elif mr["high"] >= partial_tp_px:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["high"]
                    trail_sl = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                    state = "partial"
                elif mr["low"] <= sl_px:
                    exit_px, gross, closed = sl_px, (sl_px - entry_px) * qty, True

        elif state == "partial":
            if direction == "LONG":
                old_trail = trail_sl
                if mr["open"] <= old_trail:
                    exit_px, gross, closed = mr["open"], partial_locked + (mr["open"] - entry_px) * half, True
                elif mr["open"] >= hard_tp_px:
                    exit_px, gross, closed = hard_tp_px, partial_locked + (hard_tp_px - entry_px) * half, True
                else:
                    running_ext = max(running_ext, mr["high"])
                    trail_sl = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                    if mr["low"] <= old_trail:
                        exit_px, gross, closed = old_trail, partial_locked + (old_trail - entry_px) * half, True
                    elif mr["high"] >= hard_tp_px:
                        exit_px, gross, closed = hard_tp_px, partial_locked + (hard_tp_px - entry_px) * half, True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * fee_side
            slip_acc += exit_px * exit_qty * slip_pct
            return df_1m.index[m_idx], gross, exit_px, fee_acc, slip_acc

    return None


# ─── Main backtest engine ─────────────────────────────────────────────────────

def run_lzr_calmar(symbol, df_1m, df_sig, df_zone,
                   df_weekly, df_daily,
                   use_regime, max_dd_pct):
    """
    LZR v8 LONG-only with optional:
      - use_regime: only trade when weekly trend is bullish
      - max_dd_pct: circuit breaker (pause if equity drops this % from peak)
    """
    FEE_SIDE = FEE_RT / 2
    m_ts     = df_1m.index
    n        = len(df_sig)

    # Signal-TF indicators
    atr_sig  = calc_atr(df_sig)
    ema_sig  = df_sig["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    vol_sig  = df_sig["vol"] / df_sig["vol"].rolling(VOL_MA_PERIOD).mean()

    # Zone detection
    atr_zone_s = calc_atr(df_zone)
    zl, zh     = find_zones(df_zone)
    zone_ts    = df_zone.index

    # Regime filter: weekly EMA
    weekly_ema = df_weekly["close"].ewm(span=WEEKLY_EMA_PERIOD, adjust=False).mean()
    weekly_ts  = df_weekly.index

    # Daily EMA (secondary regime filter — price above daily 50 EMA)
    daily_ema  = df_daily["close"].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean()
    daily_ts   = df_daily.index

    balance       = INITIAL_BALANCE
    peak_balance  = INITIAL_BALANCE
    equity        = [balance]
    trades        = []
    cooldown_until = 0
    last_signal_i  = -999
    paused_until   = 0          # circuit breaker
    pause_entry_balance = None  # balance when we paused
    fired_zones    = set()      # each zone bar-index fires at most once

    WARMUP = max(EMA_PERIOD, VOL_MA_PERIOD, ZONE_WINDOW * 2,
                 WEEKLY_EMA_PERIOD * 7) + 10  # weekly ema needs more bars

    i = 0
    while i < n:
        if i < WARMUP:
            equity.append(balance); i += 1; continue
        if balance < INITIAL_BALANCE * MIN_BAL_RATIO:
            equity.append(balance); i += 1; continue

        # ── Cooldown (after loss) ─────────────────────────────────────────────
        if i <= cooldown_until:
            equity.append(balance); i += 1; continue

        # ── Circuit breaker check ─────────────────────────────────────────────
        peak_balance = max(peak_balance, balance)
        if max_dd_pct is not None:
            current_dd = (balance / peak_balance) - 1.0
            if current_dd <= -max_dd_pct:
                # Trigger pause
                if i > paused_until:
                    paused_until = i + PAUSE_TIMEOUT_BARS
                    pause_entry_balance = balance
            if i <= paused_until:
                # Check for early recovery: recovered 30% of the drawdown gap
                if pause_entry_balance is not None:
                    recovery_target = pause_entry_balance + \
                        (peak_balance - pause_entry_balance) * PAUSE_RESUME_THRESHOLD
                    if balance >= recovery_target:
                        paused_until = 0  # resume early
                    else:
                        equity.append(balance); i += 1; continue
                else:
                    equity.append(balance); i += 1; continue

        row    = df_sig.iloc[i]
        sig_ts = df_sig.index[i]
        atr    = atr_sig.iloc[i]
        ema    = ema_sig.iloc[i]
        vol_r  = vol_sig.iloc[i]

        if atr <= 0 or np.isnan(atr) or np.isnan(ema) or np.isnan(vol_r):
            equity.append(balance); i += 1; continue

        # ── Regime filter (weekly trend) ──────────────────────────────────────
        if use_regime:
            w_idx = int(weekly_ts.searchsorted(sig_ts, side="right")) - 1
            if w_idx >= 1:
                w_close = df_weekly["close"].iloc[w_idx]
                w_ema   = weekly_ema.iloc[w_idx]
                if w_close < w_ema:
                    equity.append(balance); i += 1; continue

            # Secondary: daily EMA
            d_idx = int(daily_ts.searchsorted(sig_ts, side="right")) - 1
            if d_idx >= 1:
                d_close = df_daily["close"].iloc[d_idx]
                d_ema   = daily_ema.iloc[d_idx]
                if d_close < d_ema:
                    equity.append(balance); i += 1; continue

        # ── Volume filter ─────────────────────────────────────────────────────
        if vol_r < VOL_MULT:
            equity.append(balance); i += 1; continue

        # ── Zone lookup ───────────────────────────────────────────────────────
        z_idx = int(zone_ts.searchsorted(sig_ts, side="right")) - 1
        if z_idx < ZONE_WINDOW:
            equity.append(balance); i += 1; continue

        close  = row["close"]
        signal = None
        zone_lo = zone_hi = None

        # Check LONG: close in a swing-low zone (each zone fires at most once)
        fired_zone_key = None
        for zb in sorted([k for k in zl if k < z_idx and k not in fired_zones],
                         reverse=True)[:10]:
            zp    = zl[zb]
            atr_z = atr_zone_s.iloc[zb]
            z_lo  = zp - ZONE_TOUCH_MULT * atr_z
            z_hi  = zp + ZONE_TOUCH_MULT * atr_z
            if z_lo <= close <= z_hi:
                signal        = "LONG"
                zone_lo       = z_lo
                zone_hi       = z_hi
                fired_zone_key = zb
                break

        if signal is None or i == last_signal_i:
            equity.append(balance); i += 1; continue

        # ── EMA trend confirm ─────────────────────────────────────────────────
        if close < ema:
            equity.append(balance); i += 1; continue

        if i + 1 >= n:
            equity.append(balance); i += 1; continue

        # ── Entry ─────────────────────────────────────────────────────────────
        entry_px = df_sig["open"].iloc[i + 1]
        sl_px    = zone_lo - SL_MULT * atr
        sl_dist  = entry_px - sl_px
        if sl_dist <= 0:
            equity.append(balance); i += 1; continue

        partial_tp = entry_px + PARTIAL_TP_MULT * atr
        hard_tp    = entry_px + HARD_TP_MULT    * atr

        eff_bal = min(balance, MAX_BALANCE)
        qty     = min((eff_bal * RISK_PCT) / sl_dist,
                      MAX_POSITION_NOTIONAL / entry_px)

        # ── Execute on 1m ─────────────────────────────────────────────────────
        entry_ts = df_sig.index[i + 1]
        m_start  = int(m_ts.searchsorted(entry_ts))

        info = exec_1m(df_1m, m_start, entry_px, "LONG",
                       sl_px, partial_tp, hard_tp, atr, qty,
                       FEE_SIDE, SLIP_PCT)
        if info is None:
            equity.append(balance); i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip = info
        j = max(int(df_sig.index.searchsorted(close_ts, side="right")) - 1, i + 1)
        j = min(j, n - 1)

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        # Deplete the zone — it can never fire again
        if fired_zone_key is not None:
            fired_zones.add(fired_zone_key)

        # Cooldown: longer after loss, short after win
        if result == "LOSS":
            cooldown_until = j + CD_LOSS_BARS
        else:
            cooldown_until = j + 3  # 3×4h = 12h cooldown after win (prevents re-entry same zone)

        last_signal_i = i
        dur_h = (close_ts - entry_ts).total_seconds() / 3600

        trades.append({
            "ts": sig_ts, "close_ts": close_ts,
            "entry": round(entry_px, 6), "exit": round(exit_px, 6),
            "sl": round(sl_px, 6),
            "gross": round(gross, 4), "fee": round(total_fee, 4),
            "net": round(net, 4), "result": result,
            "duration_h": round(dur_h, 1),
            "balance": round(balance, 4),
        })

        equity.append(bal_open)
        for _ in range(j - i - 1):
            equity.append(bal_open)
        equity.append(balance)
        i = j + 1

    if not trades:
        return None

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]
    nw    = wins["net"].sum()
    nl    = abs(loses["net"].sum())
    npf   = round(nw / nl, 3) if nl > 0 else float("inf")
    wr    = round(len(wins) / len(df_t) * 100, 1)
    years = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    cagr  = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    eq    = pd.Series(equity)
    mdd   = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    calmar = abs(cagr / mdd) if mdd < 0 else float("inf")

    return {
        "trades": len(df_t), "wins": len(wins),
        "win_rate": wr, "net_pf": npf,
        "final_bal": round(balance, 2),
        "cagr": round(cagr, 2),
        "max_dd": round(mdd, 2),
        "calmar": round(calmar, 3),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 88
    print("=" * W)
    print("  LZR v8  CALMAR RATIO OPTIMIZATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print("  Target: Calmar ≥ 2.0  (CAGR ≥ 2 × |MaxDD|)")
    print("  LONG-only | 4h signal | 16h zone | 1m execution")
    print("  New: weekly regime filter + drawdown circuit breaker")
    print()

    # Load data
    print("  Loading data...", flush=True)
    all_data = {}
    for sym in CRYPTO_SYMBOLS:
        df_1m    = load_1m(sym)
        df_sig   = resample(df_1m, SIG_TF)
        df_zone  = resample(df_1m, ZONE_TF)
        df_weekly = resample(df_1m, "1W")
        df_daily  = resample(df_1m, "1D")
        all_data[sym] = (df_1m, df_sig, df_zone, df_weekly, df_daily)
        print(f"    {sym}: {len(df_1m):,} 1m  {len(df_sig):,} 4h  "
              f"{len(df_weekly):,} W  {len(df_daily):,} D", flush=True)
    print()

    # Run grid
    all_results = {}
    for cfg_name, use_regime, max_dd_pct in CONFIGS:
        cb_str = f"CB-{int(max_dd_pct*100)}%" if max_dd_pct else "no-CB"
        reg_str = "regime" if use_regime else "no-reg"
        print(f"  ── Config: {cfg_name:<20} [{reg_str}] [{cb_str}]")
        cfg_stats = {}
        for sym in CRYPTO_SYMBOLS:
            df_1m, df_sig, df_zone, df_weekly, df_daily = all_data[sym]
            stats = run_lzr_calmar(sym, df_1m, df_sig, df_zone,
                                   df_weekly, df_daily,
                                   use_regime, max_dd_pct)
            if stats:
                cfg_stats[sym] = stats
                s    = stats
                sign = "+" if s["cagr"] >= 0 else ""
                star = "★" if s["calmar"] >= 2.0 else ("●" if s["calmar"] >= 1.0 else " ")
                print(f"    {sym}  {s['trades']:>3}tr  WR {s['win_rate']:>5.1f}%  "
                      f"NetPF {s['net_pf']:.3f}  "
                      f"CAGR {sign}{s['cagr']:>5.1f}%  "
                      f"DD {s['max_dd']:>7.2f}%  "
                      f"Calmar {s['calmar']:.2f} {star}")
            else:
                print(f"    {sym}  no trades")
        all_results[cfg_name] = cfg_stats
        print()

    # ─── Calmar grid ──────────────────────────────────────────────────────────
    print("=" * W)
    print("  CALMAR RATIO GRID  (★ = ≥ 2.0 target  ● = ≥ 1.0  target Calmar ≥ 2.0)")
    print("=" * W)
    hdr = f"  {'Config':<22} {'BTC':>8} {'ETH':>8} {'SOL':>8} {'BNB':>8} {'XRP':>8} {'AVG':>8}"
    print(hdr)
    print("-" * W)
    for cfg_name, _, _ in CONFIGS:
        row = f"  {cfg_name:<22}"
        vals = []
        for sym in CRYPTO_SYMBOLS:
            s = all_results[cfg_name].get(sym)
            if s and s["calmar"] != float("inf"):
                star = "★" if s["calmar"] >= 2.0 else ("●" if s["calmar"] >= 1.0 else " ")
                row += f" {s['calmar']:>6.2f}{star}"
                vals.append(s["calmar"])
            elif s and s["calmar"] == float("inf"):
                row += f"   inf★"
                vals.append(5.0)
            else:
                row += f"    N/A "
        if vals:
            avg = np.mean(vals)
            star = "★" if avg >= 2.0 else ("●" if avg >= 1.0 else " ")
            row += f" {avg:>6.2f}{star}"
        print(row)

    print()
    print("  CAGR GRID")
    print("-" * W)
    for cfg_name, _, _ in CONFIGS:
        row = f"  {cfg_name:<22}"
        vals = []
        for sym in CRYPTO_SYMBOLS:
            s = all_results[cfg_name].get(sym)
            if s:
                row += f" {'+' if s['cagr'] >= 0 else ''}{s['cagr']:>6.1f}%"
                vals.append(s["cagr"])
            else:
                row += f"     N/A"
        if vals:
            row += f" {'+' if np.mean(vals) >= 0 else ''}{np.mean(vals):>6.1f}%"
        print(row)

    print()
    print("  MaxDD GRID")
    print("-" * W)
    for cfg_name, _, _ in CONFIGS:
        row = f"  {cfg_name:<22}"
        vals = []
        for sym in CRYPTO_SYMBOLS:
            s = all_results[cfg_name].get(sym)
            if s:
                row += f" {s['max_dd']:>7.1f}%"
                vals.append(s["max_dd"])
            else:
                row += f"      N/A"
        if vals:
            row += f" {np.mean(vals):>7.1f}%"
        print(row)

    # ─── Best Calmar configs ───────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  BEST CONFIG PER SYMBOL  (by Calmar ratio)")
    print("=" * W)
    for sym in CRYPTO_SYMBOLS:
        best = None
        best_calmar = -1
        for cfg_name, _, _ in CONFIGS:
            s = all_results[cfg_name].get(sym)
            if s:
                c = s["calmar"] if s["calmar"] != float("inf") else 10.0
                if c > best_calmar and s["cagr"] > 0:  # must be profitable
                    best_calmar = c
                    best = (cfg_name, s)
        if best:
            cfg_name, s = best
            sign = "+" if s["cagr"] >= 0 else ""
            print(f"  {sym}  → {cfg_name:<22} "
                  f"Calmar {best_calmar:.2f}  "
                  f"CAGR {sign}{s['cagr']:.1f}%  "
                  f"DD {s['max_dd']:.1f}%  "
                  f"WR {s['win_rate']:.1f}%  "
                  f"{s['trades']} trades")
        else:
            print(f"  {sym}  → no config with positive CAGR")

    print()
    print("  INTERPRETATION:")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  Calmar = CAGR / |MaxDD|")
    print("  ★  ≥ 2.0:  MaxDD is less than half of CAGR  (USER TARGET)")
    print("  ●  ≥ 1.0:  MaxDD roughly equals CAGR  (decent)")
    print("  < 1.0:    MaxDD exceeds CAGR  (unacceptable risk)")
    print()
    print("  Regime filter: skips all trades when weekly price < weekly EMA(20)")
    print("                 + when daily price < daily EMA(50)")
    print("  Circuit breaker: pauses trading when equity falls max_dd% from peak")
    print("                   resumes after 30% recovery OR 8 days (50×4h)")

    # Save
    rows = []
    for cfg_name, use_regime, max_dd_pct in CONFIGS:
        for sym in CRYPTO_SYMBOLS:
            s = all_results[cfg_name].get(sym)
            if s:
                rows.append({
                    "config": cfg_name, "symbol": sym,
                    "use_regime": use_regime,
                    "max_dd_pct": max_dd_pct or 0,
                    **s
                })
    if rows:
        df_out = pd.DataFrame(rows)
        out = OUTPUT_DIR / "backtest_v8_calmar.csv"
        df_out.to_csv(out, index=False)
        print(f"\n  Saved → {out}")
    print("=" * W)


if __name__ == "__main__":
    main()
