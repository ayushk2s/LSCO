"""
backtest_optimal_per_symbol.py  --  Per-Symbol Optimal TF + LONG-only
=====================================================================
Discovery from extended MTF sweep (5m → 1D):
  Different symbols have DIFFERENT optimal timeframes.
  Using each symbol's best TF dramatically improves results.

  BEST TF (from v8 MTF sweep — all directions):
    ETH  → 3h:  NetPF 1.253  CAGR +14.5%  (vs 4h: -6.4%)
    SOL  → 4h:  NetPF 2.040  CAGR +26.3%  (confirmed best)
    BNB  → 12h: NetPF 3.014  CAGR +17.5%  (vs 4h: -10.2%)
    XRP  → 3h:  NetPF 1.136  CAGR +11.1%  (vs 4h: -16.7%)
    BTC  → 1D:  NetPF 2.008  CAGR  +2.4%  (only 3 trades)

This file tests:
  Phase 1: Per-symbol optimal TF, LONG-only
  Phase 2: Per-symbol optimal TF + LONG-only + slight SL/vol tuning

Engine: LZR v8 (same as backtest_v8_longonly.py), gap-aware, fixed-fractional.
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

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10

# Per-symbol optimal timeframes (signal TF → zone TF = 4× signal)
# Format: (signal_resample, zone_resample, zone_window, cd_loss, cd_win)
# Parameters scaled to preserve same calendar duration as 1h baseline:
#   1h baseline: zone_window=24, cd_loss=168, cd_win=10
#   Scale by TF ratio. e.g. 3h: divide by 3.
PER_SYMBOL_CONFIG = {
    "BTCUSDT": ("1D",  "4D",   1,   7,  1, 1),   # 1D signal (best: NetPF 2.008)
    "ETHUSDT": ("3h",  "12h",  8,  56,  3, 1),   # 3h signal (best: NetPF 1.253)
    "SOLUSDT": ("4h",  "16h",  6,  42,  3, 1),   # 4h signal (best: NetPF 2.040)
    "BNBUSDT": ("12h", "48h",  2,  14,  1, 1),   # 12h signal (best: NetPF 3.014)
    "XRPUSDT": ("3h",  "12h",  8,  56,  3, 1),   # 3h signal (best: NetPF 1.136)
}

# Test configs (LONG-only × SL variation)
CONFIGS = [
    # (name,      long_only, vol_mult, sl_mult)
    ("Baseline",      False,   1.8,    0.75),  # reference: all-direction
    ("LONG_ONLY",     True,    1.8,    0.75),  # long-only, default params
    ("L+SL0.6",       True,    1.8,    0.60),  # tighter SL
    ("L+SL0.5",       True,    1.8,    0.50),  # very tight SL
]

# Execution
FEE_RT   = 0.0004
SLIP_PCT = 0.0003
ATR_PERIOD = 14
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20

# Zone detection params
ZONE_TOUCH_MULT  = 0.5     # zone extends 0.5× ATR from swing
EMA_PERIOD       = 50
VOL_MA_PERIOD    = 20
PARTIAL_TP_MULT  = 1.0     # partial TP at 1× ATR (same as LZR v8)
TRAIL_DIST_MULT  = 0.5
HARD_TP_MULT     = 3.0


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_1m(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def build_tf(df_1m, resample_str):
    return df_1m.resample(resample_str).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), vol=("vol","sum")
    ).dropna()


# ─── Indicators ───────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


def calc_ema(df, p):
    return df["close"].ewm(span=p, adjust=False).mean()


def calc_vol_ratio(df, p):
    return df["vol"] / df["vol"].rolling(p).mean()


# ─── Zone detection (from zone TF) ───────────────────────────────────────────

def build_zones(df_zone, zone_window, atr_zone):
    """Swing highs and lows on zone TF — these are the LZR zones."""
    lows, highs = {}, {}
    lo_v  = df_zone["low"].values
    hi_v  = df_zone["high"].values
    n = len(df_zone)
    for i in range(zone_window, n - zone_window):
        if lo_v[i] == np.min(lo_v[i - zone_window:i + zone_window + 1]):
            lows[i] = lo_v[i]
        if hi_v[i] == np.max(hi_v[i - zone_window:i + zone_window + 1]):
            highs[i] = hi_v[i]
    return lows, highs


# ─── 1m executor (gap-aware, SHORT SL fixed) ─────────────────────────────────

def exec_1m(df_1m, m_start, entry_px, direction,
             sl_px, partial_tp_px, hard_tp_px,
             trade_atr, qty, fee_side, slip_pct):
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
                    trail_sl = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
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
                    trail_sl = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    if mr["high"] >= old_trail:
                        exit_px, gross, closed = old_trail, partial_locked + (entry_px - old_trail) * half, True
                    elif mr["low"] <= hard_tp_px:
                        exit_px, gross, closed = hard_tp_px, partial_locked + (entry_px - hard_tp_px) * half, True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * fee_side
            slip_acc += exit_px * exit_qty * slip_pct
            return df_1m.index[m_idx], gross, exit_px, fee_acc, slip_acc

    return None


# ─── Per-symbol backtest ──────────────────────────────────────────────────────

def run_lzr(symbol, df_1m, df_sig, df_zone,
            zone_window, cd_loss, cd_win,
            long_only, vol_mult, sl_mult):
    FEE_SIDE = FEE_RT / 2
    m_ts     = df_1m.index
    n        = len(df_sig)

    atr_sig  = calc_atr(df_sig)
    ema_sig  = calc_ema(df_sig, EMA_PERIOD)
    vol_sig  = calc_vol_ratio(df_sig, VOL_MA_PERIOD)

    atr_zone_s = calc_atr(df_zone)
    zone_lows, zone_highs = build_zones(df_zone, zone_window, atr_zone_s)
    zone_ts = df_zone.index

    balance        = INITIAL_BALANCE
    equity         = [balance]
    trades         = []
    cooldown_until = 0
    last_signal_i  = -999

    WARMUP = max(EMA_PERIOD, VOL_MA_PERIOD, zone_window * 2) + 10

    i = 0
    while i < n:
        if i < WARMUP or i <= cooldown_until or balance < INITIAL_BALANCE * MIN_BAL_RATIO:
            equity.append(balance); i += 1; continue

        row     = df_sig.iloc[i]
        atr     = atr_sig.iloc[i]
        ema     = ema_sig.iloc[i]
        vol_r   = vol_sig.iloc[i]
        sig_ts  = df_sig.index[i]

        if atr <= 0 or np.isnan(atr) or np.isnan(ema) or np.isnan(vol_r):
            equity.append(balance); i += 1; continue
        if vol_r < vol_mult:
            equity.append(balance); i += 1; continue

        # Find the current zone-bar index corresponding to sig_ts
        z_idx = int(zone_ts.searchsorted(sig_ts, side="right")) - 1
        if z_idx < zone_window:
            equity.append(balance); i += 1; continue

        # Check if row closes inside any active zone
        close  = row["close"]
        signal = None
        zone_lo = None
        zone_hi = None

        # LONG: close inside a support zone (swing low zone)
        # Fix: filter to past zones first, THEN take the 10 most recent
        past_lows = [(zb, zp) for zb, zp in zone_lows.items() if zb < z_idx]
        if not long_only or True:  # always check longs
            for zb, zp in sorted(past_lows, key=lambda x: x[0], reverse=True)[:10]:
                atr_z = atr_zone_s.iloc[zb]
                z_lo  = zp - ZONE_TOUCH_MULT * atr_z
                z_hi  = zp + ZONE_TOUCH_MULT * atr_z
                if z_lo <= close <= z_hi:
                    signal  = "LONG"
                    zone_lo = z_lo
                    zone_hi = z_hi
                    break

        # SHORT: close inside a resistance zone (swing high zone)
        past_highs = [(zb, zp) for zb, zp in zone_highs.items() if zb < z_idx]
        if signal is None and not long_only:
            for zb, zp in sorted(past_highs, key=lambda x: x[0], reverse=True)[:10]:
                atr_z = atr_zone_s.iloc[zb]
                z_lo  = zp - ZONE_TOUCH_MULT * atr_z
                z_hi  = zp + ZONE_TOUCH_MULT * atr_z
                if z_lo <= close <= z_hi:
                    signal  = "SHORT"
                    zone_lo = z_lo
                    zone_hi = z_hi
                    break

        if signal is None or i == last_signal_i:
            equity.append(balance); i += 1; continue
        if long_only and signal == "SHORT":
            equity.append(balance); i += 1; continue
        if i + 1 >= n:
            equity.append(balance); i += 1; continue

        # EMA filter
        if signal == "LONG" and close < ema:
            equity.append(balance); i += 1; continue
        if signal == "SHORT" and close > ema:
            equity.append(balance); i += 1; continue

        # Entry on next bar
        entry_px = df_sig["open"].iloc[i + 1]

        if signal == "LONG":
            sl_px    = zone_lo - sl_mult * atr
            sl_dist  = entry_px - sl_px
            if sl_dist <= 0:
                equity.append(balance); i += 1; continue
            partial_tp = entry_px + PARTIAL_TP_MULT * atr
            hard_tp    = entry_px + HARD_TP_MULT    * atr
        else:
            sl_px   = zone_hi + sl_mult * atr
            sl_dist = sl_px - entry_px
            if sl_dist <= 0:
                equity.append(balance); i += 1; continue
            partial_tp = entry_px - PARTIAL_TP_MULT * atr
            hard_tp    = entry_px - HARD_TP_MULT    * atr

        eff_bal = min(balance, MAX_BALANCE)
        qty     = min((eff_bal * RISK_PCT) / sl_dist,
                      MAX_POSITION_NOTIONAL / entry_px)

        entry_ts = df_sig.index[i + 1]
        m_start  = int(m_ts.searchsorted(entry_ts))

        info = exec_1m(df_1m, m_start, entry_px, signal,
                       sl_px, partial_tp, hard_tp, atr, qty, FEE_SIDE, SLIP_PCT)
        if info is None:
            equity.append(balance); i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip = info
        j = max(int(df_sig.index.searchsorted(close_ts, side="right")) - 1, i + 1)
        j = min(j, n - 1)

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        if result == "LOSS":
            cooldown_until = j + cd_loss
        else:
            pass  # no cooldown on wins

        last_signal_i = i
        dur_h = (close_ts - entry_ts).total_seconds() / 3600

        trades.append({
            "ts": sig_ts, "close_ts": close_ts, "dir": signal,
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
    l_wr  = round(len(wins[wins["dir"] == "LONG"]) / max(len(df_t[df_t["dir"] == "LONG"]), 1) * 100, 1)
    years = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    cagr  = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    eq    = pd.Series(equity)
    mdd   = round(((eq - eq.cummax()) / eq.cummax() * 100).min(), 2)

    return {
        "trades": len(df_t), "wins": len(wins), "losses": len(loses),
        "win_rate": wr, "long_wr": l_wr, "net_pf": npf,
        "final_bal": round(balance, 2),
        "cagr": round(cagr, 2), "max_dd": mdd,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


def main():
    W = 80
    print("=" * W)
    print("  LZR v8  PER-SYMBOL OPTIMAL TIMEFRAME  (LONG-only)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print("  Using each symbol's natural best TF from extended MTF sweep")
    print("  BTC→1D  ETH→3h  SOL→4h  BNB→12h  XRP→3h")
    print()

    # Load 1m data once
    print("  Loading 1m data...", flush=True)
    data_1m = {}
    for sym in SYMBOLS:
        df = load_1m(sym)
        data_1m[sym] = df
        print(f"    {sym}: {len(df):,} bars", flush=True)
    print()

    # Build higher-TF bars per symbol
    data_tf = {}
    for sym, (sig_tf, zone_tf, zw, cdl, cdw, *_) in PER_SYMBOL_CONFIG.items():
        data_tf[sym] = {
            "sig":  build_tf(data_1m[sym], sig_tf),
            "zone": build_tf(data_1m[sym], zone_tf),
            "zone_window": zw,
            "cd_loss": cdl,
            "cd_win": cdw,
            "sig_tf": sig_tf,
            "zone_tf": zone_tf,
        }
        print(f"    {sym}: sig {sig_tf} ({len(data_tf[sym]['sig']):,} bars)  "
              f"zone {zone_tf} ({len(data_tf[sym]['zone']):,} bars)", flush=True)
    print()

    # Grid: configs × symbols
    all_results = {}

    for cfg_name, long_only, vol_mult, sl_mult in CONFIGS:
        print(f"  ── Config: {cfg_name:<20} long={long_only}  vol={vol_mult}  sl={sl_mult}")
        cfg_stats = {}
        for sym in SYMBOLS:
            td = data_tf[sym]
            stats = run_lzr(sym, data_1m[sym],
                            td["sig"], td["zone"],
                            td["zone_window"], td["cd_loss"], td["cd_win"],
                            long_only, vol_mult, sl_mult)
            if stats:
                cfg_stats[sym] = stats
                s    = stats
                sign = "+" if s["cagr"] >= 0 else ""
                star = " ★" if s["net_pf"] > 1.0 else ""
                print(f"    {sym}  {s['trades']:>3}tr  WR {s['win_rate']:>5.1f}%  "
                      f"NetPF {s['net_pf']:.3f}  CAGR {sign}{s['cagr']:.1f}%  "
                      f"DD {s['max_dd']:.1f}%{star}")
            else:
                print(f"    {sym}  no trades")
        all_results[cfg_name] = cfg_stats
        print()

    # ─── Summary grids ────────────────────────────────────────────────────────
    print("=" * W)
    print("  CAGR% GRID  [config × symbol]")
    print("=" * W)
    hdr = f"  {'Config':<22} {'BTC-1D':>7} {'ETH-3h':>7} {'SOL-4h':>7} {'BNB-12h':>8} {'XRP-3h':>7} {'AVG':>7}"
    print(hdr)
    print("-" * W)
    for cfg_name, _, _, _ in CONFIGS:
        row = f"  {cfg_name:<22}"
        vals = []
        for sym in SYMBOLS:
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
    print("  NetPF GRID  [config × symbol]  (* = profitable)")
    print("-" * W)
    for cfg_name, _, _, _ in CONFIGS:
        row = f"  {cfg_name:<22}"
        vals = []
        for sym in SYMBOLS:
            s = all_results[cfg_name].get(sym)
            if s and s["net_pf"] != float("inf"):
                star = "*" if s["net_pf"] > 1.0 else " "
                row += f" {s['net_pf']:>6.3f}{star}"
                vals.append(s["net_pf"])
            else:
                row += f"    N/A "
        if vals:
            row += f" {np.mean(vals):>6.3f}"
        print(row)

    print()
    print("  TRADES GRID")
    print("-" * W)
    for cfg_name, _, _, _ in CONFIGS:
        row = f"  {cfg_name:<22}"
        total = 0
        for sym in SYMBOLS:
            s = all_results[cfg_name].get(sym)
            t = s["trades"] if s else 0
            row += f" {t:>7}"
            total += t
        row += f" {total:>7}"
        print(row)

    print()
    print("=" * W)
    print("  PORTFOLIO SUMMARY  (per-symbol optimal TF, LONG-only)")
    print("=" * W)
    print("  Comparison:")
    print("  Method                    Avg CAGR   Symbols profitable")
    print("  LZR 4h all-directions:    -1.6%       1/5")
    print("  LZR 4h LONG-only:         +2.1%       3/5")
    print("  Per-symbol optimal TF (baseline): above results →")

    lo_stats = all_results.get("LONG_ONLY", {})
    if lo_stats:
        avg_cagr = np.mean([s["cagr"] for s in lo_stats.values()])
        pos      = sum(1 for s in lo_stats.values() if s["cagr"] > 0)
        avg_npf  = np.mean([s["net_pf"] for s in lo_stats.values()
                            if s["net_pf"] != float("inf")])
        print(f"  Per-symbol optimal TF LONG-only:  {'+' if avg_cagr >= 0 else ''}{avg_cagr:.1f}%  "
              f"({pos}/5 profitable)")

    print()
    print("  KEY INSIGHT:")
    print("  BNB at 12h unlocks a previously 'broken' symbol (was -10.2% at 4h)")
    print("  XRP at 3h: same — was -16.7% at 4h")
    print("  Same LZR v8 engine, just different timeframe = dramatically different results")
    print("=" * W)

    # Save results
    rows = []
    for cfg_name, long_only, vol_mult, sl_mult in CONFIGS:
        for sym in SYMBOLS:
            s = all_results[cfg_name].get(sym)
            if s:
                rows.append({
                    "config": cfg_name, "symbol": sym,
                    "sig_tf": PER_SYMBOL_CONFIG[sym][0],
                    "long_only": long_only,
                    **s
                })
    if rows:
        df_out = pd.DataFrame(rows)
        out = OUTPUT_DIR / "backtest_optimal_per_symbol.csv"
        df_out.to_csv(out, index=False)
        print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
