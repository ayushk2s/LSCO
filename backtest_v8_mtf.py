"""
backtest_v8_mtf.py  --  LZR v8  MULTI-TIMEFRAME SWEEP
=======================================================
Tests the fixed v8 engine across 5 signal timeframes:
  15m  30m  1h  2h  4h
Signal detection runs on the chosen TF; execution always stays on 1m.
Zone TF = 4x the signal TF (same ratio as v7/v8 original).
Bar-count parameters (zone_window, cooldown) are scaled to preserve
the same calendar-time duration as the 1h baseline.

SHORT SL bug is FIXED: gross = (entry_px - sl_px) * qty for shorts.
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
EMA_PERIOD      = 20
EMA_LOOKBACK    = 3
ZONE_MAX_TOUCH  = 2

CRYPTO_FEE_RT   = 0.0004
CRYPTO_SLIP_PCT = 0.0003

MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20
WARMUP_BARS           = ATR_PERIOD * 4   # 56 signal bars

CRYPTO_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# ─── Timeframe sweep config ───────────────────────────────────────────────────
# (signal_tf, zone_tf, bars_per_day, zone_window_bars, cd_loss_bars, cd_win_bars)
# zone_window and cooldowns are scaled to keep the same calendar time as 1h baseline:
#   1h baseline: zone_window=168 (7d), cd_loss=10 (10h), cd_win=3 (3h)
TIMEFRAMES = [
    ("15min", "60min",  96,  672, 40, 12),
    ("30min", "120min", 48,  336, 20,  6),
    ("1h",    "4h",     24,  168, 10,  3),
    ("2h",    "8h",     12,   84,  5,  2),
    ("4h",    "16h",     6,   42,  3,  1),
]


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_crypto(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def build_signal_and_zone(df_1m, sig_tf, zone_tf):
    """Resample 1m data to signal TF and zone TF."""
    agg = dict(open=("open","first"), high=("high","max"),
               low=("low","min"),   close=("close","last"), vol=("vol","sum"))
    df_sig  = df_1m.resample(sig_tf).agg(**agg).dropna()
    df_zone = df_1m.resample(zone_tf).agg(**agg).dropna()
    return df_sig, df_zone


# ─── Helpers ──────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


def find_zones(df_zone):
    n, lb = len(df_zone), SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_zone["high"].iloc[i] == df_zone["high"].iloc[i - lb:i + lb + 1].max():
            highs.append(df_zone["high"].iloc[i])
        if df_zone["low"].iloc[i] == df_zone["low"].iloc[i - lb:i + lb + 1].min():
            lows.append(df_zone["low"].iloc[i])
    levels, merged = sorted(set(highs + lows)), []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── 1m executor — v8 gap-aware + SHORT SL FIXED ─────────────────────────────

def _exec_1m(df_1m, m_start, entry_px, direction,
              sl_px, partial_tp_px, hard_tp_px,
              trade_atr, qty, fee_side, slip_pct):
    half    = qty / 2
    state   = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct

    n = len(df_1m)
    for m_idx in range(m_start, n):
        mr      = df_1m.iloc[m_idx]
        gross   = None
        closed  = False
        exit_px = 0.0

        if state == "full":
            if direction == "LONG":
                if mr["open"] <= sl_px:                    # gap-aware SL
                    exit_px = mr["open"]
                    gross   = (exit_px - entry_px) * qty
                    closed  = True
                elif mr["low"] <= sl_px:
                    exit_px = sl_px
                    gross   = (sl_px - entry_px) * qty
                    closed  = True
                elif mr["open"] >= partial_tp_px:          # gap-aware pTP
                    partial_locked  = (partial_tp_px - entry_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["open"]
                    trail_sl        = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"
                elif mr["high"] >= partial_tp_px:
                    partial_locked  = (partial_tp_px - entry_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["high"]
                    trail_sl        = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"
            else:  # SHORT
                if mr["open"] >= sl_px:                    # gap-aware SL
                    exit_px = mr["open"]
                    gross   = (entry_px - exit_px) * qty   # FIXED: entry - exit
                    closed  = True
                elif mr["high"] >= sl_px:
                    exit_px = sl_px
                    gross   = (entry_px - sl_px) * qty     # FIXED: entry - exit
                    closed  = True
                elif mr["open"] <= partial_tp_px:          # gap-aware pTP
                    partial_locked  = (entry_px - partial_tp_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["open"]
                    trail_sl        = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"
                elif mr["low"] <= partial_tp_px:
                    partial_locked  = (entry_px - partial_tp_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["low"]
                    trail_sl        = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"

        elif state == "partial":
            if direction == "LONG":
                old_trail = trail_sl
                if mr["open"] <= old_trail:                # gap-aware trail
                    exit_px = mr["open"]
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True
                elif mr["open"] >= hard_tp_px:             # gap-aware hTP
                    exit_px = hard_tp_px
                    gross   = partial_locked + (hard_tp_px - entry_px) * half
                    closed  = True
                else:
                    running_ext = max(running_ext, mr["high"])
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    if mr["low"] <= old_trail:
                        exit_px = old_trail
                        gross   = partial_locked + (old_trail  - entry_px) * half
                        closed  = True
                    elif mr["high"] >= hard_tp_px:
                        exit_px = hard_tp_px
                        gross   = partial_locked + (hard_tp_px - entry_px) * half
                        closed  = True
            else:  # SHORT partial
                old_trail = trail_sl
                if mr["open"] >= old_trail:                # gap-aware trail
                    exit_px = mr["open"]
                    gross   = partial_locked + (entry_px - exit_px) * half
                    closed  = True
                elif mr["open"] <= hard_tp_px:             # gap-aware hTP
                    exit_px = hard_tp_px
                    gross   = partial_locked + (entry_px - hard_tp_px) * half
                    closed  = True
                else:
                    running_ext = min(running_ext, mr["low"])
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    if mr["high"] >= old_trail:
                        exit_px = old_trail
                        gross   = partial_locked + (entry_px - old_trail)  * half
                        closed  = True
                    elif mr["low"] <= hard_tp_px:
                        exit_px = hard_tp_px
                        gross   = partial_locked + (entry_px - hard_tp_px) * half
                        closed  = True

        if closed:
            exit_qty   = qty if state == "full" else half
            exit_fee   = exit_px * exit_qty * fee_side
            exit_slip  = exit_px * exit_qty * slip_pct
            return (df_1m.index[m_idx], gross, exit_px,
                    fee_acc + exit_fee, slip_acc + exit_slip, state)

    return None


# ─── Backtest engine (parametric TF) ─────────────────────────────────────────

def run_backtest(symbol, df_1m, df_sig, df_zone,
                 bars_per_day, zone_window, cd_loss, cd_win):
    FEE_SIDE     = CRYPTO_FEE_RT / 2
    m_timestamps = df_1m.index

    atr_s = calc_atr(df_sig)

    # EMA on zone TF for trend filter
    df_zone["ema"]      = df_zone["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_zone["ema_lag"]  = df_zone["ema"].shift(1)
    df_zone["ema_prev"] = df_zone["ema"].shift(1 + EMA_LOOKBACK)
    ema_now  = df_zone["ema_lag"].reindex(df_sig.index,  method="ffill")
    ema_prev = df_zone["ema_prev"].reindex(df_sig.index, method="ffill")

    # Zone snapshots updated once per calendar day
    n_days    = len(df_sig) // bars_per_day + 2
    zone_snap = []
    for d in range(n_days):
        end_ts  = df_sig.index[min(d * bars_per_day, len(df_sig) - 1)]
        past_z  = df_zone[df_zone.index < end_ts]
        zone_snap.append(find_zones(past_z.iloc[-200:]) if len(past_z) >= SWING_LOOKBACK * 2 + 5 else [])

    balance          = INITIAL_BALANCE
    min_balance      = INITIAL_BALANCE * MIN_BAL_RATIO
    equity           = [balance]
    trades           = []
    zone_cooldown    = {}
    zone_touches     = defaultdict(list)
    last_trigger_bar = -999

    i = 0
    n_sig = len(df_sig)

    while i < n_sig:
        if i < WARMUP_BARS:
            equity.append(balance); i += 1; continue

        ts    = df_sig.index[i]
        row   = df_sig.iloc[i]
        atr   = atr_s.iloc[i]
        price = row["close"]

        if atr <= 0 or np.isnan(atr):
            equity.append(balance); i += 1; continue

        day_idx = i // bars_per_day
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            equity.append(balance); i += 1; continue
        zones = zone_snap[day_idx]

        zones_below = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
        zones_above = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]
        if zones_below:
            near_zone, near_dir = max(zones_below), "LONG"
        elif zones_above:
            near_zone, near_dir = min(zones_above), "SHORT"
        else:
            equity.append(balance); i += 1; continue

        if zone_cooldown.get(near_zone, 0) > i:
            equity.append(balance); i += 1; continue
        if i == last_trigger_bar:
            equity.append(balance); i += 1; continue
        if balance < min_balance:
            equity.append(balance); i += 1; continue

        if near_dir == "LONG":
            triggered = row["low"] <= near_zone * (1 + TOUCH_BUF) and row["close"] > near_zone
        else:
            triggered = row["high"] >= near_zone * (1 - TOUCH_BUF) and row["close"] < near_zone
        if not triggered:
            equity.append(balance); i += 1; continue

        # Volume filter
        vol_avg = df_sig["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
        if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
            equity.append(balance); i += 1; continue

        # EMA trend filter
        en, ep = ema_now.iloc[i], ema_prev.iloc[i]
        if not (pd.isna(en) or pd.isna(ep)):
            if near_dir == "LONG"  and en <= ep:
                equity.append(balance); i += 1; continue
            if near_dir == "SHORT" and en >= ep:
                equity.append(balance); i += 1; continue

        # Zone freshness
        recent = [b for b in zone_touches[near_zone] if i - b <= zone_window]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance); i += 1; continue

        if i + 1 >= n_sig:
            equity.append(balance); i += 1; continue

        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance); i += 1; continue

        eff_bal  = min(balance, MAX_BALANCE)
        risk_usd = eff_bal * RISK_PCT
        qty      = risk_usd / sl_dist
        entry_px = df_sig["open"].iloc[i + 1]
        trade_atr = atr

        qty = min(qty, MAX_POSITION_NOTIONAL / entry_px)

        if near_dir == "LONG":
            sl_px         = entry_px - SL_MULT         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT         * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        entry_ts = df_sig.index[i + 1]
        m_start  = int(m_timestamps.searchsorted(entry_ts))

        close_info = _exec_1m(df_1m, m_start, entry_px, near_dir,
                               sl_px, partial_tp_px, hard_tp_px,
                               trade_atr, qty, FEE_SIDE, CRYPTO_SLIP_PCT)
        if close_info is None:
            equity.append(balance); i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip, final_state = close_info
        close_sig_idx = int(df_sig.index.searchsorted(close_ts, side="right")) - 1
        close_sig_idx = max(close_sig_idx, i + 1)
        close_sig_idx = min(close_sig_idx, n_sig - 1)

        net      = gross - total_fee - total_slip
        result   = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        zone_cooldown[near_zone]  = close_sig_idx + (cd_loss if result == "LOSS" else cd_win)
        zone_touches[near_zone].append(i)
        last_trigger_bar = i

        trades.append({"ts": ts, "close_ts": close_ts, "dir": near_dir,
                       "entry": round(entry_px, 6), "exit": round(exit_px, 6),
                       "qty": round(qty, 8), "gross": round(gross, 4),
                       "fee": round(total_fee, 4), "slip": round(total_slip, 4),
                       "net": round(net, 4), "result": result,
                       "balance_open": round(bal_open, 4), "balance": round(balance, 4)})

        equity.append(bal_open)
        for _ in range(close_sig_idx - i - 1):
            equity.append(bal_open)
        equity.append(balance)
        i = close_sig_idx + 1

    # ─── Stats ────────────────────────────────────────────────────────────────
    if not trades:
        return None

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]

    gross_wins = wins["gross"].sum()
    gross_loss = abs(loses["gross"].sum())
    net_wins   = wins["net"].sum()
    net_loss   = abs(loses["net"].sum())

    gross_pf = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    net_pf   = round(net_wins   / net_loss,   3) if net_loss   > 0 else float("inf")
    win_rate = round(len(wins) / len(df_t) * 100, 1)

    final_bal = balance
    years     = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    cagr      = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    net_pnl   = df_t["net"].sum()

    eq       = pd.Series(equity)
    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    max_dd   = round(dd_pct.min(), 2)

    tpy      = len(df_t) / max(years, 0.1)
    pct_rets = (df_t["net"] / df_t["balance_open"]).values
    sharpe   = round((pct_rets.mean() / pct_rets.std() * np.sqrt(max(tpy, 1)))
                     if pct_rets.std() > 0 else 0, 2)

    long_trades  = df_t[df_t["dir"] == "LONG"]
    short_trades = df_t[df_t["dir"] == "SHORT"]
    long_wr  = round((long_trades["net"] > 0).mean() * 100, 1) if len(long_trades) else 0
    short_wr = round((short_trades["net"] > 0).mean() * 100, 1) if len(short_trades) else 0

    return {
        "trades": len(df_t), "wins": len(wins), "losses": len(loses),
        "win_rate": win_rate, "long_wr": long_wr, "short_wr": short_wr,
        "long_n": len(long_trades), "short_n": len(short_trades),
        "gross_pf": gross_pf, "net_pf": net_pf,
        "net_pnl": round(net_pnl, 2), "final_bal": round(final_bal, 2),
        "cagr": round(cagr, 2), "max_dd": max_dd, "sharpe": sharpe,
        "years": round(years, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 72
    print("=" * W)
    print("  LZR v8  MULTI-TIMEFRAME SWEEP  (SHORT SL bug fixed)")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print(f"  Signal TFs tested : 15m  30m  1h  2h  4h")
    print(f"  Zone TF           : 4x signal TF")
    print(f"  Execution         : 1m bars (gap-aware fills)")
    print(f"  Symbols           : {', '.join(CRYPTO_SYMBOLS)}")
    print(f"  Risk/trade        : {RISK_PCT*100:.0f}%  |  Balance cap ${MAX_BALANCE:,.0f}")
    print()

    # Collect all results: results[symbol][tf_label]
    all_results = {}

    for sym in CRYPTO_SYMBOLS:
        print(f"  Loading {sym}...", flush=True)
        try:
            df_1m = load_crypto(sym)
        except Exception as e:
            print(f"  ERROR loading {sym}: {e}"); continue

        all_results[sym] = {}

        for (sig_tf, zone_tf, bars_per_day, zone_window, cd_loss, cd_win) in TIMEFRAMES:
            t0 = datetime.now()
            try:
                df_sig, df_zone = build_signal_and_zone(df_1m, sig_tf, zone_tf)
                r = run_backtest(sym, df_1m, df_sig, df_zone,
                                 bars_per_day, zone_window, cd_loss, cd_win)
                elapsed = (datetime.now() - t0).seconds
                if r:
                    all_results[sym][sig_tf] = r
                    sign = "+" if r["cagr"] >= 0 else ""
                    print(f"    [{sig_tf:>5}]  {r['trades']:>3} trades  "
                          f"WR {r['win_rate']:>4.1f}%  "
                          f"L:{r['long_wr']:>4.1f}% S:{r['short_wr']:>4.1f}%  "
                          f"NetPF {r['net_pf']:>5.3f}  "
                          f"CAGR {sign}{r['cagr']:.1f}%  "
                          f"DD {r['max_dd']:.1f}%  ({elapsed}s)", flush=True)
                else:
                    print(f"    [{sig_tf:>5}]  no trades")
            except Exception as e:
                print(f"    [{sig_tf:>5}]  ERROR: {e}")

        print()

    # ─── Summary table per TF ─────────────────────────────────────────────────
    print("=" * W)
    print("  SUMMARY BY TIMEFRAME  (avg across symbols)")
    print("=" * W)
    tf_labels = [t[0] for t in TIMEFRAMES]
    hdr = f"  {'TF':>6}  {'Trades':>7}  {'WR%':>6}  {'L-WR%':>6}  {'S-WR%':>6}  {'NetPF':>6}  {'CAGR%':>7}  {'MaxDD%':>7}  {'Sharpe':>7}"
    print(hdr)
    print("-" * W)
    for tf in tf_labels:
        rows = [all_results[s][tf] for s in CRYPTO_SYMBOLS if tf in all_results.get(s, {})]
        if not rows: continue
        avg_tr  = np.mean([r["trades"]  for r in rows])
        avg_wr  = np.mean([r["win_rate"] for r in rows])
        avg_lwr = np.mean([r["long_wr"] for r in rows])
        avg_swr = np.mean([r["short_wr"] for r in rows])
        fin = [r["net_pf"] for r in rows if r["net_pf"] != float("inf")]
        avg_npf = np.mean(fin) if fin else 0
        avg_cagr= np.mean([r["cagr"]   for r in rows])
        avg_dd  = np.mean([r["max_dd"] for r in rows])
        avg_sh  = np.mean([r["sharpe"] for r in rows])
        pos = sum(1 for r in rows if r["cagr"] > 0)
        flag = " <-- PROFITABLE" if avg_npf > 1.0 else ""
        print(f"  {tf:>6}  {avg_tr:>7.0f}  {avg_wr:>6.1f}  "
              f"{avg_lwr:>6.1f}  {avg_swr:>6.1f}  "
              f"{avg_npf:>6.3f}  {avg_cagr:>+7.1f}  "
              f"{avg_dd:>7.2f}  {avg_sh:>7.2f}  "
              f"({pos}/{len(rows)} sym pos){flag}")

    # ─── Per-symbol grid ──────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  CAGR% GRID  [symbol x timeframe]")
    print("=" * W)
    hdr2 = f"  {'Symbol':<10}" + "".join(f"  {tf:>7}" for tf in tf_labels)
    print(hdr2)
    print("-" * W)
    for sym in CRYPTO_SYMBOLS:
        row_str = f"  {sym:<10}"
        for tf in tf_labels:
            r = all_results.get(sym, {}).get(tf)
            if r:
                row_str += f"  {r['cagr']:>+7.1f}"
            else:
                row_str += f"  {'N/A':>7}"
        print(row_str)

    print()
    print("  NetPF GRID  [symbol x timeframe]  (>1.0 = profitable before costs eaten)")
    print("-" * W)
    print(hdr2)
    print("-" * W)
    for sym in CRYPTO_SYMBOLS:
        row_str = f"  {sym:<10}"
        for tf in tf_labels:
            r = all_results.get(sym, {}).get(tf)
            if r:
                mark = "*" if r["net_pf"] > 1.0 else " "
                row_str += f"  {r['net_pf']:>6.3f}{mark}"
            else:
                row_str += f"  {'N/A':>7}"
        print(row_str)
    print("  (* = Net PF > 1.0)")

    # ─── Save CSV ─────────────────────────────────────────────────────────────
    csv_rows = []
    for sym in CRYPTO_SYMBOLS:
        for tf, r in all_results.get(sym, {}).items():
            csv_rows.append({"symbol": sym, "tf": tf, **r})
    if csv_rows:
        csv = OUTPUT_DIR / "backtest_v8_mtf.csv"
        pd.DataFrame(csv_rows).to_csv(csv, index=False)
        print(f"\n  Results saved -> {csv}")
    print("=" * W)


if __name__ == "__main__":
    main()
