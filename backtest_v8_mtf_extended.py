"""
backtest_v8_mtf_extended.py  --  LZR v8  EXTENDED TIMEFRAME SWEEP
==================================================================
Tests 5m → 1D signal timeframes (12 TFs total).
Zone TF = 4x signal TF. Execution always on 1m with gap-aware fills.
All bar-count parameters scaled to preserve the same calendar duration.
SHORT SL bug FIXED throughout.
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
WARMUP_BARS           = ATR_PERIOD * 4  # 56 signal-bars

CRYPTO_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# ─── Extended timeframe config ────────────────────────────────────────────────
# (sig_tf, zone_tf, bars_per_day, zone_window(7d), cd_loss(10h), cd_win(3h), label)
TIMEFRAMES = [
    ("5min",   "20min",   288, 2016, 120, 36, "5m"),
    ("10min",  "40min",   144, 1008,  60, 18, "10m"),
    ("15min",  "60min",    96,  672,  40, 12, "15m"),
    ("30min",  "120min",   48,  336,  20,  6, "30m"),
    ("1h",     "4h",       24,  168,  10,  3, "1h"),
    ("2h",     "8h",       12,   84,   5,  2, "2h"),
    ("3h",     "12h",       8,   56,   3,  1, "3h"),
    ("4h",     "16h",       6,   42,   3,  1, "4h"),
    ("6h",     "24h",       4,   28,   2,  1, "6h"),
    ("8h",     "32h",       3,   21,   2,  1, "8h"),
    ("12h",    "48h",       2,   14,   1,  1, "12h"),
    ("1D",     "4D",        1,    7,   1,  1, "1D"),
]


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_crypto(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def build_tfs(df_1m, sig_tf, zone_tf):
    agg = dict(open=("open","first"), high=("high","max"),
               low=("low","min"),   close=("close","last"), vol=("vol","sum"))
    return (df_1m.resample(sig_tf).agg(**agg).dropna(),
            df_1m.resample(zone_tf).agg(**agg).dropna())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


def find_zones(df_zone):
    n, lb = len(df_zone), SWING_LOOKBACK
    levels = []
    for i in range(lb, n - lb):
        hi = df_zone["high"].iloc[i]
        lo = df_zone["low"].iloc[i]
        if hi == df_zone["high"].iloc[i - lb:i + lb + 1].max():
            levels.append(hi)
        if lo == df_zone["low"].iloc[i - lb:i + lb + 1].min():
            levels.append(lo)
    merged = []
    for lvl in sorted(set(levels)):
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── 1m gap-aware executor (SHORT SL fixed) ───────────────────────────────────

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

    for m_idx in range(m_start, len(df_1m)):
        mr = df_1m.iloc[m_idx]
        closed, gross, exit_px = False, None, 0.0

        if state == "full":
            if direction == "LONG":
                if mr["open"] <= sl_px:
                    exit_px, gross, closed = mr["open"], (mr["open"] - entry_px) * qty, True
                elif mr["low"] <= sl_px:
                    exit_px, gross, closed = sl_px, (sl_px - entry_px) * qty, True
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
            else:  # SHORT
                if mr["open"] >= sl_px:
                    exit_px, gross, closed = mr["open"], (entry_px - mr["open"]) * qty, True
                elif mr["high"] >= sl_px:
                    exit_px, gross, closed = sl_px, (entry_px - sl_px) * qty, True
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
            else:  # SHORT partial
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


# ─── Per-TF backtest ──────────────────────────────────────────────────────────

def run_backtest(symbol, df_1m, df_sig, df_zone,
                 bars_per_day, zone_window, cd_loss, cd_win):
    FEE_SIDE     = CRYPTO_FEE_RT / 2
    m_timestamps = df_1m.index
    atr_s        = calc_atr(df_sig)

    df_zone["ema"]      = df_zone["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_zone["ema_lag"]  = df_zone["ema"].shift(1)
    df_zone["ema_prev"] = df_zone["ema"].shift(1 + EMA_LOOKBACK)
    ema_now  = df_zone["ema_lag"].reindex(df_sig.index,  method="ffill")
    ema_prev = df_zone["ema_prev"].reindex(df_sig.index, method="ffill")

    n_days    = len(df_sig) // bars_per_day + 2
    zone_snap = []
    for d in range(n_days):
        end_ts = df_sig.index[min(d * bars_per_day, len(df_sig) - 1)]
        past_z = df_zone[df_zone.index < end_ts]
        zone_snap.append(find_zones(past_z.iloc[-200:]) if len(past_z) >= SWING_LOOKBACK * 2 + 5 else [])

    balance       = INITIAL_BALANCE
    min_balance   = INITIAL_BALANCE * MIN_BAL_RATIO
    equity        = [balance]
    trades        = []
    zone_cooldown = {}
    zone_touches  = defaultdict(list)
    last_trigger  = -999

    i, n_sig = 0, len(df_sig)
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

        zb = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
        za = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]
        if zb:
            near_zone, direction = max(zb), "LONG"
        elif za:
            near_zone, direction = min(za), "SHORT"
        else:
            equity.append(balance); i += 1; continue

        if zone_cooldown.get(near_zone, 0) > i:
            equity.append(balance); i += 1; continue
        if i == last_trigger:
            equity.append(balance); i += 1; continue
        if balance < min_balance:
            equity.append(balance); i += 1; continue

        if direction == "LONG":
            triggered = row["low"] <= near_zone * (1 + TOUCH_BUF) and row["close"] > near_zone
        else:
            triggered = row["high"] >= near_zone * (1 - TOUCH_BUF) and row["close"] < near_zone
        if not triggered:
            equity.append(balance); i += 1; continue

        vol_avg = df_sig["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
        if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
            equity.append(balance); i += 1; continue

        en, ep = ema_now.iloc[i], ema_prev.iloc[i]
        if not (pd.isna(en) or pd.isna(ep)):
            if direction == "LONG"  and en <= ep:
                equity.append(balance); i += 1; continue
            if direction == "SHORT" and en >= ep:
                equity.append(balance); i += 1; continue

        recent = [b for b in zone_touches[near_zone] if i - b <= zone_window]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance); i += 1; continue

        if i + 1 >= n_sig:
            equity.append(balance); i += 1; continue

        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance); i += 1; continue

        entry_px  = df_sig["open"].iloc[i + 1]
        trade_atr = atr
        qty       = min((min(balance, MAX_BALANCE) * RISK_PCT) / sl_dist,
                        MAX_POSITION_NOTIONAL / entry_px)

        if direction == "LONG":
            sl_px         = entry_px - SL_MULT         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT         * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        entry_ts = df_sig.index[i + 1]
        m_start  = int(m_timestamps.searchsorted(entry_ts))

        info = _exec_1m(df_1m, m_start, entry_px, direction,
                        sl_px, partial_tp_px, hard_tp_px,
                        trade_atr, qty, FEE_SIDE, CRYPTO_SLIP_PCT)
        if info is None:
            equity.append(balance); i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip, final_state = info
        j = max(int(df_sig.index.searchsorted(close_ts, side="right")) - 1, i + 1)
        j = min(j, n_sig - 1)

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        zone_cooldown[near_zone] = j + (cd_loss if result == "LOSS" else cd_win)
        zone_touches[near_zone].append(i)
        last_trigger = i

        trades.append({"ts": ts, "close_ts": close_ts, "dir": direction,
                       "entry": round(entry_px, 6), "exit": round(exit_px, 6),
                       "qty": round(qty, 8), "gross": round(gross, 4),
                       "fee": round(total_fee, 4), "net": round(net, 4),
                       "result": result, "balance": round(balance, 4)})
        equity.append(bal_open)
        for _ in range(j - i - 1):
            equity.append(bal_open)
        equity.append(balance)
        i = j + 1

    if not trades:
        return None

    df_t = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]
    gw = wins["gross"].sum(); gl = abs(loses["gross"].sum())
    nw = wins["net"].sum();   nl = abs(loses["net"].sum())
    gross_pf = round(gw / gl, 3) if gl > 0 else float("inf")
    net_pf   = round(nw / nl, 3) if nl > 0 else float("inf")
    win_rate = round(len(wins) / len(df_t) * 100, 1)
    years    = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    cagr     = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    eq       = pd.Series(equity)
    max_dd   = round(((eq - eq.cummax()) / eq.cummax() * 100).min(), 2)
    tpy      = len(df_t) / max(years, 0.1)
    pr       = (df_t["net"] / (df_t["balance"] - df_t["net"])).values
    sharpe   = round(pr.mean() / pr.std() * np.sqrt(max(tpy, 1)) if pr.std() > 0 else 0, 2)
    long_t   = df_t[df_t["dir"] == "LONG"]
    short_t  = df_t[df_t["dir"] == "SHORT"]
    lwr      = round((long_t["net"] > 0).mean() * 100, 1) if len(long_t) else 0
    swr      = round((short_t["net"] > 0).mean() * 100, 1) if len(short_t) else 0

    return {"trades": len(df_t), "win_rate": win_rate,
            "long_wr": lwr, "short_wr": swr, "long_n": len(long_t), "short_n": len(short_t),
            "gross_pf": gross_pf, "net_pf": net_pf,
            "net_pnl": round(df_t["net"].sum(), 2), "final_bal": round(balance, 2),
            "cagr": round(cagr, 2), "max_dd": max_dd, "sharpe": sharpe, "years": round(years, 2)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 80
    print("=" * W)
    print("  LZR v8  EXTENDED MTF SWEEP  (5m → 1D, SHORT SL fixed)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(TIMEFRAMES)} timeframes  |  {len(CRYPTO_SYMBOLS)} symbols")
    print("=" * W)
    print("  NOTE: 5m and 10m bars take 5-10 min per symbol — please wait.\n")

    all_results = {}
    for sym in CRYPTO_SYMBOLS:
        print(f"\n  ── {sym} ──────────────────────────────────────────────────")
        try:
            df_1m = load_crypto(sym)
        except Exception as e:
            print(f"  ERROR: {e}"); continue

        all_results[sym] = {}
        for (sig_tf, zone_tf, bpd, zw, cdl, cdw, lbl) in TIMEFRAMES:
            t0 = datetime.now()
            try:
                df_sig, df_zone = build_tfs(df_1m, sig_tf, zone_tf)
                r = run_backtest(sym, df_1m, df_sig, df_zone, bpd, zw, cdl, cdw)
                sec = int((datetime.now() - t0).total_seconds())
                if r:
                    all_results[sym][lbl] = r
                    sign = "+" if r["cagr"] >= 0 else ""
                    star = " ★" if r["net_pf"] > 1.0 else ""
                    print(f"    [{lbl:>4}]  {r['trades']:>4}tr  "
                          f"WR {r['win_rate']:>4.1f}%  "
                          f"L:{r['long_wr']:>4.1f}% S:{r['short_wr']:>4.1f}%  "
                          f"NetPF {r['net_pf']:>5.3f}  "
                          f"CAGR {sign}{r['cagr']:>5.1f}%  "
                          f"DD {r['max_dd']:>6.1f}%  ({sec}s){star}", flush=True)
                else:
                    print(f"    [{lbl:>4}]  no trades ({int((datetime.now()-t0).total_seconds())}s)")
            except Exception as e:
                print(f"    [{lbl:>4}]  ERROR: {e}")

    labels = [t[6] for t in TIMEFRAMES]

    print(f"\n{'=' * W}")
    print("  SUMMARY BY TIMEFRAME  (avg across all symbols)")
    print(f"{'=' * W}")
    hdr = f"  {'TF':>5}  {'Trades':>6}  {'WR%':>6}  {'L-WR':>5}  {'S-WR':>5}  {'NetPF':>6}  {'CAGR%':>7}  {'MaxDD':>7}  {'Sharpe':>7}  POS"
    print(hdr)
    print("-" * W)
    for lbl in labels:
        rows = [all_results[s][lbl] for s in CRYPTO_SYMBOLS if lbl in all_results.get(s, {})]
        if not rows: continue
        avg_tr  = np.mean([r["trades"]   for r in rows])
        avg_wr  = np.mean([r["win_rate"] for r in rows])
        avg_lwr = np.mean([r["long_wr"]  for r in rows])
        avg_swr = np.mean([r["short_wr"] for r in rows])
        fin     = [r["net_pf"] for r in rows if r["net_pf"] != float("inf")]
        avg_npf = np.mean(fin) if fin else 0
        avg_cg  = np.mean([r["cagr"]    for r in rows])
        avg_dd  = np.mean([r["max_dd"]  for r in rows])
        avg_sh  = np.mean([r["sharpe"]  for r in rows])
        pos     = sum(1 for r in rows if r["net_pf"] > 1.0)
        flag    = "  ★ PROFITABLE" if avg_npf > 1.0 else ""
        print(f"  {lbl:>5}  {avg_tr:>6.0f}  {avg_wr:>6.1f}  "
              f"{avg_lwr:>5.1f}  {avg_swr:>5.1f}  "
              f"{avg_npf:>6.3f}  {avg_cg:>+7.1f}  "
              f"{avg_dd:>7.2f}  {avg_sh:>7.2f}  "
              f"{pos}/{len(rows)}{flag}")

    print(f"\n{'=' * W}")
    print("  CAGR% GRID")
    print(f"{'=' * W}")
    print(f"  {'Symbol':<10}" + "".join(f"  {l:>6}" for l in labels))
    print("-" * W)
    for sym in CRYPTO_SYMBOLS:
        row = f"  {sym:<10}"
        for lbl in labels:
            r = all_results.get(sym, {}).get(lbl)
            row += f"  {r['cagr']:>+6.1f}" if r else f"  {'N/A':>6}"
        print(row)

    print(f"\n  NetPF GRID  (* = profitable)")
    print("-" * W)
    print(f"  {'Symbol':<10}" + "".join(f"  {l:>6}" for l in labels))
    print("-" * W)
    for sym in CRYPTO_SYMBOLS:
        row = f"  {sym:<10}"
        for lbl in labels:
            r = all_results.get(sym, {}).get(lbl)
            if r:
                m = "*" if r["net_pf"] > 1.0 else " "
                row += f"  {r['net_pf']:>5.3f}{m}"
            else:
                row += f"  {'N/A':>6}"
        print(row)

    csv_rows = []
    for sym in CRYPTO_SYMBOLS:
        for lbl, r in all_results.get(sym, {}).items():
            csv_rows.append({"symbol": sym, "tf": lbl, **r})
    if csv_rows:
        out = OUTPUT_DIR / "backtest_v8_mtf_extended.csv"
        pd.DataFrame(csv_rows).to_csv(out, index=False)
        print(f"\n  Saved → {out}")
    print("=" * W)


if __name__ == "__main__":
    main()
