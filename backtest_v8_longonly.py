"""
backtest_v8_longonly.py  --  LZR v8  IMPROVEMENT SWEEP
=======================================================
Tests 7 configurations on the 4h-signal / 1m-execution engine:

  Baseline   : LONG+SHORT  vol 1.8  SL 0.75 ATR  (bug-fixed original)
  LONG_ONLY  : LONG only   vol 1.8  SL 0.75 ATR
  L+VOL2.5   : LONG only   vol 2.5  SL 0.75 ATR
  L+SL0.6    : LONG only   vol 1.8  SL 0.60 ATR  (better R:R)
  L+SL0.5    : LONG only   vol 1.8  SL 0.50 ATR  (best R:R)
  L+V2.5+SL06: LONG only   vol 2.5  SL 0.60 ATR
  L+V2.5+SL05: LONG only   vol 2.5  SL 0.50 ATR  (all three combined)

Tighter SL improves R:R automatically because qty scales with 1/SL_dist
(same fixed-risk $ per trade, but larger qty → bigger wins for same risk).

SHORT SL bug FIXED. NEVER modifies account_data.py or liq_algo.py.
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Static config (unchanged across all sweeps) ─────────────────────────────
DATA_DIR    = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR  = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10

PARTIAL_TP_MULT = 1.0
TRAIL_DIST_MULT = 0.5
HARD_TP_MULT    = 3.0
APPROACH_PCT    = 0.008
TOUCH_BUF       = 0.003
ATR_PERIOD      = 14
SWING_LOOKBACK  = 20
MIN_ZONE_GAP    = 0.005
VOL_LOOKBACK    = 20
EMA_PERIOD      = 20
EMA_LOOKBACK    = 3
ZONE_MAX_TOUCH  = 2
CRYPTO_FEE_RT   = 0.0004
CRYPTO_SLIP_PCT = 0.0003
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20
WARMUP_BARS           = ATR_PERIOD * 4  # 56 bars

# 4h signal parameters
BARS_PER_DAY = 6
ZONE_WINDOW  = 42
CD_LOSS      = 3
CD_WIN       = 1

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# ─── Config sweep ─────────────────────────────────────────────────────────────
# (name, long_only, vol_mult, sl_mult)
CONFIGS = [
    ("Baseline",     False, 1.8, 0.75),
    ("LONG_ONLY",    True,  1.8, 0.75),
    ("L+VOL2.5",     True,  2.5, 0.75),
    ("L+SL0.6",      True,  1.8, 0.60),
    ("L+SL0.5",      True,  1.8, 0.50),
    ("L+V2.5+SL0.6", True,  2.5, 0.60),
    ("L+V2.5+SL0.5", True,  2.5, 0.50),
]


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_crypto(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def build_tfs(df_1m):
    agg = dict(open=("open","first"), high=("high","max"),
               low=("low","min"), close=("close","last"), vol=("vol","sum"))
    df_sig  = df_1m.resample("4h").agg(**agg).dropna()
    df_zone = df_1m.resample("16h").agg(**agg).dropna()
    return df_sig, df_zone


# ─── Helpers ──────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


def find_zones(df_zone):
    n, lb = len(df_zone), SWING_LOOKBACK
    levels = []
    for i in range(lb, n - lb):
        if df_zone["high"].iloc[i] == df_zone["high"].iloc[i - lb:i + lb + 1].max():
            levels.append(df_zone["high"].iloc[i])
        if df_zone["low"].iloc[i] == df_zone["low"].iloc[i - lb:i + lb + 1].min():
            levels.append(df_zone["low"].iloc[i])
    merged = []
    for lvl in sorted(set(levels)):
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── 1m executor (SHORT SL fixed, gap-aware) ─────────────────────────────────

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
            else:  # SHORT — SL BUG FIXED
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


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(symbol, df_1m, df_sig, df_zone,
                 long_only, vol_mult, sl_mult):
    FEE_SIDE     = CRYPTO_FEE_RT / 2
    m_timestamps = df_1m.index
    atr_s        = calc_atr(df_sig)

    df_zone["ema"]      = df_zone["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_zone["ema_lag"]  = df_zone["ema"].shift(1)
    df_zone["ema_prev"] = df_zone["ema"].shift(1 + EMA_LOOKBACK)
    ema_now  = df_zone["ema_lag"].reindex(df_sig.index,  method="ffill")
    ema_prev = df_zone["ema_prev"].reindex(df_sig.index, method="ffill")

    n_days    = len(df_sig) // BARS_PER_DAY + 2
    zone_snap = []
    for d in range(n_days):
        end_ts = df_sig.index[min(d * BARS_PER_DAY, len(df_sig) - 1)]
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

        day_idx = i // BARS_PER_DAY
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

        # ── LONG-only filter ──────────────────────────────────────────────────
        if long_only and direction == "SHORT":
            equity.append(balance); i += 1; continue

        if zone_cooldown.get(near_zone, 0) > i or i == last_trigger:
            equity.append(balance); i += 1; continue
        if balance < min_balance:
            equity.append(balance); i += 1; continue

        if direction == "LONG":
            triggered = row["low"] <= near_zone * (1 + TOUCH_BUF) and row["close"] > near_zone
        else:
            triggered = row["high"] >= near_zone * (1 - TOUCH_BUF) and row["close"] < near_zone
        if not triggered:
            equity.append(balance); i += 1; continue

        # ── Configurable volume filter ────────────────────────────────────────
        vol_avg = df_sig["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
        if vol_avg > 0 and row["vol"] < vol_avg * vol_mult:
            equity.append(balance); i += 1; continue

        en, ep = ema_now.iloc[i], ema_prev.iloc[i]
        if not (pd.isna(en) or pd.isna(ep)):
            if direction == "LONG"  and en <= ep:
                equity.append(balance); i += 1; continue
            if direction == "SHORT" and en >= ep:
                equity.append(balance); i += 1; continue

        recent = [b for b in zone_touches[near_zone] if i - b <= ZONE_WINDOW]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance); i += 1; continue

        if i + 1 >= n_sig:
            equity.append(balance); i += 1; continue

        sl_dist = sl_mult * atr
        if sl_dist <= 0:
            equity.append(balance); i += 1; continue

        entry_px  = df_sig["open"].iloc[i + 1]
        trade_atr = atr
        qty       = min((min(balance, MAX_BALANCE) * RISK_PCT) / sl_dist,
                        MAX_POSITION_NOTIONAL / entry_px)

        if direction == "LONG":
            sl_px         = entry_px - sl_mult         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + sl_mult         * trade_atr
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

        net      = gross - total_fee - total_slip
        result   = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        zone_cooldown[near_zone] = j + (CD_LOSS if result == "LOSS" else CD_WIN)
        zone_touches[near_zone].append(i)
        last_trigger = i

        trades.append({"ts": ts, "dir": direction, "entry": entry_px,
                       "exit": exit_px, "net": round(net, 4), "result": result})
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
    gw    = wins["net"].apply(lambda x: x).sum()       # using net for PF calc is wrong
    # recompute gross from trade dict
    gw = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))
    net_pf  = round(gw / gl, 3) if gl > 0 else float("inf")
    win_rate = round(len(wins) / len(df_t) * 100, 1)
    years    = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    cagr     = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    eq       = pd.Series(equity)
    max_dd   = round(((eq - eq.cummax()) / eq.cummax() * 100).min(), 2)
    net_pnl  = round(df_t["net"].sum(), 2)

    longs  = df_t[df_t["dir"] == "LONG"]
    shorts = df_t[df_t["dir"] == "SHORT"]
    lwr    = round((longs["net"] > 0).mean() * 100, 1) if len(longs) else 0
    swr    = round((shorts["net"] > 0).mean() * 100, 1) if len(shorts) else 0

    return {
        "trades": len(df_t), "win_rate": win_rate,
        "long_n": len(longs), "short_n": len(shorts),
        "long_wr": lwr, "short_wr": swr,
        "net_pf": net_pf, "net_pnl": net_pnl,
        "final_bal": round(balance, 2),
        "cagr": round(cagr, 2), "max_dd": max_dd,
        "years": round(years, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 80
    print("=" * W)
    print("  LZR v8  IMPROVEMENT SWEEP  (4h signal / 1m execution)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print("  R:R note: tighter SL → more qty per trade (same $ risk)")
    print("    SL 0.75 ATR → partial TP at 1.0 ATR = 1.33:1 R:R")
    print("    SL 0.60 ATR → partial TP at 1.0 ATR = 1.67:1 R:R")
    print("    SL 0.50 ATR → partial TP at 1.0 ATR = 2.00:1 R:R")
    print()

    # Pre-load and pre-build signal/zone TFs (shared across all configs per symbol)
    dfs = {}
    for sym in CRYPTO_SYMBOLS:
        try:
            df_1m = load_crypto(sym)
            df_sig, df_zone = build_tfs(df_1m)
            dfs[sym] = (df_1m, df_sig, df_zone)
            print(f"  Loaded {sym}  ({len(df_1m):,} 1m bars)", flush=True)
        except Exception as e:
            print(f"  ERROR loading {sym}: {e}")
    print()

    # all_results[sym][cfg_name] = result_dict
    all_results = {sym: {} for sym in CRYPTO_SYMBOLS}

    for (cfg_name, long_only, vol_mult, sl_mult) in CONFIGS:
        print(f"  ── Config: {cfg_name:18s}  long_only={long_only}  vol={vol_mult}  sl={sl_mult} ATR")
        for sym in CRYPTO_SYMBOLS:
            if sym not in dfs:
                continue
            df_1m, df_sig, df_zone = dfs[sym]
            # Clone zone df to avoid mutation across configs
            df_zone_c = df_zone.copy()
            try:
                r = run_backtest(sym, df_1m, df_sig, df_zone_c,
                                 long_only, vol_mult, sl_mult)
                if r:
                    all_results[sym][cfg_name] = r
                    sign = "+" if r["cagr"] >= 0 else ""
                    star = " ★" if r["net_pf"] > 1.0 else ""
                    print(f"    {sym}  {r['trades']:>3}tr  "
                          f"WR {r['win_rate']:>5.1f}%  "
                          f"L:{r['long_wr']:>4.1f}%({r['long_n']}) "
                          f"S:{r['short_wr']:>4.1f}%({r['short_n']})  "
                          f"NetPF {r['net_pf']:>5.3f}  "
                          f"CAGR {sign}{r['cagr']:>5.1f}%  "
                          f"DD {r['max_dd']:>6.1f}%{star}", flush=True)
                else:
                    print(f"    {sym}  no trades")
            except Exception as e:
                print(f"    {sym}  ERROR: {e}")
        print()

    cfg_names = [c[0] for c in CONFIGS]

    # ─── Summary grids ────────────────────────────────────────────────────────
    print("=" * W)
    print("  CAGR% GRID  [config × symbol]")
    print("=" * W)
    print(f"  {'Config':<20}" + "".join(f"  {s[:7]:>7}" for s in CRYPTO_SYMBOLS) + "   AVG")
    print("-" * W)
    for cfg_name in cfg_names:
        row = f"  {cfg_name:<20}"
        cagrs = []
        for sym in CRYPTO_SYMBOLS:
            r = all_results[sym].get(cfg_name)
            if r:
                row += f"  {r['cagr']:>+7.1f}"
                cagrs.append(r["cagr"])
            else:
                row += f"  {'N/A':>7}"
        if cagrs:
            row += f"  {np.mean(cagrs):>+7.1f}"
        print(row)

    print()
    print("  NetPF GRID  [config × symbol]  (* = profitable)")
    print("-" * W)
    print(f"  {'Config':<20}" + "".join(f"  {s[:7]:>7}" for s in CRYPTO_SYMBOLS) + "   AVG")
    print("-" * W)
    for cfg_name in cfg_names:
        row = f"  {cfg_name:<20}"
        npfs = []
        for sym in CRYPTO_SYMBOLS:
            r = all_results[sym].get(cfg_name)
            if r:
                npf = r["net_pf"]
                mark = "*" if npf > 1.0 else " "
                row += f"  {npf:>6.3f}{mark}"
                if npf != float("inf"):
                    npfs.append(npf)
            else:
                row += f"  {'N/A':>7}"
        if npfs:
            avg = np.mean(npfs)
            mark = "*" if avg > 1.0 else " "
            row += f"  {avg:>6.3f}{mark}"
        print(row)

    print()
    print("  TRADES GRID  [config × symbol]")
    print("-" * W)
    print(f"  {'Config':<20}" + "".join(f"  {s[:7]:>7}" for s in CRYPTO_SYMBOLS) + "   TOT")
    print("-" * W)
    for cfg_name in cfg_names:
        row = f"  {cfg_name:<20}"
        total = 0
        for sym in CRYPTO_SYMBOLS:
            r = all_results[sym].get(cfg_name)
            if r:
                row += f"  {r['trades']:>7}"
                total += r["trades"]
            else:
                row += f"  {'N/A':>7}"
        row += f"  {total:>7}"
        print(row)

    print()
    print("  WIN RATE GRID  [config × symbol]")
    print("-" * W)
    print(f"  {'Config':<20}" + "".join(f"  {s[:7]:>7}" for s in CRYPTO_SYMBOLS) + "   AVG")
    print("-" * W)
    for cfg_name in cfg_names:
        row = f"  {cfg_name:<20}"
        wrs = []
        for sym in CRYPTO_SYMBOLS:
            r = all_results[sym].get(cfg_name)
            if r:
                row += f"  {r['win_rate']:>7.1f}"
                wrs.append(r["win_rate"])
            else:
                row += f"  {'N/A':>7}"
        if wrs:
            row += f"  {np.mean(wrs):>7.1f}"
        print(row)

    # ─── Combined portfolio simulation per config ──────────────────────────────
    print()
    print("=" * W)
    print("  EQUAL-WEIGHT PORTFOLIO STATS  (treat all symbols as one pool)")
    print("=" * W)
    print(f"  {'Config':<20}  {'Trades':>6}  {'WR%':>6}  {'NetPF':>6}  {'Avg CAGR%':>10}  {'Avg DD%':>8}")
    print("-" * W)
    for cfg_name in cfg_names:
        rows = [all_results[sym][cfg_name] for sym in CRYPTO_SYMBOLS
                if cfg_name in all_results[sym]]
        if not rows: continue
        tot_trades = sum(r["trades"] for r in rows)
        avg_wr     = np.mean([r["win_rate"] for r in rows])
        npfs       = [r["net_pf"] for r in rows if r["net_pf"] != float("inf")]
        avg_npf    = np.mean(npfs) if npfs else 0
        avg_cagr   = np.mean([r["cagr"] for r in rows])
        avg_dd     = np.mean([r["max_dd"] for r in rows])
        pos_sym    = sum(1 for r in rows if r["cagr"] > 0)
        flag = "  ← RECOMMENDED" if cfg_name == max(
            cfg_names, key=lambda c: np.mean(
                [all_results[s][c]["net_pf"] for s in CRYPTO_SYMBOLS
                 if c in all_results[s] and all_results[s][c]["net_pf"] != float("inf")]
            ) if any(c in all_results[s] for s in CRYPTO_SYMBOLS) else -999
        ) else ""
        star = " ★" if avg_npf > 1.0 else ""
        print(f"  {cfg_name:<20}  {tot_trades:>6}  {avg_wr:>6.1f}  "
              f"{avg_npf:>6.3f}{star}  {avg_cagr:>+10.1f}  {avg_dd:>8.2f}"
              f"  ({pos_sym}/{len(rows)} sym profitable){flag}")

    # ─── Best config per symbol ────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  BEST CONFIG PER SYMBOL  (by Net PF)")
    print("=" * W)
    for sym in CRYPTO_SYMBOLS:
        best_cfg  = max(all_results[sym].items(), key=lambda x: x[1]["net_pf"])
        best_name, best_r = best_cfg
        sign = "+" if best_r["cagr"] >= 0 else ""
        print(f"  {sym}  →  {best_name:<20}  "
              f"NetPF {best_r['net_pf']:.3f}  "
              f"WR {best_r['win_rate']:.1f}%  "
              f"CAGR {sign}{best_r['cagr']:.1f}%  "
              f"{best_r['trades']} trades")

    # ─── Save CSV ──────────────────────────────────────────────────────────────
    csv_rows = []
    for sym in CRYPTO_SYMBOLS:
        for cfg_name, r in all_results[sym].items():
            cfg = next(c for c in CONFIGS if c[0] == cfg_name)
            csv_rows.append({
                "symbol": sym, "config": cfg_name,
                "long_only": cfg[1], "vol_mult": cfg[2], "sl_mult": cfg[3],
                **r
            })
    if csv_rows:
        out = OUTPUT_DIR / "backtest_v8_longonly.csv"
        pd.DataFrame(csv_rows).to_csv(out, index=False)
        print(f"\n  Saved → {out}")
    print("=" * W)


if __name__ == "__main__":
    main()
