"""
backtest_walkforward.py  --  Walk-Forward Validation of EMA BTC+ETH Strategy
==============================================================================
Purpose : Prove (or disprove) that the EMA trend strategy is not curve-fitted.
Method  : Anchored walk-forward -- train on all past data, test on next 6 months.
          Parameter selection happens ONLY on training data; test data is unseen.

Walk-forward windows  (anchored: training always starts Jan 2021):
  Win 1  Train: Jan 2021 - Dec 2022   Test: Jan 2023 - Jun 2023
  Win 2  Train: Jan 2021 - Jun 2023   Test: Jul 2023 - Dec 2023
  Win 3  Train: Jan 2021 - Dec 2023   Test: Jan 2024 - Jun 2024
  Win 4  Train: Jan 2021 - Jun 2024   Test: Jul 2024 - Dec 2024
  Win 5  Train: Jan 2021 - Dec 2024   Test: Jan 2025 - Jun 2025
  Win 6  Train: Jan 2021 - Jun 2025   Test: Jul 2025 - Jun 2026

Parameter grid (12 combos searched in each training window):
  ema_slow : [15, 20, 25, 30, 35, 40]
  vol_size : [False, True]
  (momentum=4w, atr=8w, top_n=2 are fixed)

Selection criterion : Calmar ratio on training period
  -- Evaluated from a FAIR START DATE (after the longest warmup across all combos)
     so that faster/slower EMAs get an equal-length comparison window.

Output:
  1. Per-window table: which param won training, how it performed OOS
  2. Parameter stability: how often each EMA was selected
  3. OOS concatenated equity curve stats (Sharpe, Sortino, Calmar, CAGR)
  4. OOS monthly returns table
  5. Side-by-side: in-sample V3/V4 vs OOS vs BTC B&H
  6. Verdict: is the strategy robust?
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from lzr_core import load_1m, resample

DATA_DIR  = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUT       = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT.mkdir(exist_ok=True)

INITIAL   = 1_000.0
FEE_SIDE  = 0.0004
SLIP_RATES= {"BTCUSDT": 0.0005, "ETHUSDT": 0.0005}
UNIVERSE  = ["BTCUSDT", "ETHUSDT"]
RISK_FREE = 0.04
W         = "=" * 78
SEP       = "-" * 78

# ── Walk-forward windows ────────────────────────────────────────────────────────
WF_WINDOWS = [
    ("2022-12-31", "2023-01-01", "2023-06-30"),
    ("2023-06-30", "2023-07-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2024-06-30"),
    ("2024-06-30", "2024-07-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2026-06-15"),
]

# ── Parameter grid ──────────────────────────────────────────────────────────────
PARAM_GRID = []
for ema_slow in [15, 20, 25, 30, 35, 40]:
    for vol_size in [False, True]:
        PARAM_GRID.append({
            "ema_slow" : ema_slow,
            "ema_fast" : ema_slow // 2,
            "mom_weeks": 4,
            "top_n"    : 2,
            "vol_size" : vol_size,
            "atr_weeks": 8,
        })

# Longest warmup across all combos (for fair training comparison)
MAX_WARMUP = max(
    cfg["ema_slow"] + cfg["mom_weeks"] + cfg.get("atr_weeks", 8) + 2
    for cfg in PARAM_GRID
)   # = 40 + 4 + 8 + 2 = 54 weeks


# ── Trade helpers ──────────────────────────────────────────────────────────────

def slip(sym):
    return SLIP_RATES.get(sym, 0.0010)

def _buy(sym, price, dollar, fm=1.0):
    s   = slip(sym) * fm
    f   = FEE_SIDE  * fm
    qty = dollar / (price * (1 + f + s))
    return qty, dollar

def _sell(sym, price, qty, fm=1.0):
    s = slip(sym) * fm
    f = FEE_SIDE  * fm
    return qty * price * (1 - f - s)

def equity_val(cash, holdings, lc):
    return cash + sum(holdings[x] * lc.get(x, 0) for x in holdings)


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(eq, init=INITIAL):
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    final = float(eq.iloc[-1])
    yrs   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (final / init) ** (1 / max(yrs, 0.01)) - 1
    pk    = eq.cummax()
    mdd   = float(((eq - pk) / pk).min())
    cal   = abs(cagr / mdd) if mdd < -0.001 else float("inf")
    return dict(cagr=round(cagr*100,1), mdd=round(mdd*100,1),
                calmar=round(cal,2), final=round(final,2), cagr_r=cagr)

def compute_advanced(eq, risk_free=RISK_FREE):
    eq = eq.dropna()
    if len(eq) < 10:
        return {}
    # Strip flat warmup
    iv  = float(eq.iloc[0])
    chg = eq[eq != iv]
    if len(chg) > 0:
        eq = eq[chg.index[0]:]
    if len(eq) < 10:
        return {}
    init  = float(eq.iloc[0])
    final = float(eq.iloc[-1])
    yrs   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (final / init) ** (1 / max(yrs, 0.01)) - 1

    wr  = eq.pct_change().dropna()
    if len(wr) == 0:
        return {}
    ann_vol   = float(wr.std()) * np.sqrt(52)
    rf_weekly = (1 + risk_free) ** (1/52) - 1
    sharpe    = (cagr - risk_free) / ann_vol if ann_vol > 0 else 0.0

    down_w = wr[wr < rf_weekly]
    down_std = (np.sqrt((down_w**2).mean()) * np.sqrt(52)
                if len(down_w) > 0 else 1e-9)
    sortino = min((cagr - risk_free) / down_std, 99.0)

    try:
        mo_eq  = eq.resample("ME").last().dropna()
    except Exception:
        mo_eq  = eq.resample("M").last().dropna()
    mo_ret = mo_eq.pct_change().dropna()

    if len(mo_ret) == 0:
        return {}

    pos_mo   = int((mo_ret > 0).sum())
    total_mo = int(len(mo_ret))
    mo_wr    = pos_mo / total_mo * 100

    best_mo  = float(mo_ret.max()) * 100
    worst_mo = float(mo_ret.min()) * 100
    avg_win  = float(mo_ret[mo_ret > 0].mean()) * 100 if (mo_ret > 0).any() else 0.0
    avg_loss = float(mo_ret[mo_ret < 0].mean()) * 100 if (mo_ret < 0).any() else 0.0

    max_consec = 0
    curr = 0
    for v in mo_ret:
        curr = curr + 1 if v < 0 else 0
        max_consec = max(max_consec, curr)

    pk = eq.cummax()
    in_dd = False
    bot_date = None
    bot_val  = float("inf")
    max_rec  = 0
    for date, val, peak in zip(eq.index, eq.values, pk.values):
        uw = val < peak * 0.999
        if uw:
            in_dd = True
            if val < bot_val:
                bot_val  = val
                bot_date = date
        else:
            if in_dd and bot_date is not None:
                max_rec = max(max_rec, (date - bot_date).days)
            in_dd    = False
            bot_date = None
            bot_val  = float("inf")
    if in_dd and bot_date is not None:
        max_rec = max(max_rec, (eq.index[-1] - bot_date).days)

    return dict(
        sharpe=round(sharpe,2), sortino=round(sortino,2),
        ann_vol=round(ann_vol*100,1),
        monthly_wr=round(mo_wr,1),
        pos_months=pos_mo, total_months=total_mo,
        best_month=round(best_mo,1), worst_month=round(worst_mo,1),
        avg_win=round(avg_win,1), avg_loss=round(avg_loss,1),
        max_consec_loss=max_consec,
        max_rec_months=round(max_rec/30.44,1),
        monthly_ret=mo_ret,
    )

def year_by_year(eq, init=INITIAL):
    rows, bal = [], init
    for yr, g in eq.groupby(eq.index.year):
        end = float(g.iloc[-1])
        ret = (end/bal - 1)*100
        pk  = g.cummax()
        dd  = float(((g-pk)/pk).min())*100
        rows.append((yr, round(ret,1), round(dd,1), round(end,2)))
        bal = end
    return rows

def lu(yy, yr):
    for y, r, d, b in yy:
        if y == yr: return r, d, b
    return 0.0, 0.0, 0.0

def print_monthly_table(mo_ret, title=""):
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"\n  {title}")
    hdr = f"  {'Year':<6}" + "".join(f" {m:>6}" for m in MONTHS) + f"  {'Annual':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    for yr, g in mo_ret.groupby(mo_ret.index.year):
        row = f"  {yr:<6}"
        yr_r = 1.0
        for m in range(1, 13):
            md = g[g.index.month == m]
            if len(md) > 0:
                r     = float(md.iloc[0]) * 100
                yr_r *= (1 + float(md.iloc[0]))
                tag   = "^" if r >= 20 else ("v" if r <= -10 else " ")
                row  += f" {r:>+5.1f}{tag}"
            else:
                row  += f"   --- "
        row += f"  {(yr_r-1)*100:>+7.1f}%"
        print(row)
    print(f"\n  (^ >= +20%   v <= -10%)")


# ── Strategy engine ────────────────────────────────────────────────────────────

def run_strategy(weekly_data, cfg, fee_mult=1.0):
    ema_slow = cfg["ema_slow"]
    ema_fast = cfg["ema_fast"]
    mom_w    = cfg["mom_weeks"]
    top_n    = cfg["top_n"]
    vol_size = cfg.get("vol_size", False)
    atr_w    = cfg.get("atr_weeks", 8)

    slow_ema, mom, atr_ser = {}, {}, {}
    for sym in UNIVERSE:
        cl = weekly_data[sym]["close"]
        hi = weekly_data[sym]["high"]
        lo = weekly_data[sym]["low"]
        slow_ema[sym] = cl.ewm(span=ema_slow, adjust=False).mean()
        mom[sym]      = cl.pct_change(mom_w)
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_ser[sym] = tr.rolling(atr_w).mean()

    ref   = max(UNIVERSE, key=lambda s: len(weekly_data[s]))
    weeks = list(weekly_data[ref].index)
    WARMUP = ema_slow + mom_w + atr_w + 2

    cash = INITIAL
    holdings = {}
    prev_target = set()
    lc = {}
    eq_curve = {}

    for i, wk in enumerate(weeks):
        for sym in UNIVERSE:
            if wk in weekly_data[sym].index:
                c = weekly_data[sym].loc[wk, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        if i < WARMUP:
            eq_curve[wk] = INITIAL
            continue

        prev_wk = weeks[i-1]

        candidates = {}
        atr_vals   = {}
        for sym in UNIVERSE:
            df = weekly_data[sym]
            if prev_wk not in df.index:
                continue
            cp = float(df.loc[prev_wk, "close"])
            ep = float(slow_ema[sym].loc[prev_wk])
            mp = float(mom[sym].loc[prev_wk])
            av = (float(atr_ser[sym].loc[prev_wk])
                  if prev_wk in atr_ser[sym].index else np.nan)
            if np.isnan(ep) or np.isnan(mp):
                continue
            if cp > ep and mp > 0:
                candidates[sym] = mp
                if not np.isnan(av):
                    atr_vals[sym] = av

        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:top_n]
        target = set(s for s, _ in ranked)

        if target != prev_target:
            for sym in (prev_target - target):
                if sym not in holdings:
                    continue
                if wk in weekly_data[sym].index:
                    p = float(weekly_data[sym].loc[wk, "open"])
                    if np.isnan(p) or p <= 0:
                        p = lc.get(sym, 0)
                else:
                    p = lc.get(sym, 0)
                if p > 0:
                    cash += _sell(sym, p, holdings[sym], fee_mult)
                del holdings[sym]

            if target:
                opx = {}
                for sym in target:
                    if wk in weekly_data[sym].index:
                        p = float(weekly_data[sym].loc[wk, "open"])
                        if not np.isnan(p) and p > 0:
                            opx[sym] = p

                port_val = cash
                for sym in target:
                    if sym in holdings:
                        port_val += holdings[sym] * opx.get(sym, lc.get(sym, 0))

                if vol_size and atr_vals:
                    pct_vol = {}
                    for sym in target:
                        if sym in atr_vals and sym in opx and opx[sym] > 0:
                            pct_vol[sym] = atr_vals[sym] / opx[sym]
                    if pct_vol:
                        inv     = {s: 1/max(v,1e-6) for s,v in pct_vol.items()}
                        tot_inv = sum(inv.values())
                        weights = {s: inv[s]/tot_inv for s in inv}
                        avg_pv  = float(np.mean(list(pct_vol.values())))
                        scale   = min(1.0, 0.03/max(avg_pv, 1e-6))
                        per_sym_alloc = {s: port_val*weights[s]*scale for s in weights}
                    else:
                        per = port_val / len(target)
                        per_sym_alloc = {s: per for s in target}
                else:
                    per = port_val / len(target)
                    per_sym_alloc = {s: per for s in target}

                for sym in list(holdings.keys()):
                    if sym not in target:
                        continue
                    p     = opx.get(sym, lc.get(sym, 0))
                    if p <= 0:
                        continue
                    alloc = per_sym_alloc.get(sym, 0)
                    cv    = holdings[sym] * p
                    if cv > alloc * 1.02:
                        excess = (cv - alloc) / p
                        cash  += _sell(sym, p, excess, fee_mult)
                        holdings[sym] -= excess

                for sym in target:
                    p = opx.get(sym)
                    if p is None:
                        continue
                    alloc   = per_sym_alloc.get(sym, 0)
                    cv      = holdings.get(sym, 0) * p
                    deficit = alloc - cv
                    if deficit < 1.0:
                        continue
                    spend = min(deficit, cash)
                    if spend < 1.0:
                        continue
                    qty, spent = _buy(sym, p, spend, fee_mult)
                    holdings[sym] = holdings.get(sym, 0) + qty
                    cash -= spent

            prev_target = target

        eq_curve[wk] = equity_val(cash, holdings, lc)

    return pd.Series(eq_curve)


def run_bah(weekly_data, sym):
    weeks = list(weekly_data[sym].index)
    cash  = INITIAL
    qty   = 0.0
    lc    = {}
    eq    = {}
    first = True
    for wk in weeks:
        if wk in weekly_data[sym].index:
            c = weekly_data[sym].loc[wk, "close"]
            if not np.isnan(c):
                lc[sym] = float(c)
        if first and wk in weekly_data[sym].index:
            p = float(weekly_data[sym].loc[wk, "open"])
            if not np.isnan(p) and p > 0:
                qty, _ = _buy(sym, p, INITIAL)
                cash   = 0.0
                first  = False
        eq[wk] = cash + qty * lc.get(sym, 0)
    return pd.Series(eq)


# ── Walk-forward engine ────────────────────────────────────────────────────────

def compute_train_calmar(eq, fair_start, train_end):
    """
    Calmar on [fair_start, train_end].
    Uses fair_start's equity as init so all combos are on equal footing.
    """
    sl = eq[(eq.index >= fair_start) & (eq.index <= train_end)]
    if len(sl) < 5:
        return -np.inf
    init = float(sl.iloc[0])
    if init <= 0:
        return -np.inf
    st = compute_stats(sl, init=init)
    return st.get("calmar", -np.inf)


def run_walk_forward(all_eq, weekly_data, wf_windows, fair_start_date):
    """
    all_eq : dict  { cfg_idx -> pd.Series }  -- pre-computed equity curves
    Returns : (oos_equity pd.Series, wf_report list)
    """
    oos_parts   = []
    wf_report   = []
    oos_balance = INITIAL

    for w_idx, (te_str, ts_str, tend_str) in enumerate(wf_windows):
        train_end  = pd.Timestamp(te_str)
        test_start = pd.Timestamp(ts_str)
        test_end   = pd.Timestamp(tend_str)

        # ── 1. Training: pick best param by Calmar ────────────────────────────
        best_calmar  = -np.inf
        best_idx     = 0
        calmar_table = {}

        for idx in range(len(PARAM_GRID)):
            eq  = all_eq[idx]
            cal = compute_train_calmar(eq, fair_start_date, train_end)
            calmar_table[idx] = cal
            if cal > best_calmar:
                best_calmar = cal
                best_idx    = idx

        best_cfg = PARAM_GRID[best_idx]

        # ── 2. Testing: performance on unseen data ────────────────────────────
        best_eq  = all_eq[best_idx]
        test_sl  = best_eq[(best_eq.index >= test_start) & (best_eq.index <= test_end)]

        if len(test_sl) < 2:
            continue

        # Equity just before test window starts (to compute test return correctly)
        pre_test = best_eq[best_eq.index < test_start]
        eq_at_ts = float(pre_test.iloc[-1]) if len(pre_test) > 0 else INITIAL
        eq_at_te = float(test_sl.iloc[-1])
        test_ret = (eq_at_te / eq_at_ts - 1) if eq_at_ts > 0 else 0.0

        # BTC return over same period for comparison
        bah_sl   = all_eq["bah_btc"]
        bah_test = bah_sl[(bah_sl.index >= test_start) & (bah_sl.index <= test_end)]
        bah_pre  = bah_sl[bah_sl.index < test_start]
        bah_ts   = float(bah_pre.iloc[-1]) if len(bah_pre) > 0 else INITIAL
        bah_te   = float(bah_test.iloc[-1]) if len(bah_test) > 0 else bah_ts
        btc_ret  = (bah_te / bah_ts - 1) if bah_ts > 0 else 0.0

        # Chain into OOS equity
        oos_sl = test_sl / eq_at_ts * oos_balance
        oos_parts.append(oos_sl)
        oos_balance = float(oos_sl.iloc[-1])

        wf_report.append(dict(
            win        = w_idx + 1,
            train_end  = te_str,
            test_start = ts_str,
            test_end   = tend_str,
            best_ema   = best_cfg["ema_slow"],
            vol_size   = best_cfg["vol_size"],
            train_cal  = round(best_calmar, 2),
            test_ret   = round(test_ret * 100, 1),
            btc_ret    = round(btc_ret * 100, 1),
            oos_bal    = round(oos_balance, 2),
            n_weeks    = len(test_sl),
        ))

    if oos_parts:
        oos_eq = pd.concat(oos_parts)
        oos_eq = oos_eq[~oos_eq.index.duplicated(keep="first")].sort_index()
    else:
        oos_eq = pd.Series(dtype=float)

    return oos_eq, wf_report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(W)
    print("  WALK-FORWARD VALIDATION  --  EMA BTC+ETH Strategy")
    print("  Training selects EMA params on past data only.")
    print("  Test periods are UNSEEN -- this is true out-of-sample.")
    print(W)

    print("\n  Loading data...")
    weekly_data = {}
    for sym in UNIVERSE:
        df1 = load_1m(sym)
        weekly_data[sym] = resample(df1, "1W")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Fair-start date: first week after max warmup period (EMA40 combo)
    ref        = max(UNIVERSE, key=lambda s: len(weekly_data[s]))
    all_weeks  = list(weekly_data[ref].index)
    fair_start = all_weeks[MAX_WARMUP]   # week 54 ~ Feb 2022
    print(f"  Fair comparison start (after max warmup): {fair_start.date()}")
    print(f"  Parameter grid: {len(PARAM_GRID)} combos  "
          f"(EMA {[c['ema_slow'] for c in PARAM_GRID if not c['vol_size']]} x vol_size T/F)")

    # ── Pre-compute all equity curves ──────────────────────────────────────────
    print(f"\n  Pre-computing {len(PARAM_GRID)} equity curves...")
    all_eq = {}
    for idx, cfg in enumerate(PARAM_GRID):
        all_eq[idx] = run_strategy(weekly_data, cfg, fee_mult=1.0)
        vs = "V" if cfg["vol_size"] else " "
        print(f"    [{idx:02d}]  EMA{cfg['ema_slow']:02d}{vs}  "
              f"CAGR {compute_stats(all_eq[idx])['cagr']:>+5.1f}%  "
              f"Calmar {compute_stats(all_eq[idx])['calmar']:.2f}")

    all_eq["bah_btc"] = run_bah(weekly_data, "BTCUSDT")
    all_eq["bah_eth"] = run_bah(weekly_data, "ETHUSDT")
    print(f"  Pre-computation done in {time.time()-t0:.1f}s")

    # ── Run walk-forward ───────────────────────────────────────────────────────
    print(f"\n  Running walk-forward ({len(WF_WINDOWS)} windows)...")
    oos_eq, wf_report = run_walk_forward(all_eq, weekly_data, WF_WINDOWS, fair_start)

    # ── Section 1: Per-window table ────────────────────────────────────────────
    print(f"\n{W}")
    print("  1. WALK-FORWARD WINDOW RESULTS")
    print(W)
    print(f"\n  {'Win':<4} {'Train_end':<12} {'Test period':<22} "
          f"{'Best EMA':>10} {'Vol?':>5} {'TrainCal':>9} "
          f"{'OOS Ret':>9} {'BTC Ret':>9} {'OOS Bal':>10}")
    print("  " + SEP)

    oos_beats_btc = 0
    for r in wf_report:
        vs       = "Yes" if r["vol_size"] else "No"
        beat     = " <--" if r["test_ret"] > r["btc_ret"] else ""
        print(f"  {r['win']:<4} {r['train_end']:<12} "
              f"{r['test_start']} - {r['test_end']}  "
              f"EMA{r['best_ema']:02d}  {vs:>5}  "
              f"{r['train_cal']:>8.2f}  "
              f"{r['test_ret']:>+8.1f}%  "
              f"{r['btc_ret']:>+8.1f}%  "
              f"${r['oos_bal']:>8,.0f}{beat}")
        if r["test_ret"] > r["btc_ret"]:
            oos_beats_btc += 1

    total_oos_ret = (wf_report[-1]["oos_bal"] / INITIAL - 1) * 100
    print("  " + SEP)
    print(f"  OOS beats BTC: {oos_beats_btc}/{len(wf_report)} windows  |  "
          f"Total OOS return: {total_oos_ret:+.1f}%  |  Final OOS balance: "
          f"${wf_report[-1]['oos_bal']:,.0f}")

    # ── Section 2: Parameter stability ────────────────────────────────────────
    print(f"\n{W}")
    print("  2. PARAMETER STABILITY  (which EMA was selected in each window?)")
    print(W)
    from collections import Counter
    ema_count  = Counter(r["best_ema"]  for r in wf_report)
    vol_count  = Counter(r["vol_size"]  for r in wf_report)
    print(f"\n  EMA selections across {len(wf_report)} windows:")
    for ema, cnt in sorted(ema_count.items()):
        bar = "#" * cnt
        print(f"    EMA{ema:02d}: {bar}  ({cnt}/{len(wf_report)})")
    print(f"\n  Vol-sizing selected: {vol_count[True]}/{len(wf_report)} windows  |  "
          f"Plain (equal-weight): {vol_count[False]}/{len(wf_report)} windows")

    stable = len(ema_count) <= 2
    print(f"\n  Stability: {'STABLE (same EMA family dominates)' if stable else 'UNSTABLE (EMA selection varies)'}")

    # ── Section 3: OOS aggregate stats ────────────────────────────────────────
    print(f"\n{W}")
    print("  3. OOS AGGREGATE STATS  (concatenated out-of-sample equity)")
    print(f"     Period: {WF_WINDOWS[0][1]}  to  {WF_WINDOWS[-1][2]}")
    print(f"     ~{(pd.Timestamp(WF_WINDOWS[-1][2]) - pd.Timestamp(WF_WINDOWS[0][1])).days/365.25:.1f} years of TRUE out-of-sample data")
    print(W)

    oos_st  = compute_stats(oos_eq, init=INITIAL)
    oos_adv = compute_advanced(oos_eq)

    bah_btc = all_eq["bah_btc"]
    bah_eth = all_eq["bah_eth"]
    btc_oos = bah_btc[(bah_btc.index >= pd.Timestamp(WF_WINDOWS[0][1]))]
    btc_oos_st = compute_stats(btc_oos, init=float(btc_oos.iloc[0]))

    # In-sample references (V3 and V4 from full period)
    v3_cfg = {"ema_slow":30,"ema_fast":15,"mom_weeks":4,"top_n":2,"vol_size":False,"atr_weeks":8}
    v4_cfg = {"ema_slow":30,"ema_fast":15,"mom_weeks":4,"top_n":2,"vol_size":True,"atr_weeks":8}
    v3_eq  = run_strategy(weekly_data, v3_cfg)
    v4_eq  = run_strategy(weekly_data, v4_cfg)
    v3_st  = compute_stats(v3_eq)
    v4_st  = compute_stats(v4_eq)

    print(f"\n  {'Metric':<26} {'OOS Walk-fwd':>14} {'BTC B&H OOS':>14} "
          f"{'V3 in-sample':>14} {'V4 in-sample':>14}")
    print("  " + "-" * 84)

    rows3 = [
        ("CAGR",         f"{oos_st['cagr']:>+13.1f}%", f"{btc_oos_st['cagr']:>+13.1f}%",
                         f"{v3_st['cagr']:>+13.1f}%",  f"{v4_st['cagr']:>+13.1f}%"),
        ("Max Drawdown", f"{oos_st['mdd']:>+13.1f}%",  f"{btc_oos_st['mdd']:>+13.1f}%",
                         f"{v3_st['mdd']:>+13.1f}%",   f"{v4_st['mdd']:>+13.1f}%"),
        ("Calmar",       f"{oos_st['calmar']:>14.2f}",  f"{btc_oos_st['calmar']:>14.2f}",
                         f"{v3_st['calmar']:>14.2f}",   f"{v4_st['calmar']:>14.2f}"),
        ("Sharpe",       f"{oos_adv.get('sharpe',0):>14.2f}", "  --",
                         "  --", "  --"),
        ("Sortino",      f"{oos_adv.get('sortino',0):>14.2f}", "  --",
                         "  --", "  --"),
        ("Ann. Vol",     f"{oos_adv.get('ann_vol',0):>13.1f}%", "  --",
                         "  --", "  --"),
        ("Monthly WR",   f"{oos_adv.get('monthly_wr',0):>13.1f}%", "  --",
                         "  --", "  --"),
        ("Max Consec L", f"{oos_adv.get('max_consec_loss','?')}mo".rjust(14), "  --",
                         "  --", "  --"),
        ("Max Recovery", f"{oos_adv.get('max_rec_months',0):.0f}mo".rjust(14), "  --",
                         "  --", "  --"),
        ("Final Balance",f"${oos_st['final']:>12,.0f}", f"  --",
                         f"${v3_st['final']:>12,.0f}", f"${v4_st['final']:>12,.0f}"),
    ]
    for label, *vals in rows3:
        print(f"  {label:<26}" + "".join(f"{v:>14}" for v in vals))

    # ── Section 4: Year-by-year OOS vs BTC ────────────────────────────────────
    print(f"\n{W}")
    print("  4. YEAR-BY-YEAR  --  OOS Walk-forward vs BTC B&H")
    print(W)
    oos_yby = year_by_year(oos_eq, init=INITIAL)
    btc_st_full = compute_stats(bah_btc)
    btc_yby = year_by_year(bah_btc)

    all_yrs = sorted(set(y for y,*_ in oos_yby + btc_yby))
    print(f"\n  {'Year':<6} {'OOS Strat':>11} {'OOS DD':>9} "
          f"{'OOS Bal':>10} {'BTC %':>9}  Result")
    print("  " + SEP)
    for yr in all_yrs:
        ro, do, bo = lu(oos_yby, yr)
        rb, db, bb = lu(btc_yby, yr)
        if ro == 0.0 and do == 0.0 and bo == 0.0:
            continue
        note = "beats BTC" if ro > rb else "lags BTC"
        print(f"  {yr:<6} {ro:>+10.1f}% {do:>+8.1f}% ${bo:>8,.0f} "
              f"{rb:>+8.1f}%  {note}")

    # ── Section 5: OOS monthly returns table ──────────────────────────────────
    print(f"\n{W}")
    print("  5. OOS MONTHLY RETURNS TABLE")
    print(W)
    if oos_adv.get("monthly_ret") is not None:
        print_monthly_table(oos_adv["monthly_ret"], "Walk-Forward OOS Monthly Returns (%)")

    # ── Section 6: Verdict ─────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  6. VERDICT  --  Is the strategy robust?")
    print(W)

    oos_calmar = oos_st["calmar"]
    oos_cagr   = oos_st["cagr"]
    is_sr      = oos_st["calmar"] >= 0.3
    beats_btc  = oos_cagr > 0        # just: positive OOS returns
    stable_ema = stable

    checks = [
        (is_sr,      f"OOS Calmar >= 0.3         : {oos_calmar:.2f}"),
        (beats_btc,  f"OOS positive CAGR          : {oos_cagr:+.1f}%"),
        (stable_ema, f"EMA selection stable       : {'Yes' if stable_ema else 'No -- varies per window'}"),
        (oos_beats_btc >= 3,
                     f"Beats BTC in majority of windows: {oos_beats_btc}/{len(wf_report)}"),
    ]

    all_pass = all(c for c, _ in checks)
    print()
    for ok, txt in checks:
        print(f"  {'  [OK]' if ok else '  [  ]'}  {txt}")

    print()
    if all_pass:
        verdict = "ROBUST  -- OOS performance confirms the strategy is not curve-fitted."
    elif sum(c for c,_ in checks) >= 3:
        verdict = "LIKELY ROBUST  -- Most checks pass. Minor concerns noted above."
    elif sum(c for c,_ in checks) >= 2:
        verdict = "MARGINAL  -- Strategy shows some OOS signal but not conclusive."
    else:
        verdict = "WEAK  -- OOS performance does not support the in-sample results."

    print(f"  OVERALL: {verdict}")

    in_vs_oos_gap = abs(v3_st["calmar"] - oos_calmar)
    print(f"\n  In-sample V3 Calmar : {v3_st['calmar']:.2f}")
    print(f"  OOS Walk-fwd Calmar : {oos_calmar:.2f}")
    print(f"  Gap                 : {in_vs_oos_gap:.2f}  "
          f"({'small -- good sign' if in_vs_oos_gap < 0.3 else 'large -- some overfitting'})")

    # ── Save ───────────────────────────────────────────────────────────────────
    oos_eq.to_csv(OUT / "eq_oos_walkforward.csv", header=["equity"])
    v3_eq.to_csv(OUT / "eq_v3_insample.csv",      header=["equity"])
    print(f"\n  Saved: eq_oos_walkforward.csv, eq_v3_insample.csv")
    print(f"  Total runtime: {(time.time()-t0)/60:.1f} min\n")
    print(W)


if __name__ == "__main__":
    main()
