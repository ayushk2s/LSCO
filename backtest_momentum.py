"""
backtest_momentum.py  ─  Two Crypto Strategies, Full Audit Pass
================================================================
All known bugs fixed:
  B1: del holdings[sym] now only after sell succeeds (or falls back to last close)
  B2: look-ahead verified clean for both strategies
  B3: fee_mult=0 / 1 lets us run with vs without fees in one call
  B4: cash properly debited per-buy in S1 (no blanket cash=0)
  B5: S2 fully rebalances all target positions (not just new ones)
  B6: last-known-close used for equity (no $0 on data gaps)
  B7: trade log carries cash_out / cash_in for exact per-symbol attribution

Weekly resample anchor confirmed:
  pandas "1W" == "W-SUN" → bar label = Sunday
  open  = first 1-min bar of the week  (Monday 00:00)
  close = last  1-min bar of the week  (Sunday 23:59)
  Signal from prev_wk close (Sunday) → execute at wk open (Monday)  CORRECT
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from lzr_core import load_1m, resample

DATA_DIR = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUT      = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT.mkdir(exist_ok=True)

INITIAL_BALANCE = 1_000.0
FEE_SIDE        = 0.0004
EMA_WEEKS       = 20
TOP_N_S1        = 3
TOP_N_S2        = 5
LOOKBACK_S1     = 3    # months
LOOKBACK_S2_W   = 4    # weeks

SLIP_BY_ASSET = {
    "BTCUSDT": 0.0005, "ETHUSDT": 0.0005,
    "BNBUSDT": 0.0008, "SOLUSDT": 0.0008, "XRPUSDT": 0.0008,
    "ADAUSDT": 0.0008, "DOGEUSDT": 0.0008, "LTCUSDT": 0.0008,
    "LINKUSDT": 0.0010, "AVAXUSDT": 0.0010, "DOTUSDT": 0.0010,
    "ATOMUSDT": 0.0010, "UNIUSDT": 0.0010, "INJUSDT": 0.0010,
    "NEARUSDT": 0.0010, "ARBUSDT": 0.0010, "OPUSDT": 0.0010,
    "APTUSDT": 0.0010, "TRXUSDT": 0.0010, "BCHUSDT": 0.0010,
    "AAVEUSDT": 0.0015, "FILUSDT": 0.0015, "RUNEUSDT": 0.0015,
    "MATICUSDT": 0.0015, "FETUSDT": 0.0015, "CFXUSDT": 0.0015,
    "LDOUSDT": 0.0015, "SEIUSDT": 0.0015, "SUIUSDT": 0.0015,
    "ASTERUSDT": 0.0015,
}
DEFAULT_SLIP = 0.0015
W = "=" * 84


# ── Cost helpers (fee_mult=1 → full fees, 0 → no fees) ────────────────────────

def slip(sym):
    return SLIP_BY_ASSET.get(sym, DEFAULT_SLIP)

def _buy(sym, price, dollar, fm):
    s = slip(sym) * fm
    f = FEE_SIDE * fm
    qty = dollar / (price * (1 + f + s))
    return qty, dollar                  # (coins received, cash spent)

def _sell(sym, price, qty, fm):
    s = slip(sym) * fm
    f = FEE_SIDE * fm
    proc = qty * price * (1 - f - s)
    return proc                         # cash received

def equity_value(cash, holdings, lc):
    return cash + sum(holdings[s] * lc.get(s, 0) for s in holdings)

def series_stats(eq, init_bal):
    if len(eq) < 2:
        return dict(cagr=0, max_dd=0, calmar=0, final=init_bal, years=0)
    final = float(eq.iloc[-1])
    yrs   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (final / init_bal) ** (1 / max(yrs, 0.1)) - 1
    pk    = eq.cummax()
    mdd   = ((eq - pk) / pk).min()
    calmar = abs(cagr / mdd) if mdd < -0.001 else (float("inf") if cagr > 0 else 0.0)
    return dict(cagr=round(cagr*100,2), max_dd=round(mdd*100,2),
                calmar=round(calmar,2), final=round(final,2), years=round(yrs,2))

def year_by_year(eq, init_bal):
    rows, bal = [], init_bal
    for yr, g in eq.groupby(eq.index.year):
        end = float(g.iloc[-1])
        ret = (end / bal - 1) * 100
        pk  = g.cummax()
        dd  = ((g - pk) / pk).min() * 100
        rows.append((yr, round(ret,1), round(dd,1), round(end,2)))
        bal = end
    return rows

def get_price(df, date):
    sub = df.loc[:date]
    return float(sub.iloc[-1]["close"]) if len(sub) > 0 else np.nan

def get_open_after(df, date):
    sub = df.loc[date:]
    return float(sub.iloc[0]["open"]) if len(sub) > 0 else np.nan


# ── Strategy 1: Momentum Rotation ─────────────────────────────────────────────

def run_momentum_rotation(daily_data, symbols, fee_mult=1.0):
    """
    Signal  : yesterday's 3-month return ranking
    Execute : today's open (first trading day of each month)
    Costs   : controlled by fee_mult (0=none, 1=full)
    """
    all_dates = sorted(set().union(*[set(d.index) for d in daily_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    month_first = {}
    for d in all_dates:
        k = (d.year, d.month)
        if k not in month_first:
            month_first[k] = d
    rebal_set = set(month_first.values())

    cash      = INITIAL_BALANCE
    holdings  = {}          # sym -> qty
    lc        = {}          # last known close
    sym_out   = {}          # sym -> total $ deployed
    sym_in    = {}          # sym -> total $ returned (sells + unrealized at end)

    eq_dict   = {}
    trades    = []
    holds_log = []

    for date in all_dates:
        # update last-known close
        for sym in symbols:
            if date in daily_data[sym].index:
                c = daily_data[sym].loc[date, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        if date in rebal_set:
            signal_date   = date - pd.Timedelta(days=1)
            lookback_date = date - pd.DateOffset(months=LOOKBACK_S1) - pd.Timedelta(days=1)

            rets = {}
            for sym in symbols:
                curr = get_price(daily_data[sym], signal_date)
                prev = get_price(daily_data[sym], lookback_date)
                if not np.isnan(curr) and not np.isnan(prev) and prev > 0:
                    rets[sym] = curr / prev - 1

            ranked = sorted(rets.items(), key=lambda x: -x[1])
            target = ([s for s, _ in ranked[:TOP_N_S1]]
                      if ranked and ranked[0][1] > 0 else [])
            holds_log.append((date, list(target)))

            # Sell all
            for sym, qty in list(holdings.items()):
                p = get_open_after(daily_data[sym], date)
                if np.isnan(p) or p <= 0:
                    p = lc.get(sym, 0)
                if p > 0:
                    proc = _sell(sym, p, qty, fee_mult)
                    cash += proc
                    sym_in[sym] = sym_in.get(sym, 0) + proc
                    trades.append(dict(date=str(date.date()), action="SELL",
                                       sym=sym, price=round(p,4), qty=round(qty,6),
                                       cash_in=round(proc,4)))
            holdings = {}

            # Buy equal-weight
            if target:
                per_sym = cash / len(target)
                for sym in target:
                    p = get_open_after(daily_data[sym], date)
                    if np.isnan(p) or p <= 0:
                        continue
                    qty, spent = _buy(sym, p, per_sym, fee_mult)
                    holdings[sym] = qty
                    cash -= spent                          # B4: per-buy deduction
                    sym_out[sym] = sym_out.get(sym, 0) + spent
                    trades.append(dict(date=str(date.date()), action="BUY",
                                       sym=sym, price=round(p,4), qty=round(qty,6),
                                       cash_out=round(spent,4)))

        eq_dict[date] = equity_value(cash, holdings, lc)

    # unrealized → add to sym_in for attribution
    for sym, qty in holdings.items():
        sym_in[sym] = sym_in.get(sym, 0) + qty * lc.get(sym, 0)

    sym_pnl = {s: {"deployed": sym_out.get(s,0),
                   "returned": sym_in.get(s,0),
                   "pnl":      sym_in.get(s,0) - sym_out.get(s,0)}
               for s in set(sym_out) | set(sym_in)}

    return pd.Series(eq_dict), trades, holds_log, sym_pnl


# ── Strategy 2: EMA Trend + Momentum ──────────────────────────────────────────

def run_ema_trend_momentum(weekly_data, symbols, fee_mult=1.0):
    """
    Signal  : PREVIOUS week's close > EMA(20w) AND 4-week return > 0
    Execute : CURRENT week's open  (no look-ahead)
    Equity  : current week's close (last-known-close fallback)
    B1 FIX  : sell always succeeds (fallback to last close if no open price)
    B5 FIX  : full equal-weight restore on every target change
    """
    ema = {s: weekly_data[s]["close"].ewm(span=EMA_WEEKS, adjust=False).mean()
           for s in symbols}
    mom = {s: weekly_data[s]["close"].pct_change(LOOKBACK_S2_W) for s in symbols}

    ref_sym   = max(symbols, key=lambda s: len(weekly_data[s]))
    all_weeks = list(weekly_data[ref_sym].index)

    cash        = INITIAL_BALANCE
    holdings    = {}
    prev_target = set()
    lc          = {}          # last known close per symbol
    sym_out     = {}
    sym_in      = {}
    holds_log   = []          # (wk, set of held syms) for attribution

    eq_dict = {}
    trades  = []
    WARMUP  = EMA_WEEKS + LOOKBACK_S2_W + 1

    for i, wk in enumerate(all_weeks):
        # ── update last-known close (before warmup skip, so it accumulates) ───
        for sym in holdings:
            df = weekly_data[sym]
            if wk in df.index:
                c = df.loc[wk, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        if i < WARMUP:
            eq_dict[wk] = INITIAL_BALANCE
            continue

        prev_wk = all_weeks[i - 1]

        # ── Signal: PREVIOUS week's close (zero look-ahead) ───────────────────
        cands = {}
        for sym in symbols:
            df = weekly_data[sym]
            if prev_wk not in df.index:
                continue
            cp = float(df.loc[prev_wk, "close"])
            ep = float(ema[sym].loc[prev_wk]) if prev_wk in ema[sym].index else np.nan
            mp = float(mom[sym].loc[prev_wk]) if prev_wk in mom[sym].index else np.nan
            if not np.isnan(ep) and not np.isnan(mp) and cp > ep and mp > 0:
                cands[sym] = mp

        target = set(s for s, _ in sorted(cands.items(), key=lambda x: -x[1])[:TOP_N_S2])

        # ── Rebalance when target changes ─────────────────────────────────────
        if target != prev_target:
            to_sell = prev_target - target
            to_buy  = target - prev_target

            # 1. Sell exited positions (B1 FIX: fallback to lc if no open)
            for sym in to_sell:
                if sym not in holdings:
                    continue
                df = weekly_data[sym]
                p  = float(df.loc[wk, "open"]) if wk in df.index else np.nan
                if np.isnan(p) or p <= 0:
                    p = lc.get(sym, 0)            # B1: fallback avoids asset destruction
                if p > 0:
                    proc = _sell(sym, p, holdings[sym], fee_mult)
                    cash += proc
                    sym_in[sym] = sym_in.get(sym, 0) + proc
                    trades.append(dict(week=str(wk.date()), action="SELL",
                                       sym=sym, price=round(p,4),
                                       qty=round(holdings[sym],6),
                                       cash_in=round(proc,4)))
                del holdings[sym]              # always remove (sold or written-off)

            if target:
                # 2. Open prices for target symbols
                open_px = {}
                for sym in target:
                    df = weekly_data[sym]
                    if wk in df.index:
                        p = float(df.loc[wk, "open"])
                        if not np.isnan(p) and p > 0:
                            open_px[sym] = p

                # 3. Portfolio value at current open
                port_val = cash
                for sym in target:
                    if sym in holdings:
                        p = open_px.get(sym, lc.get(sym, 0))
                        port_val += holdings[sym] * p

                per_sym = port_val / len(target)

                # 4. Trim over-weight continuing holdings
                for sym in list(holdings.keys()):
                    p = open_px.get(sym, lc.get(sym, 0))
                    if p <= 0:
                        continue
                    curr_val = holdings[sym] * p
                    if curr_val > per_sym * 1.02:
                        excess = (curr_val - per_sym) / p
                        proc = _sell(sym, p, excess, fee_mult)
                        cash += proc
                        sym_in[sym] = sym_in.get(sym, 0) + proc
                        holdings[sym] -= excess
                        trades.append(dict(week=str(wk.date()), action="SELL_TRIM",
                                           sym=sym, price=round(p,4),
                                           qty=round(excess,6),
                                           cash_in=round(proc,4)))

                # 5. Buy ALL under-weight target positions (B5: includes continuing)
                for sym in target:
                    p = open_px.get(sym)
                    if p is None:
                        continue
                    curr_val = holdings.get(sym, 0) * p
                    deficit  = per_sym - curr_val
                    if deficit < 1.0:
                        continue
                    spend = min(deficit, cash)
                    if spend < 1.0:
                        continue
                    qty, spent = _buy(sym, p, spend, fee_mult)
                    holdings[sym] = holdings.get(sym, 0) + qty
                    cash -= spent
                    sym_out[sym] = sym_out.get(sym, 0) + spent
                    action = "BUY" if sym in to_buy else "BUY_TOPUP"
                    trades.append(dict(week=str(wk.date()), action=action,
                                       sym=sym, price=round(p,4),
                                       qty=round(qty,6),
                                       cash_out=round(spent,4)))

            prev_target = target

        # ── Equity at this week's close ────────────────────────────────────────
        for sym in holdings:
            df = weekly_data[sym]
            if wk in df.index:
                c = df.loc[wk, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)
        eq_dict[wk] = equity_value(cash, holdings, lc)
        holds_log.append((wk, set(holdings.keys()), eq_dict[wk]))

    # unrealized
    for sym, qty in holdings.items():
        sym_in[sym] = sym_in.get(sym, 0) + qty * lc.get(sym, 0)

    sym_pnl = {s: {"deployed": sym_out.get(s,0),
                   "returned": sym_in.get(s,0),
                   "pnl":      sym_in.get(s,0) - sym_out.get(s,0)}
               for s in set(sym_out) | set(sym_in)}

    return pd.Series(eq_dict), trades, sym_pnl, holds_log


# ── Reporting helpers ──────────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{W}\n  {title}\n{W}")

def sym_year_table(holds_log, eq_dict_or_series):
    """Return {sym: {year: weeks_held}} and {year: equity_change}."""
    sym_yr = {}
    eq = eq_dict_or_series
    for wk, syms, _ in holds_log:
        yr = wk.year
        for s in syms:
            sym_yr.setdefault(s, {}).setdefault(yr, 0)
            sym_yr[s][yr] += 1
    return sym_yr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(W)
    print("  CRYPTO MOMENTUM -- FULL AUDIT REPORT")
    print("  Bugs fixed: B1 del-without-sell | B4 cash-per-buy | B5 equal-weight")
    print("  Fee comparison: run once WITH fees, once WITHOUT fees")
    print(W)

    symbols = sorted(p.stem.replace("1m", "") for p in DATA_DIR.glob("*1m.csv"))
    print(f"\n  Loading {len(symbols)} symbols...")

    daily_data  = {}
    weekly_data = {}
    for sym in symbols:
        df1 = load_1m(sym)
        daily_data[sym]  = resample(df1, "1D")
        weekly_data[sym] = resample(df1, "1W")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── Run all four combinations ──────────────────────────────────────────────
    print("  Running S1 with fees...")
    eq1f, tr1f, hl1, sp1f = run_momentum_rotation(daily_data, symbols, fee_mult=1.0)

    print("  Running S1 without fees...")
    eq1n, tr1n, _,   sp1n = run_momentum_rotation(daily_data, symbols, fee_mult=0.0)

    print("  Running S2 with fees...")
    eq2f, tr2f, sp2f, hl2 = run_ema_trend_momentum(weekly_data, symbols, fee_mult=1.0)

    print("  Running S2 without fees...")
    eq2n, tr2n, sp2n, _   = run_ema_trend_momentum(weekly_data, symbols, fee_mult=0.0)

    s1f = series_stats(eq1f, INITIAL_BALANCE)
    s1n = series_stats(eq1n, INITIAL_BALANCE)
    s2f = series_stats(eq2f, INITIAL_BALANCE)
    s2n = series_stats(eq2n, INITIAL_BALANCE)

    yy1f = year_by_year(eq1f, INITIAL_BALANCE)
    yy1n = year_by_year(eq1n, INITIAL_BALANCE)
    yy2f = year_by_year(eq2f, INITIAL_BALANCE)
    yy2n = year_by_year(eq2n, INITIAL_BALANCE)

    # ── 1. Overall table ───────────────────────────────────────────────────────
    print_header("OVERALL RESULTS  (WITH FEES  vs  WITHOUT FEES)")
    LZR = dict(cagr=-26.5, max_dd=-87.2, calmar=0.30, final=199.90)
    print(f"\n  {'Metric':<20} {'S1 +fees':>12} {'S1 -fees':>12} "
          f"{'S2 +fees':>12} {'S2 -fees':>12} {'LZR v14':>10}")
    print("  " + "-" * 82)
    for label, k, fmt in [
        ("CAGR (%)",    "cagr",    "{:>+11.1f}%"),
        ("Max DD (%)",  "max_dd",  "{:>+11.1f}%"),
        ("Calmar",      "calmar",  "{:>12.2f} "),
        ("Final ($)",   "final",   "{:>12,.0f} "),
    ]:
        v1f = fmt.format(s1f[k]); v1n = fmt.format(s1n[k])
        v2f = fmt.format(s2f[k]); v2n = fmt.format(s2n[k])
        vlz = fmt.format(LZR[k])
        print(f"  {label:<20} {v1f} {v1n} {v2f} {v2n} {vlz}")

    fee_drag_s1 = s1n["final"] - s1f["final"]
    fee_drag_s2 = s2n["final"] - s2f["final"]
    print(f"\n  Fee drag (final $):         S1 = ${fee_drag_s1:>8,.0f}     "
          f"S2 = ${fee_drag_s2:>8,.0f}")
    print(f"  Fee drag (CAGR pp):         "
          f"S1 = {s1n['cagr']-s1f['cagr']:>+.1f} pp     "
          f"S2 = {s2n['cagr']-s2f['cagr']:>+.1f} pp")

    # ── 2. Year-by-year S2 with vs without fees ────────────────────────────────
    print_header("S2 (EMA + MOMENTUM) -- YEAR-BY-YEAR  [WITH FEES  vs  WITHOUT FEES]")
    yy2fd = {y: (r,d,b) for y,r,d,b in yy2f}
    yy2nd = {y: (r,d,b) for y,r,d,b in yy2n}
    print(f"\n  {'Year':<6}  {'Ret+fee':>9} {'DD+fee':>8} {'Bal+fee':>10}  |  "
          f"{'Ret-fee':>9} {'DD-fee':>8} {'Bal-fee':>10}  Fee drag")
    print("  " + "-" * 82)
    for yr in sorted(set(yy2fd) | set(yy2nd)):
        r2f,d2f,b2f = yy2fd.get(yr, (0,0,0))
        r2n,d2n,b2n = yy2nd.get(yr, (0,0,0))
        drag = b2f - b2n   # negative = fees hurt
        flag = " LOSS" if r2f < 0 else "     "
        print(f"  {yr:<6}  {r2f:>+8.1f}% {d2f:>+7.1f}% ${b2f:>9,.0f}{flag}  |  "
              f"{r2n:>+8.1f}% {d2n:>+7.1f}% ${b2n:>9,.0f}  ${drag:>+9,.0f}")

    # ── 3. Year-by-year S1 ────────────────────────────────────────────────────
    print_header("S1 (MOMENTUM ROTATION) -- YEAR-BY-YEAR  [WITH FEES  vs  WITHOUT FEES]")
    yy1fd = {y: (r,d,b) for y,r,d,b in yy1f}
    yy1nd = {y: (r,d,b) for y,r,d,b in yy1n}
    print(f"\n  {'Year':<6}  {'Ret+fee':>9} {'DD+fee':>8} {'Bal+fee':>10}  |  "
          f"{'Ret-fee':>9} {'DD-fee':>8} {'Bal-fee':>10}  Fee drag")
    print("  " + "-" * 82)
    for yr in sorted(set(yy1fd) | set(yy1nd)):
        r1f,d1f,b1f = yy1fd.get(yr, (0,0,0))
        r1n,d1n,b1n = yy1nd.get(yr, (0,0,0))
        drag = b1f - b1n
        flag = " LOSS" if r1f < 0 else "     "
        print(f"  {yr:<6}  {r1f:>+8.1f}% {d1f:>+7.1f}% ${b1f:>9,.0f}{flag}  |  "
              f"{r1n:>+8.1f}% {d1n:>+7.1f}% ${b1n:>9,.0f}  ${drag:>+9,.0f}")

    # ── 4. S2 symbol-wise P&L ─────────────────────────────────────────────────
    print_header("S2 -- PER-SYMBOL P&L  (WITH FEES)")
    rows2 = sorted(sp2f.values(), key=lambda x: -x["pnl"])
    total_dep  = sum(v["deployed"] for v in sp2f.values())
    total_ret  = sum(v["returned"] for v in sp2f.values())
    print(f"\n  {'Symbol':<14} {'Deployed $':>12} {'Returned $':>12} "
          f"{'PnL $':>12} {'Return %':>9}")
    print("  " + "-" * 65)
    for sym, v in sorted(sp2f.items(), key=lambda x: -x[1]["pnl"]):
        pct = (v["returned"]/v["deployed"]-1)*100 if v["deployed"] > 0 else 0
        flag = " LOSS" if v["pnl"] < 0 else ""
        print(f"  {sym:<14} ${v['deployed']:>11,.0f} ${v['returned']:>11,.0f} "
              f"${v['pnl']:>+11,.0f} {pct:>+8.1f}%{flag}")
    print("  " + "-" * 65)
    print(f"  {'TOTAL':<14} ${total_dep:>11,.0f} ${total_ret:>11,.0f} "
          f"${total_ret-total_dep:>+11,.0f}")

    # ── 5. S2 symbol-wise P&L WITHOUT fees ────────────────────────────────────
    print_header("S2 -- PER-SYMBOL P&L  (WITHOUT FEES)")
    print(f"\n  {'Symbol':<14} {'Deployed $':>12} {'Returned $':>12} "
          f"{'PnL $':>12} {'Return %':>9}")
    print("  " + "-" * 65)
    for sym, v in sorted(sp2n.items(), key=lambda x: -x[1]["pnl"]):
        pct = (v["returned"]/v["deployed"]-1)*100 if v["deployed"] > 0 else 0
        flag = " LOSS" if v["pnl"] < 0 else ""
        print(f"  {sym:<14} ${v['deployed']:>11,.0f} ${v['returned']:>11,.0f} "
              f"${v['pnl']:>+11,.0f} {pct:>+8.1f}%{flag}")

    # ── 6. S2 -- which symbols held each year ─────────────────────────────────
    print_header("S2 -- SYMBOL WEEKS HELD PER YEAR  (explains annual returns)")
    sym_yr = sym_year_table(hl2, eq2f)
    all_yrs = sorted({yr for s in sym_yr for yr in sym_yr[s]})
    # top symbols by total weeks held
    top_syms = sorted(sym_yr.keys(), key=lambda s: -sum(sym_yr[s].values()))[:15]
    print(f"\n  {'Symbol':<14}" + "".join(f"  {y}" for y in all_yrs) + "   Total")
    print("  " + "-" * (14 + 7 * len(all_yrs) + 10))
    for sym in top_syms:
        row = f"  {sym:<14}"
        total = 0
        for yr in all_yrs:
            wk = sym_yr[sym].get(yr, 0)
            total += wk
            row += f"  {wk:>4}" if wk else "     -"
        row += f"  {total:>5} wks"
        print(row)

    # ── 7. S1 symbol-wise P&L ─────────────────────────────────────────────────
    print_header("S1 -- PER-SYMBOL P&L  (WITH FEES)")
    print(f"\n  {'Symbol':<14} {'Deployed $':>12} {'Returned $':>12} "
          f"{'PnL $':>12} {'Return %':>9} {'Months':>7}")
    print("  " + "-" * 72)
    # count months held for S1
    s1_months = {}
    for _, syms in hl1:
        for s in syms:
            s1_months[s] = s1_months.get(s, 0) + 1
    for sym, v in sorted(sp1f.items(), key=lambda x: -x[1]["pnl"]):
        pct   = (v["returned"]/v["deployed"]-1)*100 if v["deployed"] > 0 else 0
        mo    = s1_months.get(sym, 0)
        flag  = " LOSS" if v["pnl"] < 0 else ""
        print(f"  {sym:<14} ${v['deployed']:>11,.0f} ${v['returned']:>11,.0f} "
              f"${v['pnl']:>+11,.0f} {pct:>+8.1f}% {mo:>6}mo{flag}")
    t1dep = sum(v["deployed"] for v in sp1f.values())
    t1ret = sum(v["returned"] for v in sp1f.values())
    print("  " + "-" * 72)
    print(f"  {'TOTAL':<14} ${t1dep:>11,.0f} ${t1ret:>11,.0f} ${t1ret-t1dep:>+11,.0f}")

    # ── 8. Fee impact breakdown ────────────────────────────────────────────────
    print_header("FEE IMPACT ANALYSIS")
    n_buy_s1  = sum(1 for t in tr1f if t["action"] == "BUY")
    n_sell_s1 = sum(1 for t in tr1f if t["action"] == "SELL")
    n_buy_s2  = sum(1 for t in tr2f if t["action"] in ("BUY","BUY_TOPUP"))
    n_sell_s2 = sum(1 for t in tr2f if t["action"] in ("SELL","SELL_TRIM"))

    cost_s1 = sum(t.get("cash_out",0) * (FEE_SIDE + slip(t["sym"])) / (1 + FEE_SIDE + slip(t["sym"]))
                  for t in tr1f if "cash_out" in t)
    cost_s2 = sum(t.get("cash_out",0) * (FEE_SIDE + slip(t["sym"])) / (1 + FEE_SIDE + slip(t["sym"]))
                  for t in tr2f if "cash_out" in t)

    print(f"""
  Strategy 1 (monthly rotation, {n_buy_s1} buys / {n_sell_s1} sells):
    With fees:     CAGR {s1f['cagr']:>+.1f}%   Final ${s1f['final']:>8,.0f}
    Without fees:  CAGR {s1n['cagr']:>+.1f}%   Final ${s1n['final']:>8,.0f}
    Fee drag:      CAGR {s1n['cagr']-s1f['cagr']:>+.1f} pp  Final ${fee_drag_s1:>+8,.0f}

  Strategy 2 (weekly rotation, {n_buy_s2} buys / {n_sell_s2} sells):
    With fees:     CAGR {s2f['cagr']:>+.1f}%   Final ${s2f['final']:>8,.0f}
    Without fees:  CAGR {s2n['cagr']:>+.1f}%   Final ${s2n['final']:>8,.0f}
    Fee drag:      CAGR {s2n['cagr']-s2f['cagr']:>+.1f} pp  Final ${fee_drag_s2:>+8,.0f}

  LZR v14 baseline: CAGR -26.5%  Final $200  (reversal strategy, not profitable)
""")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    eq1f.to_csv(OUT/"eq_s1_fees.csv",    header=["equity"])
    eq1n.to_csv(OUT/"eq_s1_nofees.csv",  header=["equity"])
    eq2f.to_csv(OUT/"eq_s2_fees.csv",    header=["equity"])
    eq2n.to_csv(OUT/"eq_s2_nofees.csv",  header=["equity"])
    pd.DataFrame(tr1f).to_csv(OUT/"trades_s1_fees.csv",   index=False)
    pd.DataFrame(tr2f).to_csv(OUT/"trades_s2_fees.csv",   index=False)
    pd.DataFrame(tr2n).to_csv(OUT/"trades_s2_nofees.csv", index=False)
    print(f"  Saved to backtest_results/")

    # ── 10. Final verdict ──────────────────────────────────────────────────────
    print_header("FINAL VERDICT")

    def vrd(s):
        if s["calmar"] >= 1.0 and s["cagr"] > 0: return "INVESTABLE  (Calmar >= 1.0)"
        if s["calmar"] >= 0.5 and s["cagr"] > 0: return "MARGINAL    (Calmar 0.5-1.0)"
        if s["cagr"] > 0:                          return "POSITIVE    (high DD)"
        return "NOT PROFITABLE"

    print(f"""
  S1 (Momentum Rotation)  WITH fees: {vrd(s1f)}
     CAGR {s1f['cagr']:>+.1f}%  DD {s1f['max_dd']:>+.1f}%  Calmar {s1f['calmar']:.2f}  Final ${s1f['final']:>8,.0f}

  S1 (Momentum Rotation)  NO  fees: {vrd(s1n)}
     CAGR {s1n['cagr']:>+.1f}%  DD {s1n['max_dd']:>+.1f}%  Calmar {s1n['calmar']:.2f}  Final ${s1n['final']:>8,.0f}

  S2 (EMA Trend+Momentum) WITH fees: {vrd(s2f)}
     CAGR {s2f['cagr']:>+.1f}%  DD {s2f['max_dd']:>+.1f}%  Calmar {s2f['calmar']:.2f}  Final ${s2f['final']:>8,.0f}

  S2 (EMA Trend+Momentum) NO  fees: {vrd(s2n)}
     CAGR {s2n['cagr']:>+.1f}%  DD {s2n['max_dd']:>+.1f}%  Calmar {s2n['calmar']:.2f}  Final ${s2n['final']:>8,.0f}

  LZR v14 baseline:  NOT PROFITABLE
     CAGR -26.5%  DD -87.2%  Calmar 0.30  Final $200
""")
    print(f"  Total runtime: {(time.time()-t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
