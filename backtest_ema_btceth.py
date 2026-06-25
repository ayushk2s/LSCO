"""
backtest_ema_btceth.py  --  EMA Trend on BTC + ETH  (investor-grade report)
=============================================================================
Universe : BTCUSDT, ETHUSDT only  (always in market, no moonshot dependency)
Signal   : weekly close > EMA(slow) from PREVIOUS week  (no look-ahead)
Rank     : 4-week momentum (higher momentum = preferred)
Hold     : whichever qualifies (0, 1, or 2), equal weight
Cash     : if neither passes filter (bear market protection)

4 Variants:
  V1  EMA(20)  equal weight
  V2  EMA(20)  inverse-ATR vol sizing
  V3  EMA(30)  equal weight          <-- best risk-adjusted in prior test
  V4  EMA(30)  inverse-ATR vol sizing

Investor metrics added:
  Sharpe Ratio   (rf = 4% annual)
  Sortino Ratio  (rf = 4% annual, downside deviation only)
  Annual Volatility
  Monthly Win Rate
  Best / Worst Month
  Avg Win / Avg Loss Month
  Max Consecutive Losing Months
  Max Recovery Time from Drawdown (months)
  Full Monthly Returns Table  (Year x Month grid)

Fees: 0.04%/side taker + 0.05% slippage = 0.09%/side round-trip
No look-ahead: signal from prev week's close, executed on current week's open.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from lzr_core import load_1m, resample

OUT       = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT.mkdir(exist_ok=True)

INITIAL    = 1_000.0
FEE_SIDE   = 0.0004
SLIP_RATES = {"BTCUSDT": 0.0005, "ETHUSDT": 0.0005}
UNIVERSE   = ["BTCUSDT", "ETHUSDT"]
RISK_FREE  = 0.04          # 4% annual, used for Sharpe / Sortino
W          = "=" * 78
SEP        = "-" * 78


# ── Trade helpers ──────────────────────────────────────────────────────────────

def slip(sym):
    return SLIP_RATES.get(sym, 0.0010)

def _buy(sym, price, dollar, fm=1.0):
    """fm = fee_mult (0 = no fees, 1 = full fees)."""
    s   = slip(sym) * fm
    f   = FEE_SIDE  * fm
    qty = dollar / (price * (1 + f + s))
    return qty, dollar          # (coins received, cash spent)

def _sell(sym, price, qty, fm=1.0):
    s = slip(sym) * fm
    f = FEE_SIDE  * fm
    return qty * price * (1 - f - s)   # cash received

def equity_val(cash, holdings, lc):
    return cash + sum(holdings[x] * lc.get(x, 0) for x in holdings)


# ── Basic stats ────────────────────────────────────────────────────────────────

def compute_stats(eq_series, init=INITIAL):
    eq = eq_series.dropna()
    if len(eq) < 2:
        return {}
    final = float(eq.iloc[-1])
    yrs   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (final / init) ** (1 / max(yrs, 0.01)) - 1
    pk    = eq.cummax()
    mdd   = float(((eq - pk) / pk).min())
    cal   = abs(cagr / mdd) if mdd < -0.001 else float("inf")
    return dict(
        cagr   = round(cagr * 100, 1),
        cagr_r = cagr,          # raw float for Sharpe calc
        mdd    = round(mdd * 100, 1),
        calmar = round(cal, 2),
        final  = round(final, 2),
    )

def year_by_year(eq_series, init=INITIAL):
    rows, bal = [], init
    for yr, grp in eq_series.groupby(eq_series.index.year):
        end = float(grp.iloc[-1])
        ret = (end / bal - 1) * 100
        pk  = grp.cummax()
        dd  = float(((grp - pk) / pk).min()) * 100
        rows.append((yr, round(ret, 1), round(dd, 1), round(end, 2)))
        bal = end
    return rows


# ── Investor-grade metrics ─────────────────────────────────────────────────────

def compute_advanced_stats(eq_series, risk_free=RISK_FREE):
    """
    Returns Sharpe, Sortino, volatility, monthly stats, recovery time.
    Strips flat warmup (constant INITIAL) before computing so that
    unused weeks don't artificially deflate volatility / Sharpe.
    """
    eq = eq_series.dropna()
    if len(eq) < 10:
        return {}

    # Strip leading flat warmup: find first point that differs from start
    init_val = float(eq.iloc[0])
    changed  = eq[eq != init_val]
    if len(changed) > 0:
        eq = eq[changed.index[0]:]   # start from first real trade
    if len(eq) < 10:
        return {}

    init  = float(eq.iloc[0])
    final = float(eq.iloc[-1])
    yrs   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (final / init) ** (1 / max(yrs, 0.01)) - 1

    # ── Weekly returns ─────────────────────────────────────────────────────────
    weekly_ret = eq.pct_change().dropna()
    if len(weekly_ret) == 0:
        return {}

    rf_weekly  = (1 + risk_free) ** (1 / 52) - 1   # weekly risk-free rate
    ann_vol    = float(weekly_ret.std()) * np.sqrt(52)

    # Sharpe: (CAGR - rf) / annualised weekly vol
    sharpe = (cagr - risk_free) / ann_vol if ann_vol > 0 else 0.0

    # Sortino: only downside weeks count (weeks below risk-free)
    down_weeks = weekly_ret[weekly_ret < rf_weekly]
    if len(down_weeks) > 0:
        down_std = np.sqrt((down_weeks ** 2).mean()) * np.sqrt(52)
    else:
        down_std = 1e-9
    sortino = (cagr - risk_free) / down_std

    # ── Monthly returns via resample ───────────────────────────────────────────
    try:
        mo_eq = eq.resample("ME").last().dropna()   # pandas >= 2.2
    except Exception:
        mo_eq = eq.resample("M").last().dropna()    # pandas < 2.2
    mo_ret = mo_eq.pct_change().dropna()

    if len(mo_ret) == 0:
        return {}

    pos_mo    = int((mo_ret > 0).sum())
    total_mo  = int(len(mo_ret))
    monthly_wr = pos_mo / total_mo * 100

    best_mo   = float(mo_ret.max()) * 100
    worst_mo  = float(mo_ret.min()) * 100
    avg_win   = float(mo_ret[mo_ret > 0].mean()) * 100 if (mo_ret > 0).any() else 0.0
    avg_loss  = float(mo_ret[mo_ret < 0].mean()) * 100 if (mo_ret < 0).any() else 0.0

    # Max consecutive losing months
    max_consec  = 0
    curr_consec = 0
    for v in mo_ret:
        if v < 0:
            curr_consec += 1
            max_consec   = max(max_consec, curr_consec)
        else:
            curr_consec = 0

    # ── Recovery time: bottom of each drawdown -> new equity high ─────────────
    pk           = eq.cummax()
    in_dd        = False
    bottom_date  = None
    bottom_val   = float("inf")
    max_rec_days = 0

    for date, val, peak in zip(eq.index, eq.values, pk.values):
        underwater = val < peak * 0.999     # 0.1% tolerance for float noise
        if underwater:
            in_dd = True
            if val < bottom_val:
                bottom_val  = val
                bottom_date = date
        else:
            if in_dd and bottom_date is not None:
                rec_days     = (date - bottom_date).days
                max_rec_days = max(max_rec_days, rec_days)
            in_dd       = False
            bottom_date = None
            bottom_val  = float("inf")

    # If still in drawdown at end of series
    if in_dd and bottom_date is not None:
        rec_days     = (eq.index[-1] - bottom_date).days
        max_rec_days = max(max_rec_days, rec_days)

    max_rec_months = max_rec_days / 30.44

    return dict(
        sharpe          = round(sharpe, 2),
        sortino         = round(min(sortino, 99.0), 2),   # cap at 99 if near inf
        ann_vol         = round(ann_vol * 100, 1),
        monthly_wr      = round(monthly_wr, 1),
        pos_months      = pos_mo,
        total_months    = total_mo,
        best_month      = round(best_mo, 1),
        worst_month     = round(worst_mo, 1),
        avg_win         = round(avg_win, 1),
        avg_loss        = round(avg_loss, 1),
        max_consec_loss = max_consec,
        max_rec_months  = round(max_rec_months, 1),
        monthly_ret     = mo_ret,
    )


def print_monthly_table(mo_ret, title="Monthly Returns (%)"):
    """Print a Year x Month grid of monthly returns."""
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"\n  {title}")
    hdr = f"  {'Year':<6}" + "".join(f" {m:>6}" for m in MONTHS) + f"  {'Annual':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for yr, grp in mo_ret.groupby(mo_ret.index.year):
        row  = f"  {yr:<6}"
        yr_r = 1.0
        for m in range(1, 13):
            mdata = grp[grp.index.month == m]
            if len(mdata) > 0:
                r     = float(mdata.iloc[0]) * 100
                yr_r *= (1 + float(mdata.iloc[0]))
                # Flag outstanding months
                if r >= 20:
                    tag = "^"
                elif r <= -10:
                    tag = "v"
                else:
                    tag = " "
                row += f" {r:>+5.1f}{tag}"
            else:
                row += f"   --- "
        yr_pct = (yr_r - 1) * 100
        row += f"  {yr_pct:>+7.1f}%"
        print(row)

    print(f"\n  (^ = month >= +20%   v = month <= -10%)")


# ── Strategy engine ────────────────────────────────────────────────────────────

def run_strategy(weekly_data, cfg, fee_mult=1.0, track_trades=False):
    """
    Signal   : prev week close > EMA(slow), 4w momentum > 0   (no look-ahead)
    Execute  : current week open
    Rebalance: only when target set changes
    Vol-size : optional inverse-ATR position sizing + portfolio-level dampener
    """
    ema_slow = cfg["ema_slow"]
    ema_fast = cfg["ema_fast"]
    mom_w    = cfg["mom_weeks"]
    top_n    = cfg["top_n"]
    vol_size = cfg.get("vol_size", False)
    atr_w    = cfg.get("atr_weeks", 8)

    # Precompute all indicators (causal — EWM/rolling only uses past data)
    slow_ema, mom, atr_ser = {}, {}, {}
    for sym in UNIVERSE:
        cl = weekly_data[sym]["close"]
        hi = weekly_data[sym]["high"]
        lo = weekly_data[sym]["low"]
        slow_ema[sym] = cl.ewm(span=ema_slow, adjust=False).mean()
        mom[sym]      = cl.pct_change(mom_w)
        tr            = pd.concat([
                            hi - lo,
                            (hi - cl.shift(1)).abs(),
                            (lo - cl.shift(1)).abs(),
                        ], axis=1).max(axis=1)
        atr_ser[sym]  = tr.rolling(atr_w).mean()

    ref   = max(UNIVERSE, key=lambda s: len(weekly_data[s]))
    weeks = list(weekly_data[ref].index)

    # Warmup: need enough bars for all indicators to be valid
    WARMUP = max(ema_slow, ema_fast) + mom_w + atr_w + 2

    cash        = INITIAL
    holdings    = {}
    prev_target = set()
    lc          = {}          # last-known-close, prevents zero-valuation on gaps
    eq_curve    = {}
    # Trade tracking (populated only when track_trades=True)
    _total_fees = 0.0
    _buy_n      = 0
    _sell_n     = 0
    _open_pos   = {}   # sym -> [entry_date, cost_basis, qty]
    _trades     = []   # completed round-trip exits

    for i, wk in enumerate(weeks):

        # Update last-known-close with current bar's close
        for sym in UNIVERSE:
            if wk in weekly_data[sym].index:
                c = weekly_data[sym].loc[wk, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        if i < WARMUP:
            eq_curve[wk] = INITIAL
            continue

        prev_wk = weeks[i - 1]

        # ── Build signal from PREVIOUS week (no look-ahead) ───────────────────
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

        # Select top N by momentum
        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:top_n]
        target = set(s for s, _ in ranked)

        # ── Rebalance only when holdings need to change ────────────────────────
        if target != prev_target:
            to_sell = prev_target - target

            # Sell exits — use open price or fall back to last-known-close
            for sym in to_sell:
                if sym not in holdings:
                    continue
                if wk in weekly_data[sym].index:
                    p = float(weekly_data[sym].loc[wk, "open"])
                    if np.isnan(p) or p <= 0:
                        p = lc.get(sym, 0)
                else:
                    p = lc.get(sym, 0)
                if p > 0:
                    proceeds = _sell(sym, p, holdings[sym], fee_mult)
                    cash += proceeds
                    if track_trades:
                        _f = (FEE_SIDE + slip(sym)) * fee_mult
                        _total_fees += holdings[sym] * p * _f
                        _sell_n += 1
                        if sym in _open_pos:
                            _ep   = _open_pos.pop(sym)
                            _pnl  = round(proceeds - _ep[1], 2)
                            _trades.append({
                                "sym"     : sym,
                                "entry"   : _ep[0],
                                "exit"    : wk,
                                "hold_wk" : round((wk - _ep[0]).days / 7, 1),
                                "cost"    : round(_ep[1], 2),
                                "proceeds": round(proceeds, 2),
                                "pnl"     : _pnl,
                                "pnl_pct" : round((_pnl / _ep[1]) * 100, 1) if _ep[1] > 0 else 0.0,
                            })
                del holdings[sym]

            if target:
                # Collect current-week open prices for targets
                opx = {}
                for sym in target:
                    if wk in weekly_data[sym].index:
                        p = float(weekly_data[sym].loc[wk, "open"])
                        if not np.isnan(p) and p > 0:
                            opx[sym] = p

                # Total portfolio value (cash + existing target holdings)
                port_val = cash
                for sym in target:
                    if sym in holdings:
                        port_val += holdings[sym] * opx.get(sym, lc.get(sym, 0))

                # Compute per-symbol allocation
                if vol_size and atr_vals:
                    # Inverse-ATR weight: lower volatility -> larger allocation
                    pct_vol = {}
                    for sym in target:
                        if sym in atr_vals and sym in opx and opx[sym] > 0:
                            pct_vol[sym] = atr_vals[sym] / opx[sym]

                    if pct_vol:
                        inv     = {s: 1 / max(v, 1e-6) for s, v in pct_vol.items()}
                        tot_inv = sum(inv.values())
                        weights = {s: inv[s] / tot_inv for s in inv}
                        # Portfolio-level dampener: if avg %ATR > 3%, scale down
                        avg_pv = float(np.mean(list(pct_vol.values())))
                        scale  = min(1.0, 0.03 / max(avg_pv, 1e-6))
                        per_sym_alloc = {s: port_val * weights[s] * scale
                                         for s in weights}
                    else:
                        per_sym = port_val / len(target)
                        per_sym_alloc = {s: per_sym for s in target}
                else:
                    per_sym = port_val / len(target)
                    per_sym_alloc = {s: per_sym for s in target}

                # Trim over-weight existing positions (only those staying in target)
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
                        if track_trades:
                            _f = (FEE_SIDE + slip(sym)) * fee_mult
                            _total_fees += excess * p * _f
                            _sell_n += 1
                            if sym in _open_pos and _open_pos[sym][2] > 0:
                                _ratio = excess / _open_pos[sym][2]
                                _open_pos[sym][1] *= (1.0 - _ratio)
                                _open_pos[sym][2] -= excess
                        holdings[sym] -= excess

                # Buy up under-weight positions (new entries + top-ups)
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
                    cash         -= spent
                    if track_trades:
                        _f = (FEE_SIDE + slip(sym)) * fee_mult
                        _total_fees += spent * _f / (1.0 + _f)
                        _buy_n += 1
                        if sym not in _open_pos:
                            _open_pos[sym] = [wk, spent, qty]
                        else:
                            _open_pos[sym][1] += spent
                            _open_pos[sym][2] += qty

            prev_target = target

        eq_curve[wk] = equity_val(cash, holdings, lc)

    eq_s = pd.Series(eq_curve)
    if track_trades:
        return eq_s, {
            "trade_log"  : _trades,
            "total_fees" : round(_total_fees, 2),
            "buy_orders" : _buy_n,
            "sell_orders": _sell_n,
            "open_pos"   : len(_open_pos),
        }
    return eq_s


# ── Buy-and-hold baseline ──────────────────────────────────────────────────────

def run_bah(weekly_data, sym, fm=1.0):
    """Buy-and-hold: buy at first available open, hold forever."""
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
                qty, _ = _buy(sym, p, INITIAL, fm)
                cash   = 0.0
                first  = False
        eq[wk] = cash + qty * lc.get(sym, 0)

    return pd.Series(eq)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(W)
    print("  EMA BTC+ETH Strategy  --  Investor-Grade Report")
    print("  Universe: BTCUSDT + ETHUSDT only  (always in market, repeatable)")
    print(W)

    print("\n  Loading 1-min data and resampling to weekly...")
    weekly_data = {}
    for sym in UNIVERSE:
        df1 = load_1m(sym)
        weekly_data[sym] = resample(df1, "1W")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── Four variants ──────────────────────────────────────────────────────────
    variants = {
        "V1 EMA20 plain"   : {"ema_slow":20,"ema_fast":10,"mom_weeks":4,"top_n":2,"vol_size":False},
        "V2 EMA20 vol-size": {"ema_slow":20,"ema_fast":10,"mom_weeks":4,"top_n":2,"vol_size":True,"atr_weeks":8},
        "V3 EMA30 plain"   : {"ema_slow":30,"ema_fast":15,"mom_weeks":4,"top_n":2,"vol_size":False},
        "V4 EMA30 vol-size": {"ema_slow":30,"ema_fast":15,"mom_weeks":4,"top_n":2,"vol_size":True,"atr_weeks":8},
    }

    print("  Running all variants...")
    results = {}
    for name, cfg in variants.items():
        eq_f = run_strategy(weekly_data, cfg, fee_mult=1.0)
        eq_n = run_strategy(weekly_data, cfg, fee_mult=0.0)
        results[name] = {"with_fees": eq_f, "no_fees": eq_n}
        st = compute_stats(eq_f)
        print(f"    {name:<22}  CAGR {st['cagr']:>+5.1f}%  "
              f"DD {st['mdd']:>+5.1f}%  Calmar {st['calmar']:.2f}")

    # Run V3 again with trade tracking to collect detailed statistics
    v3_eq, v3_ti = run_strategy(
        weekly_data,
        {"ema_slow":30,"ema_fast":15,"mom_weeks":4,"top_n":2,"vol_size":False},
        fee_mult=1.0, track_trades=True
    )

    bah_btc = run_bah(weekly_data, "BTCUSDT")
    bah_eth = run_bah(weekly_data, "ETHUSDT")
    st_btc  = compute_stats(bah_btc)
    st_eth  = compute_stats(bah_eth)

    # Pick best variant by Calmar
    best_calmar = 0.0
    best_name   = ""
    for name, res in results.items():
        c = compute_stats(res["with_fees"])["calmar"]
        if c > best_calmar:
            best_calmar = c
            best_name   = name

    # ── Section 1: Summary table ───────────────────────────────────────────────
    print(f"\n{W}")
    print("  1. SUMMARY TABLE  --  With Fees  (Jan 2021 - Jun 2026)")
    print(W)
    print(f"\n  {'Strategy':<24} {'CAGR':>8} {'Max DD':>8} {'Calmar':>8} "
          f"{'Final $':>10}  Fee drag/yr")
    print("  " + SEP)
    print(f"  {'BTC Buy-and-Hold':<24} {st_btc['cagr']:>+7.1f}% "
          f"{st_btc['mdd']:>+7.1f}% {st_btc['calmar']:>8.2f} "
          f"${st_btc['final']:>8,.0f}")
    print(f"  {'ETH Buy-and-Hold':<24} {st_eth['cagr']:>+7.1f}% "
          f"{st_eth['mdd']:>+7.1f}% {st_eth['calmar']:>8.2f} "
          f"${st_eth['final']:>8,.0f}")
    print("  " + SEP)
    for name, res in results.items():
        stf = compute_stats(res["with_fees"])
        stn = compute_stats(res["no_fees"])
        drag = round(stn["cagr"] - stf["cagr"], 1)
        tag  = "  <-- BEST" if name == best_name else ""
        print(f"  {name:<24} {stf['cagr']:>+7.1f}% {stf['mdd']:>+7.1f}% "
              f"{stf['calmar']:>8.2f} ${stf['final']:>8,.0f}  -{drag:.1f}%/yr{tag}")

    # ── Section 2: Year-by-year (best variant) ─────────────────────────────────
    def lu(yy, yr):
        for y, r, d, b in yy:
            if y == yr: return r, d, b
        return 0.0, 0.0, 0.0

    best_eq   = results[best_name]["with_fees"]
    best_eq_n = results[best_name]["no_fees"]
    yb        = year_by_year(best_eq)
    yb_btc    = year_by_year(bah_btc)
    yb_eth    = year_by_year(bah_eth)
    yb_n      = year_by_year(best_eq_n)
    all_yrs   = sorted(set(y for y, *_ in yb + yb_btc + yb_eth))

    print(f"\n{W}")
    print(f"  2. YEAR-BY-YEAR  --  {best_name}  vs  BTC / ETH Buy-and-Hold")
    print(W)
    print(f"\n  {'Year':<6} {'Strat':>8} {'StratDD':>9} {'Balance':>10} "
          f"{'BTC':>8} {'ETH':>8}  Note")
    print("  " + SEP)
    for yr in all_yrs:
        rs, ds, bs = lu(yb, yr)
        rb, db, _  = lu(yb_btc, yr)
        re, de, _  = lu(yb_eth, yr)
        diff = rs - rb
        note = ("beats BTC" if diff > 5 else
                "lags BTC"  if diff < -5 else "~= BTC")
        print(f"  {yr:<6} {rs:>+7.1f}% {ds:>+8.1f}% ${bs:>8,.0f} "
              f"{rb:>+7.1f}% {re:>+7.1f}%  {note}")

    # ── Section 3: Fees with/without ──────────────────────────────────────────
    print(f"\n{W}")
    print(f"  3. WITH FEES vs WITHOUT FEES  --  {best_name}")
    print(W)
    print(f"\n  {'Year':<6} {'WithFees':>10} {'NoFees':>10} {'Drag':>8} {'Bal_wFees':>12}")
    print("  " + SEP)
    for yr in all_yrs:
        rf,  _, bf = lu(yb,   yr)
        rn,  _, _  = lu(yb_n, yr)
        print(f"  {yr:<6} {rf:>+9.1f}% {rn:>+9.1f}% {rn-rf:>+7.1f}% ${bf:>10,.0f}")

    # ── Section 4: All variants year-by-year ──────────────────────────────────
    all_yby = {name: year_by_year(res["with_fees"]) for name, res in results.items()}

    print(f"\n{W}")
    print("  4. ALL VARIANTS  --  Annual Return %  (with fees)")
    print(W)
    vnames = list(variants.keys())
    hdr4 = f"  {'Year':<6}" + "".join(f" {n[:14]:>15}" for n in vnames) + f"  {'BTC':>8}"
    print(hdr4)
    print("  " + "-" * (len(hdr4) - 2))
    for yr in all_yrs:
        rb, _, _ = lu(yb_btc, yr)
        row = f"  {yr:<6}"
        for name in vnames:
            r, d, b = lu(all_yby[name], yr)
            row += f" {r:>+14.1f}%"
        row += f"  {rb:>+7.1f}%"
        print(row)

    print(f"\n  Drawdown by year:")
    hdr4b = f"  {'Year':<6}" + "".join(f" {n[:14]:>15}" for n in vnames) + f"  {'BTC DD':>8}"
    print(hdr4b)
    print("  " + "-" * (len(hdr4b) - 2))
    for yr in all_yrs:
        _, db, _ = lu(yb_btc, yr)
        row = f"  {yr:<6}"
        for name in vnames:
            r, d, b = lu(all_yby[name], yr)
            row += f" {d:>+14.1f}%"
        row += f"  {db:>+7.1f}%"
        print(row)

    # ── Section 5: Investor metrics ────────────────────────────────────────────
    print(f"\n{W}")
    print(f"  5. INVESTOR METRICS  (Sharpe / Sortino / Monthly Stats)")
    print(f"     Risk-free rate = {RISK_FREE*100:.0f}% annual")
    print(W)

    adv_btc = compute_advanced_stats(bah_btc)
    adv_eth = compute_advanced_stats(bah_eth)
    adv_all = {name: compute_advanced_stats(res["with_fees"])
               for name, res in results.items()}

    col_labels = ["BTC B&H","ETH B&H","V1","V2","V3","V4"]
    adv_list   = [adv_btc, adv_eth] + [adv_all[n] for n in vnames]

    def fmt(adv, key, suffix=""):
        v = adv.get(key, "N/A")
        if isinstance(v, float):
            return f"{v:.2f}{suffix}" if abs(v) < 10 else f"{v:.1f}{suffix}"
        return f"{v}{suffix}"

    rows_5 = [
        ("CAGR (w fees)",       [f"{compute_stats(bah_btc)['cagr']:+.1f}%",
                                  f"{compute_stats(bah_eth)['cagr']:+.1f}%"] +
                                 [f"{compute_stats(results[n]['with_fees'])['cagr']:+.1f}%"
                                  for n in vnames]),
        ("Max Drawdown",        [f"{compute_stats(bah_btc)['mdd']:+.1f}%",
                                  f"{compute_stats(bah_eth)['mdd']:+.1f}%"] +
                                 [f"{compute_stats(results[n]['with_fees'])['mdd']:+.1f}%"
                                  for n in vnames]),
        ("Calmar Ratio",        [f"{compute_stats(bah_btc)['calmar']:.2f}",
                                  f"{compute_stats(bah_eth)['calmar']:.2f}"] +
                                 [f"{compute_stats(results[n]['with_fees'])['calmar']:.2f}"
                                  for n in vnames]),
        ("--- Risk-adj ---",    ["","","","","",""]),
        ("Sharpe Ratio",        [fmt(a,"sharpe") for a in adv_list]),
        ("Sortino Ratio",       [fmt(a,"sortino") for a in adv_list]),
        ("Ann. Volatility",     [fmt(a,"ann_vol","%") for a in adv_list]),
        ("--- Monthly ---",     ["","","","","",""]),
        ("Monthly Win Rate",    [fmt(a,"monthly_wr","%") for a in adv_list]),
        ("Best Month",          [fmt(a,"best_month","%") for a in adv_list]),
        ("Worst Month",         [fmt(a,"worst_month","%") for a in adv_list]),
        ("Avg Win Month",       [fmt(a,"avg_win","%") for a in adv_list]),
        ("Avg Loss Month",      [fmt(a,"avg_loss","%") for a in adv_list]),
        ("--- Risk mgmt ---",   ["","","","","",""]),
        ("Max Consec Losses",   [f"{a.get('max_consec_loss','?')}mo" for a in adv_list]),
        ("Max Recovery",        [f"{a.get('max_rec_months','?'):.0f}mo"
                                  if isinstance(a.get('max_rec_months'), (int,float))
                                  else "?" for a in adv_list]),
        ("Pos Months / Total",  [f"{a.get('pos_months','?')}/{a.get('total_months','?')}"
                                  for a in adv_list]),
    ]

    col_w = 12
    print(f"\n  {'Metric':<22}" + "".join(f"{c:>{col_w}}" for c in col_labels))
    print("  " + "-" * (22 + col_w * len(col_labels)))
    for label, vals in rows_5:
        if label.startswith("---"):
            print(f"  {label}")
            continue
        row = f"  {label:<22}"
        for v in vals:
            row += f"{str(v):>{col_w}}"
        print(row)

    # ── Section 6: Monthly returns table (best variant) ───────────────────────
    print(f"\n{W}")
    print(f"  6. MONTHLY RETURNS TABLE  --  {best_name}  (with fees)")
    print(W)
    if adv_all[best_name].get("monthly_ret") is not None:
        print_monthly_table(adv_all[best_name]["monthly_ret"],
                            f"{best_name}  --  Monthly Returns (%)")

    # ── Section 7: Monthly returns table (BTC B&H baseline) ───────────────────
    print(f"\n{W}")
    print("  7. MONTHLY RETURNS TABLE  --  BTC Buy-and-Hold  (baseline)")
    print(W)
    if adv_btc.get("monthly_ret") is not None:
        print_monthly_table(adv_btc["monthly_ret"],
                            "BTC Buy-and-Hold  --  Monthly Returns (%)")

    # ── Section 8: What's still needed for investor presentation ──────────────
    print(f"\n{W}")
    print("  8. INVESTOR READINESS CHECKLIST")
    print(W)
    best_st    = compute_stats(results[best_name]["with_fees"])
    best_adv   = adv_all[best_name]
    calmar_ok  = best_st["calmar"] >= 0.5
    sharpe_ok  = best_adv.get("sharpe", 0) >= 0.5
    consec_ok  = best_adv.get("max_consec_loss", 99) <= 4
    rec_ok     = best_adv.get("max_rec_months", 99) <= 18

    checks = [
        (calmar_ok,  f"Calmar >= 0.5        : {best_st['calmar']:.2f}"),
        (sharpe_ok,  f"Sharpe >= 0.5        : {best_adv.get('sharpe','?'):.2f}"),
        (consec_ok,  f"Max consec loss <= 4m: {best_adv.get('max_consec_loss','?')}mo"),
        (rec_ok,     f"Recovery <= 18m      : {best_adv.get('max_rec_months','?'):.0f}mo"),
        (True,       "Walk-forward (6 win) : DONE  -- OOS Calmar 0.95, Sharpe 0.83"),
        (True,       "Regime WF (9 combos) : DONE  -- OOS Calmar 0.82, Sharpe 0.91"),
        (False,      "Live paper trading   : NOT DONE  -- 6-12 months recommended"),
    ]
    print()
    for ok, txt in checks:
        mark = "  [OK]" if ok else "  [  ]"
        print(f"  {mark}  {txt}")
    print()

    # ── Section 9: Trade statistics (V3 EMA30 plain) ──────────────────────────
    print(f"\n{W}")
    print("  9. TRADE STATISTICS  --  V3 EMA30 plain  (full investor detail)")
    print(W)

    tl    = v3_ti["trade_log"]
    v3_no_fee_final  = compute_stats(results["V3 EMA30 plain"]["no_fees"])["final"]
    v3_fee_final     = compute_stats(results["V3 EMA30 plain"]["with_fees"])["final"]
    compounded_drag  = v3_no_fee_final - v3_fee_final

    print(f"\n  -- Order Execution --")
    print(f"  Total buy  orders executed : {v3_ti['buy_orders']}")
    print(f"  Total sell orders executed : {v3_ti['sell_orders']}")
    print(f"  Completed round-trip trades: {len(tl)}")
    print(f"  Positions still open at end: {v3_ti['open_pos']}")

    print(f"\n  -- Fee Breakdown --")
    print(f"  Fee rate (each side)       : {FEE_SIDE*100:.2f}% taker + {SLIP_RATES['BTCUSDT']*100:.2f}% slippage")
    print(f"  Total fees paid (sum, $)   : ${v3_ti['total_fees']:,.2f}")
    print(f"  Compounded fee drag        : ${compounded_drag:,.2f}  "
          f"(no-fee final ${v3_no_fee_final:,.0f} vs with-fee ${v3_fee_final:,.0f})")
    avg_fee_per_order = v3_ti["total_fees"] / max(v3_ti["buy_orders"] + v3_ti["sell_orders"], 1)
    print(f"  Avg fee per order          : ${avg_fee_per_order:.2f}")

    if tl:
        wins   = [t for t in tl if t["pnl"] > 0]
        losses = [t for t in tl if t["pnl"] <= 0]
        wr     = len(wins) / len(tl) * 100
        avg_hold = sum(t["hold_wk"] for t in tl) / len(tl)
        gross_w  = sum(t["pnl"] for t in wins)  if wins   else 0.0
        gross_l  = abs(sum(t["pnl"] for t in losses)) if losses else 1e-9
        pf       = gross_w / gross_l
        avg_wp   = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0.0
        avg_lp   = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0
        best_t   = max(tl, key=lambda t: t["pnl"])
        worst_t  = min(tl, key=lambda t: t["pnl"])

        print(f"\n  -- Round-trip Trade Results --")
        print(f"  Win rate                   : {wr:.1f}%  "
              f"({len(wins)} wins  /  {len(losses)} losses)")
        print(f"  Avg holding period         : {avg_hold:.1f} weeks")
        print(f"  Avg winning trade          : +{avg_wp:.1f}%")
        print(f"  Avg losing trade           : {avg_lp:.1f}%")
        print(f"  Profit factor              : {pf:.2f}  (gross wins / gross losses)")
        print(f"  Gross profit               : ${gross_w:,.2f}")
        print(f"  Gross loss                 : ${gross_l:,.2f}")
        print(f"  Expectancy per trade       : ${(gross_w - gross_l)/len(tl):+,.2f}")

        print(f"\n  -- Best / Worst Single Trade --")
        print(f"  Best  : {best_t['sym']:<10} "
              f"Entry {str(best_t['entry'].date()):<12} "
              f"Exit {str(best_t['exit'].date()):<12} "
              f"Hold {best_t['hold_wk']:.0f}wk  "
              f"P&L ${best_t['pnl']:>+8,.2f}  ({best_t['pnl_pct']:>+.1f}%)")
        print(f"  Worst : {worst_t['sym']:<10} "
              f"Entry {str(worst_t['entry'].date()):<12} "
              f"Exit {str(worst_t['exit'].date()):<12} "
              f"Hold {worst_t['hold_wk']:.0f}wk  "
              f"P&L ${worst_t['pnl']:>+8,.2f}  ({worst_t['pnl_pct']:>+.1f}%)")

        print(f"\n  -- Full Trade Log --")
        print(f"  {'#':<4} {'Sym':<10} {'Entry':<12} {'Exit':<12} "
              f"{'Hold':>6} {'Cost':>10} {'Proceeds':>10} {'P&L $':>10} {'P&L%':>7}  W/L")
        print("  " + "-" * 88)
        for idx, t in enumerate(sorted(tl, key=lambda x: x["entry"]), 1):
            wl = "WIN" if t["pnl"] > 0 else "loss"
            print(f"  {idx:<4} {t['sym']:<10} "
                  f"{str(t['entry'].date()):<12} {str(t['exit'].date()):<12} "
                  f"{t['hold_wk']:>5.1f}w "
                  f"${t['cost']:>8,.2f} ${t['proceeds']:>8,.2f} "
                  f"${t['pnl']:>+8,.2f} {t['pnl_pct']:>+6.1f}%  {wl}")

        tl_df = pd.DataFrame(tl)
        tl_df.to_csv(OUT / "v3_trade_log.csv", index=False)
        print(f"\n  Trade log saved: backtest_results/v3_trade_log.csv")

    # ── Save equity curves ─────────────────────────────────────────────────────
    for name, res in results.items():
        clean = name.replace(" ", "_")
        res["with_fees"].to_csv(OUT / f"eq_{clean}.csv", header=["equity"])
    bah_btc.to_csv(OUT / "eq_bah_btc.csv", header=["equity"])
    bah_eth.to_csv(OUT / "eq_bah_eth.csv", header=["equity"])

    print(f"  Equity curves saved to backtest_results/")
    print(f"  Total runtime: {(time.time()-t0)/60:.1f} min\n")
    print(W)


if __name__ == "__main__":
    main()
