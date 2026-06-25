"""
backtest_btceth.py  ─  BTC + ETH: Strategy vs Buy-and-Hold
============================================================
User's question: "Leave INJ stuff. What about BTC and ETH,
those who were constantly in the market?"

We test THREE things on ONLY BTC and ETH:
  1. Buy-and-Hold BTC alone (baseline)
  2. Buy-and-Hold ETH alone (baseline)
  3. Buy-and-Hold equal-weight BTC+ETH (rebalance monthly)
  4. EMA Trend + Momentum on {BTC, ETH} only
       -- hold whichever is above EMA(20w) with positive 4w return
       -- hold both if both qualify, hold neither (cash) if neither does
  5. Blue-chip universe: BTC, ETH, BNB, SOL, XRP, LTC, BCH, LINK, ADA, DOGE
       -- same S2 logic but no moonshot alts
       -- the "always in market" assets, not INJ/FET type coins

All with fees. Year-by-year breakdown. No look-ahead.
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

INITIAL = 1_000.0
FEE     = 0.0004
EMA_W   = 20
MOM_W   = 4
W       = "=" * 76

SLIP = {
    "BTCUSDT": 0.0005, "ETHUSDT": 0.0005,
    "BNBUSDT": 0.0008, "SOLUSDT": 0.0008, "XRPUSDT": 0.0008,
    "ADAUSDT": 0.0008, "DOGEUSDT": 0.0008, "LTCUSDT": 0.0008,
    "LINKUSDT": 0.0010, "BCHUSDT": 0.0010,
}

BLUECHIP = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
            "LTCUSDT","BCHUSDT","LINKUSDT","ADAUSDT","DOGEUSDT"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def s(sym): return SLIP.get(sym, 0.0015)

def buy(sym, price, dollar):
    qty = dollar / (price * (1 + FEE + s(sym)))
    return qty, dollar

def sell(sym, price, qty):
    return qty * price * (1 - FEE - s(sym))

def eq_val(cash, h, lc):
    return cash + sum(h[x] * lc.get(x, 0) for x in h)

def stats(eq, init):
    if len(eq) < 2: return {}
    final = float(eq.iloc[-1])
    yrs   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (final/init)**(1/max(yrs,0.1)) - 1
    pk    = eq.cummax()
    mdd   = ((eq - pk)/pk).min()
    cal   = abs(cagr/mdd) if mdd < -0.001 else float("inf")
    return dict(cagr=round(cagr*100,1), mdd=round(mdd*100,1),
                calmar=round(cal,2), final=round(final,2))

def yby(eq, init):
    rows, bal = [], init
    for yr, g in eq.groupby(eq.index.year):
        end = float(g.iloc[-1])
        ret = (end/bal - 1)*100
        pk  = g.cummax()
        dd  = ((g-pk)/pk).min()*100
        rows.append((yr, round(ret,1), round(dd,1), round(end,2)))
        bal = end
    return rows


# ── 1. Buy-and-Hold ────────────────────────────────────────────────────────────

def run_bah(daily_data, sym_list, rebal_monthly=False):
    """Equal-weight buy-and-hold. If rebal_monthly: rebalance to equal weight monthly."""
    all_dates = sorted(set().union(*[set(daily_data[s].index) for s in sym_list]))
    all_dates = pd.DatetimeIndex(all_dates)

    # First day: buy equal weight
    cash = INITIAL
    holdings = {}
    lc = {}
    month_first = {}
    for d in all_dates:
        k = (d.year, d.month)
        if k not in month_first:
            month_first[k] = d
    rebal_set = set(month_first.values())

    # Initial buy on first available date
    first_date = all_dates[0]
    per = cash / len(sym_list)
    for sym in sym_list:
        p = float(daily_data[sym].iloc[0]["open"])
        q, spent = buy(sym, p, per)
        holdings[sym] = q
        cash -= spent

    eq = {}
    for date in all_dates:
        for sym in sym_list:
            if date in daily_data[sym].index:
                c = daily_data[sym].loc[date, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        if rebal_monthly and date in rebal_set and date != first_date:
            # Sell all, rebuy equal weight
            total = eq_val(0, holdings, lc)
            holdings = {}
            cash = total   # pretend we have the cash
            per = cash / len(sym_list)
            for sym in sym_list:
                p = lc.get(sym, 0)
                if p > 0:
                    q, spent = buy(sym, p, per)
                    holdings[sym] = q
                    cash -= spent

        eq[date] = eq_val(cash, holdings, lc)

    return pd.Series(eq)


# ── 2. EMA Trend on any symbol set ────────────────────────────────────────────

def run_ema_strategy(weekly_data, daily_data, sym_list, top_n):
    """
    Signal : prev week close > EMA(20w) AND 4w momentum > 0
    Execute: current week open
    Hold   : top_n by momentum, equal weight
    Cash   : when fewer than top_n qualify (partial or full cash)
    """
    ema = {s: weekly_data[s]["close"].ewm(span=EMA_W, adjust=False).mean()
           for s in sym_list}
    mom = {s: weekly_data[s]["close"].pct_change(MOM_W) for s in sym_list}

    ref   = max(sym_list, key=lambda s: len(weekly_data[s]))
    weeks = list(weekly_data[ref].index)

    cash = INITIAL
    holdings = {}
    prev_target = set()
    lc = {}

    eq = {}
    WARMUP = EMA_W + MOM_W + 1

    for i, wk in enumerate(weeks):
        for sym in holdings:
            if wk in weekly_data[sym].index:
                c = weekly_data[sym].loc[wk, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        if i < WARMUP:
            eq[wk] = INITIAL
            continue

        prev_wk = weeks[i-1]

        # Signal from prev week
        cands = {}
        for sym in sym_list:
            df = weekly_data[sym]
            if prev_wk not in df.index:
                continue
            cp = float(df.loc[prev_wk, "close"])
            ep = float(ema[sym].loc[prev_wk]) if prev_wk in ema[sym].index else np.nan
            mp = float(mom[sym].loc[prev_wk]) if prev_wk in mom[sym].index else np.nan
            if not np.isnan(ep) and not np.isnan(mp) and cp > ep and mp > 0:
                cands[sym] = mp

        target = set(s for s, _ in sorted(cands.items(), key=lambda x: -x[1])[:top_n])

        if target != prev_target:
            to_sell = prev_target - target

            # Sell exits (with lc fallback)
            for sym in to_sell:
                if sym not in holdings:
                    continue
                p = float(weekly_data[sym].loc[wk, "open"]) \
                    if wk in weekly_data[sym].index else lc.get(sym, 0)
                if p > 0:
                    cash += sell(sym, p, holdings[sym])
                del holdings[sym]

            if target:
                # Open prices
                opx = {}
                for sym in target:
                    if wk in weekly_data[sym].index:
                        p = float(weekly_data[sym].loc[wk, "open"])
                        if not np.isnan(p) and p > 0:
                            opx[sym] = p

                port = cash
                for sym in target:
                    if sym in holdings:
                        port += holdings[sym] * opx.get(sym, lc.get(sym, 0))

                per_sym = port / len(target)

                # Trim over-weight
                for sym in list(holdings.keys()):
                    p = opx.get(sym, lc.get(sym, 0))
                    if p <= 0: continue
                    cv = holdings[sym] * p
                    if cv > per_sym * 1.02:
                        excess = (cv - per_sym) / p
                        cash += sell(sym, p, excess)
                        holdings[sym] -= excess

                # Buy all under-weight
                for sym in target:
                    p = opx.get(sym)
                    if p is None: continue
                    cv      = holdings.get(sym, 0) * p
                    deficit = per_sym - cv
                    if deficit < 1.0: continue
                    spend   = min(deficit, cash)
                    if spend < 1.0: continue
                    q, spent = buy(sym, p, spend)
                    holdings[sym] = holdings.get(sym, 0) + q
                    cash -= spent

            prev_target = target

        for sym in holdings:
            if wk in weekly_data[sym].index:
                c = weekly_data[sym].loc[wk, "close"]
                if not np.isnan(c):
                    lc[sym] = float(c)

        eq[wk] = eq_val(cash, holdings, lc)

    return pd.Series(eq)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(W)
    print("  BTC + ETH: Strategy vs Buy-and-Hold")
    print("  'Leave INJ stuff -- what about the consistently present assets?'")
    print(W)

    # Load all needed symbols
    needed = set(BLUECHIP)
    print(f"\n  Loading {len(needed)} symbols...")
    daily_data  = {}
    weekly_data = {}
    for sym in needed:
        df1 = load_1m(sym)
        daily_data[sym]  = resample(df1, "1D")
        weekly_data[sym] = resample(df1, "1W")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── Run all strategies ─────────────────────────────────────────────────────
    print("  Running buy-and-hold baselines...")
    eq_bah_btc = run_bah(daily_data, ["BTCUSDT"])
    eq_bah_eth = run_bah(daily_data, ["ETHUSDT"])
    eq_bah_both = run_bah(daily_data, ["BTCUSDT","ETHUSDT"], rebal_monthly=True)

    print("  Running EMA strategy on BTC+ETH only (top 2)...")
    eq_ema2 = run_ema_strategy(weekly_data, daily_data, ["BTCUSDT","ETHUSDT"], top_n=2)

    print("  Running EMA strategy on Blue-chip 10 (top 3)...")
    eq_bc = run_ema_strategy(weekly_data, daily_data, BLUECHIP, top_n=3)

    print("  Running EMA strategy on Blue-chip 10 (top 5)...")
    eq_bc5 = run_ema_strategy(weekly_data, daily_data, BLUECHIP, top_n=5)

    # Stats
    sb  = stats(eq_bah_btc,  INITIAL)
    se  = stats(eq_bah_eth,  INITIAL)
    sbe = stats(eq_bah_both, INITIAL)
    s2  = stats(eq_ema2,     INITIAL)
    sbc = stats(eq_bc,       INITIAL)
    sb5 = stats(eq_bc5,      INITIAL)

    yb  = yby(eq_bah_btc,  INITIAL)
    ye  = yby(eq_bah_eth,  INITIAL)
    ybe = yby(eq_bah_both, INITIAL)
    y2  = yby(eq_ema2,     INITIAL)
    ybc = yby(eq_bc,       INITIAL)
    yb5 = yby(eq_bc5,      INITIAL)

    # ── Overall table ──────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  OVERALL RESULTS  (all with fees, Jan 2021 - Jun 2026)")
    print(W)
    print(f"\n  {'Strategy':<32} {'CAGR':>8} {'Max DD':>8} {'Calmar':>8} {'Final $':>10}")
    print("  " + "-" * 72)
    rows = [
        ("BTC Buy-and-Hold",                sb),
        ("ETH Buy-and-Hold",                se),
        ("BTC+ETH Equal-Weight B&H",        sbe),
        ("EMA on BTC+ETH only  (top 2)",    s2),
        ("EMA on Blue-chip 10  (top 3)",    sbc),
        ("EMA on Blue-chip 10  (top 5)",    sb5),
    ]
    for name, st in rows:
        print(f"  {name:<32} {st['cagr']:>+7.1f}% {st['mdd']:>+7.1f}% "
              f"{st['calmar']:>8.2f} ${st['final']:>9,.0f}")

    print(f"\n  Note: S2 full-30-symbol result for reference:")
    print(f"  {'S2 full 30 symbols (top 5)':<32} {'  +31.5%':>8} {'  -76.5%':>8} "
          f"{'    0.41':>8} ${'   4,192':>9}")
    print(f"  {'S2 full 30 symbols NO FEES':<32} {'  +38.7%':>8} {'  -74.9%':>8} "
          f"{'    0.52':>8} ${'   5,536':>9}")

    # ── Year-by-year all strategies ────────────────────────────────────────────
    print(f"\n{W}")
    print("  YEAR-BY-YEAR  (Return % / End balance $)")
    print(W)
    print(f"\n  {'Year':<6}  {'BTC B&H':>9} {'ETH B&H':>9} {'BTC+ETH':>9} "
          f"{'EMA BTC+ETH':>12} {'EMA BC-10 t3':>14} {'EMA BC-10 t5':>14}")
    print("  " + "-" * 80)

    def lookup(yy, yr):
        for y,r,d,b in yy:
            if y == yr: return r, b
        return 0.0, 0.0

    all_yrs = sorted(set(y for y,*_ in yb+ye+ybe+y2+ybc+yb5))
    for yr in all_yrs:
        rb,bb   = lookup(yb,  yr)
        re,be   = lookup(ye,  yr)
        rbe,bbe = lookup(ybe, yr)
        r2,b2   = lookup(y2,  yr)
        rc,bc   = lookup(ybc, yr)
        r5,b5   = lookup(yb5, yr)
        print(f"  {yr:<6}  "
              f"{rb:>+7.1f}% "
              f"{re:>+7.1f}% "
              f"{rbe:>+7.1f}% "
              f"{r2:>+10.1f}% "
              f"{rc:>+12.1f}% "
              f"{r5:>+12.1f}%")

    print(f"\n  Year-by-year end balances ($):")
    print(f"\n  {'Year':<6}  {'BTC B&H':>9} {'ETH B&H':>9} {'BTC+ETH':>9} "
          f"{'EMA BTC+ETH':>12} {'EMA BC-10 t3':>14} {'EMA BC-10 t5':>14}")
    print("  " + "-" * 80)
    for yr in all_yrs:
        rb,bb   = lookup(yb,  yr)
        re,be   = lookup(ye,  yr)
        rbe,bbe = lookup(ybe, yr)
        r2,b2   = lookup(y2,  yr)
        rc,bc   = lookup(ybc, yr)
        r5,b5   = lookup(yb5, yr)
        print(f"  {yr:<6}  "
              f"${bb:>8,.0f} "
              f"${be:>8,.0f} "
              f"${bbe:>8,.0f} "
              f"${b2:>11,.0f} "
              f"${bc:>13,.0f} "
              f"${b5:>13,.0f}")

    # ── Drawdown table ─────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  YEAR-BY-YEAR MAX DRAWDOWN (%)")
    print(W)
    print(f"\n  {'Year':<6}  {'BTC B&H':>9} {'ETH B&H':>9} {'BTC+ETH':>9} "
          f"{'EMA BTC+ETH':>12} {'EMA BC-10 t3':>14} {'EMA BC-10 t5':>14}")
    print("  " + "-" * 80)

    def lookup_dd(yy, yr):
        for y,r,d,b in yy:
            if y == yr: return d
        return 0.0

    for yr in all_yrs:
        print(f"  {yr:<6}  "
              f"{lookup_dd(yb, yr):>+8.1f}% "
              f"{lookup_dd(ye, yr):>+8.1f}% "
              f"{lookup_dd(ybe, yr):>+8.1f}% "
              f"{lookup_dd(y2, yr):>+11.1f}% "
              f"{lookup_dd(ybc, yr):>+13.1f}% "
              f"{lookup_dd(yb5, yr):>+13.1f}%")

    # ── Key insight ────────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  KEY FINDINGS")
    print(W)
    print(f"""
  1. BTC Buy-and-Hold: CAGR {sb['cagr']:+.1f}%  DD {sb['mdd']:+.1f}%  Calmar {sb['calmar']:.2f}
     Simple, no strategy needed. Just hold BTC.

  2. EMA on BTC+ETH only: CAGR {s2['cagr']:+.1f}%  DD {s2['mdd']:+.1f}%  Calmar {s2['calmar']:.2f}
     Does the trend filter help vs just holding BTC?

  3. EMA on Blue-chip 10 (top 3): CAGR {sbc['cagr']:+.1f}%  DD {sbc['mdd']:+.1f}%  Calmar {sbc['calmar']:.2f}
     10 established coins, no moonshots. Is it still better than BTC B&H?

  4. EMA on Blue-chip 10 (top 5): CAGR {sb5['cagr']:+.1f}%  DD {sb5['mdd']:+.1f}%  Calmar {sb5['calmar']:.2f}
     More diversification within blue chips.

  CONCLUSION:
  {'BTC B&H WINS' if sb['calmar'] >= max(s2['calmar'], sbc['calmar'], sb5['calmar']) else
   'EMA STRATEGY WINS over BTC B&H' if max(s2['calmar'], sbc['calmar'], sb5['calmar']) > sb['calmar'] else
   'MIXED - depends on metric'}
  BTC B&H Calmar {sb['calmar']:.2f}  vs  best EMA strategy Calmar {max(s2['calmar'],sbc['calmar'],sb5['calmar']):.2f}
""")

    # Save
    eq_bah_btc.to_csv(OUT/"eq_bah_btc.csv",  header=["equity"])
    eq_bah_eth.to_csv(OUT/"eq_bah_eth.csv",  header=["equity"])
    eq_ema2.to_csv(OUT/"eq_ema_btceth.csv",  header=["equity"])
    eq_bc.to_csv(OUT/"eq_ema_bc10_t3.csv",   header=["equity"])
    eq_bc5.to_csv(OUT/"eq_ema_bc10_t5.csv",  header=["equity"])
    print(f"  Saved equity curves to backtest_results/")
    print(f"\n  Runtime: {(time.time()-t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
