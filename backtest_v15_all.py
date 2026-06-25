"""
backtest_v15_all.py -- LZR v15: Wider TP/Trail  (30 symbols)
=============================================================
Same engine + look-ahead fixes as v14_all.py.
Only change: widen exit targets to grow gross edge.

v14: partial_tp=1x ATR | trail=0.8x ATR | hard_tp=6x ATR  -> gross P&L -$57
v15: partial_tp=2x ATR | trail=1.5x ATR | hard_tp=8x ATR  -> hypothesis: bigger avg winner
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from lzr_core import load_and_prepare
from backtest_v14 import CFG as CFG_V14
from backtest_v14_all import SLIP_BY_ASSET, DEFAULT_SLIP, run_portfolio_all

OUT      = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
DATA_DIR = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUT.mkdir(exist_ok=True)

CFG_V15 = {**CFG_V14,
    "partial_tp_mult": 2.0,
    "trail_dist_mult": 1.5,
    "hard_tp_mult"   : 8.0,
}

W = "=" * 90


def calc_stats(trades, equity_curve, init_bal, final_bal):
    """Compute CAGR, max DD, Calmar, PF from raw trades list + equity curve."""
    df = pd.DataFrame(trades)
    n  = len(df)
    if n == 0:
        return {}

    wins   = df[df["result"] == "WIN"]
    losses = df[df["result"] == "LOSS"]
    wr     = len(wins) / n * 100
    pf     = wins["net"].sum() / max(abs(losses["net"].sum()), 1e-9) if len(losses) else float("inf")

    # CAGR from equity curve
    eq   = np.array(equity_curve, dtype=float)
    ts0  = pd.to_datetime(df["ts"].min())
    ts1  = pd.to_datetime(df["close_ts"].max())
    yrs  = max((ts1 - ts0).days / 365.25, 0.1)
    cagr = (final_bal / init_bal) ** (1 / yrs) - 1

    # Max drawdown from equity curve
    eq_s  = pd.Series(eq)
    peak  = eq_s.cummax()
    dd    = ((eq_s - peak) / peak).min()

    calmar = abs(cagr / dd) if dd < 0 else float("inf")
    return dict(wr=wr, pf=pf, cagr=cagr, max_dd=dd, calmar=calmar, years=yrs)


def main():
    print(W)
    print("  LZR v15 -- WIDER TP/TRAIL  (30 symbols)")
    print("  partial_tp=2x ATR | trail_dist=1.5x ATR | hard_tp=8x ATR")
    print("  All look-ahead bugs fixed (same engine as v14)")
    print(W)

    symbols = sorted(p.stem.replace("1m", "") for p in DATA_DIR.glob("*1m.csv"))
    print(f"\n  {len(symbols)} symbols: {', '.join(symbols)}\n")

    print("  Loading data...")
    sym_data = load_and_prepare(symbols, CFG_V15)
    symbols  = [s for s in symbols if s in sym_data]
    print(f"  Loaded {len(symbols)} symbols.\n")

    print(f"  Running portfolio ({len(symbols)} symbols)...")
    print("  (May take 3-5 minutes)\n")

    trades, equity_curve, final_balance = run_portfolio_all(symbols, sym_data, CFG_V15)

    if not trades:
        print("  ERROR: No trades generated.")
        return

    df_t   = pd.DataFrame(trades)
    init_b = CFG_V15["initial_balance"]
    s      = calc_stats(trades, equity_curve, init_b, final_balance)

    n_trades = len(df_t)
    wr       = (df_t["net"] > 0).mean() * 100
    gross    = df_t["gross"].sum()
    fees     = df_t["fee"].sum()
    slip_tot = df_t["slip"].sum()
    fund     = df_t["funding"].sum()
    net_pnl  = df_t["net"].sum()
    cost_rt  = abs(fees + slip_tot + fund) / max(abs(gross), 1e-9) * 100

    # ── Overall results ────────────────────────────────────────────────────────
    print(W)
    print("  OVERALL RESULTS")
    print(W)
    print(f"\n  Trades:           {n_trades}")
    print(f"  Wins / Losses:    {(df_t['net']>0).sum()} W / {(df_t['net']<=0).sum()} L")
    print(f"  Win Rate:         {wr:.1f}%")
    print(f"  CAGR:             {s['cagr']*100:>+.1f}%")
    print(f"  Max Drawdown:     {s['max_dd']*100:>+.1f}%")
    print(f"  Calmar Ratio:     {s['calmar']:.2f}")
    print(f"  Profit Factor:    {s['pf']:.2f}x")
    print(f"  Final Balance:    ${final_balance:>10,.2f}  (start ${init_b:,.2f})")

    # ── Cost waterfall ─────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  COST WATERFALL")
    print(W)
    print(f"\n  Gross P&L:        ${gross:>+10,.2f}")
    print(f"  Fees (taker):     ${fees:>+10,.2f}")
    print(f"  Slippage:         ${slip_tot:>+10,.2f}")
    print(f"  Funding:          ${fund:>+10,.2f}")
    print(f"  Net P&L:          ${net_pnl:>+10,.2f}")
    print(f"  Final balance:    ${final_balance:>10,.2f}")
    print(f"\n  Total cost rate:  {cost_rt:.1f}% of gross P&L")

    # ── v14 vs v15 comparison ──────────────────────────────────────────────────
    print(f"\n{W}")
    print("  v14  vs  v15  COMPARISON")
    print(W)
    V14 = dict(tr=252, wr=66.3, cagr=-26.5, dd=-87.2, calmar=0.30,
               gross=-57.03, final=199.90, cost_rt=1303)
    V15 = dict(tr=n_trades, wr=wr, cagr=s['cagr']*100, dd=s['max_dd']*100,
               calmar=s['calmar'], gross=gross, final=final_balance, cost_rt=cost_rt)
    print(f"\n  {'Metric':<22} {'v14 (tight exits)':>18} {'v15 (wide exits)':>18} {'Delta':>10}")
    print("  " + "-" * 72)
    rows = [
        ("Trades",          f"{V14['tr']}", f"{V15['tr']}", f"{V15['tr']-V14['tr']:+d}"),
        ("Win Rate (%)",    f"{V14['wr']:.1f}", f"{V15['wr']:.1f}", f"{V15['wr']-V14['wr']:+.1f}"),
        ("CAGR (%)",        f"{V14['cagr']:.1f}", f"{V15['cagr']:.1f}", f"{V15['cagr']-V14['cagr']:+.1f}"),
        ("Max Drawdown (%)",f"{V14['dd']:.1f}", f"{V15['dd']:.1f}", f"{V15['dd']-V14['dd']:+.1f}"),
        ("Calmar",          f"{V14['calmar']:.2f}", f"{V15['calmar']:.2f}", f"{V15['calmar']-V14['calmar']:+.2f}"),
        ("Gross P&L ($)",   f"${V14['gross']:+,.0f}", f"${V15['gross']:+,.0f}", f"${V15['gross']-V14['gross']:+,.0f}"),
        ("Final Bal ($)",   f"${V14['final']:,.2f}", f"${V15['final']:,.2f}", f"${V15['final']-V14['final']:+,.2f}"),
        ("Cost Rate (%)",   f"{V14['cost_rt']:.0f}", f"{V15['cost_rt']:.0f}", f"{V15['cost_rt']-V14['cost_rt']:+.0f}"),
    ]
    for label, v14v, v15v, delta in rows:
        print(f"  {label:<22} {v14v:>18} {v15v:>18} {delta:>10}")

    # ── Year-by-year ───────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  YEAR-BY-YEAR")
    print(W)
    df_t["year"] = pd.to_datetime(df_t["close_ts"]).dt.year
    bal = init_b
    print(f"\n   {'Year':<6} {'Trades':>8} {'WR':>6} {'Return':>10} {'Net P&L':>12} {'Bal_end':>12}")
    print("  " + "-" * 62)
    profitable_yrs = 0
    for yr, g in df_t.groupby("year"):
        wr_yr  = (g["net"] > 0).mean() * 100
        net_yr = g["net"].sum()
        ret_yr = net_yr / bal * 100
        bal   += net_yr
        flag   = "" if net_yr > 0 else "  <-- LOSS YEAR"
        if net_yr > 0:
            profitable_yrs += 1
        print(f"   {yr:<6} {len(g):>8} {wr_yr:>5.0f}%  {ret_yr:>+9.1f}%  "
              f"${net_yr:>+10,.2f}  ${bal:>10,.2f}{flag}")
    print(f"\n  Profitable years: {profitable_yrs}/{df_t['year'].nunique()}")

    # ── Per-symbol breakdown ───────────────────────────────────────────────────
    print(f"\n{W}")
    print("  PER-SYMBOL BREAKDOWN  (sorted by Net P&L)")
    print(W)
    sym_stats = []
    for sym, g in df_t.groupby("symbol"):
        slip = SLIP_BY_ASSET.get(sym, DEFAULT_SLIP)
        tier = ("T1" if slip <= 0.0005 else "T2" if slip <= 0.0008
                else "T3" if slip <= 0.0010 else "T4")
        sym_stats.append(dict(
            sym=sym, tr=len(g), wr=(g["net"] > 0).mean() * 100,
            gross=g["gross"].sum(), fee=g["fee"].sum(),
            slip=g["slip"].sum(), fund=g["funding"].sum(),
            net=g["net"].sum(), per_tr=g["net"].sum() / len(g), tier=tier,
        ))
    sym_stats.sort(key=lambda x: -x["net"])

    print(f"\n  {'Symbol':<14} {'Tr':>4} {'WR':>7} {'Gross':>10} {'Fees':>8} "
          f"{'Slip':>8} {'Fund':>7} {'Net':>12} {'$/tr':>9}  Tier")
    print("  " + "-" * 95)
    for s2 in sym_stats:
        print(f"  {s2['sym']:<14} {s2['tr']:>4} {s2['wr']:>6.1f}%  ${s2['gross']:>+8,.2f}  "
              f"${s2['fee']:>+6,.2f}  ${s2['slip']:>+6,.2f}  ${s2['fund']:>+5,.2f}  "
              f"${s2['net']:>+10,.2f}  ${s2['per_tr']:>+7,.2f}  [{s2['tier']}]")
    print("  " + "-" * 95)
    print(f"  {'TOTAL':<14} {n_trades:>4} {wr:>6.1f}%  ${gross:>+8,.2f}  "
          f"${fees:>+6,.2f}  ${slip_tot:>+6,.2f}  ${fund:>+5,.2f}  "
          f"${net_pnl:>+10,.2f}")

    traded_syms = set(df_t["symbol"].unique())
    zero_syms   = [sym for sym in symbols if sym not in traded_syms]
    if zero_syms:
        print(f"\n  Symbols with 0 trades: {', '.join(zero_syms)}")

    # ── Save trade log ─────────────────────────────────────────────────────────
    df_t.to_csv(OUT / "trades_v15_all.csv", index=False)
    print(f"\n  Trade log: backtest_results/trades_v15_all.csv")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  VERDICT")
    print(W)
    calmar = s['calmar']
    if calmar >= 1.0:
        verdict = "INVESTABLE   -- Calmar >= 1.0"
    elif calmar >= 0.5:
        verdict = "MARGINAL     -- Calmar 0.5-1.0, close but needs one more improvement"
    else:
        verdict = "NOT YET      -- Calmar < 0.5, widening exits not enough alone"

    print(f"\n  {verdict}")
    print(f"  Gross P&L ${gross:>+,.0f} vs v14 -$57   ({'BETTER' if gross > -57 else 'WORSE'})")
    print(f"  CAGR      {s['cagr']*100:>+.1f}%  vs v14 -26.5%   ({'BETTER' if s['cagr'] > -0.265 else 'WORSE'})")
    print(f"  Calmar    {calmar:.2f}    vs v14  0.30   ({'BETTER' if calmar > 0.30 else 'WORSE'})")
    print()


if __name__ == "__main__":
    main()
