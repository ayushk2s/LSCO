"""
backtest_deep_analysis.py
=========================
Runs v11, v12_1, v12_2, v13 and prints:
  - Version comparison table
  - Year-by-year breakdown (per version)
  - Month-by-month P&L table (per version)
  - Asset-by-asset deep dive (per version)
  - Trade-level CSV saved to backtest_results/

Purpose: verify that ATOM+LTC gains are real, consistent, and not
from a handful of lucky months. Expose any gambling-like lumpiness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats)
from pathlib import Path

OUT = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT.mkdir(exist_ok=True)

# ── Configurations ─────────────────────────────────────────────────────────────

RUNS = [
    {
        "name"    : "v11",
        "label"   : "v11  BTC+ETH  7% risk",
        "symbols" : ["BTCUSDT", "ETHUSDT"],
        "cfg"     : {**DEFAULT_CFG, "risk_pct": 0.07, "hard_tp_mult": 6.0, "cd_win_bars": 3},
    },
    {
        "name"    : "v12_1",
        "label"   : "v12_1  BTC+ETH+ATOM+LTC  7% risk",
        "symbols" : ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"],
        "cfg"     : {**DEFAULT_CFG, "risk_pct": 0.07, "hard_tp_mult": 6.0, "cd_win_bars": 3},
    },
    {
        "name"    : "v12_2",
        "label"   : "v12_2  BTC+ETH+ATOM+LTC  15% risk",
        "symbols" : ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"],
        "cfg"     : {**DEFAULT_CFG, "risk_pct": 0.15, "hard_tp_mult": 6.0, "cd_win_bars": 3},
    },
    {
        "name"    : "v13",
        "label"   : "v13   BTC+ETH+ATOM+LTC  10% risk + spot + CB",
        "symbols" : ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"],
        "cfg"     : {**DEFAULT_CFG,
                     "risk_pct": 0.10, "hard_tp_mult": float("inf"),
                     "cd_win_bars": 1, "use_spot": True,
                     "spot_pct": 0.15, "spot_symbol": "BTCUSDT",
                     "max_dd_pct": 0.20},
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def W(n=90): return "=" * n
def S(n=90): return "-" * n
def section(title): print(f"\n{W()}\n  {title}\n{W()}")


def month_table(df_t, equity_curve, ref_idx, init_bal=1000.0):
    """Build month-by-month P&L table."""
    df_t = df_t.copy()
    df_t["ts"] = pd.to_datetime(df_t["ts"])
    df_t["ym"] = df_t["ts"].dt.to_period("M")

    all_months = pd.period_range(
        start=df_t["ts"].min().to_period("M"),
        end=df_t["ts"].max().to_period("M"),
        freq="M"
    )

    rows = []
    for ym in all_months:
        mt = df_t[df_t["ym"] == ym]
        trades = len(mt)
        wins   = mt["result"].eq("WIN").sum()
        net    = mt["net"].sum()
        wr     = (wins / trades * 100) if trades else 0.0
        rows.append(dict(month=str(ym), trades=trades, wins=int(wins),
                         wr=round(wr, 0), net=round(net, 2)))
    return pd.DataFrame(rows)


def asset_deep(df_t):
    """Per-asset stats."""
    rows = []
    for sym, g in df_t.groupby("symbol"):
        wins  = g[g["result"] == "WIN"]
        loses = g[g["result"] == "LOSS"]
        avg_w  = wins["net"].mean()   if len(wins)  else 0.0
        avg_l  = loses["net"].mean()  if len(loses) else 0.0
        rr     = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
        net    = g["net"].sum()
        trades = len(g)
        wr     = len(wins) / trades * 100 if trades else 0.0
        avg_dur = g["duration_h"].mean() if "duration_h" in g else 0.0
        rows.append(dict(symbol=sym, trades=trades, wins=len(wins),
                         losses=len(loses), win_rate=round(wr, 1),
                         avg_win=round(avg_w, 2), avg_loss=round(avg_l, 2),
                         rr=round(rr, 2), net=round(net, 2),
                         avg_dur_h=round(avg_dur, 1)))
    return pd.DataFrame(rows)


def year_asset_table(df_t):
    """Year x Asset matrix of net P&L."""
    df_t = df_t.copy()
    df_t["ts"] = pd.to_datetime(df_t["ts"])
    df_t["year"] = df_t["ts"].dt.year
    pivot = df_t.pivot_table(
        index="year", columns="symbol", values="net",
        aggfunc="sum", fill_value=0.0
    )
    return pivot


def print_month_table(mdf, version_name):
    print(f"\n  MONTH-BY-MONTH  [{version_name}]")
    print(f"  {'Month':<9}  {'Trades':>6}  {'Wins':>5}  {'WR':>5}  {'Net P&L':>10}  Bar")
    print("  " + S(65))
    cum = 0.0
    profitable_months = 0
    total_months_with_trades = 0
    for _, r in mdf.iterrows():
        cum += r["net"]
        bar_len = int(abs(r["net"]) / 5)  # scale: $5 per char
        bar_len = min(bar_len, 25)
        bar_chr = "+" if r["net"] >= 0 else "-"
        bar = bar_chr * bar_len
        flag = ""
        if r["net"] < 0 and r["trades"] > 0:
            flag = " <--"
        if r["trades"] > 0:
            total_months_with_trades += 1
            if r["net"] >= 0:
                profitable_months += 1
        print(f"  {r['month']:<9}  {r['trades']:>6}  {r['wins']:>5}  "
              f"{r['wr']:>4.0f}%  {r['net']:>+10.2f}  {bar}{flag}")
    pct = profitable_months / total_months_with_trades * 100 if total_months_with_trades else 0
    print(f"\n  Profitable months (with trades): {profitable_months}/{total_months_with_trades} "
          f"({pct:.0f}%)   Total net: ${cum:+.2f}")


def print_asset_deep(adf, version_name):
    print(f"\n  ASSET DEEP DIVE  [{version_name}]")
    print(f"  {'Symbol':<12}  {'Tr':>3}  {'W':>3}  {'L':>3}  "
          f"{'WR':>5}  {'AvgWin':>8}  {'AvgLoss':>9}  {'R:R':>5}  "
          f"{'NetP&L':>10}  {'AvgDur':>7}")
    print("  " + S(80))
    for _, r in adf.iterrows():
        print(f"  {r['symbol']:<12}  {r['trades']:>3}  {r['wins']:>3}  "
              f"{r['losses']:>3}  {r['win_rate']:>4.1f}%  "
              f"${r['avg_win']:>7.2f}  ${r['avg_loss']:>8.2f}  "
              f"{r['rr']:>5.2f}  ${r['net']:>9.2f}  {r['avg_dur_h']:>5.1f}h")


def print_year_asset(pivot, version_name):
    print(f"\n  YEAR x ASSET NET P&L  [{version_name}]")
    syms = list(pivot.columns)
    hdr = f"  {'Year':>5}  " + "  ".join(f"{s:<12}" for s in syms) + f"  {'Total':>9}"
    print(hdr)
    print("  " + S(max(len(hdr)-2, 60)))
    for yr, row in pivot.iterrows():
        total = row.sum()
        cols  = "  ".join(f"${row[s]:>+10.2f}" for s in syms)
        print(f"  {yr:>5}  {cols}  ${total:>+8.2f}")


def print_year_summary(s, version_name):
    print(f"\n  YEAR SUMMARY  [{version_name}]")
    print(f"  {'Year':>5}  {'Trades':>7}  {'WR':>5}  "
          f"{'CAGR_yr':>8}  {'MaxDD_yr':>9}  {'Bal_end':>10}")
    print("  " + S(65))
    bal = DEFAULT_CFG["initial_balance"]
    for r in s["yearly"]:
        sign = "+" if r["cagr"] >= 0 else ""
        flag = "" if r["cagr"] >= 0 else "  <--"
        bal_yr = bal * (1 + r["cagr"] / 100)
        print(f"  {r['year']:>5}  {r['trades']:>7}  {r['wr']:>4.0f}%  "
              f"  {sign}{r['cagr']:>5.1f}%  {r['max_dd']:>8.1f}%  "
              f"${bal_yr:>9,.2f}{flag}")
        bal = bal_yr


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(W())
    print("  BACKTEST DEEP ANALYSIS  v11 -> v13")
    print("  Year-wise | Month-wise | Asset-wise")
    print(W())

    results = {}

    # Load data once per unique symbol set
    loaded_cache = {}

    for run in RUNS:
        syms = run["symbols"]
        key  = tuple(syms)
        if key not in loaded_cache:
            print(f"\n  Loading data for {syms}...")
            loaded_cache[key] = load_and_prepare(syms, run["cfg"])
        sym_data = loaded_cache[key]

        print(f"\n  Simulating {run['name']}...")
        trades, equity, final_bal = run_portfolio(syms, sym_data, run["cfg"])

        ref_idx = sym_data[syms[0]]["df_sig"].index
        s = compute_stats(trades, equity, final_bal, ref_idx)
        df_t = pd.DataFrame(trades)

        results[run["name"]] = dict(run=run, s=s, df_t=df_t, equity=equity,
                                     final_bal=final_bal, ref_idx=ref_idx)

        # Save CSV
        df_t.to_csv(OUT / f"trades_{run['name']}.csv", index=False)
        print(f"    -> {len(trades)} trades  CAGR {s['cagr']:+.1f}%  DD {s['max_dd']:.1f}%  "
              f"Calmar {s['calmar']:.2f}")

    # ── Overall comparison ─────────────────────────────────────────────────────
    section("OVERALL COMPARISON (all versions)")
    print(f"\n  {'Version':<35}  {'Trades':>7}  {'WR':>5}  {'CAGR':>7}  "
          f"{'MaxDD':>7}  {'Calmar':>7}  {'Final_Bal':>12}")
    print("  " + S(85))
    for run in RUNS:
        s   = results[run["name"]]["s"]
        tag = "**" if s["calmar"] >= 2.0 else ("* " if s["calmar"] >= 1.5 else "  ")
        print(f"  {run['label']:<35}  {s['trades']:>7}  {s['win_rate']:>4.1f}%  "
              f"  {s['cagr']:>+5.1f}%  {s['max_dd']:>6.1f}%  "
              f"{s['calmar']:>7.2f} {tag}  ${s['final_bal']:>10,.2f}")

    # ── Per-version deep dive ──────────────────────────────────────────────────
    for run in RUNS:
        name  = run["name"]
        label = run["label"]
        s     = results[name]["s"]
        df_t  = results[name]["df_t"]
        eq    = results[name]["equity"]
        ref   = results[name]["ref_idx"]
        init  = run["cfg"]["initial_balance"]

        section(f"DEEP DIVE: {label}")

        # Year summary
        print_year_summary(s, name)

        # Month-by-month
        mdf = month_table(df_t, eq, ref, init)
        print_month_table(mdf, name)
        mdf.to_csv(OUT / f"monthly_{name}.csv", index=False)

        # Asset deep
        adf = asset_deep(df_t)
        print_asset_deep(adf, name)
        adf.to_csv(OUT / f"assets_{name}.csv", index=False)

        # Year x Asset
        try:
            ypivot = year_asset_table(df_t)
            print_year_asset(ypivot, name)
        except Exception as e:
            print(f"  (year x asset table error: {e})")

    # ── ATOM and LTC reliability check ────────────────────────────────────────
    section("ATOM + LTC RELIABILITY CHECK")
    for version_name in ["v12_1", "v12_2", "v13"]:
        df_t = results[version_name]["df_t"]
        df_t = df_t.copy()
        df_t["ts_dt"] = pd.to_datetime(df_t["ts"])
        df_t["year"]  = df_t["ts_dt"].dt.year

        print(f"\n  [{version_name}]  ATOM year-by-year:")
        atom = df_t[df_t["symbol"] == "ATOMUSDT"]
        if len(atom):
            for yr, g in atom.groupby("year"):
                wr = g["result"].eq("WIN").mean() * 100
                net = g["net"].sum()
                print(f"    {yr}: {len(g):>2}tr  WR {wr:>5.0f}%  net ${net:>+8.2f}")
        else:
            print("    (no ATOM trades)")

        print(f"\n  [{version_name}]  LTC year-by-year:")
        ltc = df_t[df_t["symbol"] == "LTCUSDT"]
        if len(ltc):
            for yr, g in ltc.groupby("year"):
                wr = g["result"].eq("WIN").mean() * 100
                net = g["net"].sum()
                print(f"    {yr}: {len(g):>2}tr  WR {wr:>5.0f}%  net ${net:>+8.2f}")
        else:
            print("    (no LTC trades)")

    # ── What if ATOM failed entirely? ──────────────────────────────────────────
    section("STRESS TEST: What if ATOM or LTC stopped working?")
    for version_name in ["v12_1", "v13"]:
        df_t = results[version_name]["df_t"]
        cfg  = results[version_name]["run"]["cfg"]
        s_all = results[version_name]["s"]

        for exclude_sym in ["ATOMUSDT", "LTCUSDT", "ATOMUSDT+LTCUSDT"]:
            if "+" in exclude_sym:
                excl = exclude_sym.split("+")
            else:
                excl = [exclude_sym]
            df_no = df_t[~df_t["symbol"].isin(excl)]
            if len(df_no) == 0:
                continue

            # Recompute net running balance
            df_no = df_no.copy().sort_values("ts").reset_index(drop=True)
            bal   = cfg["initial_balance"]
            peak  = bal
            mdd   = 0.0
            eq_no = [bal]
            for _, row in df_no.iterrows():
                bal  += row["net"]
                peak  = max(peak, bal)
                dd    = (bal / peak - 1) * 100
                mdd   = min(mdd, dd)
                eq_no.append(bal)

            # CAGR
            ts_start = pd.to_datetime(df_no["ts"].iloc[0])
            ts_end   = pd.to_datetime(df_no["ts"].iloc[-1])
            yrs = (ts_end - ts_start).days / 365.25
            cagr = ((bal / cfg["initial_balance"]) ** (1/yrs) - 1) * 100 if yrs > 0 else 0
            wr   = df_no["result"].eq("WIN").mean() * 100

            print(f"\n  [{version_name}] Remove {exclude_sym}:")
            print(f"    Trades: {len(df_no)}  WR: {wr:.1f}%  "
                  f"CAGR: {cagr:+.1f}%  DD: {mdd:.1f}%   "
                  f"(was: CAGR {s_all['cagr']:+.1f}%  DD {s_all['max_dd']:.1f}%)")

    print(f"\n{W()}")
    print("  Analysis complete. CSVs saved to backtest_results/")
    print(W())
