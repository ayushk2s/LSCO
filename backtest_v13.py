"""
backtest_v13.py  --  FINAL VERSION
====================================
Combines the best improvements from v12_1 through v12_5:

  v12_1: +ATOM+LTC         -> +31.4% CAGR  DD -7.2%   Calmar 4.34  (ALREADY 30%+)
  v12_2: Risk 7% -> 15%    -> +74.5% CAGR  DD -15.5%  Calmar 4.81
  v12_3: Pure trail        -> no change (trailing exits before hard cap)
  v12_4: CD_WIN 3 -> 1     -> +83.8% CAGR  DD -15.5%  Calmar 5.41  ** BEST
  v12_5: 12% + 20% spot    -> +72.5% CAGR  DD -12.3%  Calmar 5.88  ** BEST CALMAR

v13 design: balanced for investors who want 30%+ CAGR with minimal DD
  - 4 symbols: BTC + ETH + ATOM + LTC
  - RISK = 10%  (between 7% safe and 15% aggressive)
  - CD_WIN = 1  (proven: +3 trades, same WR)
  - HARD_TP = inf  (pure trail, proven neutral — keeps doors open)
  - SPOT = 15% BTC when bull regime  (reduces DD, adds passive alpha)
  - Circuit breaker at DD = 20%  (safety net for investors)

Expected outcome: ~55-65% CAGR, DD < -13%, Calmar > 4.5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats, print_version_header, print_summary)
import pandas as pd

VERSION     = "v13 FINAL"
DESCRIPTION = "BTC+ETH+ATOM+LTC  10% risk  pure trail  CD_WIN=1  +15% BTC spot  CB20%"
CHANGED     = "Best of v12_1..v12_5: 4 symbols, balanced risk, spot holding, circuit breaker"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"]

CFG = {**DEFAULT_CFG,
       "risk_pct"     : 0.10,          # balanced (7%=safe, 15%=aggressive)
       "hard_tp_mult" : float("inf"),  # pure trail
       "cd_win_bars"  : 1,             # proven: +3 trades at same WR
       "use_spot"     : True,          # passive alpha during bull
       "spot_pct"     : 0.15,          # 15% (less than 20% to preserve trading capital)
       "spot_symbol"  : "BTCUSDT",
       "max_dd_pct"   : 0.20,          # pause trading if DD reaches -20%
       }

VERSIONS_TABLE = [
    ("v11 baseline",  "BTC+ETH 7% risk",        "+13.8%", "-7.2%",  "1.91", "2/5 good",  7),
    ("v12_1",         "+ATOM+LTC 7% risk",       "+31.4%", "-7.2%",  "4.34", "5/5 **",   55),
    ("v12_2",         "+15% risk",               "+74.5%", "-15.5%", "4.81", "5/5 **",   55),
    ("v12_3",         "pure trail",              "+74.5%", "-15.5%", "4.81", "5/5 **",   55),
    ("v12_4",         "CD_WIN=1",                "+83.8%", "-15.5%", "5.41", "5/5 **",   58),
    ("v12_5",         "12% + 20% spot",          "+72.5%", "-12.3%", "5.88", "5/5 **",   58),
]


def print_comparison_table(s13, cfg):
    print()
    print("=" * 90)
    print("  VERSION COMPARISON TABLE")
    print("=" * 90)
    hdr = f"  {'Version':<14}  {'Description':<28}  {'CAGR':>7}  {'DD':>7}  {'Calmar':>6}  {'Trades':>6}"
    print(hdr)
    print("  " + "-" * 86)
    for v, d, cagr, dd, calmar, years, tr in VERSIONS_TABLE:
        flag = " <-- prev best" if calmar == "5.88" else ""
        print(f"  {v:<14}  {d:<28}  {cagr:>7}  {dd:>7}  {calmar:>6}  {tr:>6}{flag}")
    if s13:
        sign  = "+" if s13["cagr"] >= 0 else ""
        mark  = " <-- FINAL **"
        print(f"  {'v13 FINAL':<14}  {'10% + 15% spot + CB20%':<28}  "
              f"  {sign}{s13['cagr']:.1f}%  {s13['max_dd']:.1f}%  {s13['calmar']:.2f}  {s13['trades']:>6}{mark}")
    print()


if __name__ == "__main__":
    print_version_header(VERSION, DESCRIPTION, CHANGED)
    print()
    print("  STRATEGY DESIGN:")
    print("  - 4 symbols: BTC + ETH + ATOM + LTC  (proven performers)")
    print("  - 10% risk/trade  (balanced: 7% CAGR doubles, 15% triples)")
    print("  - Pure trail  (no hard cap  ->  monsters run unlimited)")
    print("  - CD_WIN=1   (re-enter 1 bar after win  ->  +3 trades)")
    print("  - 15% BTC spot during bull regime  (passive alpha + volatility buffer)")
    print("  - Circuit breaker at -20% DD  (pause, resume on 30% recovery)")
    print()

    print("  Loading data...")
    sym_data = load_and_prepare(SYMBOLS, CFG)

    print("\n  Running portfolio simulation...")
    trades, equity, final_bal = run_portfolio(SYMBOLS, sym_data, CFG)

    ref_idx = sym_data[SYMBOLS[0]]["df_sig"].index
    s = compute_stats(trades, equity, final_bal, ref_idx)

    print()
    print_summary(s, VERSION, CFG)

    if s:
        print()
        print("  INVESTOR METRICS:")
        print(f"    Starting capital:     $1,000")
        print(f"    Final balance:        ${s['final_bal']:>10,.2f}")
        mult = s["final_bal"] / 1_000.0
        print(f"    Total return:         {mult:.1f}x")
        print(f"    CAGR:                 {'+' if s['cagr']>0 else ''}{s['cagr']:.1f}%")
        print(f"    Max Drawdown:         {s['max_dd']:.1f}%")
        print(f"    Calmar Ratio:         {s['calmar']:.2f}")
        print(f"    Win Rate:             {s['win_rate']:.1f}%")
        n_pos = sum(1 for r in s["yearly"] if r["cagr"] >= 0)
        print(f"    Profitable years:     {n_pos}/{len(s['yearly'])}")
        print(f"    Trades/year avg:      {s['trades'] / max(len(s['yearly']),1):.1f}")
        print()
        if s["cagr"] >= 30.0 and n_pos == len(s["yearly"]):
            print("  [PASS]  30%+ CAGR target MET  and  0 losing years")
        elif s["cagr"] >= 30.0:
            print(f"  [PARTIAL]  30%+ CAGR target MET but {len(s['yearly'])-n_pos} losing year(s)")
        else:
            print(f"  [INFO]  CAGR {s['cagr']:.1f}% -- below 30% target (see comparison table)")

    print_comparison_table(s, CFG)
