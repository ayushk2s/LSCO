"""
backtest_v12_2.py
=================
v12_2: RISK 7% → 15% (same 4 symbols)
Change vs v12_1: risk_pct 0.07 → 0.15

Hypothesis: At 93% WR and 3:1 R:R, quarter-Kelly ≈ 20%.
Going to 15% nearly doubles expected P&L without blowing drawdown
beyond investor comfort zone.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats, print_version_header, print_summary)

VERSION     = "v12_2"
DESCRIPTION = "BTC+ETH+ATOM+LTC  15% risk per trade"
CHANGED     = "RISK 7% -> 15%  (everything else same as v12_1)"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"]

CFG = {**DEFAULT_CFG,
       "risk_pct"     : 0.15,
       "hard_tp_mult" : 6.0,
       "cd_win_bars"  : 3,
       }

if __name__ == "__main__":
    print_version_header(VERSION, DESCRIPTION, CHANGED)
    print("  Loading data...")
    sym_data = load_and_prepare(SYMBOLS, CFG)

    print("\n  Running portfolio simulation...")
    trades, equity, final_bal = run_portfolio(SYMBOLS, sym_data, CFG)

    ref_idx = sym_data[SYMBOLS[0]]["df_sig"].index
    s = compute_stats(trades, equity, final_bal, ref_idx)

    print()
    print_summary(s, VERSION, CFG)

    print()
    print("  BASELINE (v11 BTC+ETH 7%):  CAGR +13.8%  DD -7.2%  Calmar 1.91")
    print("  v12_1 (BTC+ETH+ATOM+LTC 7%): see v12_1 output")
    print()
    if s:
        delta_cagr = s["cagr"] - 13.8
        sign = "+" if delta_cagr >= 0 else ""
        print(f"  Delta CAGR vs v11:  {sign}{delta_cagr:.1f}%")
