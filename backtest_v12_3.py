"""
backtest_v12_3.py
=================
v12_3: PURE TRAIL — remove hard TP cap
Change vs v12_2: hard_tp_mult = float("inf")

Hypothesis: Bull runs (BTC 2021, 2023-2024) can run 10-20x ATR.
Hard cap at 6x ATR cuts these prematurely. Pure trail lets winners
run to their natural end, improving avg win size.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats, print_version_header, print_summary)

VERSION     = "v12_3"
DESCRIPTION = "BTC+ETH+ATOM+LTC  15% risk  PURE TRAIL (no hard TP)"
CHANGED     = "HARD_TP_MULT 6.0 -> inf  (pure trailing stop, no cap)"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"]

CFG = {**DEFAULT_CFG,
       "risk_pct"     : 0.15,
       "hard_tp_mult" : float("inf"),
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
    print()
    if s:
        delta_cagr = s["cagr"] - 13.8
        sign = "+" if delta_cagr >= 0 else ""
        print(f"  Delta CAGR vs v11:  {sign}{delta_cagr:.1f}%")
