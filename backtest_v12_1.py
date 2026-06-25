"""
backtest_v12_1.py
=================
v12_1: ADD ATOM + LTC to portfolio (4 symbols: BTC+ETH+ATOM+LTC)
Change vs v11: SYMBOLS = [BTC, ETH, ATOM, LTC]
All other parameters identical to v11 baseline.

Hypothesis: ATOM (100% WR, zero DD) and LTC (Calmar 1.07) add trade
frequency without adding drawdown, pushing CAGR higher.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats, print_version_header, print_summary)

VERSION     = "v12_1"
DESCRIPTION = "BTC+ETH+ATOM+LTC  (4 symbols, same 7% risk)"
CHANGED     = "+ATOM+LTC added; RISK=7%, HARD=6.0x ATR, CD_WIN=3"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"]

CFG = {**DEFAULT_CFG,
       "risk_pct"     : 0.07,
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
    print("  BASELINE (v11 BTC+ETH):  26tr  WR 92.3%  CAGR +13.8%  DD -7.2%  Calmar 1.91")
    print()
    if s:
        delta_cagr = s["cagr"] - 13.8
        sign = "+" if delta_cagr >= 0 else ""
        print(f"  Delta CAGR vs v11:  {sign}{delta_cagr:.1f}%")
