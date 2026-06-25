"""
backtest_v12_4.py
=================
v12_4: WIN COOLDOWN 3 → 1 bar (6h → 4h gap after wins)
Change vs v12_3: cd_win_bars = 1

Hypothesis: Winning trades confirm the zone works. Re-entering sooner
after wins (if a new signal fires) captures the momentum phase.
CD_LOSS remains 42 bars (7 days) — that is protective and stays.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats, print_version_header, print_summary)

VERSION     = "v12_4"
DESCRIPTION = "BTC+ETH+ATOM+LTC  15% risk  pure trail  CD_WIN=1"
CHANGED     = "CD_WIN_BARS 3 -> 1  (re-enter after win 1 bar sooner)"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"]

CFG = {**DEFAULT_CFG,
       "risk_pct"     : 0.15,
       "hard_tp_mult" : float("inf"),
       "cd_win_bars"  : 1,
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
