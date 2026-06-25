"""
backtest_v12_5.py
=================
v12_5: 20% SPOT HOLDING during bull regime + slight risk pullback
Change vs v12_4: use_spot=True, spot_pct=0.20, risk_pct=0.12

Hypothesis: During confirmed bull regime (weekly + daily EMA), hold 20%
of balance in BTC spot. This earns passive alpha while waiting for
LZR signals. Risk pulled from 15% to 12% to account for the capital
already working in spot (net effective risk ≈ 15%).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lzr_core import (DEFAULT_CFG, load_and_prepare, run_portfolio,
                       compute_stats, print_version_header, print_summary)

VERSION     = "v12_5"
DESCRIPTION = "BTC+ETH+ATOM+LTC  12% risk  pure trail  CD_WIN=1  +20% BTC spot"
CHANGED     = "RISK 15%->12% + USE_SPOT=True SPOT_PCT=0.20 (20% balance in BTC when bull)"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "LTCUSDT"]

CFG = {**DEFAULT_CFG,
       "risk_pct"     : 0.12,
       "hard_tp_mult" : float("inf"),
       "cd_win_bars"  : 1,
       "use_spot"     : True,
       "spot_pct"     : 0.20,
       "spot_symbol"  : "BTCUSDT",
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
