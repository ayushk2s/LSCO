"""
backtest_v5.py  --  Liquidation Zone Reversal  v5  (Compounding Edition)
=========================================================================
Exact copy of backtest_v3 with ONE change:
  FIXED_RISK = $10 per trade  →  RISK_PCT = 10% of current balance

Everything else is identical to v3:
  - Volume spike filter
  - 4h EMA20 trend filter
  - Partial exit at 1xATR + trailing stop + hard cap 3xATR
  - Zone freshness limit (max 2 touches per 168 bars)
  - Same cooldowns, same fee/slippage model
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR        = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR      = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_BALANCE = 1000.0
RISK_PCT        = 0.10          # 10% of current balance at risk per trade (compounding)

SL_MULT         = 0.75
PARTIAL_TP_MULT = 1.0
TRAIL_DIST_MULT = 0.5
HARD_TP_MULT    = 3.0

APPROACH_PCT    = 0.008
TOUCH_BUF       = 0.003
ATR_PERIOD      = 14
SWING_LOOKBACK  = 20
MIN_ZONE_GAP    = 0.005

VOL_MULT        = 1.8
VOL_LOOKBACK    = 20
EMA4H_PERIOD    = 20
EMA4H_LOOKBACK  = 3
ZONE_MAX_TOUCH  = 2
ZONE_WINDOW     = 168

COOLDOWN_LOSS   = 10
COOLDOWN_WIN    = 3

# ── Fees & Slippage ───────────────────────────────────────────────────────────
CRYPTO_FEE_RT   = 0.0004        # 0.04% round-trip exchange fee
CRYPTO_SLIP_PCT = 0.0003        # 0.03% entry slippage (DEX market impact)

FOREX_SPREAD    = {
    "EURUSD": 0.00010, "GBPUSD": 0.00010, "USDJPY": 0.00010,
    "AUDUSD": 0.00015, "USDCAD": 0.00020, "EURJPY": 0.00015, "GBPJPY": 0.00025,
}
FOREX_SLIP_PCT  = 0.0001        # 0.01% entry slippage (forex ECN)

CRYPTO_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FOREX_PAIRS     = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_crypto_1h(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date":"ts","Open":"open","High":"high",
                             "Low":"low","Close":"close","Volume":"vol"})
    df = df.set_index("ts").sort_index()
    return df.resample("1h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), vol=("vol","sum")
    ).dropna()


def download_forex_1h(ticker, name):
    import yfinance as yf
    print(f"    Downloading {name} ({ticker})...")
    raw = yf.download(ticker, period="2y", interval="1h",
                      auto_adjust=True, progress=False, actions=False)
    if raw.empty:
        raw = yf.Ticker(ticker).history(period="2y", interval="1h", auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns={"Open":"open","High":"high",
                              "Low":"low","Close":"close"})
    df.index.name = "ts"
    df = df[["open","high","low","close"]].copy()
    df["vol"] = 0.0
    return df.sort_index().dropna()


def resample_4h(df):
    return df.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last")
    ).dropna()


def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_4h):
    n, lb = len(df_4h), SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_4h["high"].iloc[i] == df_4h["high"].iloc[i-lb:i+lb+1].max():
            highs.append(df_4h["high"].iloc[i])
        if df_4h["low"].iloc[i] == df_4h["low"].iloc[i-lb:i+lb+1].min():
            lows.append(df_4h["low"].iloc[i])
    levels, merged = sorted(set(highs + lows)), []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(symbol, df_1h, fee_rt, slip_pct, category, use_vol_filter):
    FEE_SIDE = fee_rt / 2

    df_4h  = resample_4h(df_1h)
    atr_s  = calc_atr(df_1h)

    df_4h["ema"]      = df_4h["close"].ewm(span=EMA4H_PERIOD, adjust=False).mean()
    df_4h["ema_lag"]  = df_4h["ema"].shift(1)
    df_4h["ema_prev"] = df_4h["ema"].shift(1 + EMA4H_LOOKBACK)
    ema_now_1h  = df_4h["ema_lag"].reindex(df_1h.index,  method="ffill")
    ema_prev_1h = df_4h["ema_prev"].reindex(df_1h.index, method="ffill")

    n_days = len(df_1h) // 24 + 2
    zone_snap = []
    for d in range(n_days):
        end_ts  = df_1h.index[min(d * 24, len(df_1h)-1)]
        past_4h = df_4h[df_4h.index <= end_ts]
        if len(past_4h) < SWING_LOOKBACK * 2 + 5:
            zone_snap.append([])
        else:
            zone_snap.append(find_zones(past_4h.iloc[-200:]))

    balance          = INITIAL_BALANCE
    equity           = [balance]
    trades           = []
    zone_cooldown    = {}
    zone_touches     = defaultdict(list)
    last_trigger_bar = -999

    in_trade = False; trade_state = None; direction = ""
    entry_px = sl_px = qty = active_zone = trade_atr = 0.0
    partial_tp_px = hard_tp_px = trail_sl = running_ext = partial_locked = 0.0
    fee_accrued = slip_trade = 0.0
    balance_at_open = 0.0          # balance when this trade was opened (for % return)

    for i, (ts, row) in enumerate(df_1h.iterrows()):
        if i < 50:
            equity.append(balance); continue

        atr   = atr_s.iloc[i]
        price = row["close"]
        if atr <= 0 or np.isnan(atr):
            equity.append(balance); continue

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            gross = None; closed = False; exit_px = 0.0

            if trade_state == "full":
                if direction == "LONG":
                    if row["low"] <= sl_px:
                        gross = (sl_px - entry_px) * qty
                        exit_px = sl_px; closed = True
                    elif row["high"] >= partial_tp_px:
                        half = qty / 2
                        partial_locked = (partial_tp_px - entry_px) * half
                        fee_accrued   += partial_tp_px * half * FEE_SIDE
                        running_ext    = row["high"]
                        trail_sl       = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                        trade_state    = "partial"
                else:
                    if row["high"] >= sl_px:
                        gross = (entry_px - sl_px) * qty    # SHORT PnL: entry - exit
                        exit_px = sl_px; closed = True
                    elif row["low"] <= partial_tp_px:
                        half = qty / 2
                        partial_locked = (entry_px - partial_tp_px) * half
                        fee_accrued   += partial_tp_px * half * FEE_SIDE
                        running_ext    = row["low"]
                        trail_sl       = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                        trade_state    = "partial"

            elif trade_state == "partial":
                half = qty / 2
                if direction == "LONG":
                    old_trail   = trail_sl
                    running_ext = max(running_ext, row["high"])
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    if row["high"] >= hard_tp_px:
                        gross = partial_locked + (hard_tp_px - entry_px) * half
                        exit_px = hard_tp_px; closed = True
                    elif row["low"] <= old_trail:
                        gross = partial_locked + (old_trail - entry_px) * half
                        exit_px = old_trail; closed = True
                else:
                    old_trail   = trail_sl
                    running_ext = min(running_ext, row["low"])
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    if row["low"] <= hard_tp_px:
                        gross = partial_locked + (entry_px - hard_tp_px) * half
                        exit_px = hard_tp_px; closed = True
                    elif row["high"] >= old_trail:
                        gross = partial_locked + (entry_px - old_trail) * half
                        exit_px = old_trail; closed = True

            if closed and gross is not None:
                exit_fee_qty = qty if trade_state == "full" else qty / 2
                total_fee    = fee_accrued + exit_px * exit_fee_qty * FEE_SIDE
                net          = gross - total_fee - slip_trade
                result       = "WIN" if gross > 0 else "LOSS"
                balance     += net
                in_trade = False; trade_state = None
                zone_cooldown[active_zone] = i + (COOLDOWN_LOSS if result=="LOSS"
                                                   else COOLDOWN_WIN)
                trades.append({
                    "ts":           ts,
                    "dir":          direction,
                    "entry":        round(entry_px,       6),
                    "exit":         round(exit_px,        6),
                    "qty":          round(qty,             8),
                    "notional":     round(entry_px * qty,  2),
                    "gross":        round(gross,            4),
                    "fee":          round(total_fee,        4),
                    "slip":         round(slip_trade,       4),
                    "net":          round(net,              4),
                    "result":       result,
                    "balance_open": round(balance_at_open,  4),  # balance when opened
                    "balance":      round(balance,          4),  # balance after close
                })
            equity.append(balance); continue

        # ── Zone lookup ───────────────────────────────────────────────────────
        day_idx = i // 24
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            equity.append(balance); continue
        zones = zone_snap[day_idx]

        zones_below = [z for z in zones if z < price and (price-z)/z <= APPROACH_PCT]
        zones_above = [z for z in zones if z > price and (z-price)/z <= APPROACH_PCT]
        if zones_below:
            near_zone = max(zones_below); near_dir = "LONG"
        elif zones_above:
            near_zone = min(zones_above); near_dir = "SHORT"
        else:
            equity.append(balance); continue

        if zone_cooldown.get(near_zone, 0) > i:
            equity.append(balance); continue
        if i == last_trigger_bar:
            equity.append(balance); continue

        if near_dir == "LONG":
            triggered = (row["low"] <= near_zone * (1 + TOUCH_BUF)
                         and row["close"] > near_zone)
        else:
            triggered = (row["high"] >= near_zone * (1 - TOUCH_BUF)
                         and row["close"] < near_zone)
        if not triggered:
            equity.append(balance); continue

        # Filter 1: Volume spike
        if use_vol_filter:
            vol_avg = df_1h["vol"].iloc[max(0, i-VOL_LOOKBACK):i].mean()
            if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
                equity.append(balance); continue

        # Filter 2: 4h EMA trend
        ema_now  = ema_now_1h.iloc[i]
        ema_prev = ema_prev_1h.iloc[i]
        if not (pd.isna(ema_now) or pd.isna(ema_prev)):
            if near_dir == "LONG"  and ema_now <= ema_prev:
                equity.append(balance); continue
            if near_dir == "SHORT" and ema_now >= ema_prev:
                equity.append(balance); continue

        # Filter 4: Zone freshness
        recent = [b for b in zone_touches[near_zone] if i - b <= ZONE_WINDOW]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance); continue

        # ── Enter trade ───────────────────────────────────────────────────────
        last_trigger_bar = i
        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance); continue

        # CHANGED from v3: risk = 10% of current balance (compounding)
        risk_usdt      = balance * RISK_PCT
        qty            = risk_usdt / sl_dist
        entry_px       = df_1h["open"].iloc[i+1] if i+1 < len(df_1h) else price
        trade_atr      = atr
        slip_trade     = entry_px * qty * slip_pct
        balance_at_open = balance                  # record balance at entry for stats

        if near_dir == "LONG":
            sl_px         = entry_px - SL_MULT        * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT        * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        in_trade = True; trade_state = "full"; direction = near_dir
        active_zone = near_zone; partial_locked = 0.0
        fee_accrued = entry_px * qty * FEE_SIDE
        running_ext = entry_px; trail_sl = sl_px

        zone_touches[near_zone].append(i)
        equity.append(balance)

    # ─── Statistics ───────────────────────────────────────────────────────────
    if not trades:
        return _empty(symbol, category, equity, df_1h)

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"]=="WIN"]
    loses = df_t[df_t["result"]=="LOSS"]

    gross_pnl    = df_t["gross"].sum()
    total_fees   = df_t["fee"].sum()
    total_slip   = df_t["slip"].sum()
    total_costs  = total_fees + total_slip
    net_pnl      = df_t["net"].sum()
    gross_wins   = wins["gross"].sum()
    gross_loss   = abs(loses["gross"].sum())
    pf           = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    win_rate     = round(len(wins) / len(df_t) * 100, 1)

    final_bal    = balance
    years        = (df_1h.index[-1] - df_1h.index[0]).days / 365.25
    total_ret    = (final_bal / INITIAL_BALANCE - 1) * 100
    cagr         = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0

    eq       = pd.Series(equity)
    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    max_dd   = round(dd_pct.min(), 2)
    max_dd_usd = round((eq - roll_max).min(), 2)

    # Max DD duration (consecutive bars in drawdown)
    in_dd      = (dd_pct < 0).astype(int)
    dd_dur_max = 0; dd_cur = 0
    for v in in_dd:
        dd_cur = dd_cur + 1 if v else 0
        dd_dur_max = max(dd_dur_max, dd_cur)

    # Sharpe: per-trade % return on balance at entry (variable risk, use % not $)
    pct_rets    = (df_t["net"] / df_t["balance_open"]).values
    ann_factor  = (len(df_t) / years) if years > 0 else len(df_t)
    sharpe      = round((pct_rets.mean() / pct_rets.std() * np.sqrt(ann_factor))
                        if pct_rets.std() > 0 else 0, 2)

    calmar = round(cagr / abs(max_dd) if max_dd != 0 else 0, 2)

    # Win / loss streaks
    max_ws = max_ls = cur_ws = cur_ls = 0
    for r in df_t["result"]:
        if r == "WIN":  cur_ws += 1; cur_ls = 0
        else:           cur_ls += 1; cur_ws = 0
        max_ws = max(max_ws, cur_ws); max_ls = max(max_ls, cur_ls)

    return {
        "symbol":       symbol,
        "category":     category,
        "date_from":    df_1h.index[0].date(),
        "date_to":      df_1h.index[-1].date(),
        "years":        round(years, 2),
        "trades":       len(df_t),
        "wins":         len(wins),
        "losses":       len(loses),
        "win_rate":     win_rate,
        "pf":           pf,
        "gross_pnl":    round(gross_pnl,    2),
        "total_fees":   round(total_fees,   2),
        "total_slip":   round(total_slip,   2),
        "total_costs":  round(total_costs,  2),
        "net_pnl":      round(net_pnl,      2),
        "initial_bal":  INITIAL_BALANCE,
        "final_bal":    round(final_bal,    2),
        "total_ret":    round(total_ret,    2),
        "cagr":         round(cagr,         2),
        "max_dd":       max_dd,
        "max_dd_usd":   max_dd_usd,
        "dd_dur_bars":  dd_dur_max,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "avg_win":      round(wins["net"].mean(),      4) if len(wins)  else 0,
        "avg_loss":     round(loses["net"].mean(),     4) if len(loses) else 0,
        "best_trade":   round(df_t["net"].max(),       4),
        "worst_trade":  round(df_t["net"].min(),       4),
        "avg_fee_pt":   round(total_fees / len(df_t),  4),
        "avg_slip_pt":  round(total_slip / len(df_t),  4),
        "avg_cost_pt":  round(total_costs/ len(df_t),  4),
        "avg_notional": round(df_t["notional"].mean(), 2),
        "max_win_str":  max_ws,
        "max_los_str":  max_ls,
        "equity":       equity,
        "trades_df":    df_t,
    }


def _empty(symbol, category, equity, df_1h):
    years = (df_1h.index[-1] - df_1h.index[0]).days / 365.25 if len(df_1h) > 1 else 0
    return {
        "symbol": symbol, "category": category, "trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "pf": 0, "gross_pnl": 0, "total_fees": 0, "total_slip": 0,
        "total_costs": 0, "net_pnl": 0, "initial_bal": INITIAL_BALANCE,
        "final_bal": INITIAL_BALANCE, "total_ret": 0, "cagr": 0,
        "max_dd": 0, "max_dd_usd": 0, "dd_dur_bars": 0,
        "sharpe": 0, "calmar": 0, "avg_win": 0, "avg_loss": 0,
        "best_trade": 0, "worst_trade": 0, "avg_fee_pt": 0,
        "avg_slip_pt": 0, "avg_cost_pt": 0, "avg_notional": 0,
        "max_win_str": 0, "max_los_str": 0, "years": round(years, 2),
        "date_from": df_1h.index[0].date() if len(df_1h) else "",
        "date_to":   df_1h.index[-1].date() if len(df_1h) else "",
        "equity": equity, "trades_df": pd.DataFrame(),
    }


# ─── Report printer ───────────────────────────────────────────────────────────

W = 72

def div(char="="):  print(char * W)
def hdiv():         print("-" * W)
def row2(a, av, b, bv, w=28):
    print(f"  {a:<{w}} {av!s:<18}  {b:<{w}} {bv!s}")
def row1(a, av, w=28):
    print(f"  {a:<{w}} {av}")


def print_asset_block(r, fee_label, spread_label):
    yr  = r["years"]
    tpy = r["trades"] / yr if yr > 0 else 0

    div()
    print(f"  {r['symbol']}  [{r['category']}]  "
          f"{r['date_from']} -> {r['date_to']}  ({yr:.2f} years)")
    div()

    print(f"\n  PERFORMANCE")
    hdiv()
    row2("Total Trades",    f"{r['trades']}  ({tpy:.0f}/yr)",
         "Win / Loss",      f"{r['wins']} / {r['losses']}")
    row2("Win Rate",        f"{r['win_rate']}%",
         "Profit Factor",   f"{r['pf']}")
    row2("Avg Win (net)",   f"${r['avg_win']:+.2f}",
         "Avg Loss (net)",  f"${r['avg_loss']:+.2f}")
    row2("Best Trade",      f"${r['best_trade']:+.2f}",
         "Worst Trade",     f"${r['worst_trade']:+.2f}")
    row2("Max Win Streak",  f"{r['max_win_str']} trades",
         "Max Loss Streak", f"{r['max_los_str']} trades")

    print(f"\n  PnL BREAKDOWN  (compounding: 10% risk/trade)")
    hdiv()
    row1("Gross PnL (before all costs)", f"${r['gross_pnl']:>+12,.2f}")
    row1(f"  Exchange {fee_label}",       f"${-r['total_fees']:>+12,.2f}")
    row1(f"  Slippage (entry impact)",    f"${-r['total_slip']:>+12,.2f}")
    row1(f"  Total Costs",                f"${-r['total_costs']:>+12,.2f}")
    hdiv()
    row1("NET PnL (total)",              f"${r['net_pnl']:>+12,.2f}")
    row1("Initial Balance",              f"${r['initial_bal']:>12,.2f}")
    row1("Final  Balance",               f"${r['final_bal']:>12,.2f}")
    row1("Total Return",                 f"{r['total_ret']:>+11.2f}%")
    row1(f"CAGR ({yr:.1f} yrs)",         f"{r['cagr']:>+11.2f}%/yr")

    print(f"\n  PER-TRADE COST BREAKDOWN  (averages)")
    hdiv()
    row2("Avg Trade Notional",  f"${r['avg_notional']:.2f}",
         "Avg Gross/trade",     f"${r['gross_pnl']/r['trades']:+.4f}" if r['trades'] else "n/a")
    row2(f"Avg {fee_label}/trade", f"${r['avg_fee_pt']:.4f}",
         "Avg Slippage/trade",  f"${r['avg_slip_pt']:.4f}")
    row2("Avg Total Cost/trade", f"${r['avg_cost_pt']:.4f}",
         "Cost as % of gross",  f"{r['total_costs']/r['gross_pnl']*100:.1f}%"
                                  if r['gross_pnl'] > 0 else "n/a")

    print(f"\n  RISK METRICS")
    hdiv()
    row2("Max Drawdown",       f"{r['max_dd']:.2f}%",
         "Max Drawdown ($)",   f"${r['max_dd_usd']:,.2f}")
    row2("DD Duration",        f"{r['dd_dur_bars']} bars (~{r['dd_dur_bars']//24}d)",
         "Calmar Ratio",       f"{r['calmar']}")
    row2("Sharpe Ratio",       f"{r['sharpe']}",
         "",                   "")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_crypto, results_forex = [], []

    div("="); div(" ")
    print(f"  LIQUIDATION ZONE REVERSAL -- BACKTEST v5  (COMPOUNDING)")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    div(" "); div("=")

    print(f"""
  STRATEGY PARAMETERS  (identical to v3 except risk sizing)
    Starting Capital  : ${INITIAL_BALANCE:,.2f}
    Risk Per Trade    : {RISK_PCT*100:.0f}% of current balance  (COMPOUNDING)
                        (v3 used fixed ${10:.0f}/trade = 1% of starting capital)
    Stop Loss         : {SL_MULT} x ATR
    Partial TP        : {PARTIAL_TP_MULT} x ATR  (close 50%)
    Trail Stop        : {TRAIL_DIST_MULT} x ATR below running high (LONG) / above running low (SHORT)
    Hard Cap TP       : {HARD_TP_MULT} x ATR
    Zone Approach     : within {APPROACH_PCT*100:.1f}% of zone
    Touch Buffer      : {TOUCH_BUF*100:.1f}%  (cascade starts before exact zone)
    ATR Period        : {ATR_PERIOD} bars (1h)
    Swing Lookback    : {SWING_LOOKBACK} x 4h bars each side

  ACTIVE FILTERS  (unchanged from v3)
    [1] Volume Spike : Trigger bar vol > {VOL_MULT}x {VOL_LOOKBACK}-bar avg  (crypto only)
    [2] 4h EMA Trend : LONG only if EMA{EMA4H_PERIOD} rising | SHORT only if falling
    [3] Partial Exit : 50% at {PARTIAL_TP_MULT}xATR, trail {TRAIL_DIST_MULT}xATR, hard cap {HARD_TP_MULT}xATR
    [4] Zone Fresh   : Max {ZONE_MAX_TOUCH} trades per zone per {ZONE_WINDOW//24}-day window

  COST STRUCTURE
    Crypto exchange fee : {CRYPTO_FEE_RT*100:.2f}% round-trip
    Crypto slippage     : {CRYPTO_SLIP_PCT*100:.2f}% on entry
    Forex spread        : EURUSD/GBPUSD/USDJPY={FOREX_SPREAD['EURUSD']*10000:.1f}pip
                          AUDUSD/EURJPY={FOREX_SPREAD['AUDUSD']*10000:.1f}pip
                          USDCAD={FOREX_SPREAD['USDCAD']*10000:.1f}pip
                          GBPJPY={FOREX_SPREAD['GBPJPY']*10000:.1f}pip
    Forex slippage      : {FOREX_SLIP_PCT*100:.2f}% on entry
""")

    # ── CRYPTO ────────────────────────────────────────────────────────────────
    div("=")
    print("  CRYPTO  (loading 1m CSV, resampling to 1h)")
    div("=")
    for sym in CRYPTO_SYMBOLS:
        try:
            df = load_crypto_1h(sym)
            print(f"\n  [{sym}] {len(df):,} bars  |  "
                  f"{df.index[0].date()} → {df.index[-1].date()}  |  running...")
            r = run_backtest(sym, df, fee_rt=CRYPTO_FEE_RT,
                             slip_pct=CRYPTO_SLIP_PCT,
                             category="Crypto", use_vol_filter=True)
            results_crypto.append(r)
            print(f"  → {r['trades']} trades  |  WR {r['win_rate']}%  |  "
                  f"PF {r['pf']}  |  CAGR {r['cagr']:+.1f}%  |  "
                  f"Final ${r['final_bal']:,.2f}")
        except Exception as e:
            print(f"  ERROR {sym}: {e}")

    # ── FOREX ─────────────────────────────────────────────────────────────────
    div("=")
    print("  FOREX  (downloading from yfinance, 2-year 1h)")
    div("=")
    for name, ticker in FOREX_PAIRS.items():
        try:
            df = download_forex_1h(ticker, name)
            print(f"    [{name}] {len(df):,} bars  —  running...")
            spread = FOREX_SPREAD.get(name, 0.0002)
            r = run_backtest(name, df, fee_rt=spread*2,
                             slip_pct=FOREX_SLIP_PCT,
                             category="Forex", use_vol_filter=False)
            results_forex.append(r)
            print(f"    → {r['trades']} trades  |  WR {r['win_rate']}%  |  "
                  f"PF {r['pf']}  |  CAGR {r['cagr']:+.1f}%  |  "
                  f"Final ${r['final_bal']:,.2f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    all_results = results_crypto + results_forex
    if not all_results:
        print("No results."); return

    # ─── PER-ASSET DETAILED REPORTS ───────────────────────────────────────────
    print("\n\n")
    div("="); div(" ")
    print("  DETAILED RESULTS -- CRYPTO")
    div(" "); div("=")
    for r in results_crypto:
        print_asset_block(r, "fees (0.04% RT)", "0.03% entry")

    div("="); div(" ")
    print("  DETAILED RESULTS -- FOREX")
    div(" "); div("=")
    for r in results_forex:
        spread_pips = FOREX_SPREAD.get(r["symbol"], 0.0002) * 10000
        print_asset_block(r, f"spread ({spread_pips:.1f}pip RT)", "0.01% entry")

    # ─── SUMMARY TABLE ────────────────────────────────────────────────────────
    div("="); div(" ")
    print("  FULL SUMMARY TABLE  (sorted by CAGR)")
    div(" "); div("=")
    hdr = (f"  {'Symbol':<10} {'Cat':>6} {'Tr/yr':>6} {'WR%':>6} {'PF':>5}"
           f" {'Gross$':>10} {'Net$':>10} {'FinalBal':>10}"
           f" {'CAGR%':>7} {'MaxDD%':>7} {'Sharpe':>7} {'Calmar':>7}")
    print(hdr); hdiv()
    for r in sorted(all_results, key=lambda x: x["cagr"], reverse=True):
        yr  = r["years"]
        tpy = r["trades"] / yr if yr > 0 else 0
        print(f"  {r['symbol']:<10} {r['category']:>6} {tpy:>6.0f} "
              f"{r['win_rate']:>6.1f} {r['pf']:>5.2f}"
              f" {r['gross_pnl']:>+10,.2f} {r['net_pnl']:>+10,.2f}"
              f" {r['final_bal']:>10,.2f}"
              f" {r['cagr']:>+7.1f} {r['max_dd']:>7.2f}"
              f" {r['sharpe']:>7.2f} {r['calmar']:>7.2f}")

    hdiv()
    for label, rlist in [("CRYPTO AVG", results_crypto), ("FOREX AVG", results_forex)]:
        active = [r for r in rlist if r["trades"] > 0]
        if not active: continue
        avg_tpy  = np.mean([r["trades"]/r["years"] for r in active if r["years"] > 0])
        avg_wr   = np.mean([r["win_rate"] for r in active])
        avg_pf   = np.mean([r["pf"] for r in active])
        avg_cagr = np.mean([r["cagr"] for r in active])
        avg_dd   = np.mean([r["max_dd"] for r in active])
        avg_sh   = np.mean([r["sharpe"] for r in active])
        avg_cal  = np.mean([r["calmar"] for r in active])
        print(f"  {label:<10} {'':>6} {avg_tpy:>6.0f} "
              f"{avg_wr:>6.1f} {avg_pf:>5.2f}"
              f" {'':>10} {'':>10} {'':>10}"
              f" {avg_cagr:>+7.1f} {avg_dd:>7.2f}"
              f" {avg_sh:>7.2f} {avg_cal:>7.2f}")

    # ─── Save ─────────────────────────────────────────────────────────────────
    csv = OUTPUT_DIR / "backtest_v5_full.csv"
    rows = [{k: r[k] for k in r if k not in ("equity","trades_df")}
            for r in all_results]
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\n  Full results saved -> {csv}")
    div("=")


if __name__ == "__main__":
    main()
