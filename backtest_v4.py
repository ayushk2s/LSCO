"""
backtest_v4.py  —  Liquidation Zone Reversal  v4  (Compounding Edition)
========================================================================
Mirrors liq_algo_v4 logic as closely as possible on historical OHLCV data.

KEY CHANGES FROM backtest_v3:
  1. Compounding risk  : 10% of current balance per trade (not fixed $10).
                         Position size grows with balance, shrinks with losses.
  2. Regime filter     : last 3 completed 1h closes must NOT all point against
                         the trade direction.  No 4h EMA, no look-ahead.
  3. Simple exit       : TP = 1.5 × ATR, SL = 0.75 × ATR.
                         No partial exit, no trailing stop, no hard cap.
  4. No volume filter  : removed (v4 live algo removed it).
  5. No zone freshness : removed (v4 live algo removed it).
  6. Cooldowns (bars)  : 5 bars after LOSS, 1 bar after WIN.

BIAS / BUG CHECKS:
  - Zone snapshot rebuilt daily with data UP TO start of that day only.
  - ATR uses only bars 0..i (no bar-i+1 data).
  - Regime filter uses closes at [i-3], [i-2], [i-1]  (bars before signal bar).
  - Entry price = open of bar [i+1]  (next bar after signal — market order).
  - SL/TP checking starts from bar [i+1] onward (trade opened at its open).
  - On a bar where BOTH SL and TP are within range, SL is honoured first
    (conservative, avoids phantom wins).
  - Only one open position per symbol at a time (in_trade flag).
  - Balance updated only when trade CLOSES, never when it opens.
  - No T2 / second position logic — zero double-counting risk.
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Risk / sizing ────────────────────────────────────────────────────────────
INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10       # 10 % of current balance at risk each trade
                              # → if SL hit: lose exactly 10 % of balance
                              #   (before fees/slip, which add ~0.5 %)

# ─── Entry / exit ─────────────────────────────────────────────────────────────
SL_MULT      = 0.75          # stop-loss  = entry ± 0.75 × ATR
TP_MULT      = 1.50          # take-profit = entry ± 1.50 × ATR  (no partial/trail)

APPROACH_PCT = 0.008         # price must be within 0.8 % of zone to arm signal
TOUCH_BUF    = 0.003         # 0.3 % buffer: cascade can start just before exact zone

ATR_PERIOD     = 14
SWING_LOOKBACK = 20          # bars each side for swing high/low detection
MIN_ZONE_GAP   = 0.005       # merge zones closer than 0.5 %

# ─── Filters ──────────────────────────────────────────────────────────────────
REGIME_BARS   = 3            # check this many consecutive 1h closes for regime

# ─── Cooldowns (in 1h bars) ───────────────────────────────────────────────────
COOLDOWN_LOSS = 5            # 5 h after loss before re-using same zone
COOLDOWN_WIN  = 1            # 1 h after win

# ─── Fees & slippage ──────────────────────────────────────────────────────────
CRYPTO_FEE_RT   = 0.0004     # 0.04 % round-trip  (0.02 % each side)
CRYPTO_SLIP_PCT = 0.0003     # 0.03 % entry slippage (DEX market impact)

FOREX_SPREAD = {
    "EURUSD": 0.00010, "GBPUSD": 0.00010, "USDJPY": 0.00010,
    "AUDUSD": 0.00015, "USDCAD": 0.00020, "EURJPY": 0.00015, "GBPJPY": 0.00025,
}
FOREX_SLIP_PCT = 0.0001

# ─── Symbols ──────────────────────────────────────────────────────────────────
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FOREX_PAIRS    = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_crypto_1h(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date":"ts","Open":"open","High":"high",
                             "Low":"low","Close":"close","Volume":"vol"})
    df = df.set_index("ts").sort_index()
    return df.resample("1h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),  close=("close","last"), vol=("vol","sum")
    ).dropna()


def download_forex_1h(ticker: str, name: str) -> pd.DataFrame:
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


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last")
    ).dropna()


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_4h: pd.DataFrame) -> list:
    """Swing highs and lows on the supplied 4h slice (pure past data)."""
    n, lb = len(df_4h), SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_4h["high"].iloc[i] == df_4h["high"].iloc[i - lb: i + lb + 1].max():
            highs.append(df_4h["high"].iloc[i])
        if df_4h["low"].iloc[i] == df_4h["low"].iloc[i - lb: i + lb + 1].min():
            lows.append(df_4h["low"].iloc[i])
    levels, merged = sorted(set(highs + lows)), []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── Core backtest engine ─────────────────────────────────────────────────────

def run_backtest(symbol: str, df_1h: pd.DataFrame,
                 fee_rt: float, slip_pct: float,
                 category: str) -> dict:
    """
    Single-pass backtest.  Returns a stats dict.

    Bias / correctness guarantees
    ──────────────────────────────
    • zone_snap[d] built with 4h bars whose index ≤ first bar of day d.
    • atr_s.iloc[i] uses only closes up to bar i.  No future data.
    • regime check uses closes [i-3], [i-2], [i-1] — bars completed before
      the signal bar.
    • Entry price = open of bar [i+1].  SL/TP managed from bar [i+1] onward.
    • Balance updated only on trade CLOSE.  One trade at a time per symbol.
    • SL tested before TP on same bar (conservative — avoids phantom wins).
    """
    FEE_SIDE = fee_rt / 2   # half of round-trip applied each side

    df_4h = resample_4h(df_1h)
    atr_s = calc_atr(df_1h)       # ATR[i] = EWM over TR[0..i] — no look-ahead

    # ── Pre-compute daily zone snapshots (NO look-ahead) ─────────────────────
    # zone_snap[d] = zones known at the START of day d.
    # "start of day d" = df_1h.index[d * 24], so only 4h bars BEFORE that timestamp.
    n_days    = len(df_1h) // 24 + 2
    zone_snap = []
    for d in range(n_days):
        start_bar = d * 24
        if start_bar >= len(df_1h):
            zone_snap.append([])
            continue
        start_ts = df_1h.index[start_bar]
        # Strictly less than start_ts to avoid using the current day's 4h bar
        past_4h  = df_4h[df_4h.index < start_ts]
        if len(past_4h) < SWING_LOOKBACK * 2 + 5:
            zone_snap.append([])
        else:
            zone_snap.append(find_zones(past_4h.iloc[-200:]))

    # ── State ─────────────────────────────────────────────────────────────────
    balance          = INITIAL_BALANCE
    equity           = [balance]          # one value per bar (BEFORE bar processes)
    trades           = []
    zone_cooldown    = {}                 # zone_price → bar index until cool
    last_trigger_bar = -999               # prevent double-entry on same bar

    in_trade   = False
    direction  = ""
    entry_px   = sl_px = tp_px = qty = 0.0
    active_zone = 0.0
    fee_accrued = slip_trade = 0.0

    WARM_UP = max(50, SWING_LOOKBACK * 2 + 5, REGIME_BARS + 2)

    for i, (ts, row) in enumerate(df_1h.iterrows()):

        # Record equity BEFORE any action on bar i
        equity.append(balance)

        if i < WARM_UP:
            continue

        atr = atr_s.iloc[i]
        if atr <= 0 or np.isnan(atr):
            continue

        # ── 1. Manage existing open trade ────────────────────────────────────
        if in_trade:
            gross  = None
            closed = False
            exit_px = 0.0

            if direction == "LONG":
                # SL checked first (conservative)
                if row["low"] <= sl_px:
                    exit_px = sl_px
                    gross   = (sl_px - entry_px) * qty
                    closed  = True
                elif row["high"] >= tp_px:
                    exit_px = tp_px
                    gross   = (tp_px - entry_px) * qty
                    closed  = True

            else:  # SHORT
                if row["high"] >= sl_px:
                    exit_px = sl_px
                    gross   = (entry_px - sl_px) * qty
                    closed  = True
                elif row["low"] <= tp_px:
                    exit_px = tp_px
                    gross   = (entry_px - tp_px) * qty
                    closed  = True

            if closed and gross is not None:
                total_fee = fee_accrued + exit_px * qty * FEE_SIDE
                net       = gross - total_fee - slip_trade
                result    = "WIN" if gross > 0 else "LOSS"

                # Update balance ONLY on close
                balance_before = balance
                balance       += net

                in_trade = False
                zone_cooldown[active_zone] = i + (COOLDOWN_LOSS if result == "LOSS"
                                                   else COOLDOWN_WIN)
                trades.append({
                    "ts":           ts,
                    "dir":          direction,
                    "entry":        round(entry_px,       6),
                    "exit":         round(exit_px,        6),
                    "qty":          round(qty,             8),
                    "risk_usdt":    round(balance_before * RISK_PCT, 4),
                    "notional":     round(entry_px * qty,  2),
                    "gross":        round(gross,            4),
                    "fee":          round(total_fee,        4),
                    "slip":         round(slip_trade,       4),
                    "net":          round(net,              4),
                    "result":       result,
                    "balance_open": round(balance_before,  4),
                    "balance":      round(balance,         4),
                })
            continue  # whether or not trade closed, move to next bar

        # ── 2. Look for new entry signal ──────────────────────────────────────

        price = row["close"]

        # Zone lookup: use today's snapshot (built from data before today)
        day_idx = i // 24
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            continue
        zones = zone_snap[day_idx]

        zones_below = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
        zones_above = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]

        if zones_below:
            near_zone = max(zones_below);  near_dir = "LONG"
        elif zones_above:
            near_zone = min(zones_above);  near_dir = "SHORT"
        else:
            continue

        # Zone cooldown
        if zone_cooldown.get(near_zone, 0) > i:
            continue

        # Prevent double-entry on the same bar
        if i == last_trigger_bar:
            continue

        # Touch-and-reverse trigger
        if near_dir == "LONG":
            triggered = (row["low"] <= near_zone * (1 + TOUCH_BUF)
                         and row["close"] > near_zone)
        else:
            triggered = (row["high"] >= near_zone * (1 - TOUCH_BUF)
                         and row["close"] < near_zone)
        if not triggered:
            continue

        # ── Regime filter (v4): no look-ahead ────────────────────────────────
        # Check the REGIME_BARS closes COMPLETED before bar i.
        # These are bars [i-REGIME_BARS], [i-REGIME_BARS+1], ..., [i-1].
        if i < REGIME_BARS:
            continue
        regime_closes = df_1h["close"].iloc[i - REGIME_BARS: i].values
        # regime_closes[0] = oldest, regime_closes[-1] = most recent before signal

        all_descending = all(regime_closes[k] > regime_closes[k + 1]
                             for k in range(len(regime_closes) - 1))
        all_ascending  = all(regime_closes[k] < regime_closes[k + 1]
                             for k in range(len(regime_closes) - 1))

        if near_dir == "LONG"  and all_descending:
            continue
        if near_dir == "SHORT" and all_ascending:
            continue

        # ── Position sizing (compounding) ─────────────────────────────────────
        sl_dist = SL_MULT * atr
        if sl_dist <= 0 or balance <= 0:
            continue

        risk_usdt  = balance * RISK_PCT          # 10 % of current balance
        qty        = risk_usdt / sl_dist         # units such that SL hit = risk_usdt loss

        # Entry at open of NEXT bar (market order after signal bar closes)
        if i + 1 >= len(df_1h):
            continue
        entry_px   = df_1h["open"].iloc[i + 1]
        slip_trade = entry_px * qty * slip_pct   # entry slippage cost

        # ── Set TP / SL from actual entry price ───────────────────────────────
        if near_dir == "LONG":
            sl_px = entry_px - SL_MULT * atr
            tp_px = entry_px + TP_MULT  * atr
        else:
            sl_px = entry_px + SL_MULT * atr
            tp_px = entry_px - TP_MULT  * atr

        # ── Open trade ────────────────────────────────────────────────────────
        in_trade         = True
        direction        = near_dir
        active_zone      = near_zone
        fee_accrued      = entry_px * qty * FEE_SIDE   # entry-side fee
        last_trigger_bar = i

    # ─── Build stats ──────────────────────────────────────────────────────────
    if not trades:
        return _empty(symbol, category, equity, df_1h)

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]

    gross_pnl   = df_t["gross"].sum()
    total_fees  = df_t["fee"].sum()
    total_slip  = df_t["slip"].sum()
    total_costs = total_fees + total_slip
    net_pnl     = df_t["net"].sum()
    gross_wins  = wins["gross"].sum()
    gross_loss  = abs(loses["gross"].sum())
    pf          = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    win_rate    = round(len(wins) / len(df_t) * 100, 1)

    final_bal   = balance
    years       = (df_1h.index[-1] - df_1h.index[0]).days / 365.25
    total_ret   = (final_bal / INITIAL_BALANCE - 1) * 100
    cagr        = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Equity drawdown
    eq_s     = pd.Series(equity)
    roll_max = eq_s.cummax()
    dd_pct   = (eq_s - roll_max) / roll_max * 100
    max_dd   = round(dd_pct.min(), 2)
    max_dd_usd_val = round((eq_s - roll_max).min(), 2)

    # Max drawdown duration (bars in consecutive drawdown)
    in_dd_flag = (dd_pct < 0).astype(int)
    dd_dur_max = 0; dd_cur = 0
    for v in in_dd_flag:
        dd_cur = dd_cur + 1 if v else 0
        dd_dur_max = max(dd_dur_max, dd_cur)

    # Per-trade % return on balance at time of trade open (for Sharpe)
    pct_rets = (df_t["net"] / df_t["balance_open"]).values
    ann_factor = (len(df_t) / years) if years > 0 else len(df_t)
    sharpe = round((pct_rets.mean() / pct_rets.std() * np.sqrt(ann_factor))
                   if pct_rets.std() > 0 else 0, 2)

    calmar = round(cagr / abs(max_dd) if max_dd != 0 else 0, 2)

    # Win / loss streaks
    max_ws = max_ls = cur_ws = cur_ls = 0
    for r in df_t["result"]:
        if r == "WIN":  cur_ws += 1; cur_ls = 0
        else:           cur_ls += 1; cur_ws = 0
        max_ws = max(max_ws, cur_ws)
        max_ls = max(max_ls, cur_ls)

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
        "max_dd_usd":   max_dd_usd_val,
        "dd_dur_bars":  dd_dur_max,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "avg_win":      round(wins["net"].mean(),    4) if len(wins)  else 0,
        "avg_loss":     round(loses["net"].mean(),   4) if len(loses) else 0,
        "avg_win_pct":  round((wins["net"] / wins["balance_open"] * 100).mean(),   2) if len(wins)  else 0,
        "avg_loss_pct": round((loses["net"] / loses["balance_open"] * 100).mean(), 2) if len(loses) else 0,
        "best_trade":   round(df_t["net"].max(),     4),
        "worst_trade":  round(df_t["net"].min(),     4),
        "avg_fee_pt":   round(total_fees  / len(df_t), 4),
        "avg_slip_pt":  round(total_slip  / len(df_t), 4),
        "avg_cost_pt":  round(total_costs / len(df_t), 4),
        "avg_notional": round(df_t["notional"].mean(), 2),
        "avg_risk_usd": round(df_t["risk_usdt"].mean(), 4),
        "max_win_str":  max_ws,
        "max_los_str":  max_ls,
        "equity":       equity,
        "trades_df":    df_t,
    }


def _empty(symbol, category, equity, df_1h):
    years = (df_1h.index[-1] - df_1h.index[0]).days / 365.25 if len(df_1h) > 1 else 0
    return {
        "symbol": symbol, "category": category,
        "date_from": df_1h.index[0].date() if len(df_1h) else "",
        "date_to":   df_1h.index[-1].date() if len(df_1h) else "",
        "years": round(years, 2), "trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "pf": 0, "gross_pnl": 0, "total_fees": 0,
        "total_slip": 0, "total_costs": 0, "net_pnl": 0,
        "initial_bal": INITIAL_BALANCE, "final_bal": INITIAL_BALANCE,
        "total_ret": 0, "cagr": 0, "max_dd": 0, "max_dd_usd": 0,
        "dd_dur_bars": 0, "sharpe": 0, "calmar": 0,
        "avg_win": 0, "avg_loss": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
        "best_trade": 0, "worst_trade": 0, "avg_fee_pt": 0,
        "avg_slip_pt": 0, "avg_cost_pt": 0, "avg_notional": 0,
        "avg_risk_usd": 0, "max_win_str": 0, "max_los_str": 0,
        "equity": equity, "trades_df": pd.DataFrame(),
    }


# ─── Report printer ───────────────────────────────────────────────────────────
W = 72
def div(c="="): print(c * W)
def hdiv():     print("-" * W)
def row2(a, av, b, bv, w=28):
    print(f"  {a:<{w}} {av!s:<18}  {b:<{w}} {bv!s}")
def row1(a, av, w=38):
    print(f"  {a:<{w}} {av}")


def print_asset_block(r, fee_label):
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
    row2("Avg Win  (net$)", f"${r['avg_win']:+.2f}  ({r['avg_win_pct']:+.2f}% bal)",
         "Avg Loss (net$)", f"${r['avg_loss']:+.2f}  ({r['avg_loss_pct']:+.2f}% bal)")
    row2("Best Trade",      f"${r['best_trade']:+.2f}",
         "Worst Trade",     f"${r['worst_trade']:+.2f}")
    row2("Max Win Streak",  f"{r['max_win_str']} trades",
         "Max Loss Streak", f"{r['max_los_str']} trades")
    row2("Avg Risk/trade",  f"${r['avg_risk_usd']:.2f}",
         "Avg Notional",    f"${r['avg_notional']:.2f}")

    print(f"\n  COMPOUNDING PnL  (starting ${r['initial_bal']:,.0f}, "
          f"10% risk/trade)")
    hdiv()
    row1("Gross PnL (before fees/slip)", f"${r['gross_pnl']:>+12,.2f}")
    row1(f"  Exchange {fee_label}",       f"${-r['total_fees']:>+12,.2f}")
    row1( "  Slippage (entry impact)",    f"${-r['total_slip']:>+12,.2f}")
    row1( "  Total Costs",                f"${-r['total_costs']:>+12,.2f}")
    hdiv()
    row1("Net PnL (total)",              f"${r['net_pnl']:>+12,.2f}")
    row1("Initial Balance",              f"${r['initial_bal']:>12,.2f}")
    row1("Final  Balance",               f"${r['final_bal']:>12,.2f}")
    row1("Total Return",                 f"{r['total_ret']:>+11.2f}%")
    row1(f"CAGR ({yr:.1f} yrs)",         f"{r['cagr']:>+11.2f}%/yr")

    print(f"\n  RISK METRICS")
    hdiv()
    row2("Max Drawdown (%)",  f"{r['max_dd']:.2f}%",
         "Max Drawdown ($)",  f"${r['max_dd_usd']:,.2f}")
    row2("DD Duration",       f"{r['dd_dur_bars']} bars (~{r['dd_dur_bars']//24}d)",
         "Calmar Ratio",      f"{r['calmar']}")
    row2("Sharpe Ratio",      f"{r['sharpe']}",
         "",                  "")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_crypto, results_forex = [], []

    div("="); div(" ")
    print( "  LIQUIDATION ZONE REVERSAL  --  BACKTEST v4  (COMPOUNDING)")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    div(" "); div("=")

    print(f"""
  STRATEGY PARAMETERS (v4 logic)
    Starting Capital   : ${INITIAL_BALANCE:,.2f}
    Risk Per Trade     : {RISK_PCT*100:.0f}% of current balance  (COMPOUNDING)
    Stop Loss          : {SL_MULT} × ATR  (0.75 × ATR)
    Take Profit        : {TP_MULT} × ATR  (1.50 × ATR)  — simple single exit
    Zone Approach      : within {APPROACH_PCT*100:.1f}% of zone
    Touch Buffer       : {TOUCH_BUF*100:.1f}%  (cascade starts before exact zone)
    ATR Period         : {ATR_PERIOD} bars (1h)
    Swing Lookback     : {SWING_LOOKBACK} × 4h bars each side

  ACTIVE FILTERS  (v4 set)
    [1] Regime Filter  : skip LONG if last {REGIME_BARS} closes all descending
                         skip SHORT if last {REGIME_BARS} closes all ascending
    [2] Zone Cooldown  : {COOLDOWN_LOSS}h after LOSS, {COOLDOWN_WIN}h after WIN

  REMOVED vs v3
    [-] Volume spike filter  (removed in v4)
    [-] Zone freshness limit (removed in v4)
    [-] Partial exit / trail (simplified to single TP/SL in v4)

  COST STRUCTURE
    Crypto fee     : {CRYPTO_FEE_RT*100:.2f}% RT  ({CRYPTO_FEE_RT/2*100:.2f}% each side)
    Crypto slip    : {CRYPTO_SLIP_PCT*100:.2f}% entry
    Forex spread   : EURUSD/GBPUSD {FOREX_SPREAD['EURUSD']*10000:.1f}pip  |
                     EURJPY {FOREX_SPREAD['EURJPY']*10000:.1f}pip  |
                     GBPJPY {FOREX_SPREAD['GBPJPY']*10000:.1f}pip
    Forex slip     : {FOREX_SLIP_PCT*100:.2f}% entry
""")

    # ── CRYPTO ────────────────────────────────────────────────────────────────
    div("=")
    print("  CRYPTO  (loading 1m CSV  →  resample to 1h)")
    div("=")
    for sym in CRYPTO_SYMBOLS:
        try:
            df = load_crypto_1h(sym)
            print(f"\n  [{sym}]  {len(df):,} bars  |  "
                  f"{df.index[0].date()} → {df.index[-1].date()}  |  running...")
            r = run_backtest(sym, df,
                             fee_rt   = CRYPTO_FEE_RT,
                             slip_pct = CRYPTO_SLIP_PCT,
                             category = "Crypto")
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
            print(f"    [{name}]  {len(df):,} bars  —  running...")
            spread = FOREX_SPREAD.get(name, 0.0002)
            r = run_backtest(name, df,
                             fee_rt   = spread * 2,
                             slip_pct = FOREX_SLIP_PCT,
                             category = "Forex")
            results_forex.append(r)
            print(f"    → {r['trades']} trades  |  WR {r['win_rate']}%  |  "
                  f"PF {r['pf']}  |  CAGR {r['cagr']:+.1f}%  |  "
                  f"Final ${r['final_bal']:,.2f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    all_results = results_crypto + results_forex
    if not all_results:
        print("No results."); return

    # ─── Per-asset detailed blocks ────────────────────────────────────────────
    print("\n\n")
    div("="); div(" ")
    print("  DETAILED RESULTS  —  CRYPTO")
    div(" "); div("=")
    for r in results_crypto:
        print_asset_block(r, "fees (0.04% RT)")

    div("="); div(" ")
    print("  DETAILED RESULTS  —  FOREX")
    div(" "); div("=")
    for r in results_forex:
        pips = FOREX_SPREAD.get(r["symbol"], 0.0002) * 10000
        print_asset_block(r, f"spread ({pips:.1f}pip RT)")

    # ─── Summary table ────────────────────────────────────────────────────────
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
    # Crypto aggregate
    cr = [r for r in all_results if r["category"] == "Crypto" and r["trades"] > 0]
    if cr:
        avg_wr   = np.mean([r["win_rate"] for r in cr])
        avg_pf   = np.mean([r["pf"] for r in cr])
        avg_cagr = np.mean([r["cagr"] for r in cr])
        avg_dd   = np.mean([r["max_dd"] for r in cr])
        avg_sh   = np.mean([r["sharpe"] for r in cr])
        avg_cal  = np.mean([r["calmar"] for r in cr])
        avg_tpy  = np.mean([r["trades"] / r["years"] for r in cr if r["years"] > 0])
        print(f"  {'CRYPTO AVG':<10} {'':>6} {avg_tpy:>6.0f} "
              f"{avg_wr:>6.1f} {avg_pf:>5.2f}"
              f" {'':>10} {'':>10} {'':>10}"
              f" {avg_cagr:>+7.1f} {avg_dd:>7.2f}"
              f" {avg_sh:>7.2f} {avg_cal:>7.2f}")
    fr = [r for r in all_results if r["category"] == "Forex" and r["trades"] > 0]
    if fr:
        avg_wr   = np.mean([r["win_rate"] for r in fr])
        avg_pf   = np.mean([r["pf"] for r in fr])
        avg_cagr = np.mean([r["cagr"] for r in fr])
        avg_dd   = np.mean([r["max_dd"] for r in fr])
        avg_sh   = np.mean([r["sharpe"] for r in fr])
        avg_cal  = np.mean([r["calmar"] for r in fr])
        avg_tpy  = np.mean([r["trades"] / r["years"] for r in fr if r["years"] > 0])
        print(f"  {'FOREX AVG':<10} {'':>6} {avg_tpy:>6.0f} "
              f"{avg_wr:>6.1f} {avg_pf:>5.2f}"
              f" {'':>10} {'':>10} {'':>10}"
              f" {avg_cagr:>+7.1f} {avg_dd:>7.2f}"
              f" {avg_sh:>7.2f} {avg_cal:>7.2f}")

    # ─── Save CSV ─────────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        yr  = r["years"]
        tpy = r["trades"] / yr if yr > 0 else 0
        rows.append({
            "symbol": r["symbol"], "category": r["category"],
            "date_from": r["date_from"], "date_to": r["date_to"],
            "years": yr, "trades": r["trades"], "trades_yr": round(tpy,1),
            "wins": r["wins"], "losses": r["losses"],
            "win_rate": r["win_rate"], "pf": r["pf"],
            "gross_pnl": r["gross_pnl"], "total_fees": r["total_fees"],
            "total_slip": r["total_slip"], "net_pnl": r["net_pnl"],
            "initial_bal": r["initial_bal"], "final_bal": r["final_bal"],
            "total_ret_pct": r["total_ret"], "cagr_pct": r["cagr"],
            "max_dd_pct": r["max_dd"], "max_dd_usd": r["max_dd_usd"],
            "sharpe": r["sharpe"], "calmar": r["calmar"],
            "avg_win": r["avg_win"], "avg_loss": r["avg_loss"],
            "avg_win_pct": r["avg_win_pct"], "avg_loss_pct": r["avg_loss_pct"],
            "best_trade": r["best_trade"], "worst_trade": r["worst_trade"],
        })
    out = OUTPUT_DIR / "backtest_v4_full.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Results saved → {out}")
    div("=")


if __name__ == "__main__":
    main()
