"""
backtest_v7_NSE.py  --  Liquidation Zone Reversal  v7  (NSE Stocks Edition)
=============================================================================
Adapts the v7 crypto backtest engine for NSE equity intraday data.

DATA
    Source  : 1-minute OHLCV CSVs (IST timezone, 09:15 - 15:29 market hours)
    Period  : 2017 - 2021 (approx 4 years per stock)
    Symbols : All stocks found in DATA_DIR

KEY ADAPTATIONS FROM v7
    1. Market hours only     : 1m data filtered to 09:15-15:29 IST daily
    2. Zone detection        : Daily bars (not 4h) – standard for NSE charting
    3. EMA trend filter      : 20-day EMA on daily bars
    4. EOD forced close      : All positions closed by 15:20 IST (intraday)
    5. Entry cutoff          : No new entries after 14:00 IST
    6. NSE realistic costs   : 0.10% round-trip (brokerage + STT + exchange)
    7. INR denomination      : Starting capital ₹1,00,000
    8. No leverage           : qty capped so notional <= MAX_POSITION_INR

ALL v6/v7 STRICT MODE FIXES INHERITED
    [FIX 1] Zone snapshot   : strict < (no look-ahead)
    [FIX 2] Conservative    : trail wins over hard_tp when both fire same bar
    [FIX 3] Exit slippage   : on entry + partial exit + final exit
    [FIX 4] Notional cap    : per-trade and balance ceiling
    [FIX 5] WIN/LOSS        : by net P&L (after all costs)
    [FIX 6] Both PFs        : gross and net
    [FIX 7] Balance floor   : halt below 20% of initial
"""

import sys, warnings, json, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, date

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\GIGA\Downloads\NSE data\FullDataCsv")
OUTPUT_DIR = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Capital & risk ───────────────────────────────────────────────────────────
INITIAL_BALANCE      = 100_000.0   # INR ₹1,00,000
RISK_PCT             = 0.02        # 2% risk per trade
MAX_POSITION_INR     = 50_000.0    # max ₹50,000 notional per trade
MAX_BALANCE_INR      = 500_000.0   # balance ceiling
MIN_BAL_RATIO        = 0.20        # halt if balance < 20% of initial

# ─── Strategy ─────────────────────────────────────────────────────────────────
SL_MULT          = 0.75
PARTIAL_TP_MULT  = 1.0
TRAIL_DIST_MULT  = 0.5
HARD_TP_MULT     = 3.0
APPROACH_PCT     = 0.012    # 1.2% approach to zone
TOUCH_BUF        = 0.005    # 0.5% touch buffer
ATR_PERIOD       = 14
SWING_LOOKBACK   = 10       # 10 daily bars ≈ 2 trading weeks
MIN_ZONE_GAP     = 0.015    # 1.5% minimum gap between zones
VOL_MULT         = 1.5
VOL_LOOKBACK     = 20
EMA_PERIOD       = 20       # 20-day EMA
ZONE_MAX_TOUCH   = 2
ZONE_WINDOW      = 40       # 40 daily bars ≈ 2 months
COOLDOWN_LOSS    = 3        # 3-bar (1h) cooldown after loss
COOLDOWN_WIN     = 1
WARMUP_BARS      = 56       # 1h bars

# ─── NSE session ──────────────────────────────────────────────────────────────
MARKET_OPEN_H, MARKET_OPEN_M   = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 29
EOD_FORCE_H, EOD_FORCE_M       = 15, 20   # force-close if still open
ENTRY_CUTOFF_H, ENTRY_CUTOFF_M = 14,  0   # no new entries after 14:00

# ─── NSE costs ────────────────────────────────────────────────────────────────
NSE_FEE_RT   = 0.001    # 0.10% round-trip (brokerage 0.03% + STT 0.05% + exchange)
NSE_SLIP_PCT = 0.0005   # 0.05% slippage per fill


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_nse_stock(filepath: Path):
    """
    Load one NSE 1m CSV.  Returns (df_1m, df_1h, df_daily, symbol).
    All frames are market-hours only (09:15-15:29 IST), timezone-aware.
    """
    symbol = filepath.stem.split("__")[0]
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "ts", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "vol"})
    df = df.set_index("ts").sort_index()
    df = df[["open", "high", "low", "close", "vol"]].apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)

    # Filter to market hours (IST timezone-aware)
    t_open  = pd.Timestamp(f"1970-01-01 {MARKET_OPEN_H:02d}:{MARKET_OPEN_M:02d}:00").time()
    t_close = pd.Timestamp(f"1970-01-01 {MARKET_CLOSE_H:02d}:{MARKET_CLOSE_M:02d}:00").time()
    mask = (df.index.time >= t_open) & (df.index.time <= t_close)
    df_1m = df[mask].copy()

    if len(df_1m) < 2000:
        return symbol, None, None, None

    # Resample to 1h (market-hours aligned)
    df_1h = df_1m.resample("1h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"), vol=("vol", "sum")
    ).dropna()

    # Daily bars
    df_daily = df_1m.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"), vol=("vol", "sum")
    ).dropna()

    return symbol, df_1m, df_1h, df_daily


def get_all_files():
    return sorted(DATA_DIR.glob("*__EQ__NSE__NSE__MINUTE.csv"))


# ─── Indicators ───────────────────────────────────────────────────────────────

def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_d):
    """Swing highs/lows on daily bars as zone levels."""
    n, lb = len(df_d), SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_d["high"].iloc[i] == df_d["high"].iloc[i - lb: i + lb + 1].max():
            highs.append(df_d["high"].iloc[i])
        if df_d["low"].iloc[i] == df_d["low"].iloc[i - lb: i + lb + 1].min():
            lows.append(df_d["low"].iloc[i])
    levels, merged = sorted(set(highs + lows)), []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── 1m executor (NSE-aware with EOD forced close) ────────────────────────────

def _exec_1m_nse(df_1m, m_start, entry_px, direction,
                 sl_px, partial_tp_px, hard_tp_px,
                 trade_atr, qty, fee_side, slip_pct):
    """
    Same state-machine as v7 _exec_1m with one addition:
    positions are force-closed at EOD_FORCE_H:EOD_FORCE_M IST close price.
    """
    half           = qty / 2
    state          = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct
    t_eod          = pd.Timestamp(f"1970-01-01 {EOD_FORCE_H:02d}:{EOD_FORCE_M:02d}:00").time()

    n = len(df_1m)
    for m_idx in range(m_start, n):
        mr      = df_1m.iloc[m_idx]
        bar_ts  = df_1m.index[m_idx]
        gross   = None
        closed  = False
        exit_px = 0.0

        # EOD forced close
        if bar_ts.time() >= t_eod:
            exit_px = mr["close"]
            if direction == "LONG":
                gross = (exit_px - entry_px) * qty if state == "full" \
                        else partial_locked + (exit_px - entry_px) * half
            else:
                gross = (entry_px - exit_px) * qty if state == "full" \
                        else partial_locked + (entry_px - exit_px) * half
            closed = True

        elif state == "full":
            if direction == "LONG":
                if mr["low"] <= sl_px:
                    gross = (sl_px - entry_px) * qty; exit_px = sl_px; closed = True
                elif mr["high"] >= partial_tp_px:
                    partial_locked  = (partial_tp_px - entry_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["high"]
                    trail_sl        = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"
            else:
                if mr["high"] >= sl_px:
                    gross = (entry_px - sl_px) * qty; exit_px = sl_px; closed = True  # SHORT PnL
                elif mr["low"] <= partial_tp_px:
                    partial_locked  = (entry_px - partial_tp_px) * half
                    fee_acc        += partial_tp_px * half * fee_side
                    slip_acc       += partial_tp_px * half * slip_pct
                    running_ext     = mr["low"]
                    trail_sl        = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state           = "partial"

        elif state == "partial":
            if direction == "LONG":
                old_trail   = trail_sl
                running_ext = max(running_ext, mr["high"])
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                trail_hit   = mr["low"]  <= old_trail
                htp_hit     = mr["high"] >= hard_tp_px
                if trail_hit:
                    gross = partial_locked + (old_trail  - entry_px) * half
                    exit_px = old_trail;  closed = True
                elif htp_hit:
                    gross = partial_locked + (hard_tp_px - entry_px) * half
                    exit_px = hard_tp_px; closed = True
            else:
                old_trail   = trail_sl
                running_ext = min(running_ext, mr["low"])
                trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                trail_hit   = mr["high"] >= old_trail
                htp_hit     = mr["low"]  <= hard_tp_px
                if trail_hit:
                    gross = partial_locked + (entry_px - old_trail)  * half
                    exit_px = old_trail;  closed = True
                elif htp_hit:
                    gross = partial_locked + (entry_px - hard_tp_px) * half
                    exit_px = hard_tp_px; closed = True

        if closed:
            exit_qty   = qty if state == "full" else half
            exit_fee   = exit_px * exit_qty * fee_side
            exit_slip  = exit_px * exit_qty * slip_pct
            total_fee  = fee_acc  + exit_fee
            total_slip = slip_acc + exit_slip
            return (bar_ts, gross, exit_px, total_fee, total_slip, state)

    return None


# ─── Main backtest ────────────────────────────────────────────────────────────

def run_backtest_nse(symbol, df_1m, df_1h, df_daily):
    FEE_SIDE = NSE_FEE_RT / 2

    atr_1h  = calc_atr(df_1h)
    atr_d   = calc_atr(df_daily)

    # Daily EMA for trend filter (lagged by 1 day)
    df_daily["ema"]     = df_daily["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_daily["ema_lag"] = df_daily["ema"].shift(1)
    df_daily["ema_prev"]= df_daily["ema"].shift(1 + 3)   # 3 days back for slope

    # Forward-fill daily EMA onto 1h index
    ema_now_1h  = df_daily["ema_lag"].reindex(df_1h.index,  method="ffill")
    ema_prev_1h = df_daily["ema_prev"].reindex(df_1h.index, method="ffill")

    # Pre-build zone snapshots keyed by trading date (strict FIX 1)
    trading_dates = sorted(df_daily.index.normalize().unique())
    zone_snap = {}
    for d in trading_dates:
        past = df_daily[df_daily.index.normalize() < d]
        if len(past) < SWING_LOOKBACK * 2 + 5:
            zone_snap[d] = []
        else:
            zone_snap[d] = find_zones(past.iloc[-200:])

    # Pre-index 1m timestamps for searchsorted
    m_timestamps = df_1m.index

    # Entry cutoff time
    t_cutoff = pd.Timestamp(f"1970-01-01 {ENTRY_CUTOFF_H:02d}:{ENTRY_CUTOFF_M:02d}:00").time()

    balance      = INITIAL_BALANCE
    min_balance  = INITIAL_BALANCE * MIN_BAL_RATIO
    equity       = [balance]
    trades       = []
    zone_cool    = {}
    zone_touches = defaultdict(list)
    last_trig    = -999

    i    = 0
    n_1h = len(df_1h)

    while i < n_1h:

        if i < WARMUP_BARS:
            equity.append(balance)
            i += 1
            continue

        ts    = df_1h.index[i]
        row   = df_1h.iloc[i]
        atr   = atr_1h.iloc[i]
        price = row["close"]

        if atr <= 0 or np.isnan(atr):
            equity.append(balance)
            i += 1
            continue

        # Entry cutoff: no new entries after 14:00 IST
        if ts.time() >= t_cutoff:
            equity.append(balance)
            i += 1
            continue

        # Balance floor (FIX 7)
        if balance < min_balance:
            equity.append(balance)
            i += 1
            continue

        # Look up today's zone snapshot
        today = ts.normalize()
        zones = zone_snap.get(today, [])
        if not zones:
            equity.append(balance)
            i += 1
            continue

        # Nearest zone
        zones_below = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
        zones_above = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]
        if zones_below:
            near_zone = max(zones_below)
            near_dir  = "LONG"
        elif zones_above:
            near_zone = min(zones_above)
            near_dir  = "SHORT"
        else:
            equity.append(balance)
            i += 1
            continue

        # Cooldown guard
        if zone_cool.get(near_zone, 0) > i:
            equity.append(balance)
            i += 1
            continue
        if i == last_trig:
            equity.append(balance)
            i += 1
            continue

        # Trigger: price touched zone and closed back inside
        if near_dir == "LONG":
            triggered = (row["low"] <= near_zone * (1 + TOUCH_BUF) and row["close"] > near_zone)
        else:
            triggered = (row["high"] >= near_zone * (1 - TOUCH_BUF) and row["close"] < near_zone)
        if not triggered:
            equity.append(balance)
            i += 1
            continue

        # Filter 1: Volume spike
        vol_avg = df_1h["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
        if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
            equity.append(balance)
            i += 1
            continue

        # Filter 2: Daily EMA trend
        ema_now  = ema_now_1h.iloc[i]
        ema_prev = ema_prev_1h.iloc[i]
        if not (pd.isna(ema_now) or pd.isna(ema_prev)):
            if near_dir == "LONG"  and ema_now <= ema_prev:
                equity.append(balance)
                i += 1
                continue
            if near_dir == "SHORT" and ema_now >= ema_prev:
                equity.append(balance)
                i += 1
                continue

        # Filter 3: Zone freshness (max 2 touches per ZONE_WINDOW bars)
        recent = [b for b in zone_touches[near_zone] if i - b <= ZONE_WINDOW]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance)
            i += 1
            continue

        # Need next bar for entry price
        if i + 1 >= n_1h:
            equity.append(balance)
            i += 1
            continue

        # Skip if next bar is next day (gap risk) or after cutoff
        next_ts = df_1h.index[i + 1]
        if next_ts.date() != ts.date():
            equity.append(balance)
            i += 1
            continue

        # Position sizing
        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance)
            i += 1
            continue

        eff_bal   = min(balance, MAX_BALANCE_INR)
        risk_inr  = eff_bal * RISK_PCT
        qty       = risk_inr / sl_dist
        entry_px  = df_1h["open"].iloc[i + 1]
        trade_atr = atr

        # Cap position notional
        if entry_px > 0:
            qty = min(qty, MAX_POSITION_INR / entry_px)
        if qty <= 0:
            equity.append(balance)
            i += 1
            continue

        if near_dir == "LONG":
            sl_px         = entry_px - SL_MULT         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT         * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        entry_ts = df_1h.index[i + 1]
        m_start  = int(m_timestamps.searchsorted(entry_ts))

        close_info = _exec_1m_nse(df_1m, m_start, entry_px, near_dir,
                                   sl_px, partial_tp_px, hard_tp_px,
                                   trade_atr, qty, FEE_SIDE, NSE_SLIP_PCT)
        if close_info is None:
            equity.append(balance)
            i += 1
            continue

        close_ts, gross, exit_px, total_fee, total_slip, final_state = close_info
        close_1h_idx = int(df_1h.index.searchsorted(close_ts, side="right")) - 1
        close_1h_idx = max(close_1h_idx, i + 1)
        close_1h_idx = min(close_1h_idx, n_1h - 1)

        net      = gross - total_fee - total_slip
        result   = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        zone_cool[near_zone]  = close_1h_idx + (COOLDOWN_LOSS if result == "LOSS"
                                                  else COOLDOWN_WIN)
        zone_touches[near_zone].append(i)
        last_trig = i

        trades.append({
            "ts":           ts,
            "close_ts":     close_ts,
            "dir":          near_dir,
            "entry":        round(entry_px,      4),
            "exit":         round(exit_px,       4),
            "qty":          round(qty,            4),
            "notional":     round(entry_px * qty, 2),
            "gross":        round(gross,           4),
            "fee":          round(total_fee,       4),
            "slip":         round(total_slip,      4),
            "net":          round(net,             4),
            "result":       result,
            "balance_open": round(bal_open,        4),
            "balance":      round(balance,         4),
        })

        equity.append(bal_open)
        for _ in range(close_1h_idx - i - 1):
            equity.append(bal_open)
        equity.append(balance)

        i = close_1h_idx + 1

    # ── Statistics ────────────────────────────────────────────────────────────
    if not trades:
        return _empty_nse(symbol, equity, df_1h)

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]

    gross_pnl    = df_t["gross"].sum()
    total_fees   = df_t["fee"].sum()
    total_slip   = df_t["slip"].sum()
    total_costs  = total_fees + total_slip
    net_pnl      = df_t["net"].sum()

    gross_wins = wins["gross"].sum()
    gross_loss = abs(loses["gross"].sum())
    net_wins   = wins["net"].sum()
    net_loss   = abs(loses["net"].sum())

    gross_pf = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    net_pf   = round(net_wins   / net_loss,   3) if net_loss   > 0 else float("inf")
    win_rate = round(len(wins) / len(df_t) * 100, 1)

    final_bal = balance
    years     = (df_1h.index[-1] - df_1h.index[0]).days / 365.25
    total_ret = (final_bal / INITIAL_BALANCE - 1) * 100
    cagr      = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0

    eq       = pd.Series(equity)
    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    max_dd   = round(dd_pct.min(), 2)
    max_dd_inr = round((eq - roll_max).min(), 2)

    dd_dur_max = dd_cur = 0
    for v in (dd_pct < 0).astype(int):
        dd_cur = dd_cur + 1 if v else 0
        dd_dur_max = max(dd_dur_max, dd_cur)

    tpy      = len(df_t) / max(years, 0.1)
    pct_rets = (df_t["net"] / df_t["balance_open"]).values
    sharpe   = round((pct_rets.mean() / pct_rets.std() * np.sqrt(max(tpy, 1)))
                     if pct_rets.std() > 0 else 0, 2)
    calmar   = round(cagr / abs(max_dd) if max_dd != 0 else 0, 2)

    max_ws = max_ls = cur_ws = cur_ls = 0
    for r in df_t["result"]:
        if r == "WIN":   cur_ws += 1; cur_ls = 0
        else:            cur_ls += 1; cur_ws = 0
        max_ws = max(max_ws, cur_ws)
        max_ls = max(max_ls, cur_ls)

    return {
        "symbol":       symbol,
        "date_from":    df_1h.index[0].date(),
        "date_to":      df_1h.index[-1].date(),
        "years":        round(years, 2),
        "trades":       len(df_t),
        "wins":         len(wins),
        "losses":       len(loses),
        "win_rate":     win_rate,
        "gross_pf":     gross_pf,
        "net_pf":       net_pf,
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
        "max_dd_inr":   max_dd_inr,
        "dd_dur_bars":  dd_dur_max,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "avg_win":      round(wins["net"].mean(),  2) if len(wins)  else 0,
        "avg_loss":     round(loses["net"].mean(), 2) if len(loses) else 0,
        "best_trade":   round(df_t["net"].max(),   2),
        "worst_trade":  round(df_t["net"].min(),   2),
        "max_win_str":  max_ws,
        "max_los_str":  max_ls,
        "equity":       equity,
        "trades_df":    df_t,
    }


def _empty_nse(symbol, equity, df_1h):
    years = (df_1h.index[-1] - df_1h.index[0]).days / 365.25 if len(df_1h) > 1 else 0
    return {
        "symbol": symbol, "trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "gross_pf": 0, "net_pf": 0, "gross_pnl": 0,
        "total_fees": 0, "total_slip": 0, "total_costs": 0, "net_pnl": 0,
        "initial_bal": INITIAL_BALANCE, "final_bal": INITIAL_BALANCE,
        "total_ret": 0, "cagr": 0, "max_dd": 0, "max_dd_inr": 0,
        "dd_dur_bars": 0, "sharpe": 0, "calmar": 0,
        "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
        "max_win_str": 0, "max_los_str": 0, "years": round(years, 2),
        "date_from": df_1h.index[0].date() if len(df_1h) else "",
        "date_to":   df_1h.index[-1].date() if len(df_1h) else "",
        "equity": equity, "trades_df": pd.DataFrame(),
    }


# ─── HTML report ──────────────────────────────────────────────────────────────

def generate_html_report(results: list, run_time: str) -> str:
    # Filter to stocks that had trades
    active = [r for r in results if r["trades"] > 0]
    skipped = len(results) - len(active)

    if not active:
        return "<html><body><h1>No trades generated.</h1></body></html>"

    # Aggregate stats
    all_nets    = [r["net_pnl"]   for r in active]
    all_cagr    = [r["cagr"]      for r in active]
    all_wr      = [r["win_rate"]  for r in active]
    all_netpf   = [r["net_pf"]    for r in active if r["net_pf"] != float("inf")]
    all_sharpe  = [r["sharpe"]    for r in active]
    all_dd      = [r["max_dd"]    for r in active]
    total_trades= sum(r["trades"] for r in active)
    total_wins  = sum(r["wins"]   for r in active)
    avg_cagr    = round(sum(all_cagr)   / len(active), 1)
    avg_wr      = round(sum(all_wr)     / len(active), 1)
    avg_netpf   = round(sum(all_netpf)  / len(all_netpf), 3) if all_netpf else 0
    avg_sharpe  = round(sum(all_sharpe) / len(active), 2)
    avg_dd      = round(sum(all_dd)     / len(active), 1)
    profitable  = sum(1 for r in active if r["net_pnl"] > 0)

    # Sort by CAGR descending
    active_sorted = sorted(active, key=lambda r: r["cagr"], reverse=True)

    # Top 10 equity curves (by CAGR)
    top10 = active_sorted[:10]
    equity_datasets = []
    for r in top10:
        eq = r["equity"]
        step = max(1, len(eq) // 200)
        sampled = eq[::step]
        equity_datasets.append({
            "label": r["symbol"],
            "data":  [round(v, 2) for v in sampled],
        })

    # Per-symbol table rows
    sym_rows = ""
    for rank, r in enumerate(active_sorted, 1):
        cagr_c = "pos" if r["cagr"] >= 0 else "neg"
        pnl_c  = "pos" if r["net_pnl"] >= 0 else "neg"
        dd_cls = "neg" if r["max_dd"] < -20 else "warn" if r["max_dd"] < -10 else ""
        sym_rows += f"""
        <tr>
          <td>{rank}</td>
          <td><strong>{r['symbol']}</strong></td>
          <td>{r['trades']}</td>
          <td>{r['win_rate']}%</td>
          <td>{r['gross_pf']}</td>
          <td>{r['net_pf']}</td>
          <td class="{pnl_c}">{'+'if r['net_pnl']>=0 else ''}INR {r['net_pnl']:,.0f}</td>
          <td class="{cagr_c}">{'+'if r['cagr']>=0 else ''}{r['cagr']:.1f}%</td>
          <td class="{dd_cls}">{r['max_dd']:.1f}%</td>
          <td>{r['sharpe']}</td>
          <td>{r['calmar']}</td>
          <td>{r['max_win_str']} / {r['max_los_str']}</td>
          <td style="color:#64748b;font-size:11px">{r['date_from']} — {r['date_to']}</td>
        </tr>"""

    ds_json = json.dumps(equity_datasets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NSE Backtest Report — LSCO v7</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0a0a0f; --card:#111118; --border:#1e1e2e; --text:#e2e8f0;
    --muted:#64748b; --gold:#f59e0b; --green:#22c55e; --red:#ef4444;
    --blue:#60a5fa; --purple:#a78bfa; --warn:#f97316;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif; font-size:13px; }}
  .header {{ background:linear-gradient(135deg,#0f0f1e,#1a1a2e); padding:36px 48px 28px;
             border-bottom:1px solid var(--border); }}
  .logo {{ font-size:28px; font-weight:800; color:var(--gold); letter-spacing:2px; }}
  .logo span {{ color:var(--text); font-weight:400; }}
  .header-sub {{ color:var(--muted); margin-top:6px; font-size:13px; }}
  .header-meta {{ font-size:12px; color:var(--muted); line-height:1.8; text-align:right; }}
  .header-top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
  .live-badge {{ display:inline-flex; align-items:center; gap:6px; background:#16213e;
                 border:1px solid #1e3a5f; border-radius:20px; padding:4px 12px;
                 color:var(--blue); font-size:11px; font-weight:700; }}
  .container {{ max-width:1600px; margin:0 auto; padding:32px 40px; }}
  .section {{ margin-bottom:36px; }}
  .section-title {{ font-size:12px; font-weight:700; color:var(--muted); letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:16px; padding-bottom:8px;
                    border-bottom:1px solid var(--border); }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
  .kpi-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
               padding:20px 22px; }}
  .kpi-card.gold  {{ border-left:3px solid var(--gold);   }}
  .kpi-card.green {{ border-left:3px solid var(--green);  }}
  .kpi-card.blue  {{ border-left:3px solid var(--blue);   }}
  .kpi-card.purple{{ border-left:3px solid var(--purple); }}
  .kpi-card.warn  {{ border-left:3px solid var(--warn);   }}
  .kpi-label {{ font-size:11px; color:var(--muted); text-transform:uppercase;
                letter-spacing:1px; margin-bottom:8px; }}
  .kpi-value {{ font-size:28px; font-weight:800; line-height:1; }}
  .kpi-sub   {{ font-size:11px; color:var(--muted); margin-top:8px; }}
  .chart-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
                 padding:24px; }}
  .chart-wrap {{ position:relative; height:320px; }}
  table {{ width:100%; border-collapse:collapse; }}
  thead tr {{ background:#0d0d1a; }}
  th {{ padding:10px 14px; text-align:left; font-size:10px; text-transform:uppercase;
        letter-spacing:1px; color:var(--muted); font-weight:600; white-space:nowrap; }}
  td {{ padding:9px 14px; border-bottom:1px solid #0d0d1a; }}
  tbody tr:hover {{ background:#16162a; }}
  .pos  {{ color:var(--green); }}
  .neg  {{ color:var(--red);   }}
  .warn {{ color:var(--warn);  }}
  .tbl-wrap {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
               overflow:auto; max-height:600px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }}
  .stat-box {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
               padding:16px 18px; }}
  .stat-label {{ font-size:10px; color:var(--muted); text-transform:uppercase;
                 letter-spacing:1px; margin-bottom:6px; }}
  .stat-value {{ font-size:18px; font-weight:700; }}
  .disclaimer {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                 padding:16px 20px; font-size:11px; color:var(--muted); line-height:1.8; }}
  @media(max-width:1100px) {{
    .kpi-grid {{ grid-template-columns:repeat(2,1fr); }}
    .stat-grid {{ grid-template-columns:repeat(2,1fr); }}
    .container {{ padding:16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div>
      <div class="logo">LSCO<span> Trading</span></div>
      <div class="header-sub">Liquidation Zone Reversal v7 — NSE Equities Backtest Report</div>
    </div>
    <div class="header-meta">
      <span class="live-badge">BACKTEST REPORT</span><br><br>
      <strong>Strategy:</strong> LZR v7 (1m Execution)<br>
      <strong>Universe:</strong> {len(results)} NSE Stocks tested<br>
      <strong>Period:</strong> 2017 — 2021 (4 yrs)<br>
      <strong>Generated:</strong> {run_time}
    </div>
  </div>
</div>

<div class="container">

  <!-- KPI CARDS -->
  <div class="section">
    <div class="section-title">Portfolio Summary — All {len(active)} Active Stocks</div>
    <div class="kpi-grid">
      <div class="kpi-card gold">
        <div class="kpi-label">Avg CAGR / Stock</div>
        <div class="kpi-value" style="color:{'var(--green)' if avg_cagr>=0 else 'var(--red)'}">
          {'+'if avg_cagr>=0 else ''}{avg_cagr:.1f}%
        </div>
        <div class="kpi-sub">Per year, compounding on INR 1,00,000</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Avg Win Rate</div>
        <div class="kpi-value" style="color:var(--green)">{avg_wr}%</div>
        <div class="kpi-sub">{total_wins:,} wins / {total_trades - total_wins:,} losses / {total_trades:,} total</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Avg Net Profit Factor</div>
        <div class="kpi-value" style="color:var(--blue)">{avg_netpf}</div>
        <div class="kpi-sub">After 0.10% round-trip fees + 0.05% slippage</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Avg Sharpe Ratio</div>
        <div class="kpi-value" style="color:var(--purple)">{avg_sharpe}</div>
        <div class="kpi-sub">Trade-frequency annualised</div>
      </div>
      <div class="kpi-card warn">
        <div class="kpi-label">Profitable Stocks</div>
        <div class="kpi-value" style="color:var(--warn)">{profitable} / {len(active)}</div>
        <div class="kpi-sub">{round(profitable/len(active)*100,1)}% of tested universe</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Avg Max Drawdown</div>
        <div class="kpi-value" style="color:var(--red)">{avg_dd:.1f}%</div>
        <div class="kpi-sub">Average across all stocks</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value" style="color:var(--text)">{total_trades:,}</div>
        <div class="kpi-sub">Across {len(active)} stocks, ~4 years</div>
      </div>
      <div class="kpi-card gold">
        <div class="kpi-label">Skipped Stocks</div>
        <div class="kpi-value" style="color:var(--muted)">{skipped}</div>
        <div class="kpi-sub">Insufficient data or no signals</div>
      </div>
    </div>
  </div>

  <!-- ADDITIONAL STATS -->
  <div class="section">
    <div class="section-title">Strategy Parameters (NSE Edition)</div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-label">Starting Capital</div>
        <div class="stat-value">INR 1,00,000</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Risk per Trade</div>
        <div class="stat-value">2% of balance</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">SL / Partial TP</div>
        <div class="stat-value">0.75× / 1.0× ATR</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Trail / Hard Cap</div>
        <div class="stat-value">0.5× / 3.0× ATR</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Round-trip Fee</div>
        <div class="stat-value">0.10% + 0.05% slip</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Max Trade Notional</div>
        <div class="stat-value">INR 50,000</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Entry Cutoff</div>
        <div class="stat-value">14:00 IST</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">EOD Force Close</div>
        <div class="stat-value">15:20 IST</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Zone Detection</div>
        <div class="stat-value">Daily swing H/L</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Trend Filter</div>
        <div class="stat-value">20-day EMA slope</div>
      </div>
    </div>
  </div>

  <!-- TOP 10 EQUITY CURVES -->
  <div class="section">
    <div class="section-title">Top 10 Stocks — Balance Curve (INR, compounding)</div>
    <div class="chart-card">
      <div class="chart-wrap">
        <canvas id="eqChart"></canvas>
      </div>
    </div>
  </div>

  <!-- PER-SYMBOL TABLE -->
  <div class="section">
    <div class="section-title">All Stocks — Results (sorted by CAGR)</div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Symbol</th><th>Trades</th><th>Win Rate</th>
            <th>Gross PF</th><th>Net PF</th><th>Net PnL (INR)</th>
            <th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Calmar</th>
            <th>W/L Streak</th><th>Period</th>
          </tr>
        </thead>
        <tbody>{sym_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- DISCLAIMER -->
  <div class="section">
    <div class="disclaimer">
      <strong>Disclaimer:</strong>
      This is a historical backtest using the Liquidation Zone Reversal (LZR) strategy on NSE equity 1-minute OHLCV data (2017–2021).
      Results assume intraday-only trading (all positions closed by 15:20 IST), realistic costs (0.10% round-trip brokerage + STT + exchange charges, 0.05% slippage per fill),
      and no leverage. Past performance is not indicative of future results. Markets change, and strategies that worked historically may underperform in live trading.
      This report is for informational purposes only and does not constitute investment advice.
    </div>
  </div>

</div>

<script>
const datasets = {ds_json};
const palette = [
  '#f59e0b','#22c55e','#60a5fa','#a78bfa','#f97316',
  '#ec4899','#14b8a6','#84cc16','#ef4444','#6366f1'
];

new Chart(document.getElementById('eqChart'), {{
  type: 'line',
  data: {{
    labels: Array.from({{length: Math.max(...datasets.map(d=>d.data.length))}}, (_,i)=>i+1),
    datasets: datasets.map((ds,i) => ({{
      label: ds.label,
      data: ds.data,
      borderColor: palette[i % palette.length],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    }}))
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', font:{{ size:11 }} }} }},
      tooltip: {{ mode:'index', intersect:false,
        callbacks: {{ label: ctx => ' ' + ctx.dataset.label + ': INR ' + ctx.parsed.y.toLocaleString('en-IN', {{maximumFractionDigits:0}}) }} }}
    }},
    scales: {{
      x: {{ display:false }},
      y: {{
        grid: {{ color:'#1e1e2e' }},
        ticks: {{ color:'#64748b', callback: v => 'INR ' + (v/1000).toFixed(0) + 'K' }},
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    files   = get_all_files()
    results = []
    t_start = datetime.now()

    print("=" * 72)
    print("  LIQUIDATION ZONE REVERSAL v7 — NSE STOCKS BACKTEST")
    print(f"  {len(files)} stocks found in {DATA_DIR}")
    print(f"  Starting capital : INR {INITIAL_BALANCE:,.0f}")
    print(f"  Risk/trade       : {RISK_PCT*100:.0f}%")
    print(f"  Costs            : {NSE_FEE_RT*100:.2f}% fee + {NSE_SLIP_PCT*100:.2f}% slip")
    print(f"  Entry cutoff     : {ENTRY_CUTOFF_H:02d}:{ENTRY_CUTOFF_M:02d} IST")
    print(f"  EOD force-close  : {EOD_FORCE_H:02d}:{EOD_FORCE_M:02d} IST")
    print("=" * 72)
    print()

    for idx, fpath in enumerate(files, 1):
        sym_t0 = datetime.now()
        symbol, df_1m, df_1h, df_daily = load_nse_stock(fpath)

        if df_1m is None:
            print(f"  [{idx:3d}/{len(files)}] {symbol:<20} SKIP (insufficient data)")
            continue

        print(f"  [{idx:3d}/{len(files)}] {symbol:<20} "
              f"{len(df_1m):>7,} 1m  {len(df_1h):>5,} 1h  "
              f"{df_1h.index[0].date()} -> {df_1h.index[-1].date()}  ...", end="", flush=True)

        r = run_backtest_nse(symbol, df_1m, df_1h, df_daily)
        elapsed = (datetime.now() - sym_t0).seconds
        results.append(r)

        if r["trades"] == 0:
            print(f"  no trades  ({elapsed}s)")
        else:
            print(f"  {r['trades']:>4} trades  WR {r['win_rate']:.0f}%  "
                  f"NetPF {r['net_pf']:.2f}  CAGR {r['cagr']:+.1f}%  "
                  f"Final INR {r['final_bal']:>9,.0f}  ({elapsed}s)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ("equity", "trades_df")}
                            for r in results])
    csv_path = OUTPUT_DIR / "backtest_v7_NSE_full.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    active = [r for r in results if r["trades"] > 0]
    if active:
        avg_cagr   = sum(r["cagr"]     for r in active) / len(active)
        avg_wr     = sum(r["win_rate"] for r in active) / len(active)
        netpfs     = [r["net_pf"] for r in active if r["net_pf"] != float("inf")]
        avg_netpf  = sum(netpfs) / len(netpfs) if netpfs else 0
        avg_sharpe = sum(r["sharpe"]   for r in active) / len(active)
        avg_dd     = sum(r["max_dd"]   for r in active) / len(active)
        profitable = sum(1 for r in active if r["net_pnl"] > 0)

        print()
        print("=" * 72)
        print("  AGGREGATE RESULTS")
        print("=" * 72)
        print(f"  Stocks tested     : {len(results)}")
        print(f"  Stocks with trades: {len(active)}")
        print(f"  Profitable stocks : {profitable} / {len(active)} ({profitable/len(active)*100:.1f}%)")
        print(f"  Avg CAGR          : {avg_cagr:+.1f}% / yr")
        print(f"  Avg Win Rate      : {avg_wr:.1f}%")
        print(f"  Avg Net PF        : {avg_netpf:.3f}")
        print(f"  Avg Sharpe        : {avg_sharpe:.2f}")
        print(f"  Avg Max DD        : {avg_dd:.1f}%")
        print(f"  Total trades      : {sum(r['trades'] for r in active):,}")
        print("=" * 72)

        # Top 10 by CAGR
        top10 = sorted(active, key=lambda r: r["cagr"], reverse=True)[:10]
        print("\n  TOP 10 BY CAGR")
        print(f"  {'Rank':<5} {'Symbol':<20} {'CAGR':>9} {'WR':>7} {'NetPF':>7} "
              f"{'Sharpe':>8} {'MaxDD':>8} {'Final INR':>12}")
        print("  " + "-" * 78)
        for rank, r in enumerate(top10, 1):
            print(f"  {rank:<5} {r['symbol']:<20} {r['cagr']:>+8.1f}% "
                  f"{r['win_rate']:>6.1f}% {r['net_pf']:>7.3f} "
                  f"{r['sharpe']:>8.2f} {r['max_dd']:>7.1f}% "
                  f"{r['final_bal']:>12,.0f}")
        print()

    # ── HTML report ───────────────────────────────────────────────────────────
    run_time  = t_start.strftime("%Y-%m-%d %H:%M")
    html      = generate_html_report(results, run_time)
    html_path = OUTPUT_DIR / "NSE_Investor_Report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML report → {html_path}")
    print(f"  Open in browser → Print → Save as PDF\n")
    elapsed_total = (datetime.now() - t_start).seconds
    print(f"  Total run time : {elapsed_total // 60}m {elapsed_total % 60}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
