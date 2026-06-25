"""
backtest_v7_NSE_portfolio.py  --  LZR v7  |  NSE  |  PORTFOLIO MODE
=====================================================================
Answers: "If I put ₹1 lakh total, how much do I make through compounding
         running multiple trades simultaneously across all NSE stocks?"

KEY DIFFERENCES FROM backtest_v7_NSE.py
  1. SHARED ₹1L  : single portfolio balance, not ₹1L per stock
  2. MULTI-TRADE  : up to MAX_CONCURRENT=10 positions open simultaneously
  3. RISK-BASED   : 2% of PORTFOLIO per trade (scales with growth)
  4. NO NOTIONAL CAP : uses 3× intraday leverage (standard NSE margin)
  5. 5m DATA      : 2015→2026 (11 years) instead of 1m 2017-2021
  6. 5m EXECUTION : SL/TP/trail evaluated on 5-minute bars
  7. TRUE COMPOUND: each win grows the portfolio → next trade is bigger

REALISTIC COSTS
  Fee   : 0.10% round-trip (brokerage + STT + exchange + GST)
  Slip  : 0.05% per fill (entry + partial exit + final exit)
"""

import sys, warnings, json, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR_5M = Path(r"C:\Users\GIGA\Downloads\NSE data 5m")
OUTPUT_DIR  = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Portfolio config ─────────────────────────────────────────────────────────
INITIAL_BALANCE   = 1_00_000.0   # ₹1 lakh starting capital (SHARED)
RISK_PCT          = 0.02          # 2% risk per trade on portfolio balance
MAX_LEVERAGE      = 3.0           # max 3× intraday leverage (notional ≤ 3×balance)
MAX_CONCURRENT    = 10            # max simultaneous open positions
MAX_BALANCE_CAP   = 50_00_000.0  # ₹50L ceiling (prevents runaway compounding)
MIN_BAL_RATIO     = 0.20          # halt entries below 20% of initial

# ─── Strategy ─────────────────────────────────────────────────────────────────
SL_MULT          = 0.75
PARTIAL_TP_MULT  = 1.0
TRAIL_DIST_MULT  = 0.5
HARD_TP_MULT     = 3.0
APPROACH_PCT     = 0.012
TOUCH_BUF        = 0.005
ATR_PERIOD       = 14
SWING_LOOKBACK   = 10
MIN_ZONE_GAP     = 0.015
VOL_MULT         = 1.5
VOL_LOOKBACK     = 20
EMA_PERIOD       = 20
ZONE_MAX_TOUCH   = 2
ZONE_WINDOW      = 40
COOLDOWN_LOSS    = 3
COOLDOWN_WIN     = 1
WARMUP_BARS      = 56

# ─── NSE session times ────────────────────────────────────────────────────────
EOD_FORCE_H,    EOD_FORCE_M    = 15, 20
ENTRY_CUTOFF_H, ENTRY_CUTOFF_M = 14,  0

# ─── NSE costs ────────────────────────────────────────────────────────────────
NSE_FEE_RT   = 0.001
NSE_SLIP_PCT = 0.0005
FEE_SIDE     = NSE_FEE_RT / 2

import time as _time
T_EOD     = None   # set once at startup
T_CUTOFF  = None


def _init_times():
    global T_EOD, T_CUTOFF
    import datetime as _dt
    T_EOD     = _dt.time(EOD_FORCE_H,    EOD_FORCE_M)
    T_CUTOFF  = _dt.time(ENTRY_CUTOFF_H, ENTRY_CUTOFF_M)


# ─── Indicators ───────────────────────────────────────────────────────────────

def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_d):
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


# ─── 5m executor (EOD-aware) ──────────────────────────────────────────────────

def _exec_5m(df_5m, m_start, entry_px, direction,
             sl_px, partial_tp_px, hard_tp_px,
             trade_atr, qty, fee_side, slip_pct):
    half           = qty / 2
    state          = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct

    n = len(df_5m)
    for m_idx in range(m_start, n):
        mr     = df_5m.iloc[m_idx]
        bar_ts = df_5m.index[m_idx]
        gross  = None
        closed = False
        exit_px = 0.0

        # EOD forced close
        if bar_ts.time() >= T_EOD:
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
                if mr["low"] <= old_trail:
                    gross = partial_locked + (old_trail  - entry_px) * half
                    exit_px = old_trail;  closed = True
                elif mr["high"] >= hard_tp_px:
                    gross = partial_locked + (hard_tp_px - entry_px) * half
                    exit_px = hard_tp_px; closed = True
            else:
                old_trail   = trail_sl
                running_ext = min(running_ext, mr["low"])
                trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                if mr["high"] >= old_trail:
                    gross = partial_locked + (entry_px - old_trail)  * half
                    exit_px = old_trail;  closed = True
                elif mr["low"] <= hard_tp_px:
                    gross = partial_locked + (entry_px - hard_tp_px) * half
                    exit_px = hard_tp_px; closed = True

        if closed:
            exit_qty   = qty if state == "full" else half
            total_fee  = fee_acc  + exit_px * exit_qty * fee_side
            total_slip = slip_acc + exit_px * exit_qty * slip_pct
            net_pnl    = gross - total_fee - total_slip
            return {
                "exit_ts":    bar_ts,
                "gross":      gross,
                "exit_px":    exit_px,
                "total_fee":  total_fee,
                "total_slip": total_slip,
                "net_pnl":    net_pnl,
                "final_state": state,
            }

    return None


# ─── Per-stock state ──────────────────────────────────────────────────────────

class StockState:
    __slots__ = (
        "symbol", "df_5m", "df_1h", "atr_1h",
        "ema_now_1h", "ema_prev_1h", "zone_snap",
        "ts_to_idx", "zone_cool", "zone_touches", "last_trig",
    )

    def __init__(self, symbol, df_5m, df_1h, df_daily):
        self.symbol  = symbol
        self.df_5m   = df_5m
        self.df_1h   = df_1h
        self.atr_1h  = calc_atr(df_1h)

        df_daily      = df_daily.copy()
        df_daily["ema"]      = df_daily["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        df_daily["ema_lag"]  = df_daily["ema"].shift(1)
        df_daily["ema_prev"] = df_daily["ema"].shift(4)   # 3 days back for slope

        self.ema_now_1h  = df_daily["ema_lag"].reindex(df_1h.index,  method="ffill")
        self.ema_prev_1h = df_daily["ema_prev"].reindex(df_1h.index, method="ffill")

        # Zone snapshots keyed by normalized date
        trading_dates = sorted(df_daily.index.normalize().unique())
        self.zone_snap = {}
        for d in trading_dates:
            past = df_daily[df_daily.index.normalize() < d]
            if len(past) < SWING_LOOKBACK * 2 + 5:
                self.zone_snap[d] = []
            else:
                self.zone_snap[d] = find_zones(past.iloc[-200:])

        # Fast ts → bar_idx lookup
        self.ts_to_idx = {ts: i for i, ts in enumerate(df_1h.index)}

        # Trade state (per-stock)
        self.zone_cool    = {}
        self.zone_touches = defaultdict(list)
        self.last_trig    = -999


def load_stock(filepath: Path):
    """Load one NSE 5m CSV. Returns (symbol, df_5m, df_1h, df_daily) or None on failure."""
    symbol = filepath.stem.replace("_5minute", "")
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.rename(columns={"date": "ts"})
        df = df.set_index("ts").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].rename(
            columns={"volume": "vol"}
        ).apply(pd.to_numeric, errors="coerce").dropna()

        import datetime as _dt
        t_open  = _dt.time(9,  15)
        t_close = _dt.time(15, 29)
        mask    = (df.index.time >= t_open) & (df.index.time <= t_close)
        df_5m   = df[mask].copy()

        if len(df_5m) < 5000:
            return symbol, None, None, None

        df_1h = df_5m.resample("1h").agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"),   close=("close","last"), vol=("vol","sum")
        ).dropna()

        df_daily = df_5m.resample("1D").agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"),   close=("close","last"), vol=("vol","sum")
        ).dropna()

        return symbol, df_5m, df_1h, df_daily

    except Exception as e:
        return symbol, None, None, None


# ─── Signal detection ─────────────────────────────────────────────────────────

def detect_signal(st: StockState, ts, bar_idx: int):
    """
    Returns dict with signal info, or None.
    Does NOT update zone_touches / zone_cool (those are updated only on accepted entry).
    """
    if bar_idx < WARMUP_BARS:
        return None
    if ts.time() >= T_CUTOFF:
        return None

    row   = st.df_1h.iloc[bar_idx]
    atr   = st.atr_1h.iloc[bar_idx]
    price = row["close"]

    if atr <= 0 or np.isnan(atr):
        return None

    # Zone lookup
    today = pd.Timestamp(ts.date())
    zones = st.zone_snap.get(today, [])
    if not zones:
        return None

    # Nearest zone approach
    z_below = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
    z_above = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]
    if z_below:
        near_zone = max(z_below); near_dir = "LONG"
    elif z_above:
        near_zone = min(z_above); near_dir = "SHORT"
    else:
        return None

    # Cooldown guard
    if st.zone_cool.get(near_zone, 0) > bar_idx:
        return None
    if bar_idx == st.last_trig:
        return None

    # Trigger: touched zone and close back inside
    if near_dir == "LONG":
        triggered = (row["low"] <= near_zone * (1 + TOUCH_BUF) and row["close"] > near_zone)
    else:
        triggered = (row["high"] >= near_zone * (1 - TOUCH_BUF) and row["close"] < near_zone)
    if not triggered:
        return None

    # Filter 1: volume spike
    vol_avg = st.df_1h["vol"].iloc[max(0, bar_idx - VOL_LOOKBACK): bar_idx].mean()
    if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
        return None

    # Filter 2: EMA trend
    ema_now  = st.ema_now_1h.iloc[bar_idx]
    ema_prev = st.ema_prev_1h.iloc[bar_idx]
    if not (pd.isna(ema_now) or pd.isna(ema_prev)):
        if near_dir == "LONG"  and ema_now <= ema_prev: return None
        if near_dir == "SHORT" and ema_now >= ema_prev: return None

    # Filter 3: zone freshness
    recent = [b for b in st.zone_touches[near_zone] if bar_idx - b <= ZONE_WINDOW]
    if len(recent) >= ZONE_MAX_TOUCH:
        return None

    # Need a next bar
    if bar_idx + 1 >= len(st.df_1h):
        return None

    # Don't cross day boundary
    next_ts = st.df_1h.index[bar_idx + 1]
    if next_ts.date() != ts.date():
        return None

    return {
        "near_zone": near_zone,
        "near_dir":  near_dir,
        "atr":       atr,
        "bar_idx":   bar_idx,
        "ts":        ts,
    }


# ─── Portfolio simulation ──────────────────────────────────────────────────────

def run_portfolio(all_states: dict, verbose: bool = True) -> dict:
    """
    Event-driven portfolio simulation over a unified 1h timeline.
    Returns portfolio stats dict.
    """
    min_balance   = INITIAL_BALANCE * MIN_BAL_RATIO
    balance       = INITIAL_BALANCE
    open_pos      = {}     # symbol → {exit_ts, net_pnl, ...trade info}
    closed_trades = []
    equity_ts     = []
    equity_val    = []

    # Unified sorted 1h timeline across all stocks
    all_ts = sorted(set().union(*[set(st.df_1h.index) for st in all_states.values()]))
    n_ts   = len(all_ts)

    if verbose:
        print(f"\n  Unified timeline: {all_ts[0]} -> {all_ts[-1]}  ({n_ts:,} 1h bars)")
        print(f"  Running portfolio simulation...")

    # ── Main loop ────────────────────────────────────────────────────────────
    report_every = n_ts // 20  # print progress 20 times

    for t_idx, ts in enumerate(all_ts):

        if verbose and report_every and t_idx % report_every == 0:
            pct = t_idx / n_ts * 100
            print(f"    {pct:5.1f}%  ts={ts.date()}  "
                  f"bal=INR {balance:>10,.0f}  "
                  f"open={len(open_pos)}/{MAX_CONCURRENT}  "
                  f"trades={len(closed_trades)}", flush=True)

        # Step 1: Close positions whose exit has passed
        to_close = [sym for sym, pos in open_pos.items() if pos["exit_ts"] <= ts]
        for sym in to_close:
            pos = open_pos.pop(sym)
            balance += pos["net_pnl"]
            closed_trades.append(pos)

            # Update cooldown state on the stock (post-close)
            st = all_states[sym]
            result = "WIN" if pos["net_pnl"] > 0 else "LOSS"
            close_bar_idx = st.ts_to_idx.get(
                st.df_1h.index[st.df_1h.index.searchsorted(pos["exit_ts"], side="right") - 1],
                pos["open_bar_idx"]
            )
            cooldown_bars = COOLDOWN_LOSS if result == "LOSS" else COOLDOWN_WIN
            st.zone_cool[pos["near_zone"]] = close_bar_idx + cooldown_bars

        equity_ts.append(ts)
        equity_val.append(balance)

        # Step 2: Try to open new positions
        if balance < min_balance:
            continue
        if len(open_pos) >= MAX_CONCURRENT:
            continue

        eff_bal = min(balance, MAX_BALANCE_CAP)

        # Shuffle stocks each bar to avoid systematic ordering bias
        syms = list(all_states.keys())
        np.random.shuffle(syms)

        for sym in syms:
            if len(open_pos) >= MAX_CONCURRENT:
                break
            if sym in open_pos:
                continue

            st       = all_states[sym]
            bar_idx  = st.ts_to_idx.get(ts)
            if bar_idx is None:
                continue

            sig = detect_signal(st, ts, bar_idx)
            if sig is None:
                continue

            # Position sizing (risk-based, with leverage cap)
            atr       = sig["atr"]
            sl_dist   = SL_MULT * atr
            if sl_dist <= 0:
                continue

            risk_inr  = eff_bal * RISK_PCT
            entry_px  = st.df_1h["open"].iloc[bar_idx + 1]
            if entry_px <= 0:
                continue

            qty = risk_inr / sl_dist
            qty = min(qty, MAX_LEVERAGE * eff_bal / entry_px)
            qty = max(qty, 1.0)
            qty = float(int(qty))   # whole shares only

            near_dir = sig["near_dir"]
            if near_dir == "LONG":
                sl_px         = entry_px - SL_MULT         * atr
                partial_tp_px = entry_px + PARTIAL_TP_MULT * atr
                hard_tp_px    = entry_px + HARD_TP_MULT    * atr
            else:
                sl_px         = entry_px + SL_MULT         * atr
                partial_tp_px = entry_px - PARTIAL_TP_MULT * atr
                hard_tp_px    = entry_px - HARD_TP_MULT    * atr

            # Execute on 5m
            entry_ts  = st.df_1h.index[bar_idx + 1]
            m_start   = int(st.df_5m.index.searchsorted(entry_ts))
            close_info = _exec_5m(
                st.df_5m, m_start, entry_px, near_dir,
                sl_px, partial_tp_px, hard_tp_px,
                atr, qty, FEE_SIDE, NSE_SLIP_PCT
            )
            if close_info is None:
                continue

            # Update stock signal state (accepted trade)
            st.last_trig = bar_idx
            st.zone_touches[sig["near_zone"]].append(bar_idx)

            # Record open position
            open_pos[sym] = {
                "symbol":       sym,
                "open_ts":      ts,
                "entry_ts":     entry_ts,
                "exit_ts":      close_info["exit_ts"],
                "direction":    near_dir,
                "entry_px":     round(entry_px, 4),
                "exit_px":      round(close_info["exit_px"], 4),
                "qty":          qty,
                "notional":     round(entry_px * qty, 2),
                "gross":        round(close_info["gross"],      4),
                "total_fee":    round(close_info["total_fee"],  4),
                "total_slip":   round(close_info["total_slip"], 4),
                "net_pnl":      round(close_info["net_pnl"],    4),
                "bal_open":     round(balance, 2),
                "near_zone":    sig["near_zone"],
                "open_bar_idx": bar_idx,
            }

    # Close any still-open positions at last price
    for sym, pos in list(open_pos.items()):
        st      = all_states[sym]
        last_px = st.df_1h["close"].iloc[-1]
        qty     = pos["qty"]
        dir_    = pos["direction"]
        gross   = (last_px - pos["entry_px"]) * qty if dir_ == "LONG" \
                  else (pos["entry_px"] - last_px) * qty
        fee     = last_px * qty * FEE_SIDE
        slip    = last_px * qty * NSE_SLIP_PCT
        net_pnl = gross - fee - slip - pos["total_fee"] - pos["total_slip"]
        pos["net_pnl"]   = round(net_pnl, 4)
        pos["exit_ts"]   = all_ts[-1]
        pos["exit_px"]   = last_px
        balance += net_pnl
        closed_trades.append(pos)

    equity_ts.append(all_ts[-1])
    equity_val.append(balance)

    return {
        "balance":       balance,
        "equity_ts":     equity_ts,
        "equity_val":    equity_val,
        "closed_trades": closed_trades,
        "date_from":     all_ts[0],
        "date_to":       all_ts[-1],
    }


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_portfolio_stats(portfolio: dict) -> dict:
    trades = portfolio["closed_trades"]
    if not trades:
        return {}

    df_t = pd.DataFrame(trades)
    wins   = df_t[df_t["net_pnl"] > 0]
    losses = df_t[df_t["net_pnl"] <= 0]

    gross_wins = wins["gross"].sum()
    gross_loss = abs(losses["gross"].sum())
    net_wins   = wins["net_pnl"].sum()
    net_loss   = abs(losses["net_pnl"].sum())

    gross_pf = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    net_pf   = round(net_wins   / net_loss,   3) if net_loss   > 0 else float("inf")
    win_rate = round(len(wins)  / len(df_t) * 100, 1)

    eq   = pd.Series(portfolio["equity_val"])
    peak = eq.cummax()
    dd   = (eq - peak) / peak * 100
    max_dd = round(dd.min(), 2)

    years     = (portfolio["date_to"] - portfolio["date_from"]).days / 365.25
    final_bal = portfolio["balance"]
    cagr      = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret = (final_bal / INITIAL_BALANCE - 1) * 100

    net_pnls  = df_t["net_pnl"].tolist()
    tpy       = len(df_t) / max(years, 0.1)
    mean_r    = sum(net_pnls) / len(net_pnls)
    std_r     = np.std(net_pnls, ddof=1) if len(net_pnls) > 1 else 1
    sharpe    = round(mean_r / std_r * np.sqrt(tpy) if std_r > 0 else 0, 2)
    calmar    = round(cagr / abs(max_dd) if max_dd != 0 else 0, 2)

    # Per-symbol breakdown
    by_sym = df_t.groupby("symbol").agg(
        trades=("net_pnl","count"),
        wins=("net_pnl", lambda x: (x>0).sum()),
        net_pnl=("net_pnl","sum"),
    ).reset_index()
    by_sym["win_rate"] = (by_sym["wins"] / by_sym["trades"] * 100).round(1)
    by_sym = by_sym.sort_values("net_pnl", ascending=False)

    # Monthly returns
    df_eq = pd.DataFrame({"ts": portfolio["equity_ts"], "val": portfolio["equity_val"]})
    df_eq = df_eq.set_index("ts").resample("ME").last().dropna()
    df_eq["ret"] = df_eq["val"].pct_change() * 100
    monthly = df_eq["ret"].dropna()

    return {
        "total_trades": len(df_t),
        "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate,
        "gross_pf": gross_pf, "net_pf": net_pf,
        "total_gross": round(df_t["gross"].sum(), 2),
        "total_fees":  round(df_t["total_fee"].sum(), 2),
        "total_slip":  round(df_t["total_slip"].sum(), 2),
        "total_net":   round(df_t["net_pnl"].sum(), 2),
        "final_bal":   round(final_bal, 2),
        "total_ret":   round(total_ret, 2),
        "cagr":        round(cagr, 2),
        "max_dd":      max_dd,
        "sharpe":      sharpe,
        "calmar":      calmar,
        "years":       round(years, 2),
        "tpy":         round(tpy, 0),
        "avg_win":     round(wins["net_pnl"].mean(), 2)   if len(wins)   else 0,
        "avg_loss":    round(losses["net_pnl"].mean(), 2) if len(losses) else 0,
        "best_trade":  round(df_t["net_pnl"].max(), 2),
        "worst_trade": round(df_t["net_pnl"].min(), 2),
        "by_symbol":   by_sym,
        "monthly":     monthly,
        "df_trades":   df_t,
    }


# ─── HTML report ──────────────────────────────────────────────────────────────

def generate_html(portfolio: dict, stats: dict, run_time: str) -> str:
    n_stocks = len(set(t["symbol"] for t in portfolio["closed_trades"]))

    # Equity curve (sampled for chart performance)
    eq_ts  = portfolio["equity_ts"]
    eq_val = portfolio["equity_val"]
    step   = max(1, len(eq_val) // 500)
    eq_labels = json.dumps([str(eq_ts[i].date()) for i in range(0, len(eq_ts), step)])
    eq_data   = json.dumps([round(eq_val[i], 2) for i in range(0, len(eq_val), step)])

    # Monthly returns heatmap data
    monthly = stats["monthly"]
    mon_labels = json.dumps([str(m)[:7] for m in monthly.index.tolist()])
    mon_data   = json.dumps([round(v, 2) for v in monthly.values.tolist()])
    mon_colors = json.dumps(["#22c55e" if v >= 0 else "#ef4444" for v in monthly.values])

    # Per-symbol table
    by_sym    = stats["by_symbol"]
    sym_rows  = ""
    for _, r in by_sym.head(30).iterrows():
        pnl_c = "pos" if r["net_pnl"] >= 0 else "neg"
        pnl_s = f"+INR {r['net_pnl']:,.0f}" if r["net_pnl"] >= 0 else f"-INR {abs(r['net_pnl']):,.0f}"
        sym_rows += f"""
        <tr>
          <td><strong>{r['symbol']}</strong></td>
          <td>{int(r['trades'])}</td>
          <td>{r['win_rate']:.0f}%</td>
          <td class="{pnl_c}">{pnl_s}</td>
        </tr>"""

    # Recent trades (last 20)
    df_t = stats["df_trades"].sort_values("exit_ts", ascending=False)
    trade_rows = ""
    for _, t in df_t.head(20).iterrows():
        net_c = "pos" if t["net_pnl"] >= 0 else "neg"
        net_s = f"+INR {t['net_pnl']:,.2f}" if t["net_pnl"] >= 0 else f"-INR {abs(t['net_pnl']):,.2f}"
        badge = '<span style="color:#22c55e;font-size:10px;font-weight:700">WIN</span>' \
                if t["net_pnl"] > 0 else \
                '<span style="color:#ef4444;font-size:10px;font-weight:700">LOSS</span>'
        dir_c = '<span style="color:#60a5fa;font-size:10px">LONG</span>' \
                if t["direction"] == "LONG" else \
                '<span style="color:#f472b6;font-size:10px">SHORT</span>'
        trade_rows += f"""
        <tr>
          <td>{str(t['entry_ts'])[:16]}</td>
          <td><strong>{t['symbol']}</strong></td>
          <td>{dir_c}</td>
          <td>INR {t['entry_px']:,.2f}</td>
          <td>INR {t['exit_px']:,.2f}</td>
          <td>{t['qty']:.0f}</td>
          <td>INR {t['notional']:,.0f}</td>
          <td class="{net_c}">{net_s}</td>
          <td>{badge}</td>
        </tr>"""

    c = stats["cagr"]
    cagr_color = "#22c55e" if c >= 0 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NSE Portfolio Backtest — LSCO LZR v7</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root{{--bg:#0a0a0f;--card:#111118;--border:#1e1e2e;--text:#e2e8f0;
        --muted:#64748b;--gold:#f59e0b;--green:#22c55e;--red:#ef4444;
        --blue:#60a5fa;--purple:#a78bfa;--warn:#f97316;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:13px;}}
  .header{{background:linear-gradient(135deg,#0f0f1e,#1a1a2e);padding:36px 48px 28px;border-bottom:1px solid var(--border);}}
  .logo{{font-size:28px;font-weight:800;color:var(--gold);letter-spacing:2px;}}
  .logo span{{color:var(--text);font-weight:400;}}
  .header-sub{{color:var(--muted);margin-top:6px;font-size:13px;}}
  .header-meta{{font-size:12px;color:var(--muted);line-height:1.8;text-align:right;}}
  .header-top{{display:flex;justify-content:space-between;align-items:flex-start;}}
  .badge{{display:inline-flex;align-items:center;background:#16213e;border:1px solid #1e3a5f;
          border-radius:20px;padding:4px 12px;color:var(--blue);font-size:11px;font-weight:700;}}
  .container{{max-width:1400px;margin:0 auto;padding:32px 40px;}}
  .section{{margin-bottom:36px;}}
  .section-title{{font-size:11px;font-weight:700;color:var(--muted);letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:16px;padding-bottom:8px;
                  border-bottom:1px solid var(--border);}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}}
  .kpi-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 22px;}}
  .kpi-card.gold{{border-left:3px solid var(--gold);}}
  .kpi-card.green{{border-left:3px solid var(--green);}}
  .kpi-card.blue{{border-left:3px solid var(--blue);}}
  .kpi-card.purple{{border-left:3px solid var(--purple);}}
  .kpi-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}}
  .kpi-value{{font-size:28px;font-weight:800;line-height:1;}}
  .kpi-sub{{font-size:11px;color:var(--muted);margin-top:8px;}}
  .chart-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;}}
  .chart-wrap{{position:relative;height:300px;}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
  table{{width:100%;border-collapse:collapse;}}
  thead tr{{background:#0d0d1a;}}
  th{{padding:10px 14px;text-align:left;font-size:10px;text-transform:uppercase;
      letter-spacing:1px;color:var(--muted);font-weight:600;white-space:nowrap;}}
  td{{padding:9px 14px;border-bottom:1px solid #0d0d1a;}}
  tbody tr:hover{{background:#16162a;}}
  .pos{{color:var(--green);}} .neg{{color:var(--red);}}
  .tbl-wrap{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:auto;max-height:500px;}}
  .growth-box{{background:linear-gradient(135deg,#0f1e0f,#142814);border:1px solid #16532d;
               border-radius:12px;padding:28px 32px;text-align:center;}}
  .growth-title{{font-size:12px;color:#22c55e;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;}}
  .growth-val{{font-size:48px;font-weight:900;color:#22c55e;}}
  .growth-sub{{font-size:13px;color:var(--muted);margin-top:8px;}}
  .disclaimer{{background:var(--card);border:1px solid var(--border);border-radius:10px;
               padding:16px 20px;font-size:11px;color:var(--muted);line-height:1.8;}}
</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div>
      <div class="logo">LSCO<span> Trading</span></div>
      <div class="header-sub">LZR v7 — NSE Portfolio Backtest &nbsp;|&nbsp; Shared ₹1 Lakh Capital &nbsp;|&nbsp; {n_stocks} Stocks</div>
    </div>
    <div class="header-meta">
      <span class="badge">PORTFOLIO BACKTEST</span><br><br>
      <strong>Data:</strong> NSE 5m (2015 - 2026)<br>
      <strong>Strategy:</strong> LZR v7 (5m execution)<br>
      <strong>Generated:</strong> {run_time}
    </div>
  </div>
</div>

<div class="container">

  <!-- GROWTH HIGHLIGHT -->
  <div class="section">
    <div class="growth-box">
      <div class="growth-title">₹1,00,000 invested in 2015 grew to</div>
      <div class="growth-val">INR {stats['final_bal']:,.0f}</div>
      <div class="growth-sub">
        <strong style="color:#22c55e">{stats['total_ret']:+.1f}% total return</strong> &nbsp;|&nbsp;
        <strong style="color:{cagr_color}">{stats['cagr']:+.1f}% CAGR</strong> &nbsp;|&nbsp;
        {stats['years']:.1f} years &nbsp;|&nbsp; {stats['total_trades']:,} trades across {n_stocks} NSE stocks
      </div>
    </div>
  </div>

  <!-- KPI CARDS -->
  <div class="section">
    <div class="kpi-grid">
      <div class="kpi-card gold">
        <div class="kpi-label">CAGR (Annualised)</div>
        <div class="kpi-value" style="color:{cagr_color}">{stats['cagr']:+.1f}%</div>
        <div class="kpi-sub">Compounding on shared ₹1L</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value" style="color:var(--green)">{stats['win_rate']}%</div>
        <div class="kpi-sub">{stats['wins']:,}W / {stats['losses']:,}L / {stats['total_trades']:,} total</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Net Profit Factor</div>
        <div class="kpi-value" style="color:var(--blue)">{stats['net_pf']}</div>
        <div class="kpi-sub">Gross PF: {stats['gross_pf']} &nbsp;|&nbsp; After all costs</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value" style="color:var(--purple)">{stats['sharpe']}</div>
        <div class="kpi-sub">Calmar: {stats['calmar']} &nbsp;|&nbsp; Max DD: {stats['max_dd']:.1f}%</div>
      </div>
    </div>
  </div>

  <!-- EQUITY CURVE -->
  <div class="section">
    <div class="section-title">Portfolio Balance Curve — ₹1,00,000 Compounding</div>
    <div class="chart-card">
      <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
    </div>
  </div>

  <!-- MONTHLY RETURNS + SYMBOL BREAKDOWN -->
  <div class="section two-col">
    <div>
      <div class="section-title">Monthly Returns (%)</div>
      <div class="chart-card">
        <div class="chart-wrap"><canvas id="monChart"></canvas></div>
      </div>
    </div>
    <div>
      <div class="section-title">Top 30 Stocks by Contribution</div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Symbol</th><th>Trades</th><th>WR</th><th>Net PnL</th></tr></thead>
          <tbody>{sym_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- STATS SUMMARY -->
  <div class="section">
    <div class="section-title">Portfolio Statistics</div>
    <div class="chart-card">
      <table>
        <tr>
          <td style="padding:12px 24px;color:var(--muted)">Starting Capital</td>
          <td style="padding:12px 24px;font-weight:700">INR 1,00,000</td>
          <td style="padding:12px 24px;color:var(--muted)">Final Balance</td>
          <td style="padding:12px 24px;font-weight:700;color:var(--green)">INR {stats['final_bal']:,.0f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Total Return</td>
          <td style="padding:12px 24px;font-weight:700;color:var(--green)">{stats['total_ret']:+.1f}%</td>
        </tr>
        <tr>
          <td style="padding:12px 24px;color:var(--muted)">Avg Win</td>
          <td style="padding:12px 24px;color:var(--green)">INR {stats['avg_win']:,.2f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Avg Loss</td>
          <td style="padding:12px 24px;color:var(--red)">INR {stats['avg_loss']:,.2f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Trades/Year</td>
          <td style="padding:12px 24px;">{stats['tpy']:.0f}</td>
        </tr>
        <tr>
          <td style="padding:12px 24px;color:var(--muted)">Best Trade</td>
          <td style="padding:12px 24px;color:var(--green)">INR {stats['best_trade']:,.2f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Worst Trade</td>
          <td style="padding:12px 24px;color:var(--red)">INR {stats['worst_trade']:,.2f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Stocks Traded</td>
          <td style="padding:12px 24px;">{n_stocks}</td>
        </tr>
        <tr>
          <td style="padding:12px 24px;color:var(--muted)">Total Fees</td>
          <td style="padding:12px 24px;color:var(--muted)">INR {stats['total_fees']:,.0f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Total Slippage</td>
          <td style="padding:12px 24px;color:var(--muted)">INR {stats['total_slip']:,.0f}</td>
          <td style="padding:12px 24px;color:var(--muted)">Max Concurrent</td>
          <td style="padding:12px 24px;">{MAX_CONCURRENT} positions</td>
        </tr>
      </table>
    </div>
  </div>

  <!-- RECENT TRADES -->
  <div class="section">
    <div class="section-title">Last 20 Closed Trades</div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr><th>Entry Time</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
              <th>Qty</th><th>Notional</th><th>Net PnL</th><th>Result</th></tr>
        </thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="disclaimer">
    <strong>Disclaimer:</strong>
    Historical backtest (2015–2026) using LZR v7 on NSE 5m data. Shared ₹1,00,000 capital,
    max {MAX_CONCURRENT} simultaneous positions, 3× intraday leverage, 0.10% round-trip fees + 0.05% slippage.
    All positions force-closed by 15:20 IST (intraday only). Past performance does not guarantee future results.
    This is for informational purposes only and does not constitute investment advice.
  </div>

</div>

<script>
new Chart(document.getElementById('eqChart'), {{
  type:'line',
  data: {{
    labels: {eq_labels},
    datasets: [{{
      label:'Portfolio Balance (INR)',
      data: {eq_data},
      borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,0.05)',
      borderWidth:2,pointRadius:0,tension:0.3,fill:true,
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>'INR '+c.parsed.y.toLocaleString('en-IN',{{maximumFractionDigits:0}})}}}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b',maxTicksLimit:12}}, grid:{{color:'#1e1e2e'}}}},
      y:{{ticks:{{color:'#64748b',callback:v=>'₹'+Math.round(v/1000)+'K'}}, grid:{{color:'#1e1e2e'}}}}
    }}
  }}
}});

new Chart(document.getElementById('monChart'), {{
  type:'bar',
  data: {{
    labels: {mon_labels},
    datasets: [{{
      label:'Monthly Return %',
      data: {mon_data},
      backgroundColor: {mon_colors},
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b',maxTicksLimit:24,maxRotation:45}}, grid:{{display:false}}}},
      y:{{ticks:{{color:'#64748b',callback:v=>v+'%'}}, grid:{{color:'#1e1e2e'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    _init_times()

    files = sorted(DATA_DIR_5M.glob("*_5minute.csv"))
    print("=" * 72)
    print("  LZR v7 — NSE PORTFOLIO BACKTEST (5m data)")
    print(f"  {len(files)} stock files  |  Starting capital: INR {INITIAL_BALANCE:,.0f}")
    print(f"  Risk: {RISK_PCT*100:.0f}%/trade  |  Max concurrent: {MAX_CONCURRENT}  |  Leverage: {MAX_LEVERAGE}×")
    print(f"  Fee: {NSE_FEE_RT*100:.2f}%  |  Slippage: {NSE_SLIP_PCT*100:.2f}%  |  Entry cutoff: 14:00 IST")
    print("=" * 72)

    # ── Load all stocks ───────────────────────────────────────────────────────
    print("\nLoading stock data...")
    all_states = {}
    t0 = datetime.now()

    for i, f in enumerate(files, 1):
        symbol, df_5m, df_1h, df_daily = load_stock(f)
        if df_5m is None:
            print(f"  [{i:3d}/{len(files)}] {symbol:<20} SKIP")
            continue
        try:
            st = StockState(symbol, df_5m, df_1h, df_daily)
            all_states[symbol] = st
            print(f"  [{i:3d}/{len(files)}] {symbol:<20} "
                  f"{len(df_5m):>7,} 5m  {len(df_1h):>5,} 1h  "
                  f"{df_1h.index[0].date()} -> {df_1h.index[-1].date()}")
        except Exception as e:
            print(f"  [{i:3d}/{len(files)}] {symbol:<20} ERROR: {e}")

    load_sec = (datetime.now() - t0).seconds
    print(f"\n  Loaded {len(all_states)} stocks in {load_sec}s")

    if not all_states:
        print("No stocks loaded. Check DATA_DIR_5M path.")
        return

    # ── Portfolio simulation ──────────────────────────────────────────────────
    print()
    t1 = datetime.now()
    portfolio = run_portfolio(all_states, verbose=True)
    sim_sec   = (datetime.now() - t1).seconds

    # ── Statistics ────────────────────────────────────────────────────────────
    stats = compute_portfolio_stats(portfolio)
    if not stats:
        print("No trades generated.")
        return

    print(f"\n  Simulation done in {sim_sec}s")
    print()
    print("=" * 72)
    print("  PORTFOLIO RESULTS")
    print("=" * 72)
    print(f"  Starting capital     : INR {INITIAL_BALANCE:>12,.0f}")
    print(f"  Final balance        : INR {stats['final_bal']:>12,.0f}")
    print(f"  Total return         : {stats['total_ret']:>+12.1f}%")
    print(f"  CAGR                 : {stats['cagr']:>+12.1f}% / year")
    print(f"  Win Rate             : {stats['win_rate']:>11.1f}%")
    print(f"  Net Profit Factor    : {stats['net_pf']:>12.3f}")
    print(f"  Sharpe Ratio         : {stats['sharpe']:>12.2f}")
    print(f"  Calmar Ratio         : {stats['calmar']:>12.2f}")
    print(f"  Max Drawdown         : {stats['max_dd']:>12.1f}%")
    print(f"  Total Trades         : {stats['total_trades']:>12,}")
    print(f"  Trades / Year        : {stats['tpy']:>12.0f}")
    print(f"  Active Stocks        : {len(set(t['symbol'] for t in portfolio['closed_trades'])):>12,}")
    print(f"  Gross PnL            : INR {stats['total_gross']:>10,.0f}")
    print(f"  Total Fees           : INR {stats['total_fees']:>10,.0f}")
    print(f"  Total Slippage       : INR {stats['total_slip']:>10,.0f}")
    print(f"  Net PnL              : INR {stats['total_net']:>10,.0f}")
    print("=" * 72)

    # Per-symbol top 10
    print("\n  TOP 10 STOCKS BY CONTRIBUTION:")
    print(f"  {'Symbol':<20} {'Trades':>7} {'WR':>7} {'Net PnL':>14}")
    print("  " + "-" * 52)
    for _, r in stats["by_symbol"].head(10).iterrows():
        pnl_s = f"INR {r['net_pnl']:>+,.0f}"
        print(f"  {r['symbol']:<20} {int(r['trades']):>7} {r['win_rate']:>6.0f}% {pnl_s:>14}")

    # Save CSV
    csv_path = OUTPUT_DIR / "backtest_v7_NSE_portfolio_trades.csv"
    stats["df_trades"].to_csv(csv_path, index=False)
    print(f"\n  Trades CSV  -> {csv_path}")

    # Save HTML
    run_time  = datetime.now().strftime("%Y-%m-%d %H:%M")
    html      = generate_html(portfolio, stats, run_time)
    html_path = OUTPUT_DIR / "NSE_Portfolio_Report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML report -> {html_path}")
    print(f"  Open in browser  ->  Print  ->  Save as PDF")
    total_sec = (datetime.now() - t0).seconds
    print(f"\n  Total runtime: {total_sec // 60}m {total_sec % 60}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
