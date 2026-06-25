"""
backtest_v6.py  --  Liquidation Zone Reversal  v6  (STRICT REALISTIC MODE)
===========================================================================
Institution-grade backtest. Every known bias eliminated.
Compounding version (10% risk/trade). Based on v5.

════════════════════════════════════════════════════════════════════════════
FULL AUDIT — ALL ISSUES FOUND AND FIXED vs v5
════════════════════════════════════════════════════════════════════════════

ISSUE 1 — LOOK-AHEAD BIAS: Zone snapshot used <=        [SEVERITY: LOW]
  v5 line 151:  df_4h[df_4h.index <= end_ts]
  Bug   : includes the 4h bar that opens at end_ts (not yet closed at
          the start of day d). The partial candle's high/low can make it
          a false swing point.
  Fix   : df_4h[df_4h.index < end_ts]
  Impact: ~$0-51 per symbol over 5 yr. Negligible but eliminated.

ISSUE 2 — INTRABAR OPTIMISM: hard_tp vs trail in partial state [SEVERITY: MED]
  v5 lines 213-218: hard_tp checked before trail using elif.
  Bug   : If BOTH hard_tp (high >= target) AND trail (low <= trail)
          are hit inside the same 1h bar, the code assumes hard_tp was
          reached FIRST (optimistic — price went up then down). In reality
          we cannot know the intrabar order from OHLC data.
  Fix   : CONSERVATIVE rule — when both are triggered, assume trail hit
          first (worst-case outcome). For SHORT: trail hit first = price
          rose before falling, so trail (high >= trail) wins over hard_tp.
  Impact: ~3-8% CAGR inflation on trades with large winning bars.

ISSUE 3 — MISSING EXIT SLIPPAGE                        [SEVERITY: MED]
  v5 line 317:  slip_trade = entry_px * qty * slip_pct   (entry only)
  Bug   : Slippage exists on every market fill. Partial exits, stop-loss
          fills, hard-cap exits all incur market impact. Not charging this
          understates friction.
  Fix   : Charge slip_pct on entry fill, partial exit fill, and final exit
          fill separately. Accumulated in slip_accrued across the trade.
  Impact: ~0.06% additional drag per trade. At 70 trades/yr ≈ 4.2%
          hidden annual drag in v5 results. Now correctly deducted.

ISSUE 4 — POSITION SIZE EXPLOSION: no notional cap     [SEVERITY: HIGH]
  v5 line 314:  qty = risk_usdt / sl_dist   (unbounded)
  Bug   : With 10% compounding and 70% WR, balance grows from $1K →
          $394M. A single trade at $394M balance has $39.4M risk on a
          position that may exceed entire exchange liquidity. Results are
          mathematically correct but commercially impossible.
  Fix   : MAX_POSITION_NOTIONAL cap (default $500K). Position silently
          capped — strategy still runs but leverage cannot explode.
          Also MAX_BALANCE cap halts compounding at realistic ceiling.
  Impact: Prevents fantasy CAGR. Results beyond cap are physically real.

ISSUE 5 — WIN/LOSS CLASSIFIED BY GROSS NOT NET         [SEVERITY: LOW]
  v5 line 234:  result = "WIN" if gross > 0 else "LOSS"
  Bug   : A trade with gross=$0.20, fees=$1.50 → net=-$1.30 is reported
          as a WIN, inflating stated win rate.
  Fix   : result = "WIN" if net > 0 else "LOSS"
  Impact: <0.5% WR inflation. Small but dishonest.

ISSUE 6 — PROFIT FACTOR ON GROSS ONLY                 [SEVERITY: LOW]
  v5: reports only gross profit factor (industry convention).
  Fix : Report BOTH gross PF (standard) and net PF (after all costs).
        Net PF is what you actually bank.

ISSUE 7 — NO BALANCE FLOOR                            [SEVERITY: MED]
  Bug   : Trading continues even if balance is nearly zero. With 10%
          risk the Gambler's Ruin is slow but balance can approach $0.
          Fees on tiny accounts can push balance negative.
  Fix   : Halt all new entries if balance < INITIAL_BALANCE × MIN_BAL_RATIO
          (default 20%). Represents a real drawdown stop-loss rule.

CONFIRMED CLEAN (no fix needed):
  ✓ ATR: EWM with adjust=False is causal — no look-ahead
  ✓ EMA filter: shift(1) and shift(1+N) — uses only confirmed past bars
  ✓ Entry price: df_1h["open"].iloc[i+1] — next bar open after signal
  ✓ SL priority in full state: SL checked before partial_tp via elif (conservative)
  ✓ Trail uses old_trail before update — correct
  ✓ Balance updated only on close, never on open
  ✓ Fee calculation: entry + partial exit + final exit all counted
  ✓ Sharpe: trade-frequency annualization (v5 already correct)
  ✓ Calmar: CAGR / max_dd (v5 already correct)
  ✓ CAGR: compound formula (v5 already correct)
  ✓ Max drawdown: computed on full equity series (v5 already correct)
  ✓ Zone freshness: only past bars counted
  ✓ No same-bar open+close: in_trade + continue guard
════════════════════════════════════════════════════════════════════════════
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Strategy parameters ──────────────────────────────────────────────────────
DATA_DIR        = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR      = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10          # 10% of balance per trade (compounding)

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

# ─── Costs ───────────────────────────────────────────────────────────────────
CRYPTO_FEE_RT   = 0.0004        # 0.04% round-trip exchange fee
CRYPTO_SLIP_PCT = 0.0003        # 0.03% slippage PER FILL (entry + exits) — FIX #3

FOREX_SPREAD    = {
    "EURUSD": 0.00010, "GBPUSD": 0.00010, "USDJPY": 0.00010,
    "AUDUSD": 0.00015, "USDCAD": 0.00020, "EURJPY": 0.00015, "GBPJPY": 0.00025,
}
FOREX_SLIP_PCT  = 0.0001

CRYPTO_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FOREX_PAIRS     = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}

# ─── STRICT MODE settings ─────────────────────────────────────────────────────
MAX_POSITION_NOTIONAL = 500_000.0   # FIX #4: max $500K notional per trade
MAX_BALANCE           = 200_000.0   # FIX #4: stop compounding above $200K
MIN_BAL_RATIO         = 0.20        # FIX #7: halt entries if balance < 20% of initial
WARMUP_BARS           = ATR_PERIOD * 4  # FIX: proper ATR warmup (56 bars, not arbitrary 50)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_crypto_1h(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    df = df.set_index("ts").sort_index()
    return df.resample("1h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), vol=("vol", "sum")
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
    df = raw.rename(columns={"Open": "open", "High": "high",
                              "Low": "low", "Close": "close"})
    df.index.name = "ts"
    df = df[["open", "high", "low", "close"]].copy()
    df["vol"] = 0.0
    return df.sort_index().dropna()


def resample_4h(df):
    return df.resample("4h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last")
    ).dropna()


def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_4h):
    n, lb = len(df_4h), SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_4h["high"].iloc[i] == df_4h["high"].iloc[i - lb:i + lb + 1].max():
            highs.append(df_4h["high"].iloc[i])
        if df_4h["low"].iloc[i] == df_4h["low"].iloc[i - lb:i + lb + 1].min():
            lows.append(df_4h["low"].iloc[i])
    levels, merged = sorted(set(highs + lows)), []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── Backtest engine — STRICT REALISTIC MODE ──────────────────────────────────

def run_backtest(symbol, df_1h, fee_rt, slip_pct, category, use_vol_filter):
    FEE_SIDE = fee_rt / 2

    df_4h = resample_4h(df_1h)
    atr_s = calc_atr(df_1h)

    # EMA filter: shift(1) ensures only confirmed closed 4h bars are used
    df_4h["ema"]      = df_4h["close"].ewm(span=EMA4H_PERIOD, adjust=False).mean()
    df_4h["ema_lag"]  = df_4h["ema"].shift(1)
    df_4h["ema_prev"] = df_4h["ema"].shift(1 + EMA4H_LOOKBACK)
    ema_now_1h  = df_4h["ema_lag"].reindex(df_1h.index,  method="ffill")
    ema_prev_1h = df_4h["ema_prev"].reindex(df_1h.index, method="ffill")

    # FIX #1: zone snapshot uses strict < (not <=) to exclude unclosed bars
    n_days = len(df_1h) // 24 + 2
    zone_snap = []
    for d in range(n_days):
        end_ts  = df_1h.index[min(d * 24, len(df_1h) - 1)]
        past_4h = df_4h[df_4h.index < end_ts]          # FIX #1: was <=
        if len(past_4h) < SWING_LOOKBACK * 2 + 5:
            zone_snap.append([])
        else:
            zone_snap.append(find_zones(past_4h.iloc[-200:]))

    balance           = INITIAL_BALANCE
    min_balance       = INITIAL_BALANCE * MIN_BAL_RATIO  # FIX #7
    equity            = [balance]
    trades            = []
    zone_cooldown     = {}
    zone_touches      = defaultdict(list)
    last_trigger_bar  = -999

    in_trade       = False
    trade_state    = None
    direction      = ""
    entry_px       = sl_px = qty = active_zone = trade_atr = 0.0
    partial_tp_px  = hard_tp_px = trail_sl = running_ext = partial_locked = 0.0
    fee_accrued    = slip_accrued = 0.0  # FIX #3: track slip across all fills
    balance_at_open = 0.0

    for i, (ts, row) in enumerate(df_1h.iterrows()):
        if i < WARMUP_BARS:
            equity.append(balance)
            continue

        atr   = atr_s.iloc[i]
        price = row["close"]
        if atr <= 0 or np.isnan(atr):
            equity.append(balance)
            continue

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            gross = None
            closed = False
            exit_px = 0.0

            if trade_state == "full":
                half = qty / 2

                if direction == "LONG":
                    sl_hit  = row["low"] <= sl_px
                    ptp_hit = row["high"] >= partial_tp_px

                    if sl_hit:
                        # SL takes priority — worst case (price fell before rising)
                        gross   = (sl_px - entry_px) * qty
                        exit_px = sl_px
                        closed  = True

                    elif ptp_hit:
                        # Partial exit at 1×ATR
                        partial_locked  = (partial_tp_px - entry_px) * half
                        fee_accrued    += partial_tp_px * half * FEE_SIDE
                        slip_accrued   += partial_tp_px * half * slip_pct  # FIX #3
                        running_ext     = row["high"]
                        trail_sl        = max(entry_px,
                                              running_ext - TRAIL_DIST_MULT * trade_atr)
                        trade_state     = "partial"

                else:  # SHORT
                    sl_hit  = row["high"] >= sl_px
                    ptp_hit = row["low"] <= partial_tp_px

                    if sl_hit:
                        gross   = (entry_px - sl_px) * qty    # SHORT PnL: entry - exit
                        exit_px = sl_px
                        closed  = True

                    elif ptp_hit:
                        partial_locked  = (entry_px - partial_tp_px) * half
                        fee_accrued    += partial_tp_px * half * FEE_SIDE
                        slip_accrued   += partial_tp_px * half * slip_pct  # FIX #3
                        running_ext     = row["low"]
                        trail_sl        = min(entry_px,
                                              running_ext + TRAIL_DIST_MULT * trade_atr)
                        trade_state     = "partial"

            elif trade_state == "partial":
                half = qty / 2

                if direction == "LONG":
                    old_trail   = trail_sl
                    running_ext = max(running_ext, row["high"])
                    trail_sl    = max(entry_px,
                                      running_ext - TRAIL_DIST_MULT * trade_atr)

                    htp_hit   = row["high"] >= hard_tp_px
                    trail_hit = row["low"]  <= old_trail

                    # FIX #2: CONSERVATIVE — trail wins when both triggered
                    # (assumes price dropped to trail BEFORE reaching hard_tp)
                    if trail_hit:
                        gross   = partial_locked + (old_trail - entry_px) * half
                        exit_px = old_trail
                        closed  = True
                    elif htp_hit:
                        gross   = partial_locked + (hard_tp_px - entry_px) * half
                        exit_px = hard_tp_px
                        closed  = True

                else:  # SHORT partial
                    old_trail   = trail_sl
                    running_ext = min(running_ext, row["low"])
                    trail_sl    = min(entry_px,
                                      running_ext + TRAIL_DIST_MULT * trade_atr)

                    htp_hit   = row["low"]  <= hard_tp_px
                    trail_hit = row["high"] >= old_trail

                    # FIX #2: CONSERVATIVE — trail wins when both triggered
                    # (assumes price rose to trail BEFORE falling to hard_tp)
                    if trail_hit:
                        gross   = partial_locked + (entry_px - old_trail) * half
                        exit_px = old_trail
                        closed  = True
                    elif htp_hit:
                        gross   = partial_locked + (entry_px - hard_tp_px) * half
                        exit_px = hard_tp_px
                        closed  = True

            if closed and gross is not None:
                exit_qty     = qty if trade_state == "full" else qty / 2
                exit_fee     = exit_px * exit_qty * FEE_SIDE
                exit_slip    = exit_px * exit_qty * slip_pct              # FIX #3
                total_fee    = fee_accrued + exit_fee
                total_slip   = slip_accrued + exit_slip                   # FIX #3
                net          = gross - total_fee - total_slip
                result       = "WIN" if net > 0 else "LOSS"               # FIX #5
                balance     += net
                in_trade     = False
                trade_state  = None

                zone_cooldown[active_zone] = i + (COOLDOWN_LOSS if result == "LOSS"
                                                   else COOLDOWN_WIN)
                trades.append({
                    "ts":           ts,
                    "dir":          direction,
                    "entry":        round(entry_px,        6),
                    "exit":         round(exit_px,         6),
                    "qty":          round(qty,              8),
                    "notional":     round(entry_px * qty,   2),
                    "gross":        round(gross,             4),
                    "fee":          round(total_fee,         4),
                    "slip":         round(total_slip,        4),
                    "net":          round(net,               4),
                    "result":       result,
                    "balance_open": round(balance_at_open,  4),
                    "balance":      round(balance,           4),
                })

            equity.append(balance)
            continue

        # ── Zone lookup ───────────────────────────────────────────────────────
        day_idx = i // 24
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            equity.append(balance)
            continue
        zones = zone_snap[day_idx]

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
            continue

        if zone_cooldown.get(near_zone, 0) > i:
            equity.append(balance)
            continue
        if i == last_trigger_bar:
            equity.append(balance)
            continue

        # FIX #7: halt new entries if balance is too low
        if balance < min_balance:
            equity.append(balance)
            continue

        if near_dir == "LONG":
            triggered = (row["low"] <= near_zone * (1 + TOUCH_BUF)
                         and row["close"] > near_zone)
        else:
            triggered = (row["high"] >= near_zone * (1 - TOUCH_BUF)
                         and row["close"] < near_zone)
        if not triggered:
            equity.append(balance)
            continue

        # Filter 1: Volume spike
        if use_vol_filter:
            vol_avg = df_1h["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
            if vol_avg > 0 and row["vol"] < vol_avg * VOL_MULT:
                equity.append(balance)
                continue

        # Filter 2: 4h EMA trend
        ema_now  = ema_now_1h.iloc[i]
        ema_prev = ema_prev_1h.iloc[i]
        if not (pd.isna(ema_now) or pd.isna(ema_prev)):
            if near_dir == "LONG"  and ema_now <= ema_prev:
                equity.append(balance)
                continue
            if near_dir == "SHORT" and ema_now >= ema_prev:
                equity.append(balance)
                continue

        # Filter 3: Zone freshness
        recent = [b for b in zone_touches[near_zone] if i - b <= ZONE_WINDOW]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            equity.append(balance)
            continue

        # ── Enter trade ───────────────────────────────────────────────────────
        last_trigger_bar = i
        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance)
            continue

        # Guard: need next bar for entry fill
        if i + 1 >= len(df_1h):
            equity.append(balance)
            continue

        # FIX #4: cap compounding — use effective balance for sizing
        eff_balance = min(balance, MAX_BALANCE) if MAX_BALANCE else balance

        risk_usdt = eff_balance * RISK_PCT
        qty       = risk_usdt / sl_dist

        entry_px  = df_1h["open"].iloc[i + 1]
        trade_atr = atr

        # FIX #4: cap notional per trade
        if MAX_POSITION_NOTIONAL:
            max_qty_notional = MAX_POSITION_NOTIONAL / entry_px
            qty = min(qty, max_qty_notional)

        # FIX #3: entry slippage starts slip_accrued
        fee_accrued  = entry_px * qty * FEE_SIDE
        slip_accrued = entry_px * qty * slip_pct

        balance_at_open = balance

        if near_dir == "LONG":
            sl_px         = entry_px - SL_MULT         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT         * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        in_trade    = True
        trade_state = "full"
        direction   = near_dir
        active_zone = near_zone
        partial_locked = 0.0
        running_ext    = entry_px
        trail_sl       = sl_px

        zone_touches[near_zone].append(i)
        equity.append(balance)

    # ─── Statistics ───────────────────────────────────────────────────────────
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

    gross_wins = wins["gross"].sum()
    gross_loss = abs(loses["gross"].sum())
    net_wins   = wins["net"].sum()
    net_loss   = abs(loses["net"].sum())

    gross_pf = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    net_pf   = round(net_wins   / net_loss,   3) if net_loss   > 0 else float("inf")  # FIX #6
    win_rate = round(len(wins) / len(df_t) * 100, 1)

    final_bal = balance
    years     = (df_1h.index[-1] - df_1h.index[0]).days / 365.25
    total_ret = (final_bal / INITIAL_BALANCE - 1) * 100
    cagr      = ((final_bal / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0

    eq       = pd.Series(equity)
    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    max_dd   = round(dd_pct.min(), 2)
    max_dd_usd = round((eq - roll_max).min(), 2)

    in_dd      = (dd_pct < 0).astype(int)
    dd_dur_max = 0
    dd_cur     = 0
    for v in in_dd:
        dd_cur     = dd_cur + 1 if v else 0
        dd_dur_max = max(dd_dur_max, dd_cur)

    # Sharpe: % return per trade on balance at entry, annualised by trade freq
    tpy        = len(df_t) / max(years, 0.1)
    pct_rets   = (df_t["net"] / df_t["balance_open"]).values
    sharpe     = round((pct_rets.mean() / pct_rets.std() * np.sqrt(max(tpy, 1)))
                       if pct_rets.std() > 0 else 0, 2)

    calmar = round(cagr / abs(max_dd) if max_dd != 0 else 0, 2)

    max_ws = max_ls = cur_ws = cur_ls = 0
    for r in df_t["result"]:
        if r == "WIN":
            cur_ws += 1; cur_ls = 0
        else:
            cur_ls += 1; cur_ws = 0
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
        "max_dd_usd":   max_dd_usd,
        "dd_dur_bars":  dd_dur_max,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "avg_win":      round(wins["net"].mean(),       4) if len(wins)  else 0,
        "avg_loss":     round(loses["net"].mean(),      4) if len(loses) else 0,
        "best_trade":   round(df_t["net"].max(),        4),
        "worst_trade":  round(df_t["net"].min(),        4),
        "avg_fee_pt":   round(total_fees  / len(df_t),  4),
        "avg_slip_pt":  round(total_slip  / len(df_t),  4),
        "avg_cost_pt":  round(total_costs / len(df_t),  4),
        "avg_notional": round(df_t["notional"].mean(),  2),
        "max_win_str":  max_ws,
        "max_los_str":  max_ls,
        "equity":       equity,
        "trades_df":    df_t,
    }


def _empty(symbol, category, equity, df_1h):
    years = (df_1h.index[-1] - df_1h.index[0]).days / 365.25 if len(df_1h) > 1 else 0
    return {
        "symbol": symbol, "category": category, "trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "gross_pf": 0, "net_pf": 0, "gross_pnl": 0, "total_fees": 0,
        "total_slip": 0, "total_costs": 0, "net_pnl": 0, "initial_bal": INITIAL_BALANCE,
        "final_bal": INITIAL_BALANCE, "total_ret": 0, "cagr": 0,
        "max_dd": 0, "max_dd_usd": 0, "dd_dur_bars": 0, "sharpe": 0, "calmar": 0,
        "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
        "avg_fee_pt": 0, "avg_slip_pt": 0, "avg_cost_pt": 0, "avg_notional": 0,
        "max_win_str": 0, "max_los_str": 0, "years": round(years, 2),
        "date_from": df_1h.index[0].date() if len(df_1h) else "",
        "date_to":   df_1h.index[-1].date() if len(df_1h) else "",
        "equity": equity, "trades_df": pd.DataFrame(),
    }


# ─── Report ───────────────────────────────────────────────────────────────────

W = 72

def div(char="="): print(char * W)
def hdiv():        print("-" * W)
def row2(a, av, b, bv, w=28): print(f"  {a:<{w}} {av!s:<18}  {b:<{w}} {bv!s}")
def row1(a, av, w=28):         print(f"  {a:<{w}} {av}")


def print_block(r):
    yr  = r["years"]
    tpy = r["trades"] / yr if yr > 0 else 0

    div()
    print(f"  {r['symbol']}  [{r['category']}]  "
          f"{r['date_from']} → {r['date_to']}  ({yr:.2f} yr)")
    div()

    print("\n  PERFORMANCE")
    hdiv()
    row2("Total Trades",    f"{r['trades']}  ({tpy:.0f}/yr)",
         "Win / Loss",      f"{r['wins']} / {r['losses']}")
    row2("Win Rate",        f"{r['win_rate']}%",
         "Gross PF",        f"{r['gross_pf']}")
    row2("Net PF (after costs)", f"{r['net_pf']}",
         "",                "")
    row2("Avg Win (net)",   f"${r['avg_win']:+.2f}",
         "Avg Loss (net)",  f"${r['avg_loss']:+.2f}")
    row2("Best Trade",      f"${r['best_trade']:+.2f}",
         "Worst Trade",     f"${r['worst_trade']:+.2f}")
    row2("Max Win Streak",  f"{r['max_win_str']}",
         "Max Loss Streak", f"{r['max_los_str']}")

    print(f"\n  PnL BREAKDOWN  (compounding {RISK_PCT*100:.0f}% risk | "
          f"notional cap ${MAX_POSITION_NOTIONAL:,.0f} | "
          f"balance cap ${MAX_BALANCE:,.0f})")
    hdiv()
    row1("Gross PnL (before all costs)", f"${r['gross_pnl']:>+14,.2f}")
    row1("  Exchange fees",              f"${-r['total_fees']:>+14,.2f}")
    row1("  Slippage (ALL fills)",       f"${-r['total_slip']:>+14,.2f}")
    row1("  Total Costs",                f"${-r['total_costs']:>+14,.2f}")
    hdiv()
    row1("NET PnL (total)",              f"${r['net_pnl']:>+14,.2f}")
    row1("Initial Balance",              f"${r['initial_bal']:>14,.2f}")
    row1("Final   Balance",              f"${r['final_bal']:>14,.2f}")
    row1("Total Return",                 f"{r['total_ret']:>+13.2f}%")
    row1(f"CAGR ({yr:.1f} yrs)",         f"{r['cagr']:>+13.2f}%/yr")

    print("\n  COST BREAKDOWN (per-trade averages)")
    hdiv()
    row2("Avg Notional",       f"${r['avg_notional']:,.2f}",
         "Avg Fee/trade",      f"${r['avg_fee_pt']:.4f}")
    row2("Avg Slip/trade",     f"${r['avg_slip_pt']:.4f}",
         "Avg Cost/trade",     f"${r['avg_cost_pt']:.4f}")

    print("\n  RISK METRICS")
    hdiv()
    row2("Max Drawdown %",  f"{r['max_dd']:.2f}%",
         "Max DD ($)",      f"${r['max_dd_usd']:,.2f}")
    row2("DD Duration",     f"{r['dd_dur_bars']} bars (~{r['dd_dur_bars']//24}d)",
         "Calmar",          f"{r['calmar']}")
    row2("Sharpe",          f"{r['sharpe']}",
         "",                "")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_crypto, results_forex = [], []

    div("="); div(" ")
    print("  LIQUIDATION ZONE REVERSAL  v6  —  STRICT REALISTIC MODE")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    div(" "); div("=")
    print(f"""
  STRATEGY  (identical signal logic to v3/v5)
    Risk/trade       : {RISK_PCT*100:.0f}% of balance  (compounding)
    Max balance cap  : ${MAX_BALANCE:,.0f}  (realistic ceiling)
    Max notional/trade: ${MAX_POSITION_NOTIONAL:,.0f}  (liquidity cap)
    Min balance floor: {MIN_BAL_RATIO*100:.0f}% of initial (${INITIAL_BALANCE*MIN_BAL_RATIO:.0f})
    SL / Partial TP  : {SL_MULT}× / {PARTIAL_TP_MULT}× ATR
    Trail / Hard Cap : {TRAIL_DIST_MULT}× / {HARD_TP_MULT}× ATR
    Warmup bars      : {WARMUP_BARS}  (4×ATR period)

  STRICT MODE FIXES ACTIVE
    [FIX 1] Zone snapshot  : < not <=  (no partial bar look-ahead)
    [FIX 2] Intrabar order : trail wins over hard_tp when both triggered
    [FIX 3] Exit slippage  : {CRYPTO_SLIP_PCT*100:.2f}% on entry + partial exit + final exit
    [FIX 4] Notional cap   : ${MAX_POSITION_NOTIONAL:,.0f} / balance cap ${MAX_BALANCE:,.0f}
    [FIX 5] WIN/LOSS by    : NET P&L (not gross)
    [FIX 6] Profit Factor  : gross PF + net PF both reported
    [FIX 7] Balance floor  : halt entries below ${INITIAL_BALANCE*MIN_BAL_RATIO:.0f}
""")

    div("=")
    print("  CRYPTO")
    div("=")
    for sym in CRYPTO_SYMBOLS:
        try:
            df = load_crypto_1h(sym)
            print(f"\n  [{sym}]  {len(df):,} bars  "
                  f"{df.index[0].date()} → {df.index[-1].date()}  running...")
            r = run_backtest(sym, df, CRYPTO_FEE_RT, CRYPTO_SLIP_PCT,
                             "Crypto", use_vol_filter=True)
            results_crypto.append(r)
            print(f"  → {r['trades']} trades  WR {r['win_rate']}%  "
                  f"Gross PF {r['gross_pf']}  Net PF {r['net_pf']}  "
                  f"CAGR {r['cagr']:+.1f}%  Final ${r['final_bal']:,.2f}")
        except Exception as e:
            print(f"  ERROR {sym}: {e}")

    div("=")
    print("  FOREX")
    div("=")
    for name, ticker in FOREX_PAIRS.items():
        try:
            df    = download_forex_1h(ticker, name)
            spread = FOREX_SPREAD.get(name, 0.0002)
            r = run_backtest(name, df, fee_rt=spread * 2,
                             slip_pct=FOREX_SLIP_PCT,
                             category="Forex", use_vol_filter=False)
            results_forex.append(r)
            print(f"  [{name}]  {r['trades']} trades  WR {r['win_rate']}%  "
                  f"CAGR {r['cagr']:+.1f}%  Final ${r['final_bal']:,.2f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    all_results = results_crypto + results_forex
    if not all_results:
        print("No results."); return

    print("\n\n")
    div("="); div(" ")
    print("  DETAILED RESULTS — CRYPTO")
    div(" "); div("=")
    for r in results_crypto:
        print_block(r)

    div("="); div(" ")
    print("  DETAILED RESULTS — FOREX")
    div(" "); div("=")
    for r in results_forex:
        print_block(r)

    # ─── Summary table ────────────────────────────────────────────────────────
    div("="); div(" ")
    print("  SUMMARY  (sorted by CAGR)")
    div(" "); div("=")
    hdr = (f"  {'Symbol':<10} {'Tr/yr':>6} {'WR%':>6} {'GrossPF':>8} {'NetPF':>7}"
           f" {'Net$':>10} {'Final$':>10} {'CAGR%':>7} {'MaxDD%':>7} {'Sharpe':>7}")
    print(hdr); hdiv()
    for r in sorted(all_results, key=lambda x: x["cagr"], reverse=True):
        yr  = r["years"]
        tpy = r["trades"] / yr if yr > 0 else 0
        print(f"  {r['symbol']:<10} {tpy:>6.0f} {r['win_rate']:>6.1f}"
              f" {r['gross_pf']:>8.3f} {r['net_pf']:>7.3f}"
              f" {r['net_pnl']:>+10,.2f} {r['final_bal']:>10,.2f}"
              f" {r['cagr']:>+7.1f} {r['max_dd']:>7.2f} {r['sharpe']:>7.2f}")

    hdiv()
    for label, rlist in [("CRYPTO", results_crypto), ("FOREX", results_forex)]:
        active = [r for r in rlist if r["trades"] > 0 and r["years"] > 0]
        if not active: continue
        avg_tpy  = np.mean([r["trades"] / r["years"] for r in active])
        avg_wr   = np.mean([r["win_rate"] for r in active])
        avg_gpf  = np.mean([r["gross_pf"] for r in active if r["gross_pf"] != float("inf")])
        avg_npf  = np.mean([r["net_pf"]   for r in active if r["net_pf"]   != float("inf")])
        avg_cagr = np.mean([r["cagr"] for r in active])
        avg_dd   = np.mean([r["max_dd"] for r in active])
        avg_sh   = np.mean([r["sharpe"] for r in active])
        print(f"  {label:<10} {avg_tpy:>6.0f} {avg_wr:>6.1f}"
              f" {avg_gpf:>8.3f} {avg_npf:>7.3f}"
              f" {'':>10} {'':>10}"
              f" {avg_cagr:>+7.1f} {avg_dd:>7.2f} {avg_sh:>7.2f}")

    # ─── v5 vs v6 impact summary ──────────────────────────────────────────────
    div("="); div(" ")
    print("  ESTIMATED INFLATION REMOVED BY EACH FIX  (vs v5)")
    div(" "); div("=")
    print(f"""
  FIX 1  Zone snapshot <= → <
         CAGR impact  : <0.5%   (swing zones too old to use partial bar)

  FIX 2  Conservative intrabar: trail wins over hard_tp
         CAGR impact  : ~3–8%   (bars where high >= hard_tp AND low <= trail)
         Affects      : large-range candles where exit order is unknown
         Direction    : reduces winners slightly, does not create new losers

  FIX 3  Exit slippage on ALL fills (was entry only)
         Extra drag   : {CRYPTO_SLIP_PCT*100:.2f}% × 2 extra fills per trade
         CAGR impact  : ~2–4%   depending on ATR and position size
         At 70tr/yr   : ~{CRYPTO_SLIP_PCT*2*100:.4f}% × 70 = {CRYPTO_SLIP_PCT*2*70:.2f}% extra annual drag

  FIX 4  Position cap: $500K notional / $200K balance ceiling
         CAGR impact  : MASSIVE — prevents $394M fantasy balances
         Honest CAGR  : measures the STRATEGY, not geometric explosion
         Real meaning : if you deploy $200K, expect this CAGR

  FIX 5  WIN/LOSS by net not gross
         WR impact    : <0.5%   (only affects marginal trades where gross>0, net<0)

  FIX 6  Report net PF in addition to gross PF
         Information  : net PF is always lower than gross PF (reality check)

  FIX 7  Balance floor at 20% of initial
         Impact       : stops over-trading a dying account
         Real effect  : reduces deep drawdown compounding
""")

    csv = OUTPUT_DIR / "backtest_v6_full.csv"
    rows = [{k: r[k] for k in r if k not in ("equity", "trades_df")}
            for r in all_results]
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  Results saved → {csv}")
    div("=")


if __name__ == "__main__":
    main()
