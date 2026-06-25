"""
compare_markets.py -- LZR Strategy: 3-Market Comparison
========================================================
Tests LZR liquidation-zone reversal on:
  Crypto:  4 symbols (from v14 FINAL results -- BTC/ETH/ATOM/LTC)
  Forex:   7 pairs   (1m data, C:\\Users\\GIGA\\Documents\\forex)
  Stocks:  12 indices (1h data, C:\\Users\\GIGA\\Documents\\Stock)

Key adaptations per market:
  Crypto:  Results loaded from v14 CSV (already run with full bug fixes)
  Forex:   VOL_MULT=0 (vol=0 in all CSV files); risk-based capital reservation
  Stocks:  1h execution bars (no 1m data); risk-based capital reservation
           timezone stripped from index
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import heapq
import pandas as pd
import numpy as np
from pathlib import Path
from lzr_core import DEFAULT_CFG, calc_atr, find_zones, resample

# ── Paths ──────────────────────────────────────────────────────────────────────
CRYPTO_CSV = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results\trades_v14_final.csv")
FOREX_DIR  = Path(r"C:\Users\GIGA\Documents\forex")
STOCK_DIR  = Path(r"C:\Users\GIGA\Documents\Stock")
OUT        = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT.mkdir(exist_ok=True)

INITIAL_BAL = 1_000.0

FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY"]
STOCK_IDXS  = ["SP500", "NAS100", "DOW", "DAX", "FTSE100", "NIKKEI",
               "ASX200", "RUS2000", "HSI", "CAC40", "ESTX50", "SMI"]

# ── Market configs ─────────────────────────────────────────────────────────────
FOREX_CFG = {**DEFAULT_CFG,
    "initial_balance": INITIAL_BAL,
    "risk_pct"       : 0.07,
    "fee_rt"         : 0.0002,   # 0.01%/side (spread cost, no exchange fee)
    "slip_pct"       : 0.0002,   # fallback
    "vol_mult"       : 0,        # DISABLED -- vol=0 in all forex CSVs
    "hard_tp_mult"   : 6.0,
    "cd_win_bars"    : 3,
    "cd_loss_bars"   : 42,
    "max_dd_pct"     : None,
}

STOCK_CFG = {**DEFAULT_CFG,
    "initial_balance": INITIAL_BAL,
    "risk_pct"       : 0.07,
    "fee_rt"         : 0.0004,   # 0.02%/side (retail CFD/ETF)
    "slip_pct"       : 0.0005,
    "vol_mult"       : 1.5,
    "hard_tp_mult"   : 6.0,
    "cd_win_bars"    : 3,
    "cd_loss_bars"   : 42,
    "max_dd_pct"     : None,
}

FOREX_SLIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0002, "USDJPY": 0.0001,
    "AUDUSD": 0.0002, "USDCAD": 0.0002, "EURJPY": 0.0003, "GBPJPY": 0.0003,
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_forex_1m(pair):
    path = FOREX_DIR / f"{pair}_1m.csv"
    df = pd.read_csv(path, parse_dates=["dt"])
    df = df.rename(columns={"dt": "ts"})
    df = df.set_index("ts").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def load_stock_1h(sym):
    path = STOCK_DIR / f"{sym}_1h.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "ts", "volume": "vol"})
    df = df.set_index("ts").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_generic(df_base, cfg):
    """
    Build all indicator/zone structures from base bars.
    df_base: 1m for forex, 1h for stocks.
    Signal TF = 4h (resampled). Zone TF = 16h.
    """
    ZW  = cfg["zone_window"]
    EP  = cfg["ema_period"]
    VMP = cfg["vol_ma_period"]
    AP  = cfg["atr_period"]
    WEP = cfg["weekly_ema_period"]
    DEP = cfg["daily_ema_period"]

    df_sig    = resample(df_base, "4h")
    df_zone   = resample(df_base, "16h")
    df_weekly = resample(df_base, "1W")
    df_daily  = resample(df_base, "1D")

    atr_sig   = calc_atr(df_sig,  AP)
    ema_sig   = df_sig["close"].ewm(span=EP,  adjust=False).mean()
    vol_ratio = df_sig["vol"] / df_sig["vol"].rolling(VMP).mean()

    atr_zone_s = calc_atr(df_zone, AP)
    zl          = find_zones(df_zone, ZW)

    weekly_ema = df_weekly["close"].ewm(span=WEP, adjust=False).mean()
    daily_ema  = df_daily["close"].ewm(span=DEP,  adjust=False).mean()

    # Shift daily by 1 bar: eliminates intraday look-ahead (same fix as lzr_core)
    daily_cls_s = df_daily["close"].shift(1)
    daily_ema_s = daily_ema.shift(1)

    return dict(
        df_base=df_base,
        df_sig=df_sig, df_zone=df_zone,
        df_weekly=df_weekly, df_daily=df_daily,
        atr_sig=atr_sig, ema_sig=ema_sig, vol_ratio=vol_ratio,
        atr_zone_s=atr_zone_s, zl=zl,
        weekly_ema=weekly_ema, daily_ema=daily_ema,
        daily_cls_s=daily_cls_s, daily_ema_s=daily_ema_s,
        zone_ts=df_zone.index, weekly_ts=df_weekly.index,
        daily_ts=df_daily.index,
        exec_ts=df_base.index,
    )


# ── Generic execution (no funding fees) ────────────────────────────────────────

def exec_bars(df_bars, b_start, entry_px, sl_px, partial_tp_px, hard_tp_px,
              atr, qty, fee_side, slip_pct, trail_dist_mult):
    """
    LONG-only execution on any bar granularity. No funding fees.
    SL checked before TP in same bar (conservative, matches v14 Bug #8 fix).
    Returns (close_ts, gross, exit_px, fee_acc, slip_acc) or None.
    """
    half           = qty / 2.0
    state          = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px

    fee_acc  = entry_px * qty * fee_side
    slip_acc = entry_px * qty * slip_pct

    for b_idx in range(b_start, len(df_bars)):
        br      = df_bars.iloc[b_idx]
        closed  = False
        gross   = 0.0
        exit_px = 0.0

        if state == "full":
            if br["open"] <= sl_px:
                exit_px = br["open"]
                gross   = (exit_px - entry_px) * qty
                closed  = True
            elif br["open"] >= partial_tp_px:
                partial_locked = (partial_tp_px - entry_px) * half
                fee_acc  += partial_tp_px * half * fee_side
                slip_acc += partial_tp_px * half * slip_pct
                running_ext = br["open"]
                trail_sl    = max(entry_px, running_ext - trail_dist_mult * atr)
                state = "partial"
            else:
                sl_hit = br["low"]  <= sl_px
                tp_hit = br["high"] >= partial_tp_px
                if sl_hit:
                    exit_px = sl_px
                    gross   = (exit_px - entry_px) * qty
                    closed  = True
                elif tp_hit:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = br["high"]
                    trail_sl    = max(entry_px, running_ext - trail_dist_mult * atr)
                    state = "partial"

        elif state == "partial":
            old_trail = trail_sl
            if br["open"] <= old_trail:
                exit_px = br["open"]
                gross   = partial_locked + (exit_px - entry_px) * half
                closed  = True
            elif not np.isinf(hard_tp_px) and br["open"] >= hard_tp_px:
                exit_px = hard_tp_px
                gross   = partial_locked + (exit_px - entry_px) * half
                closed  = True
            else:
                running_ext = max(running_ext, br["high"])
                trail_sl    = max(entry_px, running_ext - trail_dist_mult * atr)
                trail_hit = br["low"]  <= old_trail
                hard_hit  = not np.isinf(hard_tp_px) and br["high"] >= hard_tp_px
                if trail_hit:
                    exit_px = old_trail
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True
                elif hard_hit:
                    exit_px = hard_tp_px
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * fee_side
            slip_acc += exit_px * exit_qty * slip_pct
            return df_bars.index[b_idx], gross, exit_px, fee_acc, slip_acc

    return None


# ── Generic portfolio engine ───────────────────────────────────────────────────

def run_portfolio_generic(symbols, sym_data, cfg, slip_by_sym=None):
    """
    Shared-capital portfolio for forex/stocks.

    vs v14 crypto:
    - No funding fees
    - Risk-based reservation: avail = balance - n_open * RISK_PCT * balance
      (vs margin-based: avail = balance - sum(notional * 10%))
    - VOL_MULT=0 disables volume filter entirely (forex has vol=0)
    - Works with any bar granularity (1m or 1h)
    """
    INITIAL_BAL = cfg["initial_balance"]
    RISK_PCT    = cfg["risk_pct"]
    MAX_BAL     = cfg["max_balance"]
    MAX_POS_N   = cfg["max_pos_notional"]
    MIN_BAL_R   = cfg["min_bal_ratio"]
    ZW          = cfg["zone_window"]
    CD_LOSS     = cfg["cd_loss_bars"]
    CD_WIN      = cfg["cd_win_bars"]
    VOL_MULT    = cfg["vol_mult"]
    SL_MULT     = cfg["sl_mult"]
    ZT_MULT     = cfg["zone_touch_mult"]
    PTP_MULT    = cfg["partial_tp_mult"]
    HTP_MULT    = cfg["hard_tp_mult"]
    TRAIL_MULT  = cfg["trail_dist_mult"]
    FEE_SIDE    = cfg["fee_rt"] / 2.0
    SLIP        = cfg.get("slip_pct", 0.0005)
    MAX_DD      = cfg["max_dd_pct"]
    PAUSE_RES   = cfg["pause_resume_thresh"]
    PAUSE_TOUT  = cfg["pause_timeout_bars"]

    WARMUP = max(cfg["ema_period"], cfg["vol_ma_period"],
                 ZW * 2, cfg["weekly_ema_period"] * 7) + 10

    ref_sig = sym_data[symbols[0]]["df_sig"]
    n       = len(ref_sig)

    balance      = INITIAL_BAL
    peak_balance = INITIAL_BAL
    peak_equity  = INITIAL_BAL
    paused_until = -1
    pause_entry_eq = None

    sym_state = {s: {
        "cooldown_until": 0,
        "last_signal_i" : -999,
        "fired_zones"   : set(),
        "in_trade"      : False,
        "entry_px"      : 0.0,
        "entry_qty"     : 0.0,
    } for s in symbols}

    pending      = []
    trade_ctr    = 0
    all_trades   = []
    equity_curve = []

    for i in range(n):
        sig_ts = ref_sig.index[i]

        # Settle closed trades
        while pending and pending[0][0] <= i:
            cb, _, sym, net, td = heapq.heappop(pending)
            balance += net
            peak_balance = max(peak_balance, balance)
            ss = sym_state[sym]
            ss["in_trade"]  = False
            ss["entry_px"]  = 0.0
            ss["entry_qty"] = 0.0
            all_trades.append({**td, "balance": round(balance, 4)})

        # Mark-to-market equity (simplified -- full qty pre/post partial TP)
        unrealized = 0.0
        for sym in symbols:
            ss = sym_state[sym]
            if not ss["in_trade"] or ss["entry_qty"] == 0:
                continue
            curr_px = sym_data[sym]["df_sig"]["close"].iloc[i]
            unrealized += (curr_px - ss["entry_px"]) * ss["entry_qty"]

        equity = balance + unrealized
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

        if i < WARMUP or balance < INITIAL_BAL * MIN_BAL_R:
            continue

        # Circuit breaker
        if MAX_DD is not None:
            current_dd = (equity / peak_equity) - 1.0
            if current_dd <= -MAX_DD:
                if i > paused_until:
                    paused_until   = i + PAUSE_TOUT
                    pause_entry_eq = equity
            if i <= paused_until:
                if pause_entry_eq is not None:
                    rec_target = (pause_entry_eq +
                                  (peak_equity - pause_entry_eq) * PAUSE_RES)
                    if equity >= rec_target:
                        paused_until = -1
                    else:
                        continue
                else:
                    continue

        # Risk-based capital reservation
        n_open    = sum(1 for s in symbols if sym_state[s]["in_trade"])
        avail_bal = max(balance - n_open * RISK_PCT * balance, 0.0)

        for sym in symbols:
            ss = sym_state[sym]
            sd = sym_data[sym]

            if ss["in_trade"] or i <= ss["cooldown_until"]:
                continue
            # Each symbol may have a slightly different number of 4h bars
            sym_n = len(sd["df_sig"])
            if i >= sym_n or i + 1 >= sym_n:
                continue

            atr   = sd["atr_sig"].iloc[i]
            ema   = sd["ema_sig"].iloc[i]
            vol_r = sd["vol_ratio"].iloc[i]

            if atr <= 0 or np.isnan(atr) or np.isnan(ema):
                continue

            # Volume filter: disabled when VOL_MULT=0 (forex)
            if VOL_MULT > 0:
                if np.isnan(vol_r) or vol_r < VOL_MULT:
                    continue

            # Weekly regime filter (look-ahead fix: Monday boundary)
            week_floor = sig_ts - pd.Timedelta(days=sig_ts.dayofweek)
            week_floor = pd.Timestamp(week_floor.date())
            w_idx = int(sd["weekly_ts"].searchsorted(week_floor, side="left")) - 1
            if w_idx < 1:
                continue
            if sd["df_weekly"]["close"].iloc[w_idx] < sd["weekly_ema"].iloc[w_idx]:
                continue

            # Daily regime filter (look-ahead fix: shifted series)
            d_idx = int(sd["daily_ts"].searchsorted(sig_ts, side="right")) - 1
            if d_idx < 1:
                continue
            d_cls = sd["daily_cls_s"].iloc[d_idx]
            d_ema = sd["daily_ema_s"].iloc[d_idx]
            if np.isnan(d_cls) or np.isnan(d_ema) or d_cls < d_ema:
                continue

            # Zone lookup
            z_idx = int(sd["zone_ts"].searchsorted(sig_ts, side="right")) - 1
            if z_idx < ZW:
                continue

            close = sd["df_sig"].iloc[i]["close"]
            zl    = sd["zl"]
            fired = ss["fired_zones"]

            fired_zone_key = None
            zone_lo        = None

            for zb in sorted([k for k in zl if k < z_idx and k not in fired],
                             reverse=True)[:10]:
                zp    = zl[zb]
                atr_z = sd["atr_zone_s"].iloc[zb]
                z_lo  = zp - ZT_MULT * atr_z
                z_hi  = zp + ZT_MULT * atr_z
                if z_lo <= close <= z_hi:
                    fired_zone_key = zb
                    zone_lo        = z_lo
                    break

            if fired_zone_key is None or i == ss["last_signal_i"]:
                continue

            # 4h EMA trend confirm
            if close < ema:
                continue

            entry_px = sd["df_sig"]["open"].iloc[i + 1]
            sl_px    = zone_lo - SL_MULT * atr
            sl_dist  = entry_px - sl_px
            if sl_dist <= 0:
                continue

            partial_tp = entry_px + PTP_MULT * atr
            hard_tp    = entry_px + HTP_MULT * atr

            eff_bal = min(balance, MAX_BAL)
            if eff_bal < INITIAL_BAL * MIN_BAL_R:
                continue

            # Need at least one more risk unit of free capital
            if avail_bal < balance * RISK_PCT:
                continue

            qty = min((balance * RISK_PCT) / sl_dist, MAX_POS_N / entry_px)
            if qty <= 0:
                continue

            entry_ts = sd["df_sig"].index[i + 1]
            b_start  = int(sd["exec_ts"].searchsorted(entry_ts))
            slip     = slip_by_sym.get(sym, SLIP) if slip_by_sym else SLIP

            info = exec_bars(sd["df_base"], b_start, entry_px, sl_px,
                             partial_tp, hard_tp, atr, qty,
                             FEE_SIDE, slip, TRAIL_MULT)
            if info is None:
                continue

            close_ts, gross, exit_px, total_fee, total_slip = info
            net    = gross - total_fee - total_slip
            result = "WIN" if net > 0 else "LOSS"

            close_bar = int(sd["df_sig"].index.searchsorted(close_ts, side="right")) - 1
            close_bar = max(close_bar, i + 1)
            close_bar = min(close_bar, n - 1)

            dur_h = (close_ts - entry_ts).total_seconds() / 3600.0

            ss["fired_zones"].add(fired_zone_key)
            ss["cooldown_until"] = close_bar + (CD_LOSS if result == "LOSS" else CD_WIN)
            ss["last_signal_i"]  = i
            ss["in_trade"]       = True
            ss["entry_px"]       = entry_px
            ss["entry_qty"]      = qty

            # Reduce available capital for next symbol this bar
            avail_bal -= balance * RISK_PCT
            avail_bal  = max(avail_bal, 0.0)

            td = dict(symbol=sym, ts=sig_ts, close_ts=close_ts,
                      entry=round(entry_px, 8), exit=round(exit_px, 8),
                      sl=round(sl_px, 8),
                      gross=round(gross, 4), fee=round(total_fee, 4),
                      slip=round(total_slip, 4), net=round(net, 4),
                      result=result, duration_h=round(dur_h, 1))

            heapq.heappush(pending, (close_bar, trade_ctr, sym, net, td))
            trade_ctr += 1

    # Flush remaining open trades
    while pending:
        cb, _, sym, net, td = heapq.heappop(pending)
        balance += net
        peak_balance = max(peak_balance, balance)
        ss = sym_state[sym]
        ss["in_trade"]  = False
        ss["entry_px"]  = 0.0
        ss["entry_qty"] = 0.0
        all_trades.append({**td, "balance": round(balance, 4)})

    return all_trades, equity_curve, balance


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_market_stats(trades, equity_curve, final_balance):
    if not trades:
        return None

    df    = pd.DataFrame(trades)
    wins  = df[df["result"] == "WIN"]
    loses = df[df["result"] == "LOSS"]
    nw    = wins["net"].sum()
    nl    = abs(loses["net"].sum()) if len(loses) else 0.0

    wr  = len(wins) / len(df) * 100
    npf = round(nw / nl, 3) if nl > 0 else float("inf")

    INIT  = equity_curve[0] if equity_curve else final_balance
    # Use timestamp range from trades for CAGR
    t0    = pd.to_datetime(df["ts"].min())
    t1    = pd.to_datetime(df["close_ts"].max())
    years = max((t1 - t0).days / 365.25, 0.5)
    cagr  = ((final_balance / INIT) ** (1 / years) - 1) * 100

    eq  = pd.Series(equity_curve)
    mdd = round(float(((eq - eq.cummax()) / eq.cummax() * 100).min()), 2)
    calmar = round(abs(cagr / mdd), 3) if mdd < 0 else float("inf")

    df["year"] = pd.to_datetime(df["ts"]).dt.year
    yearly = []
    bal = INITIAL_BAL
    for yr in sorted(df["year"].unique()):
        yr_t   = df[df["year"] == yr]
        net_yr = yr_t["net"].sum()
        start  = bal
        bal   += net_yr
        ret_yr = (bal / start - 1) * 100 if start > 0 else 0.0
        yr_wr  = yr_t["result"].eq("WIN").mean() * 100
        yearly.append(dict(year=yr, trades=len(yr_t),
                           wr=round(yr_wr, 1), ret=round(ret_yr, 1),
                           net=round(net_yr, 2)))

    per_sym = {}
    for sym, grp in df.groupby("symbol"):
        per_sym[sym] = dict(
            trades=len(grp),
            wr=round(grp["result"].eq("WIN").mean() * 100, 1),
            net=round(grp["net"].sum(), 2),
        )

    gross = df["gross"].sum() if "gross" in df.columns else 0.0
    fees  = df["fee"].sum()   if "fee"   in df.columns else 0.0
    slip  = df["slip"].sum()  if "slip"  in df.columns else 0.0
    n_pos = sum(1 for y in yearly if y["net"] >= 0)

    return dict(
        trades=len(df), wins=len(wins), losses=len(loses),
        win_rate=round(wr, 1), profit_factor=npf,
        cagr=round(cagr, 1), max_dd=mdd, calmar=calmar,
        final_bal=round(final_balance, 2),
        gross=round(gross, 2), fees=round(fees, 2), slip=round(slip, 2),
        yearly=yearly, per_sym=per_sym,
        profitable_years=n_pos, total_years=len(yearly),
    )


def load_crypto_stats():
    """Load crypto v14 results from CSV and compute comparison stats."""
    if not CRYPTO_CSV.exists():
        return None

    df = pd.read_csv(CRYPTO_CSV)
    # Use incremental balance column for final value
    final_balance = df["balance"].iloc[-1]

    wins  = df[df["result"] == "WIN"]
    loses = df[df["result"] == "LOSS"]
    nw    = wins["net"].sum()
    nl    = abs(loses["net"].sum()) if len(loses) else 0.0
    wr    = len(wins) / len(df) * 100
    npf   = round(nw / nl, 3) if nl > 0 else float("inf")

    years = 5.0
    cagr  = ((final_balance / INITIAL_BAL) ** (1 / years) - 1) * 100
    # Use known v14 DD from full equity curve (MTM-aware)
    mdd   = -11.6
    calmar = round(abs(cagr / mdd), 3) if mdd < 0 else float("inf")

    df["year"] = pd.to_datetime(df["ts"]).dt.year
    yearly = []
    bal = INITIAL_BAL
    for yr in sorted(df["year"].unique()):
        yr_t   = df[df["year"] == yr]
        net_yr = yr_t["net"].sum()
        start  = bal
        bal   += net_yr
        ret_yr = (bal / start - 1) * 100 if start > 0 else 0.0
        yr_wr  = yr_t["result"].eq("WIN").mean() * 100
        yearly.append(dict(year=yr, trades=len(yr_t),
                           wr=round(yr_wr, 1), ret=round(ret_yr, 1),
                           net=round(net_yr, 2)))

    per_sym = {}
    for sym, grp in df.groupby("symbol"):
        per_sym[sym] = dict(
            trades=len(grp),
            wr=round(grp["result"].eq("WIN").mean() * 100, 1),
            net=round(grp["net"].sum(), 2),
        )

    gross   = df["gross"].sum()
    fees    = df["fee"].sum()
    slip    = df["slip"].sum()
    funding = df["funding"].sum() if "funding" in df.columns else 0.0
    n_pos   = sum(1 for y in yearly if y["net"] >= 0)

    return dict(
        trades=len(df), wins=len(wins), losses=len(loses),
        win_rate=round(wr, 1), profit_factor=npf,
        cagr=round(cagr, 1), max_dd=mdd, calmar=calmar,
        final_bal=round(final_balance, 2),
        gross=round(gross, 2), fees=round(fees, 2), slip=round(slip, 2),
        funding=round(funding, 2),
        yearly=yearly, per_sym=per_sym,
        profitable_years=n_pos, total_years=len(yearly),
    )


# ── Display helpers ────────────────────────────────────────────────────────────

def print_market_stats(market_name, stats, extra_costs=None):
    if stats is None:
        print(f"  {market_name}: No results.")
        return
    W2 = "-" * 60
    print(f"\n  {market_name} OVERVIEW")
    print("  " + W2)
    print(f"  Trades:           {stats['trades']:>6}")
    print(f"  Wins / Losses:    {stats['wins']:>3} W / {stats['losses']:>3} L")
    print(f"  Win Rate:         {stats['win_rate']:>5.1f}%")
    print(f"  CAGR:             {stats['cagr']:>+5.1f}%")
    print(f"  Max Drawdown:     {stats['max_dd']:>5.1f}%")
    print(f"  Calmar Ratio:     {stats['calmar']:>6.2f}")
    pf = stats["profit_factor"]
    pf_str = f"{pf:.2f}x" if pf != float("inf") else "inf"
    print(f"  Profit Factor:    {pf_str:>7}")
    print(f"  Final Balance:    ${stats['final_bal']:>9,.2f}  "
          f"({stats['final_bal']/INITIAL_BAL:.2f}x / "
          f"+{(stats['final_bal']/INITIAL_BAL-1)*100:.0f}%)")

    print(f"\n  YEAR-BY-YEAR:")
    for y in stats["yearly"]:
        flag = "  <-- LOSS YEAR" if y["net"] < 0 else ""
        print(f"    {y['year']}: {y['trades']:>3}tr  "
              f"WR {y['wr']:>5.1f}%  "
              f"Ret {y['ret']:>+6.1f}%  "
              f"Net ${y['net']:>+8,.2f}{flag}")
    print(f"\n  Profitable years: {stats['profitable_years']}/{stats['total_years']}")

    if stats.get("gross", 0) != 0:
        gross  = stats["gross"]
        fees   = abs(stats["fees"])
        slip   = abs(stats["slip"])
        fund   = abs(extra_costs) if extra_costs else 0.0
        costs  = fees + slip + fund
        net_pl = stats["final_bal"] - INITIAL_BAL
        print(f"\n  COST WATERFALL:")
        print(f"    Gross P&L:  ${gross:>+10,.2f}")
        print(f"    Fees:       ${-fees:>+10,.2f}")
        print(f"    Slippage:   ${-slip:>+10,.2f}")
        if fund > 0:
            print(f"    Funding:    ${-fund:>+10,.2f}")
        print(f"    Net P&L:    ${net_pl:>+10,.2f}")
        if gross != 0:
            print(f"    Cost rate:  {costs/abs(gross)*100:.1f}% of gross")


def print_per_symbol(market_name, stats):
    if stats is None or not stats.get("per_sym"):
        return
    print(f"\n  {market_name} PER-SYMBOL:")
    print(f"  {'Symbol':<12}  {'Tr':>4}  {'WR':>6}  {'Net P&L':>12}")
    print("  " + "-" * 40)
    for sym, ps in sorted(stats["per_sym"].items(), key=lambda x: -x[1]["net"]):
        sign = "+" if ps["net"] >= 0 else ""
        print(f"  {sym:<12}  {ps['trades']:>4}  "
              f"{ps['wr']:>5.1f}%  {sign}${ps['net']:>9,.2f}")


def print_comparison(crypto, forex, stocks):
    W = "=" * 90
    print(W)
    print("  3-MARKET COMPARISON  (LZR Liquidation Zone Reversal, $1,000 start)")
    print(W)

    def fv(s, key, fmt="{:.1f}"):
        if s is None:
            return "   N/A"
        v = s.get(key)
        if v is None:
            return "   N/A"
        if v == float("inf"):
            return "   inf"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    rows = [
        ("Metric",            "CRYPTO (4 sym)",  "FOREX (7 pairs)", "STOCKS (12 idx)"),
        ("---",               "---",              "---",              "---"),
        ("Trades",            fv(crypto, "trades", "{:.0f}"),
                              fv(forex,  "trades", "{:.0f}"),
                              fv(stocks, "trades", "{:.0f}")),
        ("Win Rate (%)",      fv(crypto, "win_rate"),
                              fv(forex,  "win_rate"),
                              fv(stocks, "win_rate")),
        ("CAGR (%)",          fv(crypto, "cagr", "{:+.1f}"),
                              fv(forex,  "cagr", "{:+.1f}"),
                              fv(stocks, "cagr", "{:+.1f}")),
        ("Max Drawdown (%)",  fv(crypto, "max_dd"),
                              fv(forex,  "max_dd"),
                              fv(stocks, "max_dd")),
        ("Calmar Ratio",      fv(crypto, "calmar", "{:.2f}"),
                              fv(forex,  "calmar", "{:.2f}"),
                              fv(stocks, "calmar", "{:.2f}")),
        ("Profit Factor",     fv(crypto, "profit_factor", "{:.2f}"),
                              fv(forex,  "profit_factor", "{:.2f}"),
                              fv(stocks, "profit_factor", "{:.2f}")),
        ("Profitable Yrs",    fv(crypto, "profitable_years", "{:.0f}"),
                              fv(forex,  "profitable_years", "{:.0f}"),
                              fv(stocks, "profitable_years", "{:.0f}")),
        ("Final Balance ($)", f"${crypto['final_bal']:,.2f}" if crypto else "N/A",
                              f"${forex['final_bal']:,.2f}"  if forex  else "N/A",
                              f"${stocks['final_bal']:,.2f}" if stocks else "N/A"),
        ("Total Return",      f"{crypto['final_bal']/INITIAL_BAL:.2f}x" if crypto else "N/A",
                              f"{forex['final_bal']/INITIAL_BAL:.2f}x"  if forex  else "N/A",
                              f"{stocks['final_bal']/INITIAL_BAL:.2f}x" if stocks else "N/A"),
    ]

    print(f"\n  {'Metric':<22}  {'CRYPTO (4 sym)':>16}  {'FOREX (7 pairs)':>16}  {'STOCKS (12 idx)':>16}")
    print("  " + "-" * 78)
    for row in rows[2:]:
        label, cv, fv_, sv = row
        print(f"  {label:<22}  {cv:>16}  {fv_:>16}  {sv:>16}")

    # Year-by-year comparison
    print(f"\n  YEAR-BY-YEAR RETURN:")
    print(f"  {'Year':<6}  {'CRYPTO':>10}  {'FOREX':>10}  {'STOCKS':>10}")
    print("  " + "-" * 45)

    all_years = set()
    for s in [crypto, forex, stocks]:
        if s and s.get("yearly"):
            all_years.update(y["year"] for y in s["yearly"])

    def get_yr(s, yr):
        if not s or not s.get("yearly"):
            return None
        for y in s["yearly"]:
            if y["year"] == yr:
                return y
        return None

    for yr in sorted(all_years):
        def fmt_yr(s, yr=yr):
            y = get_yr(s, yr)
            if y is None:
                return "       N/A"
            sign = "+" if y["ret"] >= 0 else ""
            return f"{sign}{y['ret']:.1f}%".rjust(10)
        print(f"  {yr:<6}  {fmt_yr(crypto)}  {fmt_yr(forex)}  {fmt_yr(stocks)}")

    # Verdict
    print(f"\n  VERDICT:")
    available = [(n, s) for n, s in
                 [("Crypto", crypto), ("Forex", forex), ("Stocks", stocks)]
                 if s and s["trades"] > 0]
    if not available:
        print("    No results to compare.")
        return

    best_calmar = max(available,
                      key=lambda x: (x[1]["calmar"] if x[1]["calmar"] != float("inf")
                                     else 9999))
    best_wr     = max(available, key=lambda x: x[1]["win_rate"])
    best_cagr   = max(available, key=lambda x: x[1]["cagr"])
    best_dd     = max(available, key=lambda x: x[1]["max_dd"])   # max_dd is negative; closest to 0 = best

    print(f"    Best Calmar Ratio:  {best_calmar[0]} ({best_calmar[1]['calmar']:.2f})")
    print(f"    Best Win Rate:      {best_wr[0]} ({best_wr[1]['win_rate']:.1f}%)")
    print(f"    Best CAGR:          {best_cagr[0]} ({best_cagr[1]['cagr']:+.1f}%)")
    print(f"    Lowest Drawdown:    {best_dd[0]} ({best_dd[1]['max_dd']:.1f}%)")

    # Rank all three
    if len(available) >= 2:
        print(f"\n  MARKET RANKING (by Calmar):")
        ranked = sorted(available,
                        key=lambda x: (x[1]["calmar"] if x[1]["calmar"] != float("inf")
                                       else 9999),
                        reverse=True)
        medals = ["1st", "2nd", "3rd"]
        for idx, (name, s) in enumerate(ranked):
            print(f"    {medals[idx]}  {name:<10}  "
                  f"CAGR {s['cagr']:+.1f}%  "
                  f"DD {s['max_dd']:.1f}%  "
                  f"Calmar {s['calmar']:.2f}  "
                  f"WR {s['win_rate']:.1f}%  "
                  f"Final ${s['final_bal']:,.2f}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    W = "=" * 90

    # ── Section 1: Crypto ─────────────────────────────────────────────────────
    print(W)
    print("  SECTION 1: CRYPTO -- BTC / ETH / ATOM / LTC  (v14 FINAL)")
    print(W)

    crypto_stats = load_crypto_stats()
    if crypto_stats:
        print(f"  Loaded {crypto_stats['trades']} trades from {CRYPTO_CSV.name}")
        print_market_stats("CRYPTO", crypto_stats,
                           extra_costs=crypto_stats.get("funding", 0))
        print_per_symbol("CRYPTO", crypto_stats)
    else:
        print(f"  ERROR: {CRYPTO_CSV} not found. Run backtest_v14.py first.")

    # ── Section 2: Forex ──────────────────────────────────────────────────────
    print()
    print(W)
    print("  SECTION 2: FOREX -- 7 PAIRS  (1m data, vol filter DISABLED)")
    print(W)

    available_forex = [p for p in FOREX_PAIRS
                       if (FOREX_DIR / f"{p}_1m.csv").exists()]
    forex_stats = None

    if not available_forex:
        print(f"  ERROR: No forex CSV files found in {FOREX_DIR}")
        print("         Expected: EURUSD_1m.csv, GBPUSD_1m.csv, ...")
    else:
        print(f"  Loading {len(available_forex)} pairs: {', '.join(available_forex)}")
        forex_data = {}
        for pair in available_forex:
            print(f"    {pair}: loading...", end="", flush=True)
            try:
                df_1m = load_forex_1m(pair)
                forex_data[pair] = prepare_generic(df_1m, FOREX_CFG)
                print(f"\r    {pair}: {len(df_1m):,} 1m bars  |  "
                      f"{len(forex_data[pair]['df_sig']):,} 4h signal bars")
            except Exception as e:
                print(f"\r    {pair}: SKIP -- {e}")

        if not forex_data:
            print("  No forex data loaded successfully.")
        else:
            syms = list(forex_data.keys())
            print(f"\n  Running forex portfolio ({len(syms)} pairs)...")
            forex_trades, forex_eq, forex_final = run_portfolio_generic(
                syms, forex_data, FOREX_CFG, slip_by_sym=FOREX_SLIP)

            if forex_trades:
                df_fx = pd.DataFrame(forex_trades)
                df_fx.to_csv(OUT / "trades_forex.csv", index=False)
                print(f"  Saved {len(forex_trades)} trades -> trades_forex.csv")

                ref_idx = forex_data[syms[0]]["df_sig"].index
                forex_stats = compute_market_stats(forex_trades, forex_eq, forex_final)
                print_market_stats("FOREX", forex_stats)
                print_per_symbol("FOREX", forex_stats)
            else:
                print("  No forex trades generated.")
                print("  (Check regime filter -- forex pairs may be in downtrend)")

    # ── Section 3: Stocks ─────────────────────────────────────────────────────
    print()
    print(W)
    print("  SECTION 3: STOCKS / INDICES -- 12 INDICES  (1h data)")
    print(W)

    available_stocks = [s for s in STOCK_IDXS
                        if (STOCK_DIR / f"{s}_1h.csv").exists()]
    stock_stats = None

    if not available_stocks:
        print(f"  ERROR: No stock CSV files found in {STOCK_DIR}")
        print("         Expected: SP500_1h.csv, NAS100_1h.csv, ...")
    else:
        print(f"  Loading {len(available_stocks)} indices: {', '.join(available_stocks)}")
        stock_data = {}
        for sym in available_stocks:
            print(f"    {sym}: loading...", end="", flush=True)
            try:
                df_1h = load_stock_1h(sym)
                stock_data[sym] = prepare_generic(df_1h, STOCK_CFG)
                print(f"\r    {sym}: {len(df_1h):,} 1h bars  |  "
                      f"{len(stock_data[sym]['df_sig']):,} 4h signal bars")
            except Exception as e:
                print(f"\r    {sym}: SKIP -- {e}")

        if not stock_data:
            print("  No stock data loaded successfully.")
        else:
            syms = list(stock_data.keys())
            print(f"\n  Running stock portfolio ({len(syms)} indices)...")
            stock_trades, stock_eq, stock_final = run_portfolio_generic(
                syms, stock_data, STOCK_CFG, slip_by_sym=None)

            if stock_trades:
                df_st = pd.DataFrame(stock_trades)
                df_st.to_csv(OUT / "trades_stocks.csv", index=False)
                print(f"  Saved {len(stock_trades)} trades -> trades_stocks.csv")

                stock_stats = compute_market_stats(stock_trades, stock_eq, stock_final)
                print_market_stats("STOCKS", stock_stats)
                print_per_symbol("STOCKS", stock_stats)
            else:
                print("  No stock trades generated.")
                print("  (Check regime filter -- indices may spend most time below EMA)")

    # ── Section 4: 3-Market Comparison ────────────────────────────────────────
    print()
    print_comparison(crypto_stats, forex_stats, stock_stats)
    print()
