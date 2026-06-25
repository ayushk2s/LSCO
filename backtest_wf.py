"""
backtest_wf.py -- Comprehensive Walk-Forward Parameter Search
=============================================================
Loads 30 symbols once, then sweeps 126 combinations of:
  9 entry variants  x  7 exit presets  x  2 risk levels  =  126 combos

Entry variants
--------------
  base        : current LZR signal (vol filter, regime filter, zone touch)
  bull_bar    : + require signal bar closed UP  (close > open = initial bounce confirmed)
  bear_bar    : + require signal bar closed DOWN (price still in zone = deeper support test)
  rsi40       : + RSI(14) on 4h must be < 40   (oversold entry)
  rsi45       : + RSI(14) on 4h must be < 45   (mild oversold)
  bull+rsi40  : bull_bar AND rsi < 40           (strong reversal confirmation)
  no_vol      : volume filter disabled (vol_mult=0)
  hi_vol      : volume filter tightened to 2.5x  (only high-volume reversals)
  no_regime   : regime filter (weekly/daily EMA) disabled (trade all market conditions)

Exit presets
------------
  tight       : partial_tp=0.5x, trail=0.5x, sl=0.75x, hard=4x ATR
  v14         : partial_tp=1.0x, trail=0.8x, sl=0.75x, hard=6x ATR  [baseline]
  v14_wtr     : partial_tp=1.0x, trail=1.2x, sl=0.75x, hard=6x ATR  [wider trail]
  medium      : partial_tp=1.5x, trail=1.0x, sl=0.75x, hard=6x ATR
  wide        : partial_tp=2.0x, trail=1.5x, sl=0.75x, hard=8x ATR
  tight_sl    : partial_tp=1.0x, trail=0.8x, sl=0.50x, hard=6x ATR  [tighter stop]
  hold_to_htp : partial_tp=99x  (disabled), trail=0.8x, sl=0.75x, hard=6x ATR

Risk
----
  r5: 5% of balance risked per trade
  r7: 7% of balance risked per trade  [baseline]

Walk-forward methodology
------------------------
  Run each combo on FULL period (2021-2026) with no date restriction.
  Split trades at TRAIN_END = 2023-01-01.
  Compute TRAIN metrics from 2021-2022 trades and equity.
  Compute TEST  metrics from 2023-2026 trades and equity.
  State (fired_zones, balance, cooldowns) carries over: realistic walk-forward.
  OVF_ratio = test_calmar / train_calmar: values near 1.0 = generalizes well.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import heapq, time
import pandas as pd
import numpy as np
from pathlib import Path
from lzr_core import load_and_prepare, DEFAULT_CFG
from backtest_v14 import exec_1m_v14, MARGIN_RATE, FUNDING_RATE_8H
from backtest_v14_all import SLIP_BY_ASSET, DEFAULT_SLIP

DATA_DIR   = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUT        = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT.mkdir(exist_ok=True)

TRAIN_END  = pd.Timestamp("2023-01-01")   # split date

# Base costs (same as v14)
BASE_CFG = {**DEFAULT_CFG,
    "initial_balance": 1_000.0,
    "cd_win_bars"    : 3,
    "cd_loss_bars"   : 42,
    "fee_rt"         : 0.0008,    # 0.04%/side taker
    "max_dd_pct"     : None,
}

# ── Entry variants ─────────────────────────────────────────────────────────────
ENTRY_VARIANTS = {
    "base"      : dict(vol_mult=1.8, bull_f=False, bear_f=False, rsi=None,  no_reg=False),
    "bull_bar"  : dict(vol_mult=1.8, bull_f=True,  bear_f=False, rsi=None,  no_reg=False),
    "bear_bar"  : dict(vol_mult=1.8, bull_f=False, bear_f=True,  rsi=None,  no_reg=False),
    "rsi40"     : dict(vol_mult=1.8, bull_f=False, bear_f=False, rsi=40.0,  no_reg=False),
    "rsi45"     : dict(vol_mult=1.8, bull_f=False, bear_f=False, rsi=45.0,  no_reg=False),
    "bull+rsi40": dict(vol_mult=1.8, bull_f=True,  bear_f=False, rsi=40.0,  no_reg=False),
    "no_vol"    : dict(vol_mult=0.0, bull_f=False, bear_f=False, rsi=None,  no_reg=False),
    "hi_vol"    : dict(vol_mult=2.5, bull_f=False, bear_f=False, rsi=None,  no_reg=False),
    "no_regime" : dict(vol_mult=1.8, bull_f=False, bear_f=False, rsi=None,  no_reg=True),
}

# ── Exit presets ───────────────────────────────────────────────────────────────
EXIT_PRESETS = {
    "tight"      : dict(partial_tp_mult=0.5,  trail_dist_mult=0.5, sl_mult=0.75, hard_tp_mult=4.0),
    "v14"        : dict(partial_tp_mult=1.0,  trail_dist_mult=0.8, sl_mult=0.75, hard_tp_mult=6.0),
    "v14_wtr"    : dict(partial_tp_mult=1.0,  trail_dist_mult=1.2, sl_mult=0.75, hard_tp_mult=6.0),
    "medium"     : dict(partial_tp_mult=1.5,  trail_dist_mult=1.0, sl_mult=0.75, hard_tp_mult=6.0),
    "wide"       : dict(partial_tp_mult=2.0,  trail_dist_mult=1.5, sl_mult=0.75, hard_tp_mult=8.0),
    "tight_sl"   : dict(partial_tp_mult=1.0,  trail_dist_mult=0.8, sl_mult=0.50, hard_tp_mult=6.0),
    "hold_to_htp": dict(partial_tp_mult=99.0, trail_dist_mult=0.8, sl_mult=0.75, hard_tp_mult=6.0),
}

RISK_PRESETS = {
    "r5": dict(risk_pct=0.05),
    "r7": dict(risk_pct=0.07),
}


# ── RSI computation ────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / (loss + 1e-12)
    return (100 - 100 / (1 + rs)).fillna(50.0)


# ── Period metrics ─────────────────────────────────────────────────────────────

def period_metrics(trades, equity_slice, init_bal):
    """
    Compute key metrics for a sub-period of trades + equity.
    init_bal: balance at start of this period.
    """
    if not trades:
        return dict(n=0, wr=0.0, cagr=0.0, max_dd=0.0, calmar=0.0,
                    pf=0.0, gross=0.0, net=0.0, final=init_bal)

    df    = pd.DataFrame(trades)
    n     = len(df)
    wins  = df[df["result"] == "WIN"]
    losss = df[df["result"] != "WIN"]
    wr    = len(wins) / n * 100
    gross = df["gross"].sum()
    net   = df["net"].sum()
    final = init_bal + net
    pf    = (wins["net"].sum() / max(abs(losss["net"].sum()), 1e-9)
             if len(losss) else float("inf"))

    # CAGR
    ts0  = pd.to_datetime(df["ts"].min())
    ts1  = pd.to_datetime(df["close_ts"].max())
    yrs  = max((ts1 - ts0).days / 365.25, 0.05)
    cagr = (final / init_bal) ** (1.0 / yrs) - 1.0

    # Max drawdown from equity slice
    if equity_slice and len(equity_slice) > 1:
        eq  = pd.Series(equity_slice, dtype=float)
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    else:
        mdd = 0.0

    calmar = abs(cagr / mdd) if mdd < -0.001 else (float("inf") if cagr > 0 else 0.0)

    return dict(n=n, wr=round(wr, 1), cagr=round(cagr * 100, 1),
                max_dd=round(mdd * 100, 1), calmar=round(calmar, 2),
                pf=round(pf, 2), gross=round(gross, 0), net=round(net, 0),
                final=round(final, 2))


# ── Walk-forward portfolio runner ──────────────────────────────────────────────

def run_portfolio_wf(symbols, sym_data, rsi_data, cfg, ev):
    """
    Full-period portfolio runner with entry filter support.

    ev: entry variant dict with keys:
        vol_mult  : float (0 = volume filter disabled)
        bull_f    : bool  (require close > open at signal bar)
        bear_f    : bool  (require close < open at signal bar)
        rsi       : float or None (require RSI < this value)
        no_reg    : bool  (skip weekly + daily regime filter)

    Returns (all_trades, equity_curve, final_balance)
    """
    INITIAL_BAL = cfg["initial_balance"]
    RISK_PCT    = cfg["risk_pct"]
    MAX_BAL     = cfg["max_balance"]
    MAX_POS_N   = cfg["max_pos_notional"]
    MIN_BAL_R   = cfg["min_bal_ratio"]
    ZW          = cfg["zone_window"]
    CD_LOSS     = cfg["cd_loss_bars"]
    CD_WIN      = cfg["cd_win_bars"]
    SL_MULT     = cfg["sl_mult"]
    ZT_MULT     = cfg["zone_touch_mult"]
    PTP_MULT    = cfg["partial_tp_mult"]
    HTP_MULT    = cfg["hard_tp_mult"]
    TRAIL_MULT  = cfg["trail_dist_mult"]
    FEE_SIDE    = cfg["fee_rt"] / 2.0

    VOL_MULT   = ev["vol_mult"]
    BULL_F     = ev["bull_f"]
    BEAR_F     = ev["bear_f"]
    RSI_THRESH = ev["rsi"]
    NO_REGIME  = ev["no_reg"]

    WARMUP = max(cfg["ema_period"], cfg["vol_ma_period"],
                 ZW * 2, cfg["weekly_ema_period"] * 7) + 10

    ref_sig = sym_data[symbols[0]]["df_sig"]
    n       = len(ref_sig)

    balance      = INITIAL_BAL
    peak_balance = INITIAL_BAL

    sym_state = {s: dict(
        cooldown_until=0, last_signal_i=-999,
        fired_zones=set(), in_trade=False,
        entry_px=0.0, entry_qty=0.0, notional=0.0,
        partial_tp_ts=None, partial_locked=0.0,
    ) for s in symbols}

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
            ss["in_trade"]       = False
            ss["entry_px"]       = 0.0
            ss["entry_qty"]      = 0.0
            ss["notional"]       = 0.0
            ss["partial_tp_ts"]  = None
            ss["partial_locked"] = 0.0
            all_trades.append(td)

        # Mark-to-market equity
        unrealized = 0.0
        for sym in symbols:
            ss = sym_state[sym]
            if not ss["in_trade"] or ss["entry_qty"] == 0:
                continue
            sd = sym_data[sym]
            if i >= len(sd["df_sig"]):
                continue
            curr_px = sd["df_sig"]["close"].iloc[i]
            ptp_ts  = ss["partial_tp_ts"]
            if ptp_ts is not None and sig_ts >= ptp_ts:
                unrealized += ss["partial_locked"] + (curr_px - ss["entry_px"]) * ss["entry_qty"] / 2.0
            else:
                unrealized += (curr_px - ss["entry_px"]) * ss["entry_qty"]

        equity_curve.append(balance + unrealized)

        if i < WARMUP or balance < INITIAL_BAL * MIN_BAL_R:
            continue

        # Margin-based available balance
        margin_used = sum(sym_state[s]["notional"] * MARGIN_RATE
                          for s in symbols if sym_state[s]["in_trade"])
        avail_bal = max(balance - margin_used, 0.0)

        for sym in symbols:
            ss = sym_state[sym]
            sd = sym_data[sym]

            if ss["in_trade"] or i <= ss["cooldown_until"]:
                continue

            sym_n = len(sd["df_sig"])
            if i >= sym_n or i + 1 >= sym_n:
                continue

            atr   = sd["atr_sig"].iloc[i]
            ema   = sd["ema_sig"].iloc[i]
            vol_r = sd["vol_ratio"].iloc[i]

            if atr <= 0 or np.isnan(atr) or np.isnan(ema) or np.isnan(vol_r):
                continue

            # ── Regime filter (skipped if no_regime) ──────────────────────────
            if not NO_REGIME:
                week_floor = sig_ts - pd.Timedelta(days=sig_ts.dayofweek)
                week_floor = pd.Timestamp(week_floor.date())
                w_idx = int(sd["weekly_ts"].searchsorted(week_floor, side="left")) - 1
                if w_idx < 1:
                    continue
                if sd["df_weekly"]["close"].iloc[w_idx] < sd["weekly_ema"].iloc[w_idx]:
                    continue

                d_idx = int(sd["daily_ts"].searchsorted(sig_ts, side="right")) - 1
                if d_idx < 1:
                    continue
                d_cls = sd["daily_cls_s"].iloc[d_idx]
                d_ema = sd["daily_ema_s"].iloc[d_idx]
                if np.isnan(d_cls) or np.isnan(d_ema) or d_cls < d_ema:
                    continue

            # ── Volume filter ──────────────────────────────────────────────────
            if VOL_MULT > 0 and (np.isnan(vol_r) or vol_r < VOL_MULT):
                continue

            # ── Zone check ─────────────────────────────────────────────────────
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

            if close < ema:
                continue

            # ── Candle direction filter ────────────────────────────────────────
            sig_bar = sd["df_sig"].iloc[i]
            if BULL_F and sig_bar["close"] <= sig_bar["open"]:
                continue
            if BEAR_F and sig_bar["close"] >= sig_bar["open"]:
                continue

            # ── RSI filter ─────────────────────────────────────────────────────
            if RSI_THRESH is not None:
                rsi_val = rsi_data[sym].iloc[i] if i < len(rsi_data[sym]) else np.nan
                if np.isnan(rsi_val) or rsi_val > RSI_THRESH:
                    continue

            # ── Entry ──────────────────────────────────────────────────────────
            entry_px = sd["df_sig"]["open"].iloc[i + 1]
            sl_px    = zone_lo - SL_MULT * atr
            sl_dist  = entry_px - sl_px
            if sl_dist <= 0:
                continue

            partial_tp = entry_px + PTP_MULT * atr
            hard_tp    = entry_px + HTP_MULT * atr

            eff_bal = min(avail_bal, MAX_BAL)
            if eff_bal < INITIAL_BAL * MIN_BAL_R:
                continue

            qty      = min((eff_bal * RISK_PCT) / sl_dist, MAX_POS_N / entry_px)
            if qty <= 0:
                continue

            notional = qty * entry_px
            margin   = notional * MARGIN_RATE
            if margin > avail_bal:
                continue

            entry_ts = sd["df_sig"].index[i + 1]
            slip     = SLIP_BY_ASSET.get(sym, DEFAULT_SLIP)
            m_start  = int(sd["m_ts"].searchsorted(entry_ts))

            info = exec_1m_v14(sd["df_1m"], m_start, entry_ts, entry_px, sl_px,
                               partial_tp, hard_tp, atr, qty,
                               FEE_SIDE, slip, TRAIL_MULT)
            if info is None:
                continue

            (close_ts, gross, exit_px, total_fee, total_slip,
             funding, ptp_ts, ptp_locked) = info

            net    = gross - total_fee - total_slip - funding
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
            ss["notional"]       = notional
            ss["partial_tp_ts"]  = ptp_ts
            ss["partial_locked"] = ptp_locked

            avail_bal -= margin
            avail_bal  = max(avail_bal, 0.0)

            td = dict(
                symbol=sym,
                ts=str(sig_ts), close_ts=str(close_ts),
                entry=round(entry_px, 6), exit=round(exit_px, 6),
                sl=round(sl_px, 6), notional=round(notional, 2),
                gross=round(gross, 4), fee=round(total_fee, 4),
                slip=round(total_slip, 4), funding=round(funding, 4),
                net=round(net, 4), result=result,
                duration_h=round(dur_h, 1),
                had_ptp=(ptp_ts is not None),
            )

            heapq.heappush(pending, (close_bar, trade_ctr, sym, net, td))
            trade_ctr += 1

    # Flush remaining open trades
    while pending:
        cb, _, sym, net, td = heapq.heappop(pending)
        balance += net
        ss = sym_state[sym]
        ss["in_trade"]       = False
        ss["entry_px"]       = 0.0
        ss["entry_qty"]      = 0.0
        ss["notional"]       = 0.0
        ss["partial_tp_ts"]  = None
        ss["partial_locked"] = 0.0
        all_trades.append(td)

    return all_trades, equity_curve, balance


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    W = "=" * 100
    t0 = time.time()

    print(W)
    print("  LZR WALK-FORWARD PARAMETER SEARCH")
    print("  9 entry variants  x  7 exit presets  x  2 risk levels  =  126 combinations")
    print(f"  Train: 2021-01-01 to {TRAIN_END.date()}   |   Test: {TRAIN_END.date()} to 2026-03-28")
    print(W)

    # ── Load data (once) ───────────────────────────────────────────────────────
    symbols = sorted(p.stem.replace("1m", "") for p in DATA_DIR.glob("*1m.csv"))
    print(f"\n  Loading {len(symbols)} symbols (this takes 3-5 min, done only once)...\n")
    sym_data = load_and_prepare(symbols, BASE_CFG)
    symbols  = [s for s in symbols if s in sym_data]

    # ── Pre-compute RSI for every symbol ──────────────────────────────────────
    print("\n  Computing RSI(14) for all symbols...")
    rsi_data = {}
    for sym in symbols:
        rsi_data[sym] = compute_rsi(sym_data[sym]["df_sig"]["close"])
    print(f"  RSI computed. Data loading done in {(time.time()-t0)/60:.1f} min.\n")

    # ── Find equity curve split index ──────────────────────────────────────────
    ref_idx   = sym_data[symbols[0]]["df_sig"].index
    split_bar = int(ref_idx.searchsorted(TRAIN_END, side="left"))
    print(f"  Train/test split at bar {split_bar} of {len(ref_idx)}  ({TRAIN_END.date()})\n")

    # ── Build combo list ───────────────────────────────────────────────────────
    combos = []
    for en, ev in ENTRY_VARIANTS.items():
        for xn, xv in EXIT_PRESETS.items():
            for rn, rv in RISK_PRESETS.items():
                cfg = {**BASE_CFG, **xv, **rv}
                combos.append((en, xn, rn, ev, cfg))

    print(f"  Running {len(combos)} combinations...\n")
    print(f"  {'#':>4}  {'Entry':<12} {'Exit':<12} {'Risk':<5}  "
          f"{'Tr_N':>5} {'Tr_WR':>6} {'Tr_CAGR':>8} {'Tr_DD':>7} {'Tr_Cal':>7}  |  "
          f"{'Te_N':>5} {'Te_WR':>6} {'Te_CAGR':>8} {'Te_DD':>7} {'Te_Cal':>7}  OVF")
    print("  " + "-" * 98)

    results = []

    for ci, (en, xn, rn, ev, cfg) in enumerate(combos, 1):
        t_c = time.time()
        trades, eq_curve, final_bal = run_portfolio_wf(symbols, sym_data, rsi_data, cfg, ev)

        # Split at TRAIN_END
        train_trades = [t for t in trades if pd.Timestamp(t["ts"]) <  TRAIN_END]
        test_trades  = [t for t in trades if pd.Timestamp(t["ts"]) >= TRAIN_END]

        train_eq = eq_curve[:split_bar]
        test_eq  = eq_curve[split_bar:]

        # Balance split: sum of net from train trades
        init_b       = cfg["initial_balance"]
        train_net    = sum(t["net"] for t in train_trades)
        train_final  = init_b + train_net
        # test starts from whatever train ended at
        test_init    = train_final

        tr = period_metrics(train_trades, train_eq, init_b)
        te = period_metrics(test_trades,  test_eq,  test_init)

        # Overfitting ratio
        if tr["calmar"] > 0 and tr["calmar"] != float("inf"):
            ovf = round(te["calmar"] / tr["calmar"], 2)
        else:
            ovf = 0.0

        elapsed_c = time.time() - t_c

        row = dict(
            rank=ci, entry=en, exit=xn, risk=rn,
            tr_n=tr["n"], tr_wr=tr["wr"], tr_cagr=tr["cagr"],
            tr_dd=tr["max_dd"], tr_cal=tr["calmar"],
            te_n=te["n"], te_wr=te["wr"], te_cagr=te["cagr"],
            te_dd=te["max_dd"], te_cal=te["calmar"],
            ovf=ovf, elapsed=round(elapsed_c, 1),
            tr_final=tr["final"], te_final=te["final"],
            tr_gross=tr["gross"], te_gross=te["gross"],
            tr_net=tr["net"], te_net=te["net"],
        )
        results.append(row)

        eta = (time.time() - t0) / ci * (len(combos) - ci)
        print(f"  {ci:>4}  {en:<12} {xn:<12} {rn:<5}  "
              f"{tr['n']:>5} {tr['wr']:>5.1f}% {tr['cagr']:>+7.1f}% {tr['max_dd']:>+6.1f}% {tr['calmar']:>7.2f}  |  "
              f"{te['n']:>5} {te['wr']:>5.1f}% {te['cagr']:>+7.1f}% {te['max_dd']:>+6.1f}% {te['calmar']:>7.2f}  "
              f"{ovf:>5.2f}   ETA {eta/60:.0f}m")

    # ── Sort and report ────────────────────────────────────────────────────────
    results.sort(key=lambda x: -x["tr_cal"])

    print(f"\n\n{W}")
    print("  FINAL RESULTS -- SORTED BY TRAIN CALMAR (highest = best in-sample)")
    print(W)
    print(f"\n  {'Rk':>3}  {'Entry':<12} {'Exit':<12} {'R':>3}  "
          f"{'Tr_N':>5} {'Tr_WR':>6} {'Tr_CAGR':>8} {'Tr_DD':>7} {'Tr_Cal':>7}  |  "
          f"{'Te_N':>5} {'Te_WR':>6} {'Te_CAGR':>8} {'Te_DD':>7} {'Te_Cal':>7}  {'OVF':>5}  Status")
    print("  " + "-" * 110)

    for ri, r in enumerate(results, 1):
        # Status flag
        if r["te_cagr"] > 0 and r["te_cal"] >= 1.0:
            status = "*** INVESTABLE"
        elif r["te_cagr"] > 0 and r["te_cal"] >= 0.5:
            status = " ** MARGINAL"
        elif r["te_cagr"] > 0:
            status = "  * POSITIVE"
        elif r["ovf"] >= 0.7:
            status = "  ~ generalized"
        else:
            status = ""

        flag = ">>>" if ri <= 10 else "   "
        print(f"  {flag}{ri:>3}  {r['entry']:<12} {r['exit']:<12} {r['risk']:<3}  "
              f"{r['tr_n']:>5} {r['tr_wr']:>5.1f}% {r['tr_cagr']:>+7.1f}% {r['tr_dd']:>+6.1f}% {r['tr_cal']:>7.2f}  |  "
              f"{r['te_n']:>5} {r['te_wr']:>5.1f}% {r['te_cagr']:>+7.1f}% {r['te_dd']:>+6.1f}% {r['te_cal']:>7.2f}  "
              f"{r['ovf']:>5.2f}  {status}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  SUMMARY")
    print(W)

    pos_test  = [r for r in results if r["te_cagr"] > 0]
    inv_test  = [r for r in results if r["te_cagr"] > 0 and r["te_cal"] >= 1.0]
    good_gen  = [r for r in results if r["ovf"] >= 0.5 and r["tr_cal"] > 0]

    print(f"\n  Total combos tested:          {len(results)}")
    print(f"  Positive test CAGR:           {len(pos_test)}/{len(results)}")
    print(f"  Test Calmar >= 1.0 (investable): {len(inv_test)}/{len(results)}")
    print(f"  OVF ratio >= 0.5 (good generalization): {len(good_gen)}/{len(results)}")

    if pos_test:
        best_te = max(pos_test, key=lambda x: x["te_cal"])
        print(f"\n  BEST by TEST CALMAR:")
        print(f"    Entry={best_te['entry']}  Exit={best_te['exit']}  Risk={best_te['risk']}")
        print(f"    Train: CAGR {best_te['tr_cagr']:+.1f}%  DD {best_te['tr_dd']:+.1f}%  Calmar {best_te['tr_cal']:.2f}")
        print(f"    Test:  CAGR {best_te['te_cagr']:+.1f}%  DD {best_te['te_dd']:+.1f}%  Calmar {best_te['te_cal']:.2f}")
        print(f"    OVF ratio: {best_te['ovf']:.2f}")

    if good_gen:
        best_gen = max(good_gen, key=lambda x: x["te_cal"])
        print(f"\n  BEST by TEST CALMAR (with OVF >= 0.5):")
        print(f"    Entry={best_gen['entry']}  Exit={best_gen['exit']}  Risk={best_gen['risk']}")
        print(f"    Train: CAGR {best_gen['tr_cagr']:+.1f}%  DD {best_gen['tr_dd']:+.1f}%  Calmar {best_gen['tr_cal']:.2f}")
        print(f"    Test:  CAGR {best_gen['te_cagr']:+.1f}%  DD {best_gen['te_dd']:+.1f}%  Calmar {best_gen['te_cal']:.2f}")

    # Best test across all
    best_any = max(results, key=lambda x: x["te_cagr"])
    print(f"\n  HIGHEST test CAGR across all combos:")
    print(f"    {best_any['entry']} / {best_any['exit']} / {best_any['risk']}: "
          f"test CAGR {best_any['te_cagr']:+.1f}%  Calmar {best_any['te_cal']:.2f}")

    # ── Breakdown by entry variant ─────────────────────────────────────────────
    print(f"\n{W}")
    print("  ENTRY VARIANT SUMMARY (avg TEST metrics across all exit/risk combos)")
    print(W)
    print(f"\n  {'Entry':<12} {'Combos':>7} {'Avg_CAGR':>9} {'Avg_Cal':>8} {'Pos_CAGR':>9} {'Best_Cal':>9}")
    print("  " + "-" * 60)
    for en in ENTRY_VARIANTS:
        rows = [r for r in results if r["entry"] == en]
        avg_cagr = np.mean([r["te_cagr"] for r in rows])
        avg_cal  = np.mean([r["te_cal"]  for r in rows])
        pos_c    = sum(1 for r in rows if r["te_cagr"] > 0)
        best_cal = max(r["te_cal"] for r in rows)
        print(f"  {en:<12} {len(rows):>7} {avg_cagr:>+8.1f}% {avg_cal:>8.2f} {pos_c:>8}/{len(rows)} {best_cal:>9.2f}")

    # ── Breakdown by exit preset ───────────────────────────────────────────────
    print(f"\n{W}")
    print("  EXIT PRESET SUMMARY (avg TEST metrics across all entry/risk combos)")
    print(W)
    print(f"\n  {'Exit':<14} {'Combos':>7} {'Avg_CAGR':>9} {'Avg_Cal':>8} {'Pos_CAGR':>9} {'Best_Cal':>9}")
    print("  " + "-" * 64)
    for xn in EXIT_PRESETS:
        rows = [r for r in results if r["exit"] == xn]
        avg_cagr = np.mean([r["te_cagr"] for r in rows])
        avg_cal  = np.mean([r["te_cal"]  for r in rows])
        pos_c    = sum(1 for r in rows if r["te_cagr"] > 0)
        best_cal = max(r["te_cal"] for r in rows)
        print(f"  {xn:<14} {len(rows):>7} {avg_cagr:>+8.1f}% {avg_cal:>8.2f} {pos_c:>8}/{len(rows)} {best_cal:>9.2f}")

    # ── Save results ───────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("tr_cal", ascending=False).reset_index(drop=True)
    df_res.index += 1
    out_path = OUT / "wf_results.csv"
    df_res.to_csv(out_path)
    print(f"\n  Full results saved: {out_path}")

    total_min = (time.time() - t0) / 60
    print(f"  Total runtime: {total_min:.1f} min")
    print()


if __name__ == "__main__":
    main()
