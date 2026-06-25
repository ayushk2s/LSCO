"""
trade_analyzer_v8.py  --  LZR v8 Deep Trade Analyzer
======================================================
For every trade in the 4h-signal / 1m-execution backtest:

  LOSING trades:
    • Root cause  (why it lost)
    • Which filters would have SKIPPED this trade (saved the loss)
    • Whether a wider/tighter SL would have helped

  WINNING trades:
    • Why it won  (what made this setup work)
    • Whether a wider TP would have captured more profit
    • How much was left on the table

SUMMARY:
    • Win rate split by: direction / volume / zone-freshness / ATR environment / EMA slope
    • For each filter: net_pf improvement if applied
    • Top 3 actionable rule changes to recover edge

SHORT SL bug FIXED throughout.
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
DATA_DIR    = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR  = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYZE_TF  = "4h"     # signal timeframe to analyze (best from MTF sweep)
ZONE_TF     = "16h"    # zone timeframe = 4× signal

INITIAL_BALANCE = 1_000.0
RISK_PCT        = 0.10
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
EMA_PERIOD      = 20
EMA_LOOKBACK    = 3
ZONE_MAX_TOUCH  = 2
CRYPTO_FEE_RT   = 0.0004
CRYPTO_SLIP_PCT = 0.0003
MAX_POSITION_NOTIONAL = 500_000.0
MAX_BALANCE           = 200_000.0
MIN_BAL_RATIO         = 0.20
WARMUP_BARS           = ATR_PERIOD * 4

# 4h → bars_per_day=6, zone_window=42 bars(7d), cd_loss=3(18h), cd_win=1
BARS_PER_DAY = 6
ZONE_WINDOW  = 42
CD_LOSS      = 3
CD_WIN       = 1

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# Hypothetical filter thresholds to test
HYP_VOL_HIGH   = 2.5    # stricter volume threshold (vs current 1.8)
HYP_SL_WIDE    = 1.0    # wider SL in ATR units (vs current 0.75)
HYP_SL_TIGHT   = 0.5    # tighter SL in ATR units
HYP_ATR_MAX    = 0.025  # skip if ATR > 2.5% of price
HYP_EMA_SLOPE  = 0.002  # skip if EMA slope < 0.2% (weak trend)
HYP_TP_WIDE    = 5.0    # hypothetical wider hard TP (vs current 3.0)


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_crypto(symbol):
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


def find_zones(df_zone):
    n, lb = len(df_zone), SWING_LOOKBACK
    levels = []
    for i in range(lb, n - lb):
        hi = df_zone["high"].iloc[i]
        lo = df_zone["low"].iloc[i]
        if hi == df_zone["high"].iloc[i - lb:i + lb + 1].max():
            levels.append(hi)
        if lo == df_zone["low"].iloc[i - lb:i + lb + 1].min():
            levels.append(lo)
    merged = []
    for lvl in sorted(set(levels)):
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


# ─── 1m executor with exit-type tracking (SHORT SL fixed) ─────────────────────

def _exec_1m_ext(df_1m, m_start, entry_px, direction,
                  sl_px, partial_tp_px, hard_tp_px,
                  trade_atr, qty, fee_side, slip_pct):
    """Returns (close_ts, gross, exit_px, fee, slip, state, exit_type)"""
    half    = qty / 2
    state   = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px
    fee_acc        = entry_px * qty * fee_side
    slip_acc       = entry_px * qty * slip_pct

    for m_idx in range(m_start, len(df_1m)):
        mr = df_1m.iloc[m_idx]
        closed, gross, exit_px, exit_type = False, None, 0.0, ""

        if state == "full":
            if direction == "LONG":
                if mr["open"] <= sl_px:
                    exit_px, gross, exit_type, closed = mr["open"], (mr["open"] - entry_px) * qty, "gap_sl", True
                elif mr["low"] <= sl_px:
                    exit_px, gross, exit_type, closed = sl_px, (sl_px - entry_px) * qty, "sl", True
                elif mr["open"] >= partial_tp_px:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["open"]
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["high"] >= partial_tp_px:
                    partial_locked = (partial_tp_px - entry_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["high"]
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
            else:  # SHORT
                if mr["open"] >= sl_px:
                    exit_px, gross, exit_type, closed = mr["open"], (entry_px - mr["open"]) * qty, "gap_sl", True
                elif mr["high"] >= sl_px:
                    exit_px, gross, exit_type, closed = sl_px, (entry_px - sl_px) * qty, "sl", True
                elif mr["open"] <= partial_tp_px:
                    partial_locked = (entry_px - partial_tp_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["open"]
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state = "partial"
                elif mr["low"] <= partial_tp_px:
                    partial_locked = (entry_px - partial_tp_px) * half
                    fee_acc  += partial_tp_px * half * fee_side
                    slip_acc += partial_tp_px * half * slip_pct
                    running_ext = mr["low"]
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    state = "partial"

        elif state == "partial":
            if direction == "LONG":
                old_trail = trail_sl
                if mr["open"] <= old_trail:
                    exit_px, gross, exit_type, closed = mr["open"], partial_locked + (mr["open"] - entry_px) * half, "gap_trail", True
                elif mr["open"] >= hard_tp_px:
                    exit_px, gross, exit_type, closed = hard_tp_px, partial_locked + (hard_tp_px - entry_px) * half, "gap_hardtp", True
                else:
                    running_ext = max(running_ext, mr["high"])
                    trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * trade_atr)
                    if mr["low"] <= old_trail:
                        exit_px, gross, exit_type, closed = old_trail, partial_locked + (old_trail - entry_px) * half, "trail", True
                    elif mr["high"] >= hard_tp_px:
                        exit_px, gross, exit_type, closed = hard_tp_px, partial_locked + (hard_tp_px - entry_px) * half, "hard_tp", True
            else:  # SHORT partial
                old_trail = trail_sl
                if mr["open"] >= old_trail:
                    exit_px, gross, exit_type, closed = mr["open"], partial_locked + (entry_px - mr["open"]) * half, "gap_trail", True
                elif mr["open"] <= hard_tp_px:
                    exit_px, gross, exit_type, closed = hard_tp_px, partial_locked + (entry_px - hard_tp_px) * half, "gap_hardtp", True
                else:
                    running_ext = min(running_ext, mr["low"])
                    trail_sl    = min(entry_px, running_ext + TRAIL_DIST_MULT * trade_atr)
                    if mr["high"] >= old_trail:
                        exit_px, gross, exit_type, closed = old_trail, partial_locked + (entry_px - old_trail) * half, "trail", True
                    elif mr["low"] <= hard_tp_px:
                        exit_px, gross, exit_type, closed = hard_tp_px, partial_locked + (entry_px - hard_tp_px) * half, "hard_tp", True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * fee_side
            slip_acc += exit_px * exit_qty * slip_pct
            return (df_1m.index[m_idx], gross, exit_px, fee_acc, slip_acc, state, exit_type)

    return None


def _scan_maefme(df_1m, m_start, m_end, entry_px, direction, trade_atr):
    """Compute MAE and MFE in ATR units over the 1m bars of a trade."""
    sub = df_1m.iloc[m_start:min(m_end + 1, len(df_1m))]
    if sub.empty or trade_atr <= 0:
        return 0.0, 0.0
    if direction == "LONG":
        mfe = max(0, sub["high"].max() - entry_px)
        mae = max(0, entry_px - sub["low"].min())
    else:
        mfe = max(0, entry_px - sub["low"].min())
        mae = max(0, sub["high"].max() - entry_px)
    return round(mae / trade_atr, 3), round(mfe / trade_atr, 3)


def _wider_sl_would_survive(mae_atr, mfe_atr, direction, sl_wide=HYP_SL_WIDE):
    """Check if a wider SL would have kept the trade alive to reach partial TP."""
    # Only relevant if the trade was a full-SL loss (mfe < partial_tp)
    if mfe_atr >= PARTIAL_TP_MULT:
        return False  # partial TP was hit anyway
    # Wider SL survives if MAE didn't reach the wider SL
    return mae_atr < sl_wide


# ─── Trade classification ─────────────────────────────────────────────────────

def classify_loss(t):
    """Return primary root cause of a losing trade."""
    mae    = t["mae_atr"]
    mfe    = t["mfe_atr"]
    vol_r  = t["vol_ratio"]
    atr_pct = t["atr_pct"]
    slope  = t["ema_slope_pct"]
    touch  = t["zone_touch"]
    etype  = t["exit_type"]

    reasons = []

    # Did price move against us immediately?
    if mfe < 0.15:
        reasons.append("ZONE_REJECTED — price moved against us from bar 1, zone had no support")
    elif mfe < 0.40:
        reasons.append("WEAK_BOUNCE — zone gave a tiny bounce before collapsing")

    # Was the confirmation volume weak?
    if vol_r < 2.0:
        reasons.append("WEAK_VOLUME — vol ratio {:.2f} barely above the 1.8 threshold".format(vol_r))

    # Was the market too volatile at entry?
    if atr_pct > 0.025:
        reasons.append("HIGH_VOLATILITY — ATR was {:.1f}% of price (noisy conditions)".format(atr_pct * 100))

    # Was the zone already used?
    if touch >= 2:
        reasons.append("AGED_ZONE — this was the {}nd+ touch of the zone (less reliable)".format(touch))

    # Was the EMA trend weak?
    if abs(slope) < 0.001:
        reasons.append("WEAK_TREND — EMA slope nearly flat ({:.3f}%), weak trend context".format(slope * 100))

    # Gap stop-loss
    if "gap" in etype:
        reasons.append("GAP_SL — a price gap filled the SL at a worse price than expected")

    if not reasons:
        reasons.append("GENUINE_REVERSAL — no obvious filter would have skipped this, zone failed")

    return reasons


def classify_win(t):
    """Return primary reasons a trade won."""
    mfe   = t["mfe_atr"]
    vol_r = t["vol_ratio"]
    slope = t["ema_slope_pct"]
    touch = t["zone_touch"]
    etype = t["exit_type"]
    dur   = t["duration_h"]

    reasons = []

    if touch == 1:
        reasons.append("FRESH_ZONE — first touch of this zone level (cleanest signal)")
    if vol_r > 3.0:
        reasons.append("STRONG_VOLUME — vol ratio {:.1f}x confirms institutional interest".format(vol_r))
    if abs(slope) > 0.003:
        reasons.append("STRONG_TREND — EMA slope {:.2f}% (clear directional momentum)".format(slope * 100))
    if etype == "hard_tp":
        reasons.append("FULL_TARGET_HIT — price ran all the way to 3.0 ATR hard TP")
    elif etype in ("trail", "gap_trail"):
        reasons.append("TRAIL_WIN — partial banked at 1 ATR, remainder trailed out above break-even")
    if mfe > HARD_TP_MULT + 0.5:
        reasons.append("MONEY_LEFT — MFE {:.1f} ATR exceeded hard TP; wider target would earn more".format(mfe))
    if dur < 12:
        reasons.append("QUICK_WIN — resolved within {:.0f}h (strong momentum)".format(dur))

    if not reasons:
        reasons.append("SOLID_SETUP — clean trade with no standout factor")

    return reasons


# ─── Hypothetical filter checks ───────────────────────────────────────────────

def check_hypotheticals(t):
    """
    Returns dict of hypothetical filter name -> action and outcome.
    action: "SKIP" (would not have taken this trade)
    outcome for SKIP: "SAVE_LOSS" / "MISS_WIN" depending on actual result
    """
    hyp = {}
    r   = t["result"]

    # Filter 1: LONG-only (skip shorts)
    if t["dir"] == "SHORT":
        hyp["LONG_ONLY"] = {"action": "SKIP", "effect": "SAVE_LOSS" if r == "LOSS" else "MISS_WIN"}
    else:
        hyp["LONG_ONLY"] = {"action": "KEEP", "effect": None}

    # Filter 2: Higher volume threshold (2.5 instead of 1.8)
    if t["vol_ratio"] < HYP_VOL_HIGH:
        hyp["VOL_2.5"] = {"action": "SKIP", "effect": "SAVE_LOSS" if r == "LOSS" else "MISS_WIN"}
    else:
        hyp["VOL_2.5"] = {"action": "KEEP", "effect": None}

    # Filter 3: First-touch only (skip 2nd zone touch)
    if t["zone_touch"] >= 2:
        hyp["1ST_TOUCH"] = {"action": "SKIP", "effect": "SAVE_LOSS" if r == "LOSS" else "MISS_WIN"}
    else:
        hyp["1ST_TOUCH"] = {"action": "KEEP", "effect": None}

    # Filter 4: ATR% cap (skip high-volatility environments)
    if t["atr_pct"] > HYP_ATR_MAX:
        hyp["ATR_CAP_2.5%"] = {"action": "SKIP", "effect": "SAVE_LOSS" if r == "LOSS" else "MISS_WIN"}
    else:
        hyp["ATR_CAP_2.5%"] = {"action": "KEEP", "effect": None}

    # Filter 5: Strong EMA slope required
    if abs(t["ema_slope_pct"]) < HYP_EMA_SLOPE:
        hyp["EMA_SLOPE_0.2%"] = {"action": "SKIP", "effect": "SAVE_LOSS" if r == "LOSS" else "MISS_WIN"}
    else:
        hyp["EMA_SLOPE_0.2%"] = {"action": "KEEP", "effect": None}

    # For winning trades: would wider TP have earned more?
    if r == "WIN" and t["mfe_atr"] > HYP_TP_WIDE:
        hyp["WIDER_TP_5x"] = {"action": "MORE_PROFIT",
                               "effect": f"MFE {t['mfe_atr']:.1f} ATR > 5 ATR — wider TP would earn more"}
    elif r == "WIN":
        hyp["WIDER_TP_5x"] = {"action": "NO_BENEFIT",
                               "effect": f"MFE {t['mfe_atr']:.1f} ATR didn't reach 5 ATR anyway"}

    # For losing full-SL trades: would wider SL have helped?
    if r == "LOSS" and t["exit_type"] in ("sl", "gap_sl"):
        survived = _wider_sl_would_survive(t["mae_atr"], t["mfe_atr"], t["dir"])
        hyp["WIDER_SL_1.0x"] = {
            "action": "SURVIVE" if survived else "STILL_LOSS",
            "effect": ("MAE {:.2f} ATR < 1.0 SL — wider SL would have kept trade open"
                       .format(t["mae_atr"])) if survived else
                      ("MAE {:.2f} ATR > 1.0 SL — wider SL still gets hit"
                       .format(t["mae_atr"]))
        }

    return hyp


# ─── Main backtest + analysis engine ─────────────────────────────────────────

def run_analyzed_backtest(symbol, df_1m):
    agg = dict(open=("open","first"), high=("high","max"),
               low=("low","min"), close=("close","last"), vol=("vol","sum"))
    df_sig  = df_1m.resample(ANALYZE_TF).agg(**agg).dropna()
    df_zone = df_1m.resample(ZONE_TF).agg(**agg).dropna()

    atr_s = calc_atr(df_sig)
    df_zone["ema"]      = df_zone["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_zone["ema_lag"]  = df_zone["ema"].shift(1)
    df_zone["ema_prev"] = df_zone["ema"].shift(1 + EMA_LOOKBACK)
    ema_now  = df_zone["ema_lag"].reindex(df_sig.index,  method="ffill")
    ema_prev = df_zone["ema_prev"].reindex(df_sig.index, method="ffill")

    n_days    = len(df_sig) // BARS_PER_DAY + 2
    zone_snap = []
    for d in range(n_days):
        end_ts = df_sig.index[min(d * BARS_PER_DAY, len(df_sig) - 1)]
        past_z = df_zone[df_zone.index < end_ts]
        zone_snap.append(find_zones(past_z.iloc[-200:]) if len(past_z) >= SWING_LOOKBACK * 2 + 5 else [])

    FEE_SIDE     = CRYPTO_FEE_RT / 2
    m_timestamps = df_1m.index
    balance      = INITIAL_BALANCE
    min_balance  = INITIAL_BALANCE * MIN_BAL_RATIO
    trades       = []
    zone_cooldown = {}
    zone_touches  = defaultdict(list)
    last_trigger  = -999

    i, n_sig = 0, len(df_sig)
    while i < n_sig:
        if i < WARMUP_BARS:
            i += 1; continue

        ts    = df_sig.index[i]
        row   = df_sig.iloc[i]
        atr   = atr_s.iloc[i]
        price = row["close"]

        if atr <= 0 or np.isnan(atr):
            i += 1; continue

        day_idx = i // BARS_PER_DAY
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            i += 1; continue
        zones = zone_snap[day_idx]

        zb = [z for z in zones if z < price and (price - z) / z <= APPROACH_PCT]
        za = [z for z in zones if z > price and (z - price) / z <= APPROACH_PCT]
        if zb:
            near_zone, direction = max(zb), "LONG"
        elif za:
            near_zone, direction = min(za), "SHORT"
        else:
            i += 1; continue

        if zone_cooldown.get(near_zone, 0) > i or i == last_trigger:
            i += 1; continue
        if balance < min_balance:
            i += 1; continue

        if direction == "LONG":
            triggered = row["low"] <= near_zone * (1 + TOUCH_BUF) and row["close"] > near_zone
        else:
            triggered = row["high"] >= near_zone * (1 - TOUCH_BUF) and row["close"] < near_zone
        if not triggered:
            i += 1; continue

        vol_avg = df_sig["vol"].iloc[max(0, i - VOL_LOOKBACK):i].mean()
        vol_ratio = row["vol"] / vol_avg if vol_avg > 0 else 0
        if vol_avg > 0 and vol_ratio < VOL_MULT:
            i += 1; continue

        en, ep = ema_now.iloc[i], ema_prev.iloc[i]
        ema_slope = 0.0
        if not (pd.isna(en) or pd.isna(ep)):
            ema_slope = (en - ep) / ep if ep != 0 else 0
            if direction == "LONG"  and en <= ep:
                i += 1; continue
            if direction == "SHORT" and en >= ep:
                i += 1; continue

        recent = [b for b in zone_touches[near_zone] if i - b <= ZONE_WINDOW]
        zone_touches[near_zone] = recent
        if len(recent) >= ZONE_MAX_TOUCH:
            i += 1; continue

        if i + 1 >= n_sig:
            i += 1; continue

        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            i += 1; continue

        entry_px  = df_sig["open"].iloc[i + 1]
        trade_atr = atr
        qty       = min((min(balance, MAX_BALANCE) * RISK_PCT) / sl_dist,
                        MAX_POSITION_NOTIONAL / entry_px)

        if direction == "LONG":
            sl_px         = entry_px - SL_MULT         * trade_atr
            partial_tp_px = entry_px + PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px + HARD_TP_MULT    * trade_atr
        else:
            sl_px         = entry_px + SL_MULT         * trade_atr
            partial_tp_px = entry_px - PARTIAL_TP_MULT * trade_atr
            hard_tp_px    = entry_px - HARD_TP_MULT    * trade_atr

        entry_ts = df_sig.index[i + 1]
        m_start  = int(m_timestamps.searchsorted(entry_ts))

        info = _exec_1m_ext(df_1m, m_start, entry_px, direction,
                             sl_px, partial_tp_px, hard_tp_px,
                             trade_atr, qty, FEE_SIDE, CRYPTO_SLIP_PCT)
        if info is None:
            i += 1; continue

        close_ts, gross, exit_px, total_fee, total_slip, final_state, exit_type = info
        j = max(int(df_sig.index.searchsorted(close_ts, side="right")) - 1, i + 1)
        j = min(j, n_sig - 1)
        m_end = int(m_timestamps.searchsorted(close_ts, side="right"))

        mae_atr, mfe_atr = _scan_maefme(df_1m, m_start, m_end, entry_px, direction, trade_atr)

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"
        bal_open = balance
        balance += net

        zone_cooldown[near_zone] = j + (CD_LOSS if result == "LOSS" else CD_WIN)
        zone_touches[near_zone].append(i)
        last_trigger = i

        duration_h = (close_ts - entry_ts).total_seconds() / 3600

        t_record = {
            "num": len(trades) + 1,
            "symbol": symbol,
            "ts": ts, "entry_ts": entry_ts, "close_ts": close_ts,
            "dir": direction,
            "entry": round(entry_px, 6),
            "exit": round(exit_px, 6),
            "qty": round(qty, 8),
            "sl_px": round(sl_px, 6),
            "partial_tp": round(partial_tp_px, 6),
            "hard_tp": round(hard_tp_px, 6),
            "gross": round(gross, 4),
            "fee": round(total_fee, 4),
            "net": round(net, 4),
            "result": result,
            "exit_type": exit_type,
            "balance_open": round(bal_open, 4),
            "balance": round(balance, 4),
            # Signal context
            "atr": round(trade_atr, 6),
            "atr_pct": round(trade_atr / entry_px, 5),
            "vol_ratio": round(vol_ratio, 3),
            "ema_slope_pct": round(ema_slope, 6),
            "zone_level": round(near_zone, 6),
            "zone_touch": len(recent) + 1,
            # Price action
            "mae_atr": mae_atr,
            "mfe_atr": mfe_atr,
            "duration_h": round(duration_h, 1),
        }
        trades.append(t_record)
        i = j + 1

    return trades


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_trade_detail(t, num_trades):
    """Print detailed analysis for a single trade."""
    result_icon = "WIN " if t["result"] == "WIN" else "LOSS"
    print(f"\n  ┌─ #{t['num']:02d}/{num_trades}  {result_icon}  {t['symbol']}  "
          f"{t['dir']}  {t['entry_ts'].strftime('%Y-%m-%d')} → "
          f"{t['close_ts'].strftime('%Y-%m-%d')}")
    print(f"  │  Entry ${t['entry']:.4f}  SL ${t['sl_px']:.4f}  "
          f"Exit ${t['exit']:.4f}  Net ${t['net']:+.2f}  [{t['exit_type']}]")

    print(f"  │  Signal:  ATR {t['atr_pct']*100:.2f}%    "
          f"Vol {t['vol_ratio']:.2f}x    "
          f"EMA slope {t['ema_slope_pct']*100:+.3f}%    "
          f"Zone touch #{t['zone_touch']}")
    print(f"  │  Action:  MAE {t['mae_atr']:.2f} ATR    "
          f"MFE {t['mfe_atr']:.2f} ATR    "
          f"Duration {t['duration_h']:.0f}h")

    if t["result"] == "LOSS":
        reasons = classify_loss(t)
        print(f"  │  Root cause(s):")
        for r in reasons:
            print(f"  │    ✗ {r}")

        hyp = check_hypotheticals(t)
        saves = [(k, v) for k, v in hyp.items()
                 if v["action"] == "SKIP" and v["effect"] == "SAVE_LOSS"]
        wider_sl = hyp.get("WIDER_SL_1.0x")

        if saves:
            print(f"  │  These filters would have SKIPPED this loss:")
            for k, v in saves:
                print(f"  │    → {k}  ({v['effect']})")
        if wider_sl and wider_sl["action"] == "SURVIVE":
            print(f"  │  → WIDER_SL_1.0x: {wider_sl['effect']}")
        if not saves and (not wider_sl or wider_sl["action"] != "SURVIVE"):
            print(f"  │  No filter would have helped — genuine reversal, accept the loss")
    else:
        reasons = classify_win(t)
        print(f"  │  Why it won:")
        for r in reasons:
            print(f"  │    ✓ {r}")

        hyp = check_hypotheticals(t)
        wider_tp = hyp.get("WIDER_TP_5x")
        if wider_tp and wider_tp["action"] == "MORE_PROFIT":
            print(f"  │  💡 {wider_tp['effect']}")

    print(f"  └──────────────────────────────────────────────────────")


def compute_filter_stats(trades):
    """For each hypothetical filter, compute what net PF would be."""
    filters = ["LONG_ONLY", "VOL_2.5", "1ST_TOUCH", "ATR_CAP_2.5%", "EMA_SLOPE_0.2%"]
    results = {}

    for f in filters:
        kept = []
        for t in trades:
            hyp = check_hypotheticals(t)
            if f in hyp and hyp[f]["action"] == "SKIP":
                continue
            kept.append(t)

        if not kept:
            continue

        wins  = [t for t in kept if t["result"] == "WIN"]
        loses = [t for t in kept if t["result"] == "LOSS"]
        gw = sum(t["gross"] for t in wins)
        gl = abs(sum(t["gross"] for t in loses))
        nw = sum(t["net"] for t in wins)
        nl = abs(sum(t["net"] for t in loses))
        n_skipped_losses = sum(1 for t in trades
                               if t["result"] == "LOSS"
                               and f in check_hypotheticals(t)
                               and check_hypotheticals(t)[f]["action"] == "SKIP")
        n_skipped_wins   = sum(1 for t in trades
                               if t["result"] == "WIN"
                               and f in check_hypotheticals(t)
                               and check_hypotheticals(t)[f]["action"] == "SKIP")

        results[f] = {
            "kept": len(kept),
            "wr": round(len(wins) / len(kept) * 100, 1) if kept else 0,
            "gross_pf": round(gw / gl, 3) if gl > 0 else float("inf"),
            "net_pf":   round(nw / nl, 3) if nl > 0 else float("inf"),
            "skipped_losses": n_skipped_losses,
            "skipped_wins":   n_skipped_wins,
        }

    return results


def print_symbol_summary(symbol, trades):
    if not trades:
        print(f"\n  {symbol}: no trades\n")
        return

    wins  = [t for t in trades if t["result"] == "WIN"]
    loses = [t for t in trades if t["result"] == "LOSS"]
    longs  = [t for t in trades if t["dir"] == "LONG"]
    shorts = [t for t in trades if t["dir"] == "SHORT"]

    gw = sum(t["gross"] for t in wins)
    gl = abs(sum(t["gross"] for t in loses))
    nw = sum(t["net"] for t in wins)
    nl = abs(sum(t["net"] for t in loses))

    print(f"\n{'─'*72}")
    print(f"  SYMBOL: {symbol}  |  {ANALYZE_TF} signal / {ZONE_TF} zone  |  {len(trades)} trades")
    print(f"  Overall: WR {len(wins)/len(trades)*100:.1f}%  "
          f"GrossPF {gw/gl:.3f}  NetPF {nw/nl:.3f}  "
          f"Net ${sum(t['net'] for t in trades):+.2f}")
    print(f"  LONG  {len(longs):2d} trades  WR {(sum(1 for t in longs if t['result']=='WIN')/len(longs)*100 if longs else 0):.1f}%")
    print(f"  SHORT {len(shorts):2d} trades  WR {(sum(1 for t in shorts if t['result']=='WIN')/len(shorts)*100 if shorts else 0):.1f}%")

    # Signal quality splits
    print(f"\n  SIGNAL QUALITY SPLITS:")
    for bucket_name, filt in [
        ("Vol < 2.5x (borderline)", lambda t: t["vol_ratio"] < 2.5),
        ("Vol >= 2.5x (strong)",    lambda t: t["vol_ratio"] >= 2.5),
        ("ATR > 2.5% (volatile)",   lambda t: t["atr_pct"] > 0.025),
        ("ATR <= 2.5% (calm)",      lambda t: t["atr_pct"] <= 0.025),
        ("Zone touch #1 (fresh)",   lambda t: t["zone_touch"] == 1),
        ("Zone touch #2 (aged)",    lambda t: t["zone_touch"] >= 2),
        ("EMA slope weak (<0.2%)",  lambda t: abs(t["ema_slope_pct"]) < 0.002),
        ("EMA slope strong(>=0.2%)",lambda t: abs(t["ema_slope_pct"]) >= 0.002),
        ("MAE < 0.3 ATR (instant)", lambda t: t["mae_atr"] < 0.30),
        ("MAE 0.3-0.8 ATR (fair)",  lambda t: 0.30 <= t["mae_atr"] < 0.80),
        ("MAE > 0.8 ATR (deep)",    lambda t: t["mae_atr"] >= 0.80),
    ]:
        sub = [t for t in trades if filt(t)]
        if not sub:
            continue
        w = sum(1 for t in sub if t["result"] == "WIN")
        nw2 = sum(t["net"] for t in sub if t["result"] == "WIN")
        nl2 = abs(sum(t["net"] for t in sub if t["result"] == "LOSS"))
        npf  = round(nw2 / nl2, 3) if nl2 > 0 else float("inf")
        print(f"    {bucket_name:<35} {len(sub):3d} trades  "
              f"WR {w/len(sub)*100:>5.1f}%  NetPF {npf:>6.3f}")

    # Filter improvement table
    print(f"\n  HYPOTHETICAL FILTER RESULTS:")
    baseline_nw = sum(t["net"] for t in wins)
    baseline_nl = abs(sum(t["net"] for t in loses))
    baseline_npf = round(baseline_nw / baseline_nl, 3) if baseline_nl > 0 else float("inf")
    print(f"    {'Baseline (no filter)':<28} {len(trades):3d} trades  "
          f"WR {len(wins)/len(trades)*100:>5.1f}%  NetPF {baseline_npf:>6.3f}")

    fstats = compute_filter_stats(trades)
    for fname, fs in fstats.items():
        improvement = fs["net_pf"] - baseline_npf
        arrow = "↑" if improvement > 0.01 else ("↓" if improvement < -0.01 else "→")
        print(f"    {fname:<28} {fs['kept']:3d} trades  "
              f"WR {fs['wr']:>5.1f}%  NetPF {fs['net_pf']:>6.3f}  "
              f"{arrow}{improvement:+.3f}  "
              f"(skip {fs['skipped_losses']}L / {fs['skipped_wins']}W)")

    # MFE analysis for winners — money left on table
    win_mfe = [t["mfe_atr"] for t in wins]
    if win_mfe:
        print(f"\n  WINNER MFE ANALYSIS (money left on table):")
        print(f"    Avg MFE of winners:    {np.mean(win_mfe):.2f} ATR")
        print(f"    Median MFE of winners: {np.median(win_mfe):.2f} ATR")
        print(f"    Winners where MFE > 3 ATR (exceeded hard TP level): "
              f"{sum(1 for m in win_mfe if m > 3)}/{len(win_mfe)}")
        print(f"    Winners where MFE > 5 ATR (would benefit from wider TP): "
              f"{sum(1 for m in win_mfe if m > 5)}/{len(win_mfe)}")

    loss_mfe = [t["mfe_atr"] for t in loses]
    if loss_mfe:
        print(f"\n  LOSER MFE DISTRIBUTION (how far price moved our way before stopping):")
        print(f"    MFE < 0.15 ATR (instant rejection):  "
              f"{sum(1 for m in loss_mfe if m < 0.15)}/{len(loss_mfe)}")
        print(f"    MFE 0.15–0.50 ATR (weak bounce):     "
              f"{sum(1 for m in loss_mfe if 0.15 <= m < 0.5)}/{len(loss_mfe)}")
        print(f"    MFE 0.50–1.00 ATR (near partial TP): "
              f"{sum(1 for m in loss_mfe if 0.5 <= m < 1.0)}/{len(loss_mfe)}")
        print(f"    MFE > 1.00 ATR (hit partial TP):     "
              f"{sum(1 for m in loss_mfe if m >= 1.0)}/{len(loss_mfe)}")

    print()


def print_aggregate(all_trades):
    print(f"\n{'='*72}")
    print("  AGGREGATE ANALYSIS  (all symbols combined)")
    print(f"{'='*72}")

    wins  = [t for t in all_trades if t["result"] == "WIN"]
    loses = [t for t in all_trades if t["result"] == "LOSS"]
    if not all_trades:
        return

    print(f"  Total trades: {len(all_trades)}  |  Wins: {len(wins)}  |  Losses: {len(loses)}")
    print(f"  Overall WR: {len(wins)/len(all_trades)*100:.1f}%")
    nw = sum(t["net"] for t in wins)
    nl = abs(sum(t["net"] for t in loses))
    print(f"  Overall NetPF: {nw/nl:.3f}" if nl > 0 else "  Overall NetPF: ∞")

    print(f"\n  TOP IMPROVEMENT OPPORTUNITIES:")
    fstats = compute_filter_stats(all_trades)
    baseline_npf = (nw / nl) if nl > 0 else float("inf")
    ranked = sorted(fstats.items(), key=lambda x: x[1]["net_pf"], reverse=True)
    for fname, fs in ranked:
        delta = fs["net_pf"] - baseline_npf
        sign  = "+" if delta >= 0 else ""
        ratio = fs["skipped_losses"] / max(fs["skipped_wins"], 1)
        print(f"    {fname:<25}  NetPF {fs['net_pf']:.3f} ({sign}{delta:.3f})  "
              f"Skip {fs['skipped_losses']} losses / {fs['skipped_wins']} wins  "
              f"(ratio {ratio:.1f}:1)")

    print(f"\n  LONG vs SHORT split:")
    longs  = [t for t in all_trades if t["dir"] == "LONG"]
    shorts = [t for t in all_trades if t["dir"] == "SHORT"]
    lw = sum(t["net"] for t in longs if t["result"] == "WIN")
    ll = abs(sum(t["net"] for t in longs if t["result"] == "LOSS"))
    sw = sum(t["net"] for t in shorts if t["result"] == "WIN")
    sl = abs(sum(t["net"] for t in shorts if t["result"] == "LOSS"))
    print(f"    LONG:  {len(longs):3d} trades  WR {sum(1 for t in longs if t['result']=='WIN')/max(len(longs),1)*100:.1f}%  "
          f"NetPF {lw/ll:.3f}" if ll > 0 else f"    LONG:  {len(longs):3d} trades")
    print(f"    SHORT: {len(shorts):3d} trades  WR {sum(1 for t in shorts if t['result']=='WIN')/max(len(shorts),1)*100:.1f}%  "
          f"NetPF {sw/sl:.3f}" if sl > 0 else f"    SHORT: {len(shorts):3d} trades")

    print(f"\n  FINAL RECOMMENDATIONS:")
    recs = sorted(fstats.items(), key=lambda x: x[1]["net_pf"], reverse=True)
    for i, (fname, fs) in enumerate(recs[:3], 1):
        delta = fs["net_pf"] - baseline_npf
        sign  = "+" if delta >= 0 else ""
        print(f"    {i}. Apply {fname} filter → NetPF {fs['net_pf']:.3f} ({sign}{delta:.3f}), "
              f"removes {fs['skipped_losses']} losing trades at cost of {fs['skipped_wins']} winners")

    # Best combined filter suggestion
    print(f"\n  TRY COMBINING top 2 filters for better precision:")
    f1, f2 = recs[0][0], recs[1][0]
    combined = []
    for t in all_trades:
        h = check_hypotheticals(t)
        if f1 in h and h[f1]["action"] == "SKIP":
            continue
        if f2 in h and h[f2]["action"] == "SKIP":
            continue
        combined.append(t)
    if combined:
        cw = [t for t in combined if t["result"] == "WIN"]
        cl = [t for t in combined if t["result"] == "LOSS"]
        cnw = sum(t["net"] for t in cw)
        cnl = abs(sum(t["net"] for t in cl))
        cnpf = cnw / cnl if cnl > 0 else float("inf")
        print(f"    {f1} + {f2}: {len(combined)} trades  "
              f"WR {len(cw)/len(combined)*100:.1f}%  NetPF {cnpf:.3f}  "
              f"(vs baseline {baseline_npf:.3f})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 72
    print("=" * W)
    print(f"  LZR v8  DEEP TRADE ANALYZER")
    print(f"  Signal TF: {ANALYZE_TF}  |  Zone TF: {ZONE_TF}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print(f"  Hypothetical filters tested:")
    print(f"    LONG_ONLY      — skip all SHORT trades")
    print(f"    VOL_2.5        — skip if volume ratio < 2.5 (vs current 1.8)")
    print(f"    1ST_TOUCH      — skip if zone has been touched before")
    print(f"    ATR_CAP_2.5%   — skip if ATR > 2.5% of price (high volatility)")
    print(f"    EMA_SLOPE_0.2% — skip if EMA slope < 0.2% (weak trend)")
    print()

    all_trades = []
    csv_trades = []

    for sym in CRYPTO_SYMBOLS:
        print(f"  Running {sym}...", flush=True)
        try:
            df_1m = load_crypto(sym)
            trades = run_analyzed_backtest(sym, df_1m)
            print(f"    → {len(trades)} trades completed", flush=True)
            all_trades.extend(trades)
        except Exception as e:
            print(f"    ERROR: {e}"); continue

        print_symbol_summary(sym, trades)

        # Print individual trade details
        print(f"\n  INDIVIDUAL TRADE BREAKDOWN  [{sym}  {ANALYZE_TF}]")
        for t in trades:
            print_trade_detail(t, len(trades))

        csv_trades.extend(trades)

    print_aggregate(all_trades)

    # Save CSV
    if csv_trades:
        csv = OUTPUT_DIR / "trade_analysis_v8_4h.csv"
        rows = []
        for t in csv_trades:
            row = {k: v for k, v in t.items() if k not in ("ts", "entry_ts", "close_ts")}
            row["ts"]        = str(t["ts"])
            row["entry_ts"]  = str(t["entry_ts"])
            row["close_ts"]  = str(t["close_ts"])
            # Add hypothetical outcomes to CSV
            hyp = check_hypotheticals(t)
            for hname, hval in hyp.items():
                row[f"hyp_{hname}"] = hval["action"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\n  Full trade detail saved → {csv}")

    print("=" * W)


if __name__ == "__main__":
    main()
