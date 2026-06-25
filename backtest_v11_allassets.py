"""
backtest_v11_allassets.py  --  LZR v11  Universal Asset Scanner
================================================================
Runs the proven LZR v11 engine on EVERY available asset:
  - 29 crypto perpetuals  (candlestick data/1m/)
  - 7  forex pairs        (forex/)

Adaptations per asset type:
  CRYPTO : regime filter (weekly EMA20 + daily EMA50) + volume spike filter
  FOREX  : no weekly regime (forex has no crypto bull/bear cycles),
           daily EMA50 trend confirm only, volume filter OFF (vol=0 in data),
           lower fee (0.01% per side vs crypto 0.02%)

Output: ranked table by Calmar ratio + portfolio recommendation.
Same LZR v11 core: LONG-only, 4h signal, 16h zone, 1m execution,
partial TP 1.0x ATR, trail 0.8x ATR, hard TP 6.0x ATR.

DOES NOT modify account_data.py or liq_algo.py.
"""

import sys, warnings, heapq
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
CRYPTO_DIR = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
FOREX_DIR  = Path(r"C:\Users\GIGA\Documents\forex")
OUTPUT_DIR = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Timeframes ─────────────────────────────────────────────────────────────────
SIG_TF  = "4h"
ZONE_TF = "16h"

# ── Capital ────────────────────────────────────────────────────────────────────
INITIAL_BALANCE       = 10_000.0   # larger base for realistic pip values in forex
RISK_PCT              = 0.10       # 10% per trade for individual asset scan
MAX_BALANCE           = 500_000.0
MAX_POSITION_NOTIONAL = 2_000_000.0
MIN_BAL_RATIO         = 0.20

# ── LZR v11 core parameters ────────────────────────────────────────────────────
ZONE_WINDOW     = 6
CD_LOSS_BARS    = 42      # 42 x 4h = 7 day cooldown after loss
CD_WIN_BARS     = 3       # 3 x 4h = 12h cooldown after win
EMA_PERIOD      = 50
VOL_MA_PERIOD   = 20
VOL_MULT        = 1.8
SL_MULT         = 0.75
ZONE_TOUCH_MULT = 0.5
ATR_PERIOD      = 14

# ── v11 TP (wider hard TP, wider trail) ───────────────────────────────────────
PARTIAL_TP_MULT = 1.0
HARD_TP_MULT    = 6.0
TRAIL_DIST_MULT = 0.8

# ── Regime filter (crypto only) ────────────────────────────────────────────────
WEEKLY_EMA_PERIOD = 20
DAILY_EMA_PERIOD  = 50

# ── Fees (per side) ───────────────────────────────────────────────────────────
FEE_CRYPTO = 0.0002   # 0.02% per side (taker, 0.04% round trip)
FEE_FOREX  = 0.00005  # 0.005% per side (spread ~1 pip on majors)
SLIP_PCT   = 0.0002

# ── WARMUP bars ────────────────────────────────────────────────────────────────
WARMUP = max(EMA_PERIOD, VOL_MA_PERIOD, ZONE_WINDOW * 2, WEEKLY_EMA_PERIOD * 7) + 10

# ── Asset catalogue ───────────────────────────────────────────────────────────
CRYPTO_SYMBOLS = [
    "AAVEUSDT", "ADAUSDT",  "APTUSDT",  "ARBUSDT",  "ATOMUSDT",
    "AVAXUSDT", "BCHUSDT",  "BNBUSDT",  "BTCUSDT",  "CFXUSDT",
    "DOGEUSDT", "DOTUSDT",  "ETHUSDT",  "FETUSDT",  "FILUSDT",
    "INJUSDT",  "LDOUSDT",  "LINKUSDT", "LTCUSDT",  "MATICUSDT",
    "NEARUSDT", "OPUSDT",   "RUNEUSDT", "SEIUSDT",  "SOLUSDT",
    "SUIUSDT",  "TRXUSDT",  "UNIUSDT",  "XRPUSDT",
]
FOREX_SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "USDCAD", "USDJPY"]


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_crypto_1m(symbol):
    path = CRYPTO_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "vol"})
    return df.set_index("ts").sort_index()


def load_forex_1m(symbol):
    path = FOREX_DIR / f"{symbol}_1m.csv"
    df = pd.read_csv(path, parse_dates=["dt"])
    df = df.rename(columns={"dt": "ts"})
    return df.set_index("ts").sort_index()


def resample(df_1m, freq):
    return df_1m.resample(freq).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),  close=("close", "last"), vol=("vol", "sum")
    ).dropna()


# ── Indicators ─────────────────────────────────────────────────────────────────

def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()


# ── Zone detection ─────────────────────────────────────────────────────────────

def find_zone_lows(df_zone):
    lo_v = df_zone["low"].values
    n    = len(df_zone)
    zl   = {}
    for i in range(ZONE_WINDOW, n - ZONE_WINDOW):
        if lo_v[i] == np.min(lo_v[i - ZONE_WINDOW: i + ZONE_WINDOW + 1]):
            zl[i] = lo_v[i]
    return zl


# ── 1m execution (LONG-only, gap-aware) ───────────────────────────────────────

def exec_1m(df_1m, m_start, entry_px, sl_px, partial_tp_px, hard_tp_px,
            atr, qty, fee_side):
    half           = qty / 2.0
    state          = "full"
    partial_locked = 0.0
    running_ext    = entry_px
    trail_sl       = sl_px

    fee_acc  = entry_px * qty * fee_side
    slip_acc = entry_px * qty * SLIP_PCT

    for m_idx in range(m_start, len(df_1m)):
        mr     = df_1m.iloc[m_idx]
        closed = False
        gross  = 0.0
        exit_px = 0.0

        if state == "full":
            if mr["open"] <= sl_px:
                exit_px = mr["open"]
                gross   = (exit_px - entry_px) * qty
                closed  = True
            elif mr["open"] >= partial_tp_px:
                partial_locked = (partial_tp_px - entry_px) * half
                fee_acc  += partial_tp_px * half * fee_side
                slip_acc += partial_tp_px * half * SLIP_PCT
                running_ext = mr["open"]
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                state = "partial"
            elif mr["high"] >= partial_tp_px:
                partial_locked = (partial_tp_px - entry_px) * half
                fee_acc  += partial_tp_px * half * fee_side
                slip_acc += partial_tp_px * half * SLIP_PCT
                running_ext = mr["high"]
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                state = "partial"
            elif mr["low"] <= sl_px:
                exit_px = sl_px
                gross   = (exit_px - entry_px) * qty
                closed  = True

        elif state == "partial":
            old_trail = trail_sl
            if mr["open"] <= old_trail:
                exit_px = mr["open"]
                gross   = partial_locked + (exit_px - entry_px) * half
                closed  = True
            elif mr["open"] >= hard_tp_px:
                exit_px = hard_tp_px
                gross   = partial_locked + (exit_px - entry_px) * half
                closed  = True
            else:
                running_ext = max(running_ext, mr["high"])
                trail_sl    = max(entry_px, running_ext - TRAIL_DIST_MULT * atr)
                if mr["low"] <= old_trail:
                    exit_px = old_trail
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True
                elif mr["high"] >= hard_tp_px:
                    exit_px = hard_tp_px
                    gross   = partial_locked + (exit_px - entry_px) * half
                    closed  = True

        if closed:
            exit_qty  = qty if state == "full" else half
            fee_acc  += exit_px * exit_qty * fee_side
            slip_acc += exit_px * exit_qty * SLIP_PCT
            return df_1m.index[m_idx], gross, exit_px, fee_acc, slip_acc

    return None


# ── Single-asset backtest engine ───────────────────────────────────────────────

def run_single(symbol, asset_type, df_1m, use_regime=True, use_vol_filter=True):
    """
    Run LZR v11 LONG-only on one asset.
    asset_type: 'crypto' or 'forex'
    use_regime: apply weekly EMA20 + daily EMA50 regime filter
    use_vol_filter: apply 1.8x volume spike filter
    Returns dict of stats or None if no trades.
    """
    fee_side = FEE_CRYPTO if asset_type == "crypto" else FEE_FOREX

    df_sig   = resample(df_1m, SIG_TF)
    df_zone  = resample(df_1m, ZONE_TF)
    df_daily = resample(df_1m, "1D")
    df_weekly = resample(df_1m, "1W") if use_regime else None

    n = len(df_sig)
    if n < WARMUP + 50:
        return None

    # Indicators
    atr_sig   = calc_atr(df_sig)
    ema_sig   = df_sig["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    vol_ma    = df_sig["vol"].rolling(VOL_MA_PERIOD).mean()
    vol_ratio = df_sig["vol"] / vol_ma  # will be NaN or inf for forex (vol=0)

    atr_zone_s = calc_atr(df_zone)
    zl         = find_zone_lows(df_zone)
    zone_ts    = df_zone.index
    m_ts       = df_1m.index

    daily_ema  = df_daily["close"].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean()
    daily_ts   = df_daily.index

    weekly_ema = None
    weekly_ts  = None
    if use_regime and df_weekly is not None:
        weekly_ema = df_weekly["close"].ewm(span=WEEKLY_EMA_PERIOD, adjust=False).mean()
        weekly_ts  = df_weekly.index

    balance      = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    fired_zones  = set()
    cooldown_until = 0
    last_signal_i  = -999
    equity         = []
    all_trades     = []

    i = 0
    while i < n:
        sig_ts = df_sig.index[i]

        if i < WARMUP or balance < INITIAL_BALANCE * MIN_BAL_RATIO:
            equity.append(balance)
            i += 1
            continue

        # ── Weekly regime filter (crypto only) ───────────────────────────────
        if use_regime and weekly_ema is not None:
            w_idx = int(weekly_ts.searchsorted(sig_ts, side="right")) - 1
            if w_idx >= 1:
                if df_weekly["close"].iloc[w_idx] < weekly_ema.iloc[w_idx]:
                    equity.append(balance)
                    i += 1
                    continue

        # ── Daily EMA regime (both crypto + forex) ────────────────────────────
        d_idx = int(daily_ts.searchsorted(sig_ts, side="right")) - 1
        if d_idx >= 1:
            if df_daily["close"].iloc[d_idx] < daily_ema.iloc[d_idx]:
                equity.append(balance)
                i += 1
                continue

        # ── Cooldown ─────────────────────────────────────────────────────────
        if i <= cooldown_until:
            equity.append(balance)
            i += 1
            continue

        row   = df_sig.iloc[i]
        atr   = atr_sig.iloc[i]
        ema   = ema_sig.iloc[i]
        vol_r = vol_ratio.iloc[i]

        if atr <= 0 or np.isnan(atr) or np.isnan(ema):
            equity.append(balance)
            i += 1
            continue

        # ── Volume filter (crypto only, skip if vol is zero/NaN) ─────────────
        if use_vol_filter:
            if np.isnan(vol_r) or vol_r < VOL_MULT:
                equity.append(balance)
                i += 1
                continue

        # ── Zone lookup ───────────────────────────────────────────────────────
        z_idx = int(zone_ts.searchsorted(sig_ts, side="right")) - 1
        if z_idx < ZONE_WINDOW:
            equity.append(balance)
            i += 1
            continue

        close          = row["close"]
        fired_zone_key = None
        zone_lo        = None

        past_lows = sorted(
            [k for k in zl if k < z_idx and k not in fired_zones],
            reverse=True
        )[:10]

        for zb in past_lows:
            zp    = zl[zb]
            atr_z = atr_zone_s.iloc[zb]
            z_lo  = zp - ZONE_TOUCH_MULT * atr_z
            z_hi  = zp + ZONE_TOUCH_MULT * atr_z
            if z_lo <= close <= z_hi:
                fired_zone_key = zb
                zone_lo        = z_lo
                break

        if fired_zone_key is None or i == last_signal_i:
            equity.append(balance)
            i += 1
            continue

        # ── EMA trend confirm ─────────────────────────────────────────────────
        if close < ema:
            equity.append(balance)
            i += 1
            continue

        if i + 1 >= n:
            equity.append(balance)
            i += 1
            continue

        # ── Entry ─────────────────────────────────────────────────────────────
        entry_px   = df_sig["open"].iloc[i + 1]
        sl_px      = zone_lo - SL_MULT * atr
        sl_dist    = entry_px - sl_px
        if sl_dist <= 0:
            equity.append(balance)
            i += 1
            continue

        partial_tp = entry_px + PARTIAL_TP_MULT * atr
        hard_tp    = entry_px + HARD_TP_MULT    * atr

        eff_bal = min(balance, MAX_BALANCE)
        qty     = min(
            (eff_bal * RISK_PCT) / sl_dist,
            MAX_POSITION_NOTIONAL / entry_px
        )
        if qty <= 0:
            equity.append(balance)
            i += 1
            continue

        # ── 1m execution ──────────────────────────────────────────────────────
        entry_ts = df_sig.index[i + 1]
        m_start  = int(m_ts.searchsorted(entry_ts))

        info = exec_1m(df_1m, m_start, entry_px, sl_px, partial_tp, hard_tp,
                       atr, qty, fee_side)
        if info is None:
            equity.append(balance)
            i += 1
            continue

        close_ts, gross, exit_px, total_fee, total_slip = info

        net    = gross - total_fee - total_slip
        result = "WIN" if net > 0 else "LOSS"

        j = int(df_sig.index.searchsorted(close_ts, side="right")) - 1
        j = max(j, i + 1)
        j = min(j, n - 1)

        bal_open = balance
        balance += net
        peak_balance = max(peak_balance, balance)

        fired_zones.add(fired_zone_key)

        if result == "LOSS":
            cooldown_until = j + CD_LOSS_BARS
        else:
            cooldown_until = j + CD_WIN_BARS

        last_signal_i = i
        dur_h = (close_ts - entry_ts).total_seconds() / 3600.0

        all_trades.append({
            "ts": sig_ts, "close_ts": close_ts,
            "entry": round(entry_px, 8), "exit": round(exit_px, 8),
            "net": round(net, 6), "result": result,
            "duration_h": round(dur_h, 1),
        })

        equity.append(bal_open)
        for _ in range(j - i - 1):
            equity.append(bal_open)
        equity.append(balance)
        i = j + 1

    if not all_trades:
        return None

    df_t  = pd.DataFrame(all_trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]
    nw    = wins["net"].sum()
    nl    = abs(loses["net"].sum()) if len(loses) else 0.0
    npf   = round(nw / nl, 3) if nl > 0 else float("inf")
    wr    = round(len(wins) / len(df_t) * 100, 1)

    years  = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    cagr   = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    eq     = pd.Series(equity)
    mdd    = round(float(((eq - eq.cummax()) / eq.cummax() * 100).min()), 2)
    calmar = round(abs(cagr / mdd), 3) if mdd < 0 else float("inf")

    # Year-by-year
    df_t["year"] = pd.to_datetime(df_t["ts"]).dt.year
    yearly = {}
    for yr, grp in df_t.groupby("year"):
        w = grp[grp["result"] == "WIN"]
        yearly[yr] = {
            "trades": len(grp),
            "wr":     round(len(w) / len(grp) * 100, 0),
            "net":    round(grp["net"].sum(), 4),
        }

    return {
        "symbol":    symbol,
        "type":      asset_type,
        "trades":    len(df_t),
        "wins":      len(wins),
        "win_rate":  wr,
        "net_pf":    npf,
        "cagr":      round(cagr, 2),
        "max_dd":    mdd,
        "calmar":    calmar,
        "final_bal": round(balance, 4),
        "years":     round(years, 1),
        "yearly":    yearly,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    W = 96
    print("=" * W)
    print("  LZR v11  UNIVERSAL ASSET SCANNER  —  Crypto + Forex")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * W)
    print(f"  LONG-only | 4h signal | 16h zone | 1m execution")
    print(f"  TP: partial {PARTIAL_TP_MULT}x ATR -> trail {TRAIL_DIST_MULT}x -> hard {HARD_TP_MULT}x ATR")
    print(f"  Crypto: regime (weekly EMA{WEEKLY_EMA_PERIOD} + daily EMA{DAILY_EMA_PERIOD}) + vol filter")
    print(f"  Forex:  daily EMA{DAILY_EMA_PERIOD} trend confirm only (no weekly regime, no vol filter)")
    print(f"  Risk: {RISK_PCT*100:.0f}% per trade | Balance: ${INITIAL_BALANCE:,.0f}")
    print()

    results = []
    errors  = []

    # ── Crypto scan ───────────────────────────────────────────────────────────
    print(f"  CRYPTO ({len(CRYPTO_SYMBOLS)} symbols)  —  regime filter ON")
    print("  " + "-" * (W - 2))

    for sym in CRYPTO_SYMBOLS:
        print(f"  {sym:<14} loading...", end="", flush=True)
        try:
            df_1m = load_crypto_1m(sym)
            stats = run_single(sym, "crypto", df_1m,
                               use_regime=True, use_vol_filter=True)
            if stats is None:
                print(f"\r  {sym:<14} no trades")
                errors.append(sym)
            else:
                s    = stats
                sign = "+" if s["cagr"] >= 0 else ""
                star = ("★★" if s["calmar"] >= 2.0 else
                        ("★ " if s["calmar"] >= 1.5 else
                         ("● " if s["calmar"] >= 1.0 else "  ")))
                print(f"\r  {sym:<14} {s['trades']:>3}tr  "
                      f"WR {s['win_rate']:>5.1f}%  "
                      f"CAGR {sign}{s['cagr']:>6.1f}%  "
                      f"DD {s['max_dd']:>7.2f}%  "
                      f"Calmar {s['calmar']:>5.2f} {star}")
                results.append(stats)
        except FileNotFoundError:
            print(f"\r  {sym:<14} FILE NOT FOUND — skipped")
            errors.append(sym)
        except Exception as e:
            print(f"\r  {sym:<14} ERROR: {e}")
            errors.append(sym)
        finally:
            # Free memory
            try:
                del df_1m
            except Exception:
                pass

    print()

    # ── Forex scan ────────────────────────────────────────────────────────────
    print(f"  FOREX ({len(FOREX_SYMBOLS)} pairs)  —  daily EMA only, no vol filter")
    print("  " + "-" * (W - 2))

    for sym in FOREX_SYMBOLS:
        print(f"  {sym:<14} loading...", end="", flush=True)
        try:
            df_1m = load_forex_1m(sym)
            stats = run_single(sym, "forex", df_1m,
                               use_regime=False, use_vol_filter=False)
            if stats is None:
                print(f"\r  {sym:<14} no trades")
                errors.append(sym)
            else:
                s    = stats
                sign = "+" if s["cagr"] >= 0 else ""
                star = ("★★" if s["calmar"] >= 2.0 else
                        ("★ " if s["calmar"] >= 1.5 else
                         ("● " if s["calmar"] >= 1.0 else "  ")))
                print(f"\r  {sym:<14} {s['trades']:>3}tr  "
                      f"WR {s['win_rate']:>5.1f}%  "
                      f"CAGR {sign}{s['cagr']:>6.1f}%  "
                      f"DD {s['max_dd']:>7.2f}%  "
                      f"Calmar {s['calmar']:>5.2f} {star}")
                results.append(stats)
        except FileNotFoundError:
            print(f"\r  {sym:<14} FILE NOT FOUND — skipped")
            errors.append(sym)
        except Exception as e:
            print(f"\r  {sym:<14} ERROR: {e}")
            errors.append(sym)
        finally:
            try:
                del df_1m
            except Exception:
                pass

    # ── Ranked results table ──────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  RANKED BY CALMAR  (all assets)")
    print("=" * W)
    print(f"  {'#':>3}  {'Symbol':<12} {'Type':>6}  {'Tr':>4}  "
          f"{'WR':>6}  {'CAGR':>7}  {'MaxDD':>7}  {'Calmar':>7}  {'Yrs':>4}")
    print("  " + "-" * (W - 2))

    profitable = [r for r in results if r["cagr"] > 0]
    unprofitable = [r for r in results if r["cagr"] <= 0]
    profitable.sort(key=lambda x: x["calmar"] if x["calmar"] != float("inf") else 99, reverse=True)
    unprofitable.sort(key=lambda x: x["cagr"], reverse=True)
    ranked = profitable + unprofitable

    for rank, r in enumerate(ranked, 1):
        sign = "+" if r["cagr"] >= 0 else ""
        star = ("★★" if r["calmar"] >= 2.0 else
                ("★ " if r["calmar"] >= 1.5 else
                 ("● " if r["calmar"] >= 1.0 else "  ")))
        sep  = " |" if rank == len(profitable) and rank < len(ranked) else "  "
        print(f"{sep} {rank:>2}  {r['symbol']:<12} {r['type']:>6}  "
              f"{r['trades']:>4}  {r['win_rate']:>5.1f}%  "
              f"{sign}{r['cagr']:>6.1f}%  {r['max_dd']:>6.1f}%  "
              f"{r['calmar']:>6.2f}{star}  {r['years']:>3.1f}y")

    # ── Year-by-year for top 5 ────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  YEAR-BY-YEAR  (top 5 by Calmar)")
    print("=" * W)

    for r in profitable[:5]:
        print(f"\n  [{r['symbol']}]  CAGR {'+' if r['cagr']>=0 else ''}{r['cagr']:.1f}%  "
              f"Calmar {r['calmar']:.2f}  {r['trades']} trades")
        for yr, yd in sorted(r["yearly"].items()):
            sign = "+" if yd["net"] >= 0 else ""
            flag = ""  if yd["net"] >= 0 else " <--"
            print(f"    {yr}: {yd['trades']:>2} trades  WR {yd['wr']:>3.0f}%  "
                  f"net {sign}${yd['net']:>8.2f}{flag}")

    # ── Portfolio recommendation ───────────────────────────────────────────────
    print()
    print("=" * W)
    print("  PORTFOLIO RECOMMENDATION")
    print("=" * W)

    top_crypto = [r for r in profitable if r["type"] == "crypto"][:5]
    top_forex  = [r for r in profitable if r["type"] == "forex"][:3]

    print()
    print("  Best crypto assets (by Calmar, CAGR > 0):")
    for r in top_crypto:
        star = "★★" if r["calmar"] >= 2.0 else ("★" if r["calmar"] >= 1.5 else "●" if r["calmar"] >= 1.0 else " ")
        print(f"    {r['symbol']:<14}  CAGR +{r['cagr']:.1f}%  DD {r['max_dd']:.1f}%  "
              f"Calmar {r['calmar']:.2f} {star}  WR {r['win_rate']:.0f}%")

    if top_forex:
        print()
        print("  Best forex pairs (by Calmar, CAGR > 0):")
        for r in top_forex:
            star = "★★" if r["calmar"] >= 2.0 else ("★" if r["calmar"] >= 1.5 else "●" if r["calmar"] >= 1.0 else " ")
            print(f"    {r['symbol']:<14}  CAGR +{r['cagr']:.1f}%  DD {r['max_dd']:.1f}%  "
                  f"Calmar {r['calmar']:.2f} {star}  WR {r['win_rate']:.0f}%")

    print()
    print("  Suggested portfolio (top performers with CAGR > 0, Calmar > 0.5):")
    portfolio_picks = [r for r in ranked if r["cagr"] > 0 and r["calmar"] > 0.5][:6]
    for r in portfolio_picks:
        print(f"    {r['symbol']:<14}  [{r['type']:>6}]  CAGR +{r['cagr']:.1f}%  Calmar {r['calmar']:.2f}")

    # ── Save results ──────────────────────────────────────────────────────────
    save_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "yearly"}
        save_rows.append(row)

    if save_rows:
        out = OUTPUT_DIR / "backtest_v11_allassets.csv"
        pd.DataFrame(save_rows).sort_values("calmar", ascending=False).to_csv(out, index=False)
        print(f"\n  Saved -> {out}")

    if errors:
        print(f"\n  Skipped ({len(errors)}): {', '.join(errors)}")

    # Final count
    n_prof = len(profitable)
    n_total = len(results)
    print(f"\n  {n_prof}/{n_total} assets profitable with CAGR > 0")
    print(f"  {len([r for r in profitable if r['calmar'] >= 1.0])}/{n_total} assets with Calmar >= 1.0")
    print(f"  {len([r for r in profitable if r['calmar'] >= 1.5])}/{n_total} assets with Calmar >= 1.5")
    print("=" * W)


if __name__ == "__main__":
    main()
