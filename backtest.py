"""
backtest.py  --  Liquidation Zone Reversal Strategy Backtest  v2
================================================================
BUGS FIXED vs v1:
  1. Fixed risk per trade ($10 constant, not % of growing balance)
  2. T2 removed (was a parallel duplicate position = 2x PnL bug)
  3. Trading fees added (crypto 0.04% RT, forex pip spread)
  4. Approach filter fixed (LONG needs zone below, SHORT above)
  5. net_pnl now matches equity (both use fixed risk)

Outputs per symbol:
  trades, win_rate, profit_factor, gross_pnl, total_fees,
  net_pnl, max_drawdown, sharpe, avg_win, avg_loss

Usage:
    python backtest.py
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR         = Path(r"C:\Users\GIGA\Documents\candlestick data\1m")
OUTPUT_DIR       = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_BALANCE  = 1000.0    # normalised starting capital (USDT)
FIXED_RISK_USDT  = 10.0      # fixed $10 risk per trade (1% of $1000, never changes)

APPROACH_PCT     = 0.008     # 0.8%  -- price within this of zone -> approaching
TOUCH_BUF        = 0.003     # 0.3%  -- buffer: cascade starts before exact zone
TP_MULT          = 1.5       # TP = entry +/- 1.5 x ATR
SL_MULT          = 0.75      # SL = entry +/- 0.75 x ATR  (max loss = FIXED_RISK_USDT)
ATR_PERIOD       = 14
SWING_LOOKBACK   = 20        # 4h bars each side to confirm swing
MIN_ZONE_GAP     = 0.005     # 0.5%  -- merge nearby zones
COOLDOWN_LOSS    = 10        # bars to skip same zone after LOSS  (10h)
COOLDOWN_WIN     = 3         # bars to skip same zone after WIN   (3h)

# Fees (round-trip)
CRYPTO_FEE_PCT   = 0.0004    # 0.04% RT: 0.02% entry + 0.02% exit (taker)
FOREX_SPREAD_PCT = {         # one-way spread as fraction of price
    "EURUSD": 0.00010,       # ~1.0 pip
    "GBPUSD": 0.00010,       # ~1.0 pip
    "USDJPY": 0.00010,       # ~0.01 JPY/100
    "AUDUSD": 0.00015,       # ~1.5 pip
    "USDCAD": 0.00020,       # ~2.0 pip
    "EURJPY": 0.00015,       # ~1.5 pip
    "GBPJPY": 0.00025,       # ~2.5 pip
}

CRYPTO_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FOREX_PAIRS      = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_crypto_1h(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}1m.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date":"ts","Open":"open","High":"high",
                             "Low":"low","Close":"close","Volume":"vol"})
    df = df.set_index("ts").sort_index()
    df_1h = df.resample("1h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"), vol=("vol","sum")
    ).dropna()
    return df_1h


def download_forex_1h(ticker: str, name: str) -> pd.DataFrame:
    import yfinance as yf
    print(f"  Downloading {name} ({ticker}) ...")
    # yfinance 1.4+ changed API; try download() first, fallback to Ticker.history
    raw = yf.download(ticker, period="2y", interval="1h", auto_adjust=True,
                      progress=False, actions=False)
    if raw.empty:
        raw = yf.Ticker(ticker).history(period="2y", interval="1h", auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns={"Open":"open","High":"high",
                              "Low":"low","Close":"close","Volume":"vol"})
    df.index.name = "ts"
    df = df[["open","high","low","close"]].copy()
    df["vol"] = 0.0
    return df.sort_index().dropna()


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last")
    ).dropna()


# ─── Indicators ───────────────────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period=ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_zones(df_4h: pd.DataFrame) -> list:
    """Swing highs/lows from 4h chart -- proxy for liquidation zone clusters."""
    n = len(df_4h)
    lb = SWING_LOOKBACK
    highs, lows = [], []
    for i in range(lb, n - lb):
        if df_4h["high"].iloc[i] == df_4h["high"].iloc[i-lb:i+lb+1].max():
            highs.append(df_4h["high"].iloc[i])
        if df_4h["low"].iloc[i] == df_4h["low"].iloc[i-lb:i+lb+1].min():
            lows.append(df_4h["low"].iloc[i])
    levels = sorted(set(highs + lows))
    merged = []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > MIN_ZONE_GAP:
            merged.append(lvl)
    return merged


def regime_ok(df_1h: pd.DataFrame, idx: int, direction: str) -> bool:
    if idx < 3:
        return True
    c = df_1h["close"].iloc
    if direction == "LONG":
        return not (c[idx-1] < c[idx-2] < c[idx-3])   # skip if 3 descending closes
    else:
        return not (c[idx-1] > c[idx-2] > c[idx-3])   # skip if 3 ascending closes


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(symbol: str, df_1h: pd.DataFrame,
                 fee_rt: float, category: str) -> dict:
    """
    fee_rt: round-trip fee as fraction of trade value
            crypto  = 0.0004
            forex   = 2 x FOREX_SPREAD_PCT[symbol]
    """
    df_4h  = resample_4h(df_1h)
    atr_s  = calc_atr(df_1h)

    # Pre-compute zone snapshots once per day (no look-ahead: use only past 4h bars)
    n_days = len(df_1h) // 24 + 2
    zone_snap = []
    for d in range(n_days):
        end_1h  = min(d * 24, len(df_1h) - 1)
        end_ts  = df_1h.index[end_1h]
        slice4h = df_4h[df_4h.index <= end_ts].iloc[-200:]
        if len(slice4h) < SWING_LOOKBACK * 2 + 5:
            zone_snap.append([])
        else:
            zone_snap.append(find_zones(slice4h))

    balance          = INITIAL_BALANCE
    equity           = [balance]
    trades           = []
    zone_cooldown    = {}       # zone_price -> bar_idx when cooldown expires
    last_trigger_bar = -999

    in_trade  = False
    direction = entry_px = tp_px = sl_px = qty = active_zone = 0.0

    for i, (ts, row) in enumerate(df_1h.iterrows()):
        if i < 50:
            equity.append(balance); continue

        atr = atr_s.iloc[i]
        if atr <= 0 or np.isnan(atr):
            equity.append(balance); continue

        price = row["close"]

        # ── Manage open trade ────────────────────────────────────────────────
        if in_trade:
            closed = False
            hit_tp = False

            if direction == "LONG":
                if row["high"] >= tp_px:
                    gross = (tp_px - entry_px) * qty
                    hit_tp = True; closed = True
                elif row["low"] <= sl_px:
                    gross = (sl_px - entry_px) * qty
                    closed = True
            else:
                if row["low"] <= tp_px:
                    gross = (entry_px - tp_px) * qty
                    hit_tp = True; closed = True
                elif row["high"] >= sl_px:
                    gross = (entry_px - sl_px) * qty
                    closed = True

            if closed:
                exit_px  = tp_px if hit_tp else sl_px
                fee      = (entry_px + exit_px) * qty * (fee_rt / 2)   # half each side
                net_pnl  = gross - fee
                result   = "WIN" if gross > 0 else "LOSS"
                balance += net_pnl
                in_trade = False
                zone_cooldown[active_zone] = i + (COOLDOWN_LOSS if result == "LOSS"
                                                   else COOLDOWN_WIN)
                trades.append({
                    "ts":       ts,
                    "dir":      direction,
                    "entry":    round(entry_px, 6),
                    "exit":     round(exit_px,  6),
                    "qty":      round(qty,       6),
                    "gross":    round(gross,     4),
                    "fee":      round(fee,       4),
                    "net":      round(net_pnl,   4),
                    "result":   result,
                    "balance":  round(balance,   4),
                })
            equity.append(balance)
            continue

        # ── Get zone snapshot for today ──────────────────────────────────────
        day_idx = i // 24
        if day_idx >= len(zone_snap) or not zone_snap[day_idx]:
            equity.append(balance); continue
        zones = zone_snap[day_idx]

        # ── BUG-FIX: separate above/below zone search ────────────────────────
        # LONG: nearest zone BELOW price (within 0.8%)
        # SHORT: nearest zone ABOVE price (within 0.8%)
        near_zone = None
        near_dir  = None

        zones_below = [z for z in zones if z < price and (price-z)/z <= APPROACH_PCT]
        zones_above = [z for z in zones if z > price and (z-price)/z <= APPROACH_PCT]

        if zones_below:
            near_zone = max(zones_below)   # closest below
            near_dir  = "LONG"
        elif zones_above:
            near_zone = min(zones_above)   # closest above
            near_dir  = "SHORT"

        if near_zone is None:
            equity.append(balance); continue

        # ── Cooldown check ───────────────────────────────────────────────────
        if zone_cooldown.get(near_zone, 0) > i:
            equity.append(balance); continue

        # ── Same-candle guard ────────────────────────────────────────────────
        if i == last_trigger_bar:
            equity.append(balance); continue

        # ── Trigger: touch zone (+/- buffer) AND close back inside ───────────
        if near_dir == "LONG":
            touched   = row["low"]  <= near_zone * (1 + TOUCH_BUF)
            triggered = touched and row["close"] > near_zone
        else:
            touched   = row["high"] >= near_zone * (1 - TOUCH_BUF)
            triggered = touched and row["close"] < near_zone

        if not triggered:
            equity.append(balance); continue

        # ── Regime filter ────────────────────────────────────────────────────
        if not regime_ok(df_1h, i, near_dir):
            equity.append(balance); continue

        # ── Size and fire (FIXED $10 risk -- never changes) ──────────────────
        last_trigger_bar = i
        sl_dist = SL_MULT * atr
        if sl_dist <= 0:
            equity.append(balance); continue

        qty      = FIXED_RISK_USDT / sl_dist          # units (BTC, ETH, EUR, etc.)
        entry_px = df_1h["open"].iloc[i+1] if i+1 < len(df_1h) else price

        if near_dir == "LONG":
            tp_px = entry_px + TP_MULT * atr
            sl_px = entry_px - SL_MULT * atr
        else:
            tp_px = entry_px - TP_MULT * atr
            sl_px = entry_px + SL_MULT * atr

        in_trade     = True
        direction    = near_dir
        active_zone  = near_zone
        equity.append(balance)

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {"symbol": symbol, "category": category, "trades": 0,
                "win_rate": 0, "pf": 0, "gross_pnl": 0, "total_fees": 0,
                "net_pnl": 0, "net_pct": 0, "max_dd": 0,
                "sharpe": 0, "avg_win": 0, "avg_loss": 0}

    df_t       = pd.DataFrame(trades)
    wins       = df_t[df_t["result"] == "WIN"]
    losses     = df_t[df_t["result"] == "LOSS"]

    gross_pnl  = df_t["gross"].sum()
    total_fees = df_t["fee"].sum()
    net_pnl    = df_t["net"].sum()
    gross_wins = wins["gross"].sum()
    gross_loss = abs(losses["gross"].sum())
    pf         = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    win_rate   = round(len(wins) / len(df_t) * 100, 1)

    eq         = pd.Series(equity)
    roll_max   = eq.cummax()
    max_dd     = round(((eq - roll_max) / roll_max * 100).min(), 2)

    trade_rets = df_t["net"] / FIXED_RISK_USDT        # normalised per-trade returns
    sharpe     = round((trade_rets.mean() / trade_rets.std() * np.sqrt(252))
                       if trade_rets.std() > 0 else 0, 2)

    return {
        "symbol":     symbol,
        "category":   category,
        "trades":     len(df_t),
        "win_rate":   win_rate,
        "pf":         pf,
        "gross_pnl":  round(gross_pnl,  2),
        "total_fees": round(total_fees, 2),
        "net_pnl":    round(net_pnl,    2),
        "net_pct":    round(net_pnl / INITIAL_BALANCE * 100, 2),
        "max_dd":     max_dd,
        "sharpe":     sharpe,
        "avg_win":    round(wins["net"].mean(),   4) if len(wins)   else 0,
        "avg_loss":   round(losses["net"].mean(), 4) if len(losses) else 0,
        "equity":     equity,
        "trades_df":  df_t,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = []

    print("\n" + "="*65)
    print("  CRYPTO BACKTEST  (fixed $10 risk/trade, 0.04% RT fee)")
    print("="*65)
    for sym in CRYPTO_SYMBOLS:
        try:
            print(f"\n[{sym}] loading ...")
            df = load_crypto_1h(sym)
            print(f"  {len(df):,} bars  ({df.index[0].date()} -> {df.index[-1].date()})")
            r = run_backtest(sym, df, fee_rt=CRYPTO_FEE_PCT, category="Crypto")
            results.append(r)
            print(f"  trades={r['trades']}  WR={r['win_rate']}%  PF={r['pf']}"
                  f"  gross={r['gross_pnl']:+.2f}  fees=-{r['total_fees']:.2f}"
                  f"  NET={r['net_pnl']:+.2f}  DD={r['max_dd']:.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "="*65)
    print("  FOREX BACKTEST  (fixed $10 risk/trade, pip spread)")
    print("="*65)
    for name, ticker in FOREX_PAIRS.items():
        spread_pct = FOREX_SPREAD_PCT.get(name, 0.0002)
        fee_rt     = spread_pct * 2          # entry + exit spread
        try:
            df = download_forex_1h(ticker, name)
            print(f"  {len(df):,} bars  ({df.index[0].date()} -> {df.index[-1].date()})")
            r = run_backtest(name, df, fee_rt=fee_rt, category="Forex")
            results.append(r)
            print(f"  trades={r['trades']}  WR={r['win_rate']}%  PF={r['pf']}"
                  f"  gross={r['gross_pnl']:+.2f}  fees=-{r['total_fees']:.2f}"
                  f"  NET={r['net_pnl']:+.2f}  DD={r['max_dd']:.1f}%")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    if not results:
        print("No results."); return

    # ─── Summary table ────────────────────────────────────────────────────────
    cols = ["symbol","category","trades","win_rate","pf",
            "gross_pnl","total_fees","net_pnl","net_pct","max_dd","sharpe","avg_win","avg_loss"]
    df_r = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    df_r = df_r.sort_values("net_pnl", ascending=False)

    print("\n" + "="*65)
    print("  FINAL RESULTS  (all on $1,000 base, fixed $10 risk/trade)")
    print("="*65)

    # Per-asset detail
    print(f"\n{'Symbol':<10} {'Cat':>6} {'Trades':>7} {'WR%':>6} {'PF':>5} "
          f"{'Gross$':>9} {'Fees$':>8} {'Net$':>8} {'Net%':>7} {'MaxDD':>7} {'Sharpe':>7}")
    print("-" * 85)
    for _, row in df_r.iterrows():
        print(f"{row['symbol']:<10} {row['category']:>6} {row['trades']:>7} "
              f"{row['win_rate']:>6.1f} {row['pf']:>5.3f} "
              f"{row['gross_pnl']:>+9.2f} {-row['total_fees']:>8.2f} "
              f"{row['net_pnl']:>+8.2f} {row['net_pct']:>+7.2f}% "
              f"{row['max_dd']:>7.2f}% {row['sharpe']:>7.2f}")

    # Category averages
    print("\n" + "-"*85)
    for cat in ["Crypto", "Forex"]:
        sub = df_r[df_r["category"] == cat]
        if sub.empty: continue
        print(f"\n{cat} average ({len(sub)} pairs):")
        print(f"  Win Rate  : {sub['win_rate'].mean():.1f}%")
        print(f"  PF        : {sub['pf'].mean():.3f}")
        print(f"  Gross PnL : ${sub['gross_pnl'].mean():+.2f}")
        print(f"  Fees      : -${sub['total_fees'].mean():.2f}")
        print(f"  Net PnL   : ${sub['net_pnl'].mean():+.2f}  ({sub['net_pct'].mean():+.2f}%)")
        print(f"  Max DD    : {sub['max_dd'].mean():.2f}%")
        print(f"  Sharpe    : {sub['sharpe'].mean():.2f}")
        print(f"  Avg Win   : ${sub['avg_win'].mean():.4f}")
        print(f"  Avg Loss  : ${sub['avg_loss'].mean():.4f}")

    # Verdict
    c_net  = df_r[df_r["category"]=="Crypto"]["net_pnl"].mean()
    f_net  = df_r[df_r["category"]=="Forex"]["net_pnl"].mean()
    c_dd   = df_r[df_r["category"]=="Crypto"]["max_dd"].mean()
    f_dd   = df_r[df_r["category"]=="Forex"]["max_dd"].mean()
    print("\n" + "="*65)
    print("  VERDICT: CRYPTO vs FOREX")
    print("="*65)
    print(f"  Better returns  : {'Crypto' if c_net > f_net else 'Forex'}"
          f"  (Crypto avg net ${c_net:+.2f}  vs  Forex ${f_net:+.2f})")
    print(f"  Lower drawdown  : {'Crypto' if abs(c_dd)<abs(f_dd) else 'Forex'}"
          f"  (Crypto {c_dd:.1f}%  vs  Forex {f_dd:.1f}%)")

    # Save
    csv_path = OUTPUT_DIR / "backtest_v2.csv"
    df_r.drop(columns=["equity","trades_df"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\n  Results saved -> {csv_path}")

    # Equity sparklines
    print("\n--- EQUITY CURVES ($1,000 start) ---")
    for r in sorted(results, key=lambda x: x["net_pnl"], reverse=True):
        eq    = r["equity"]
        final = eq[-1]
        pct   = (final - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        bar   = int(min(max(pct / 4, -25), 25))
        mark  = ("+" * max(bar,0)) if bar >= 0 else ("-" * abs(min(bar,0)))
        print(f"  {r['symbol']:<10} [{r['category']:<6}]  ${final:>8.2f}  ({pct:+.1f}%)  {mark}")

    print("\n  Backtest v2 complete.")


if __name__ == "__main__":
    main()
