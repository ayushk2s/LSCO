"""
live_market_data.py  --  Real-time market microstructure data
=============================================================
Fetches OI, order book, and trade flow from AsterDEX.
Falls back to Binance Futures API if AsterDEX endpoint returns nothing.

Use this alongside the ICT/LZR signal to confirm entries:
  - OI rising at a swept swing low → trapped shorts → LONG confirmed
  - Delta positive (more aggressive buys) → buyers absorbing the sweep
  - Order book bid wall within 0.3% of sweep low → institutional support

DOES NOT modify account_data.py or liq_algo.py.
"""

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

# ─── API endpoints ────────────────────────────────────────────────────────────
ASTERDEX_BASE   = "https://fapi.asterdex.com"
BINANCE_FAPI    = "https://fapi.binance.com"
TIMEOUT_SEC     = 8

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(url, params=None, timeout=TIMEOUT_SEC):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def _ts_now():
    return int(time.time() * 1000)


# ─── Open Interest ────────────────────────────────────────────────────────────

def get_oi(symbol: str) -> dict:
    """
    Current Open Interest from AsterDEX.
    Falls back to Binance if unavailable.
    Returns: {symbol, oi_usd, oi_qty, source, ts}
    """
    # Try AsterDEX first
    data = _get(f"{ASTERDEX_BASE}/fapi/v1/openInterest", {"symbol": symbol})
    if data and "openInterest" in data:
        oi_qty = float(data["openInterest"])
        price  = float(data.get("markPrice") or get_mark_price(symbol) or 1)
        return {
            "symbol": symbol,
            "oi_qty": round(oi_qty, 4),
            "oi_usd": round(oi_qty * price, 2),
            "source": "asterdex",
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    # Fallback: Binance
    data = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    if data and "openInterest" in data:
        oi_qty = float(data["openInterest"])
        price  = float(data.get("markPrice") or get_mark_price_binance(symbol) or 1)
        return {
            "symbol": symbol,
            "oi_qty": round(oi_qty, 4),
            "oi_usd": round(oi_qty * price, 2),
            "source": "binance",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    return {"symbol": symbol, "oi_qty": None, "oi_usd": None, "source": "error"}


def get_oi_history(symbol: str, period: str = "1h", limit: int = 48) -> Optional[pd.DataFrame]:
    """
    Historical OI from Binance (AsterDEX may not have this endpoint).
    period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
    Returns DataFrame: [ts, sumOpenInterest, sumOpenInterestValue]
    Up to 500 periods back (Binance limit).
    """
    data = _get(
        f"{BINANCE_FAPI}/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": min(limit, 500)},
    )
    if not data:
        return None
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi_qty"] = df["sumOpenInterest"].astype(float)
    df["oi_usd"] = df["sumOpenInterestValue"].astype(float)
    return df[["ts", "oi_qty", "oi_usd"]].set_index("ts")


def oi_trend(symbol: str, lookback_periods: int = 12, period: str = "1h") -> dict:
    """
    Compute OI trend: how much has OI changed over the last N periods?
    Returns: {oi_now, oi_then, change_pct, trend}
      trend: "rising" / "falling" / "flat"
    """
    df = get_oi_history(symbol, period=period, limit=lookback_periods + 2)
    if df is None or len(df) < 2:
        return {"trend": "unknown", "change_pct": None}

    oi_now  = df["oi_qty"].iloc[-1]
    oi_then = df["oi_qty"].iloc[0]
    chg     = (oi_now - oi_then) / oi_then * 100 if oi_then != 0 else 0
    trend   = "rising" if chg > 1.0 else ("falling" if chg < -1.0 else "flat")
    return {
        "oi_now": round(oi_now, 2),
        "oi_then": round(oi_then, 2),
        "change_pct": round(chg, 3),
        "trend": trend,
        "periods": lookback_periods,
        "period_size": period,
    }


# ─── Mark price ───────────────────────────────────────────────────────────────

def get_mark_price(symbol: str) -> Optional[float]:
    data = _get(f"{ASTERDEX_BASE}/fapi/v1/premiumIndex", {"symbol": symbol})
    if data and "markPrice" in data:
        return float(data["markPrice"])
    return get_mark_price_binance(symbol)


def get_mark_price_binance(symbol: str) -> Optional[float]:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if data and "markPrice" in data:
        return float(data["markPrice"])
    return None


# ─── Order book ───────────────────────────────────────────────────────────────

def get_orderbook(symbol: str, depth: int = 20) -> dict:
    """
    Fetch order book snapshot.
    Returns: {bids, asks, mid, bid_wall, ask_wall, imbalance, source}
      bid_wall: largest single bid (price, qty) in top N levels
      ask_wall: largest single ask in top N levels
      imbalance: bid_total_qty / (bid + ask total) → >0.6 = bid-heavy
    """
    data = _get(
        f"{ASTERDEX_BASE}/fapi/v1/depth",
        {"symbol": symbol, "limit": depth},
    )
    src = "asterdex"
    if not data or "bids" not in data:
        data = _get(f"{BINANCE_FAPI}/fapi/v1/depth", {"symbol": symbol, "limit": depth})
        src  = "binance"
    if not data:
        return {"symbol": symbol, "source": "error"}

    bids = [(float(p), float(q)) for p, q in data["bids"]]
    asks = [(float(p), float(q)) for p, q in data["asks"]]

    bid_total = sum(q for _, q in bids)
    ask_total = sum(q for _, q in asks)
    total     = bid_total + ask_total
    imbalance = bid_total / total if total > 0 else 0.5

    mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None

    bid_wall = max(bids, key=lambda x: x[1]) if bids else None
    ask_wall = max(asks, key=lambda x: x[1]) if asks else None

    bid_wall_pct = abs(bid_wall[0] - mid) / mid * 100 if bid_wall and mid else None
    ask_wall_pct = abs(ask_wall[0] - mid) / mid * 100 if ask_wall and mid else None

    return {
        "symbol": symbol,
        "mid": round(mid, 6) if mid else None,
        "bid_top": bids[0] if bids else None,
        "ask_top": asks[0] if asks else None,
        "bid_total_qty": round(bid_total, 4),
        "ask_total_qty": round(ask_total, 4),
        "imbalance": round(imbalance, 4),
        "bid_wall": {"price": bid_wall[0], "qty": bid_wall[1],
                     "pct_from_mid": round(bid_wall_pct, 4)} if bid_wall else None,
        "ask_wall": {"price": ask_wall[0], "qty": ask_wall[1],
                     "pct_from_mid": round(ask_wall_pct, 4)} if ask_wall else None,
        "bids_top5": bids[:5],
        "asks_top5": asks[:5],
        "source": src,
    }


def find_book_walls(symbol: str, near_price: float, pct_range: float = 0.5) -> dict:
    """
    Find large limit orders near a specific price (e.g., near a swing low).
    Useful for confirming institutional support before a LONG entry.
    pct_range: look within ±pct_range% of near_price
    """
    ob = get_orderbook(symbol, depth=50)
    if "error" in ob.get("source", ""):
        return {}

    lo = near_price * (1 - pct_range / 100)
    hi = near_price * (1 + pct_range / 100)

    bids_near = [(p, q) for p, q in ob.get("bids_top5", []) if lo <= p <= hi]
    asks_near = [(p, q) for p, q in ob.get("asks_top5", []) if lo <= p <= hi]

    # Actually need all levels, not just top 5 — re-fetch
    raw = _get(f"{ASTERDEX_BASE}/fapi/v1/depth", {"symbol": symbol, "limit": 100})
    if not raw:
        raw = _get(f"{BINANCE_FAPI}/fapi/v1/depth", {"symbol": symbol, "limit": 100})
    if not raw:
        return {"bids_near": [], "asks_near": []}

    all_bids = [(float(p), float(q)) for p, q in raw.get("bids", []) if lo <= float(p) <= hi]
    all_asks = [(float(p), float(q)) for p, q in raw.get("asks", []) if lo <= float(p) <= hi]

    bid_total_near = sum(q for _, q in all_bids)
    ask_total_near = sum(q for _, q in all_asks)

    return {
        "near_price": near_price,
        "range_pct": pct_range,
        "bids_near": all_bids,
        "asks_near": all_asks,
        "bid_qty_near": round(bid_total_near, 4),
        "ask_qty_near": round(ask_total_near, 4),
        "support_strong": bid_total_near > ask_total_near * 2,
    }


# ─── Recent trades / CVD ──────────────────────────────────────────────────────

def get_recent_trades(symbol: str, count: int = 500) -> pd.DataFrame:
    """
    Fetch recent trades. Returns DataFrame with columns:
      [ts, price, qty, is_buyer_maker, buy_vol, sell_vol]
    isBuyerMaker=True  → seller was aggressor → SELL volume
    isBuyerMaker=False → buyer was aggressor  → BUY volume
    """
    data = _get(
        f"{ASTERDEX_BASE}/fapi/v1/trades",
        {"symbol": symbol, "limit": min(count, 1000)},
    )
    src = "asterdex"
    if not data:
        data = _get(
            f"{BINANCE_FAPI}/fapi/v1/trades",
            {"symbol": symbol, "limit": min(count, 1000)},
        )
        src = "binance"
    if not data:
        return pd.DataFrame()

    rows = []
    for t in data:
        price     = float(t["price"])
        qty       = float(t["qty"])
        is_maker  = t.get("isBuyerMaker", False)
        buy_vol   = 0 if is_maker else qty
        sell_vol  = qty if is_maker else 0
        rows.append({
            "ts": pd.Timestamp(t["time"], unit="ms", tz="UTC"),
            "price": price,
            "qty": qty,
            "is_buyer_maker": is_maker,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "notional": price * qty,
        })

    df = pd.DataFrame(rows)
    df["source"] = src
    return df.sort_values("ts")


def compute_cvd(symbol: str, count: int = 500) -> dict:
    """
    Cumulative Volume Delta over the last N trades.
    CVD = buy_volume - sell_volume
    Positive → buyers dominant → bullish pressure
    Negative → sellers dominant → bearish pressure
    Also returns: buy_vol, sell_vol, delta_pct, large_print threshold
    """
    df = get_recent_trades(symbol, count)
    if df.empty:
        return {"cvd": None, "error": "no data"}

    buy_vol   = df["buy_vol"].sum()
    sell_vol  = df["sell_vol"].sum()
    total_vol = buy_vol + sell_vol
    cvd       = buy_vol - sell_vol
    delta_pct = cvd / total_vol * 100 if total_vol > 0 else 0

    # Large prints: trades in top 5% by notional
    threshold = df["notional"].quantile(0.95)
    large     = df[df["notional"] >= threshold]
    lp_buys   = large["buy_vol"].sum()
    lp_sells  = large["sell_vol"].sum()

    bias = "BULLISH" if delta_pct > 10 else ("BEARISH" if delta_pct < -10 else "NEUTRAL")

    return {
        "symbol": symbol,
        "buy_vol": round(buy_vol, 4),
        "sell_vol": round(sell_vol, 4),
        "cvd": round(cvd, 4),
        "delta_pct": round(delta_pct, 2),
        "bias": bias,
        "large_print_buys": round(lp_buys, 4),
        "large_print_sells": round(lp_sells, 4),
        "large_print_bias": "BUY" if lp_buys > lp_sells else "SELL",
        "trade_count": len(df),
    }


def get_agg_trades_since(symbol: str, since_ts_ms: int) -> pd.DataFrame:
    """
    Aggregate trades since a given timestamp (for measuring flow at a specific event).
    Useful for: measuring buy/sell delta AFTER price sweeps a swing low.
    """
    data = _get(
        f"{ASTERDEX_BASE}/fapi/v1/aggTrades",
        {"symbol": symbol, "startTime": since_ts_ms, "limit": 1000},
    )
    if not data:
        data = _get(
            f"{BINANCE_FAPI}/fapi/v1/aggTrades",
            {"symbol": symbol, "startTime": since_ts_ms, "limit": 1000},
        )
    if not data:
        return pd.DataFrame()

    rows = []
    for t in data:
        price    = float(t["p"])
        qty      = float(t["q"])
        is_maker = t.get("m", False)
        rows.append({
            "ts": pd.Timestamp(t["T"], unit="ms", tz="UTC"),
            "price": price,
            "qty": qty,
            "is_buyer_maker": is_maker,
            "buy_vol": 0 if is_maker else qty,
            "sell_vol": qty if is_maker else 0,
        })
    return pd.DataFrame(rows)


# ─── Funding rate ─────────────────────────────────────────────────────────────

def get_funding_rate(symbol: str) -> dict:
    """
    Current funding rate. Positive = longs pay shorts (crowded LONG).
    Extreme positive → contrarian SHORT signal (or LONG at discount).
    Extreme negative → contrarian LONG signal.
    """
    data = _get(f"{ASTERDEX_BASE}/fapi/v1/premiumIndex", {"symbol": symbol})
    if not data:
        data = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if not data:
        return {"symbol": symbol, "funding_rate": None}

    rate = float(data.get("lastFundingRate", 0))
    rate_8h_pct = rate * 100
    annual_pct  = rate * 100 * 3 * 365  # 3 funding events/day

    return {
        "symbol": symbol,
        "funding_rate_pct_8h": round(rate_8h_pct, 4),
        "funding_annual_pct": round(annual_pct, 2),
        "bias": "crowded_long" if rate > 0.001 else
                ("crowded_short" if rate < -0.001 else "neutral"),
        "next_funding_ts": data.get("nextFundingTime"),
    }


# ─── Long/Short ratio ─────────────────────────────────────────────────────────

def get_ls_ratio(symbol: str, period: str = "1h", limit: int = 12) -> dict:
    """
    Long/Short account ratio from Binance (number of accounts long vs short).
    High L/S (>1.5) = crowded longs = potential long squeeze if price falls.
    Low L/S (<0.7) = crowded shorts = potential short squeeze on sweep up.
    """
    data = _get(
        f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": period, "limit": limit},
    )
    if not data:
        return {"symbol": symbol, "ls_ratio": None}

    latest = data[-1]
    return {
        "symbol": symbol,
        "ls_ratio": float(latest.get("longShortRatio", 1)),
        "long_pct": float(latest.get("longAccount", 0.5)) * 100,
        "short_pct": float(latest.get("shortAccount", 0.5)) * 100,
        "bias": "crowded_long" if float(latest.get("longShortRatio", 1)) > 1.5 else
                ("crowded_short" if float(latest.get("longShortRatio", 1)) < 0.7 else "balanced"),
        "ts": latest.get("timestamp"),
    }


# ─── Composite signal ─────────────────────────────────────────────────────────

def compute_market_bias(symbol: str, near_price: Optional[float] = None) -> dict:
    """
    Composite market microstructure signal.
    Runs all checks and returns a LONG / SHORT / NEUTRAL composite bias.

    near_price: if provided, also checks order book depth near that price
                (useful when checking confirmation near a swept swing low).

    LONG confirmation signals (for ICT sweep reversal setup):
      1. OI rising (new longs being added, or shorts trapped)
      2. CVD positive (aggressive buyers dominating after the sweep)
      3. Order book: bid-heavy near the swept level (institutional support)
      4. Funding: negative or neutral (not crowded long = less squeeze risk)
      5. L/S ratio: not extreme (below 1.5)

    Returns: {composite_bias, score, details}
      score: -5 to +5 (positive = bullish, negative = bearish)
    """
    print(f"  Fetching market bias for {symbol}...", flush=True)
    details = {}
    score   = 0

    # 1. OI trend
    oi_t = oi_trend(symbol, lookback_periods=6, period="1h")
    details["oi"] = oi_t
    if oi_t["trend"] == "rising":
        score += 1
        details["oi"]["signal"] = "+1 (rising OI = positioning building)"
    elif oi_t["trend"] == "falling":
        score -= 1
        details["oi"]["signal"] = "-1 (falling OI = liquidation / unwinding)"
    else:
        details["oi"]["signal"] = "0 (flat OI)"

    # 2. CVD (buy/sell delta)
    cvd = compute_cvd(symbol, count=500)
    details["cvd"] = cvd
    if cvd.get("delta_pct", 0) > 15:
        score += 2
        cvd["signal"] = "+2 (strong buy pressure)"
    elif cvd.get("delta_pct", 0) > 5:
        score += 1
        cvd["signal"] = "+1 (mild buy pressure)"
    elif cvd.get("delta_pct", 0) < -15:
        score -= 2
        cvd["signal"] = "-2 (strong sell pressure)"
    elif cvd.get("delta_pct", 0) < -5:
        score -= 1
        cvd["signal"] = "-1 (mild sell pressure)"
    else:
        cvd["signal"] = "0 (neutral)"

    # 3. Order book imbalance
    ob = get_orderbook(symbol, depth=20)
    details["orderbook"] = ob
    imbalance = ob.get("imbalance", 0.5)
    if imbalance > 0.65:
        score += 1
        ob["signal"] = "+1 (bid-heavy book)"
    elif imbalance < 0.35:
        score -= 1
        ob["signal"] = "-1 (ask-heavy book)"
    else:
        ob["signal"] = "0 (balanced book)"

    # 3b. Walls near swept level
    if near_price is not None:
        walls = find_book_walls(symbol, near_price, pct_range=0.3)
        details["walls_near_price"] = walls
        if walls.get("support_strong"):
            score += 1
            walls["signal"] = "+1 (strong bid wall near swept level)"
        else:
            walls["signal"] = "0 (no notable bid wall)"

    # 4. Funding rate
    fr = get_funding_rate(symbol)
    details["funding"] = fr
    if fr.get("funding_rate_pct_8h", 0) is not None:
        if fr["funding_rate_pct_8h"] < -0.01:
            score += 1
            fr["signal"] = "+1 (negative funding = crowded shorts, long friendly)"
        elif fr["funding_rate_pct_8h"] > 0.05:
            score -= 1
            fr["signal"] = "-1 (high positive funding = crowded longs)"
        else:
            fr["signal"] = "0 (neutral funding)"

    # 5. L/S ratio
    ls = get_ls_ratio(symbol, period="1h", limit=1)
    details["ls_ratio"] = ls
    if ls.get("ls_ratio") is not None:
        if ls["ls_ratio"] < 0.8:
            score += 1
            ls["signal"] = "+1 (crowded shorts = potential squeeze)"
        elif ls["ls_ratio"] > 1.8:
            score -= 1
            ls["signal"] = "-1 (crowded longs = squeeze risk)"
        else:
            ls["signal"] = "0 (balanced positioning)"

    # Composite
    if score >= 3:
        composite = "STRONG_LONG"
    elif score >= 1:
        composite = "LONG"
    elif score <= -3:
        composite = "STRONG_SHORT"
    elif score <= -1:
        composite = "SHORT"
    else:
        composite = "NEUTRAL"

    return {
        "symbol": symbol,
        "composite_bias": composite,
        "score": score,
        "max_score": 6 if near_price else 5,
        "ts": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }


# ─── ICT pre-trade filter ────────────────────────────────────────────────────

def ict_entry_confirm(symbol: str, direction: str,
                       swept_level: float,
                       min_score: int = 2) -> dict:
    """
    Run before entering an ICT Liquidity Sweep trade.
    direction: "LONG" or "SHORT"

    Returns: {approved, score, reason, details}
    """
    bias = compute_market_bias(symbol, near_price=swept_level)
    score = bias["score"]

    if direction == "LONG":
        approved = score >= min_score
        reason = (f"Score {score}: {bias['composite_bias']} — LONG {'APPROVED' if approved else 'FILTERED'}")
    else:
        approved = score <= -min_score
        reason = (f"Score {score}: {bias['composite_bias']} — SHORT {'APPROVED' if approved else 'FILTERED'}")

    return {
        "symbol": symbol,
        "direction": direction,
        "swept_level": swept_level,
        "approved": approved,
        "score": score,
        "composite_bias": bias["composite_bias"],
        "reason": reason,
        "details": bias["details"],
    }


# ─── Quick check — print a summary ───────────────────────────────────────────

def market_snapshot(symbol: str) -> None:
    """Print a formatted market microstructure snapshot for a symbol."""
    print(f"\n{'='*60}")
    print(f"  MARKET SNAPSHOT: {symbol}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)

    px = get_mark_price(symbol)
    print(f"  Mark price: {px}")

    oi = get_oi(symbol)
    print(f"  OI:  {oi.get('oi_qty', 'N/A')} contracts  "
          f"(≈ ${oi.get('oi_usd', 'N/A'):,.0f})  [{oi['source']}]")

    oi_t = oi_trend(symbol)
    print(f"  OI trend (12h):  {oi_t['trend']}  "
          f"change {oi_t.get('change_pct', 'N/A')}%")

    cvd = compute_cvd(symbol, 500)
    print(f"  CVD (500 trades): buy {cvd.get('buy_vol','?'):.2f}  "
          f"sell {cvd.get('sell_vol','?'):.2f}  "
          f"delta {cvd.get('delta_pct','?'):.1f}%  → {cvd.get('bias','?')}")
    print(f"  Large prints: {cvd.get('large_print_bias','?')}")

    ob = get_orderbook(symbol, 20)
    print(f"  Order book imbalance: {ob.get('imbalance',0.5):.3f}  "
          f"(>0.6 bid-heavy, <0.4 ask-heavy)")
    if ob.get("bid_wall"):
        bw = ob["bid_wall"]
        print(f"  Largest bid wall: {bw['qty']:.2f} @ {bw['price']}  "
              f"({bw['pct_from_mid']:.3f}% below mid)")
    if ob.get("ask_wall"):
        aw = ob["ask_wall"]
        print(f"  Largest ask wall: {aw['qty']:.2f} @ {aw['price']}  "
              f"({aw['pct_from_mid']:.3f}% above mid)")

    fr = get_funding_rate(symbol)
    print(f"  Funding: {fr.get('funding_rate_pct_8h','?')}%/8h  "
          f"({fr.get('bias','?')})")

    ls = get_ls_ratio(symbol)
    print(f"  L/S ratio: {ls.get('ls_ratio','?')}  "
          f"long {ls.get('long_pct','?'):.1f}%  short {ls.get('short_pct','?'):.1f}%  "
          f"({ls.get('bias','?')})")
    print()


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    for sym in symbols:
        market_snapshot(sym)

    # Demo: composite bias
    if symbols:
        sym = symbols[0]
        print(f"\n  Computing composite bias for {sym}...")
        bias = compute_market_bias(sym)
        print(f"  Score: {bias['score']}/{bias['max_score']}  →  {bias['composite_bias']}")

        # Demo ICT entry check (simulate a swept level at current price - 0.5%)
        px = get_mark_price(sym)
        if px:
            swept = px * 0.995
            print(f"\n  Simulating ICT LONG entry check (swept level @ {swept:.4f}):")
            result = ict_entry_confirm(sym, "LONG", swept_level=swept)
            print(f"  → {result['reason']}")
