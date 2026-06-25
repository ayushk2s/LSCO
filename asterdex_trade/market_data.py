"""
market_data.py
Public market-data helpers — no authentication needed.

Tries AsterDEX endpoints first (so OB prices match where we're trading).
Falls back to Binance Futures for klines / ATR when AsterDEX data is unavailable.
"""

import requests

ASTER_HOST   = "https://fapi.asterdex.com"
BINANCE_HOST = "https://fapi.binance.com"
TIMEOUT      = 6   # seconds per HTTP request


# ══════════════════════════════════════════════════════════════════════════════
# ORDER BOOK  — used by SmartLimitOrder to know where to place/replace orders
# ══════════════════════════════════════════════════════════════════════════════
def get_ob(symbol: str, limit: int = 5) -> dict:
    """
    Return order-book top levels from AsterDEX.
    Result: {"best_bid": float, "best_ask": float,
             "bids": [(price, qty), ...], "asks": [(price, qty), ...]}

    'best_bid' = highest price someone is willing to buy at (we sell here as maker)
    'best_ask' = lowest price someone is willing to sell at  (we buy here as maker)
    """
    try:
        r = requests.get(
            f"{ASTER_HOST}/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        ob = r.json()
        bids = [(float(p), float(q)) for p, q in ob["bids"]]
        asks = [(float(p), float(q)) for p, q in ob["asks"]]
        return {
            "best_bid": bids[0][0] if bids else 0.0,
            "best_ask": asks[0][0] if asks else 0.0,
            "bids": bids,
            "asks": asks,
        }
    except Exception as e:
        raise RuntimeError(f"[OB] AsterDEX depth failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CURRENT PRICE
# ══════════════════════════════════════════════════════════════════════════════
def get_price(symbol: str) -> float:
    """
    Get latest mark/index price from AsterDEX.
    Falls back to Binance mark price if AsterDEX fails.
    """
    # Try AsterDEX ticker price
    try:
        r = requests.get(
            f"{ASTER_HOST}/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        pass

    # Fall back to Binance mark price (very reliable)
    r = requests.get(
        f"{BINANCE_HOST}/fapi/v1/ticker/price",
        params={"symbol": symbol},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return float(r.json()["price"])


# ══════════════════════════════════════════════════════════════════════════════
# KLINES (candlestick data)  — used for ATR calculation
# ══════════════════════════════════════════════════════════════════════════════
def get_klines(symbol: str, interval: str = "5m", limit: int = 50) -> list:
    """
    Fetch candlestick data.  Returns raw kline list (Binance format):
    [openTime, open, high, low, close, volume, closeTime, ...]

    Tries AsterDEX first, falls back to Binance.
    interval examples: "1m", "5m", "15m", "1h"
    """
    for host in (ASTER_HOST, BINANCE_HOST):
        try:
            r = requests.get(
                f"{host}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    raise RuntimeError(f"[KLINES] Both AsterDEX and Binance failed for {symbol} {interval}")


# ══════════════════════════════════════════════════════════════════════════════
# ATR  (Average True Range)  — used for TP / SL distance calculation
# ══════════════════════════════════════════════════════════════════════════════
def calc_atr(klines: list, period: int = 14) -> float:
    """
    Wilder's ATR on completed klines.
    ATR measures average price volatility; used to set TP = 1.5×ATR, SL = 0.75×ATR.

    klines: list of raw kline rows [open_ts, open, high, low, close, ...]
    Returns: ATR value in price units (e.g. $350 for BTC on 5m chart)
    """
    if len(klines) < period + 1:
        # Not enough data: rough estimate using last few candles
        trs = [float(k[2]) - float(k[3]) for k in klines]
        return sum(trs) / len(trs) if trs else 0.0

    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    true_ranges = []
    for i in range(1, len(klines)):
        hi  = float(klines[i][2])
        lo  = float(klines[i][3])
        pc  = float(klines[i - 1][4])   # previous candle close
        tr  = max(hi - lo, abs(hi - pc), abs(lo - pc))
        true_ranges.append(tr)

    # First ATR = simple average of first `period` true ranges
    atr = sum(true_ranges[:period]) / period

    # Wilder's smoothing for the rest
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr


# ══════════════════════════════════════════════════════════════════════════════
# LAST COMPLETED 1-MINUTE CANDLE  — for entry trigger detection
# ══════════════════════════════════════════════════════════════════════════════
def get_last_1m_candle(symbol: str) -> dict:
    """
    Returns OHLC of the most recently COMPLETED 1-minute candle.
    Used to check if price closed back inside the liq zone (entry trigger).
    """
    # Fetch 3 candles: [2 ago, last complete, current incomplete]
    klines = get_klines(symbol, "1m", 3)
    k = klines[-2]   # second-to-last = last completed candle
    return {
        "open":  float(k[1]),
        "high":  float(k[2]),
        "low":   float(k[3]),
        "close": float(k[4]),
        "ts":    int(k[0]),
    }
