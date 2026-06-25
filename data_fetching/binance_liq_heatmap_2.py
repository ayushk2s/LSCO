"""
BTC/USDT Liquidation Heatmap v2
Exchanges : Binance futures + Bybit (fallback-graceful)
Key fixes vs v1:
  • BUCKET = $10  (was $100) — entry prices at natural floats map to
    realistic liq levels like 78,985 / 79,015, NOT artificial 78,900 / 79,000
  • liq price computed at full float precision, THEN rounded to bucket
    (was: floor the entry price first — wrong, biases all levels down)
  • 7-point OHLCVWAP entry distribution per candle instead of single midpoint
  • Bybit OI added as ~25 % weight secondary source
  • Vectorized sweep-clearing (fast even with 40 k price rows)
  • Console prints EVERY level ≥ $1 M with the EXACT price annotated on the image
  • Saves a JSON file with all levels for external/trading use
"""

import argparse
import json
import os
import sys
# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime

# ── Per-symbol config ─────────────────────────────────────────────────────────
_SYMBOL_CFG = {
    "BTCUSDT": {"bucket": 100, "min_print": 5_000_000},
    "ETHUSDT": {"bucket":   5, "min_print":   500_000},
    "XAUUSDT": {"bucket":   5, "min_print":   100_000},
}

_HERE       = os.path.dirname(os.path.abspath(__file__))

# These are set in main() from CLI args / symbol config
SYMBOL      = "BTCUSDT"
OUTPUT_PNG  = os.path.join(_HERE, "binance_liq_heatmap_BTCUSDT.png")
OUTPUT_JSON = os.path.join(_HERE, "binance_liq_heatmap_BTCUSDT.json")
BUCKET      = 100
VIS_PAD     = 0.09
LIQ_THRESH  = 0.25
MIN_PRINT   = 5_000_000

# Leverage distribution: % of open interest at each leverage level
LEVERAGE_DIST = {
    2:   0.04,
    3:   0.06,
    5:   0.14,
    10:  0.30,
    20:  0.24,
    25:  0.08,
    50:  0.10,
    100: 0.04,
}

# ── HTTP helper ───────────────────────────────────────────────────────────────
def _get(url, params=None, timeout=12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ── Fetch Binance ─────────────────────────────────────────────────────────────
def fetch_binance():
    print("Binance: 4h OI history …")
    oi_raw = _get("https://fapi.binance.com/futures/data/openInterestHist",
                   {"symbol": SYMBOL, "period": "4h", "limit": 186})

    print("Binance: 4h klines …")
    kl_raw = _get("https://fapi.binance.com/fapi/v1/klines",
                   {"symbol": SYMBOL, "interval": "4h", "limit": 187})

    print("Binance: 4h L/S ratio …")
    ls_raw = _get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                   {"symbol": SYMBOL, "period": "4h", "limit": 186})

    print("Binance: 5m klines (24h display) …")
    kl5_raw = _get("https://fapi.binance.com/fapi/v1/klines",
                    {"symbol": SYMBOL, "interval": "5m", "limit": 289})

    oi = [{"ts": int(r["timestamp"]),    "usd": float(r["sumOpenInterestValue"])} for r in oi_raw]
    kl = [{"ts": int(k[0]), "o": float(k[1]), "h": float(k[2]),
            "l": float(k[3]), "c": float(k[4])} for k in kl_raw[:-1]]
    ls = {int(r["timestamp"]): float(r["longAccount"]) for r in ls_raw}
    kl5 = kl5_raw[:-1]   # drop incomplete current candle
    return oi, kl, ls, kl5

# ── Fetch Bybit (optional) ────────────────────────────────────────────────────
def fetch_bybit():
    print("Bybit: 4h OI history …")
    r1 = _get("https://api.bybit.com/v5/market/open-interest",
               {"category": "linear", "symbol": SYMBOL,
                "intervalTime": "4h", "limit": 200})
    oi_list = list(reversed(r1["result"]["list"]))   # oldest → newest

    print("Bybit: 4h klines …")
    r2 = _get("https://api.bybit.com/v5/market/kline",
               {"category": "linear", "symbol": SYMBOL,
                "interval": "240", "limit": 200})
    kl_list = list(reversed(r2["result"]["list"]))   # oldest → newest

    oi = [{"ts": int(x["timestamp"]), "usd": float(x["openInterest"])}
          for x in oi_list]
    # Bybit kline: [startTime, open, high, low, close, volume, turnover]
    kl = {int(k[0]): {"ts": int(k[0]), "o": float(k[1]), "h": float(k[2]),
                       "l": float(k[3]), "c": float(k[4])} for k in kl_list}
    return oi, kl

# ── Entry-price distribution within a 4h candle ───────────────────────────────
def entry_prices(o, h, l, c):
    """
    7 weighted entry prices spread across the candle's OHLC + VWAP.
    Traders don't all enter at the midpoint — they're distributed.
    This is what makes liq levels fall at non-round prices.
    Weights sum to 1.0.
    """
    vwap = (h + l + c) / 3
    return (
        (o,              0.10),
        ((o + vwap) / 2, 0.16),
        (vwap,           0.30),
        ((vwap + c) / 2, 0.16),
        (c,              0.18),
        (h,              0.05),
        (l,              0.05),
    )

# ── Build heat matrix ─────────────────────────────────────────────────────────
def build_heatmap(bn_oi, bn_kl, bn_ls, kl5,
                  bb_oi=None, bb_kl=None, bybit_weight=0.25):
    N5  = len(kl5)
    ts0 = int(kl5[0][0])

    # ── Price grid ────────────────────────────────────────────────────────────
    all_h = [k["h"] for k in bn_kl]
    all_l = [k["l"] for k in bn_kl]
    if bb_kl:
        for k in bb_kl.values():
            all_h.append(k["h"]); all_l.append(k["l"])
    mid  = (max(all_h) + min(all_l)) / 2
    pad  = mid * 0.14
    p_lo = round((min(all_l) - pad) / BUCKET) * BUCKET
    p_hi = round((max(all_h) + pad) / BUCKET) * BUCKET + BUCKET
    prices = list(range(p_lo, p_hi + BUCKET, BUCKET))
    P      = len(prices)
    p2i    = {p: i for i, p in enumerate(prices)}

    # ── 5m column index ───────────────────────────────────────────────────────
    ts5 = [int(k[0]) for k in kl5]

    def col_for_ts(ts_ms):
        if ts_ms <= ts0: return 0
        for j, t in enumerate(ts5):
            if t >= ts_ms: return j
        return N5 - 1

    heat = np.zeros((P, N5), dtype=np.float64)

    # ── Process one OI series ─────────────────────────────────────────────────
    def process(oi_list, oi_map, kl_dict, kl_ts_sorted,
                 ls_dict, ls_ts_sorted, weight):
        oi_ts = [r["ts"] for r in oi_list]
        for i in range(1, len(oi_ts)):
            t_c, t_p = oi_ts[i], oi_ts[i - 1]
            oi_c, oi_p = oi_map[t_c], oi_map[t_p]
            delta = oi_c - oi_p

            kts = min(kl_ts_sorted, key=lambda x: abs(x - t_c))
            kl  = kl_dict.get(kts)
            if kl is None:
                continue

            # Long/short ratio (default 50/50 when unavailable)
            if ls_ts_sorted:
                lts  = min(ls_ts_sorted, key=lambda x: abs(x - t_c))
                lpct = ls_dict.get(lts, 0.50)
            else:
                lpct = 0.50
            spct = 1.0 - lpct
            col  = col_for_ts(t_c)

            if delta >= 500:
                for ep, ep_wt in entry_prices(kl["o"], kl["h"], kl["l"], kl["c"]):
                    for lev, lev_wt in LEVERAGE_DIST.items():
                        # ── KEY FIX: exact liq price first, THEN round to bucket
                        ll = ep * (1.0 - 1.0 / lev)
                        sl = ep * (1.0 + 1.0 / lev)
                        ll_bkt = round(ll / BUCKET) * BUCKET
                        sl_bkt = round(sl / BUCKET) * BUCKET
                        amt    = delta * ep_wt * lev_wt * weight
                        if ll_bkt in p2i:
                            heat[p2i[ll_bkt], col:] += amt * lpct
                        if sl_bkt in p2i:
                            heat[p2i[sl_bkt], col:] += amt * spct

            elif delta < -500 and oi_p > 0:
                frac = min(abs(delta) / oi_p, 0.95)
                heat[:, col:] *= (1.0 - frac * weight)

    bn_oi_map = {r["ts"]: r["usd"] for r in bn_oi}
    bn_kl_map = {k["ts"]: k for k in bn_kl}
    bn_kl_ts  = sorted(bn_kl_map)
    bn_ls_ts  = sorted(bn_ls)

    process(bn_oi, bn_oi_map, bn_kl_map, bn_kl_ts, bn_ls, bn_ls_ts, weight=1.0)

    if bb_oi and bb_kl:
        bb_oi_map = {r["ts"]: r["usd"] for r in bb_oi}
        bb_kl_ts  = sorted(bb_kl)
        process(bb_oi, bb_oi_map, bb_kl, bb_kl_ts, {}, [], weight=bybit_weight)

    # ── Calibrate ─────────────────────────────────────────────────────────────
    current_oi = bn_oi_map[max(bn_oi_map)]
    if bb_oi:
        bb_oi_map2 = {r["ts"]: r["usd"] for r in bb_oi}
        current_oi += bb_oi_map2[max(bb_oi_map2)] * bybit_weight
    tracked = heat[:, -1].sum()
    if tracked > 0:
        scale = (current_oi * 0.58) / tracked
        heat *= min(scale, 5.0)

    # ── Vectorised sweep clearing ─────────────────────────────────────────────
    # When price trades through a bucket, those positions are gone.
    mids  = np.array(prices, dtype=np.float64) + BUCKET / 2
    lows  = np.array([float(k[3]) for k in kl5], dtype=np.float64)
    highs = np.array([float(k[2]) for k in kl5], dtype=np.float64)

    first_j = np.full(P, N5, dtype=np.int64)   # N5 = "never swept"
    for j in range(N5):
        mask = (lows[j] <= mids) & (mids <= highs[j]) & (first_j == N5)
        first_j[mask] = j

    for pi in np.where(first_j < N5)[0]:
        heat[pi, int(first_j[pi]):] = 0

    return heat, prices

# ── Console print + JSON save ─────────────────────────────────────────────────
def print_and_save(heat, prices, kl5, sources_used):
    cp     = float(kl5[-1][4])
    ts_str = datetime.fromtimestamp(int(kl5[-1][0]) / 1000).strftime("%d %b %Y, %H:%M")
    prof   = heat[:, -1]
    total  = prof.sum()
    peak   = max(prof.max(), 1.0)
    BAR    = 26

    def _bar(usd):
        n = max(1, int(usd / peak * BAR))
        return "█" * n + "░" * (BAR - n)

    # Use BUCKET MIDPOINTS as the reported price — these are the realistic prices
    # (e.g., bucket starting at 78,980 → midpoint 78,985)
    def level_list(above: bool):
        out = []
        for i, p in enumerate(prices):
            mid = p + BUCKET / 2
            v   = prof[i]
            if v < MIN_PRINT:
                continue
            if above and mid > cp:
                out.append((mid, v))
            elif not above and mid <= cp:
                out.append((mid, v))
        return out

    short_liq = sorted(level_list(True),  key=lambda x:  x[0])   # nearest first
    long_liq  = sorted(level_list(False), key=lambda x: -x[0])   # nearest first

    W = 78
    print("\n" + "=" * W)
    print(f"  {SYMBOL}  Liquidation Heatmap v2  [{ts_str}]")
    print(f"  Sources       : {sources_used}")
    print(f"  Current Price : ${cp:,.2f}")
    print(f"  Total modeled : ${total/1e9:.3f} B   (30-day OI, calibrated)")
    print(f"  Bucket width  : ${BUCKET}  ->  prices like ${int(cp/BUCKET)*BUCKET+BUCKET//2:,}")
    print("=" * W)

    hdr = f"  {'#':<4}  {'Price':>10}  {'Amount':>12}  {'Dist':>7}  Bar"
    div = "  " + "-" * 66

    print(f"\n  SHORT LIQUIDATIONS (UP)  -- bulls push price up, shorts get liq'd")
    print(hdr); print(div)
    for rank, (p, usd) in enumerate(short_liq[:25], 1):
        dist = (p - cp) / cp * 100
        print(f"  {rank:<4}  ${p:>9,.1f}  ${usd/1e6:>9.3f} M  {dist:>+6.2f}%  {_bar(usd)}")

    print(f"\n  LONG LIQUIDATIONS (DOWN) -- bears push price down, longs get liq'd")
    print(hdr); print(div)
    for rank, (p, usd) in enumerate(long_liq[:25], 1):
        dist = (p - cp) / cp * 100
        print(f"  {rank:<4}  ${p:>9,.1f}  ${usd/1e6:>9.3f} M  {dist:>+6.2f}%  {_bar(usd)}")

    print("=" * W + "\n")

    # ── JSON ──────────────────────────────────────────────────────────────────
    data = {
        "generated_at":     datetime.now().isoformat(),
        "sources":          sources_used,
        "current_price":    cp,
        "total_modeled_usd": total,
        "bucket_usd":       BUCKET,
        "short_liquidations": [
            {"price": round(p, 1), "usd": round(u, 0),
             "dist_pct": round((p - cp) / cp * 100, 3)}
            for p, u in short_liq
        ],
        "long_liquidations": [
            {"price": round(p, 1), "usd": round(u, 0),
             "dist_pct": round((p - cp) / cp * 100, 3)}
            for p, u in long_liq
        ],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved → {OUTPUT_JSON}")

    return short_liq, long_liq

# ── Colormap ──────────────────────────────────────────────────────────────────
def make_cmap():
    return mcolors.LinearSegmentedColormap.from_list("cg2", [
        (0.00, "#0a0118"), (0.18, "#1f0a5c"), (0.35, "#1a3a8a"),
        (0.50, "#0a6e7b"), (0.64, "#128a50"), (0.76, "#5ab52a"),
        (0.87, "#b5cc1a"), (0.94, "#d4e100"), (1.00, "#f0f000"),
    ])

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(heat, prices, kl5, short_liq, long_liq, sources_used):
    cmap = make_cmap()
    N5   = heat.shape[1]
    P    = len(prices)
    cp   = float(kl5[-1][4])
    ts5  = [int(k[0]) for k in kl5]

    h    = heat.copy()
    vmax = np.percentile(h[h > 0], 99.5) if (h > 0).any() else 1.0
    h[h < vmax * LIQ_THRESH] = 0
    h    = np.clip(h, 0, vmax)

    row_lo = max(0, int((cp * (1 - VIS_PAD) - prices[0]) // BUCKET))
    row_hi = min(P, int((cp * (1 + VIS_PAD) - prices[0]) // BUCKET) + 2)
    h_vis  = h[row_lo:row_hi, :]
    p_vis  = prices[row_lo:row_hi]

    fig   = plt.figure(figsize=(24, 12), facecolor="#080114")
    ax    = fig.add_axes([0.04, 0.09, 0.60, 0.84])
    ax_r  = fig.add_axes([0.67, 0.09, 0.25, 0.84])
    ax_cb = fig.add_axes([0.955, 0.09, 0.012, 0.84])

    # Heatmap (bilinear smoothing looks better with fine buckets)
    ax.imshow(h_vis, aspect="auto", cmap=cmap, origin="lower",
              extent=[0, N5, p_vis[0], p_vis[-1] + BUCKET],
              interpolation="bilinear", vmin=0, vmax=vmax)

    # 5m Candlesticks
    for i, k in enumerate(kl5):
        o, hi, lo, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i + .5, i + .5], [lo, hi], color=col, lw=0.5, alpha=0.9, zorder=5)
        b_lo, b_hi = min(o, c), max(o, c)
        b_hi = max(b_hi, b_lo + BUCKET * 1.5)
        ax.add_patch(mpatches.Rectangle((i + .08, b_lo), .84, b_hi - b_lo,
                                         color=col, alpha=0.95, zorder=6))

    # Current price line
    ax.axhline(cp, color="#ffffff", lw=1.0, ls="--", alpha=0.7, zorder=7)
    ax.text(N5 + .3, cp, f"${cp:,.0f}", color="white", fontsize=8.5,
            va="center", ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#222244", ec="white", lw=0.7))

    # ── Annotate chart: TOP levels from each side (exact prices) ─────────────
    # Combine short + long, pick 14 most significant by USD within visible range
    vis_lo = p_vis[0]; vis_hi = p_vis[-1] + BUCKET
    all_levels = (
        [(p, u, "SHT") for p, u in short_liq if vis_lo <= p <= vis_hi] +
        [(p, u, "LNG") for p, u in long_liq  if vis_lo <= p <= vis_hi]
    )
    all_levels.sort(key=lambda x: -x[1])

    labeled = 0
    for p_mid, usd, side in all_levels:
        if labeled >= 14: break
        dist = (p_mid - cp) / cp * 100
        col  = "#00eeff" if side == "SHT" else "#ffaa00"
        ax.axhline(p_mid, color=col, lw=0.35, ls=":", alpha=0.35, zorder=4)
        ax.text(N5 - 2, p_mid + BUCKET * 0.6,
                f"${p_mid:,.0f}  ${usd/1e6:.2f}M  {side} {dist:+.2f}%",
                color=col, fontsize=6.0, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.15", fc="#0a0118",
                          ec="#334455", lw=0.5, alpha=0.88), zorder=9)
        labeled += 1

    # Axes
    y_rng  = p_vis[-1] - p_vis[0]
    raw_step = y_rng / 8
    y_step = max(BUCKET * 10, round(raw_step / (BUCKET * 10)) * BUCKET * 10)
    yticks = range(int(p_vis[0] // y_step) * y_step, p_vis[-1] + y_step, y_step)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"${y:,}" for y in yticks], color="#aaaacc", fontsize=8)
    step = max(1, N5 // 12)
    xtk  = list(range(0, N5, step))
    ax.set_xticks(xtk)
    ax.set_xticklabels(
        [datetime.fromtimestamp(ts5[i] / 1000).strftime("%d, %H:%M") for i in xtk],
        color="#aaaacc", fontsize=7, rotation=25, ha="right")
    ax.set_xlim(0, N5)
    ax.set_ylim(p_vis[0], p_vis[-1] + BUCKET)
    ax.set_facecolor("#080114")
    for sp in ax.spines.values(): sp.set_color("#222244")
    ax.tick_params(colors="#aaaacc")

    # ── Right profile panel ───────────────────────────────────────────────────
    prof_full = heat[:, -1]
    pmax      = prof_full.max() if prof_full.max() > 0 else 1.0
    prof_vis  = prof_full[row_lo:row_hi]
    bar_cols  = [cmap(v / pmax) for v in prof_vis]
    ax_r.barh(p_vis, prof_vis / 1e6, height=BUCKET * 0.88, color=bar_cols, alpha=0.92)

    for p, usd in zip(p_vis, prof_vis):
        if usd < 2e6: continue
        p_mid = p + BUCKET / 2
        dist  = (p_mid - cp) / cp * 100
        side  = "LNG" if p_mid < cp else "SHT"
        col_t = "#f0f000" if usd / pmax > 0.65 else "#aaffcc"
        ax_r.text(usd / 1e6 + 0.05, p + BUCKET * 0.45,
                  f"${p_mid:,.0f}  ${usd/1e6:.2f}M  {dist:+.1f}%",
                  color=col_t, fontsize=6.0, va="center", ha="left")

    ax_r.axhline(cp, color="white", lw=1.0, ls="--", alpha=0.6)
    now_str = datetime.now().strftime("%d %b %Y, %H:%M")
    ax_r.text(0.98, 0.98,
              f"{now_str}\nPrice   ${cp:,.2f}\nBucket  ${BUCKET}\n{sources_used}",
              transform=ax_r.transAxes, color="white", fontsize=7.2,
              va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.5", fc="#111133", ec="#aaaacc", lw=0.8))

    ax_r.set_facecolor("#080114")
    ax_r.set_ylim(p_vis[0], p_vis[-1] + BUCKET)
    ax_r.set_yticks(list(yticks))
    ax_r.set_yticklabels([f"${y:,}" for y in yticks], color="#aaaacc", fontsize=7)
    ax_r.set_xlabel("Liquidation Leverage ($M) →", color="#aaaacc", fontsize=8)
    ax_r.set_title("Current Profile", color="#aaaacc", fontsize=8, pad=4)
    for sp in ax_r.spines.values(): sp.set_color("#222244")
    ax_r.tick_params(colors="#aaaacc")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax / 1e6))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("$M at risk", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

    lev_str = "  ".join(f"{l}×" for l in LEVERAGE_DIST)
    fig.suptitle(
        f"{SYMBOL} Liquidation Heatmap v2  "
        f"[{sources_used} · ${BUCKET} buckets · 30d model · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
        color="white", fontsize=12, fontweight="bold", x=0.48)
    ax.text(0.01, 0.012, f"Leverages modeled: {lev_str}",
            transform=ax.transAxes, color="#8888aa", fontsize=6.5)

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="#080114")
    print(f"Chart saved → {OUTPUT_PNG}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global SYMBOL, BUCKET, MIN_PRINT, OUTPUT_PNG, OUTPUT_JSON

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT",
                        choices=list(_SYMBOL_CFG.keys()),
                        help="Futures symbol to compute heatmap for")
    args = parser.parse_args()
    SYMBOL    = args.symbol
    cfg       = _SYMBOL_CFG[SYMBOL]
    BUCKET    = cfg["bucket"]
    MIN_PRINT = cfg["min_print"]
    OUTPUT_PNG  = os.path.join(_HERE, f"binance_liq_heatmap_{SYMBOL}.png")
    OUTPUT_JSON = os.path.join(_HERE, f"binance_liq_heatmap_{SYMBOL}.json")
    print(f"Heatmap v2  symbol={SYMBOL}  bucket=${BUCKET}  → {OUTPUT_JSON}")

    bn_oi, bn_kl, bn_ls, kl5 = fetch_binance()
    print(f"  Binance: {len(bn_oi)} OI bars  |  {len(kl5)} 5m bars")

    bb_oi = bb_kl = None
    try:
        bb_oi, bb_kl = fetch_bybit()
        print(f"  Bybit  : {len(bb_oi)} OI bars  |  {len(bb_kl)} kline entries")
        sources = "Binance + Bybit"
    except Exception as e:
        print(f"  Bybit fetch failed ({e}) — Binance only")
        sources = "Binance"

    print("Building heatmap …")
    heat, prices = build_heatmap(bn_oi, bn_kl, bn_ls, kl5, bb_oi, bb_kl)

    cp      = float(kl5[-1][4])
    prof    = heat[:, -1]
    print(f"  Price range  : ${prices[0]:,} – ${prices[-1]:,}")
    print(f"  Peak liq     : ${prof.max()/1e6:.2f} M")
    print(f"  Total modeled: ${prof.sum()/1e9:.3f} B")

    short_liq, long_liq = print_and_save(heat, prices, kl5, sources)

    print("Rendering chart …")
    plot(heat, prices, kl5, short_liq, long_liq, sources)


if __name__ == "__main__":
    main()
