"""
Binance BTC/USDT Liquidation Heatmap — CoinGlass-style
Uses 4h OI history (30 days) for the heat model so old positions are captured,
overlays a 24h 5-min candlestick chart, and prints a clear console table.
"""

import json
import os
import sys
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

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL              = "BTCUSDT"
_HERE               = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE         = os.path.join(_HERE, "binance_liq_heatmap.png")
OUTPUT_JSON         = os.path.join(_HERE, "binance_liq_heatmap.json")
BUCKET              = 100          # $100 price-bucket width
LIQUIDITY_THRESHOLD = 0.25         # hide heat below 25 % of max — removes noise

# Realistic leverage distribution (% of open interest at each level)
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

# ── Fetch ─────────────────────────────────────────────────────────────────────
def _get(url, params, timeout=12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_data():
    # ── Long-term heat model: 4h bars × 186 ≈ 31 days ───────────────────────
    print("Fetching 4h OI history (30 days) …")
    oi_4h = _get("https://fapi.binance.com/futures/data/openInterestHist",
                  {"symbol": SYMBOL, "period": "4h", "limit": 186})

    print("Fetching 4h klines (30 days) …")
    klines_4h = _get("https://fapi.binance.com/fapi/v1/klines",
                     {"symbol": SYMBOL, "interval": "4h", "limit": 187})

    print("Fetching 4h L/S ratio (30 days) …")
    ls_4h = _get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                  {"symbol": SYMBOL, "period": "4h", "limit": 186})

    # ── Short-term display chart: 5m bars × 288 = 24h ────────────────────────
    print("Fetching 5m klines (24h) …")
    klines_5m = _get("https://fapi.binance.com/fapi/v1/klines",
                     {"symbol": SYMBOL, "interval": "5m", "limit": 289})

    return oi_4h, klines_4h[:-1], ls_4h, klines_5m[:-1]

# ── Build heat matrix ─────────────────────────────────────────────────────────
def build_heatmap(oi_4h, klines_4h, ls_4h, klines_5m):
    N5  = len(klines_5m)    # columns in the display chart (288)
    ts0 = int(klines_5m[0][0])  # start timestamp of 24h display window (ms)

    # Index 4h OI and L/S by timestamp
    oi_map = {int(r["timestamp"]): float(r["sumOpenInterestValue"]) for r in oi_4h}
    ls_map = {int(r["timestamp"]): float(r["longAccount"])          for r in ls_4h}
    oi_ts  = sorted(oi_map)
    ls_ts  = sorted(ls_map)

    # Index 4h klines by open timestamp for price look-up
    kl4_map = {int(k[0]): k for k in klines_4h}
    kl4_ts  = sorted(kl4_map)

    def nearest(ts_list, t):
        return min(ts_list, key=lambda x: abs(x - t))

    # Price range: use 4h klines (30 days) + 12 % padding for liq spread
    all_h = [float(k[2]) for k in klines_4h]
    all_l = [float(k[3]) for k in klines_4h]
    mid   = (max(all_h) + min(all_l)) / 2
    pad   = mid * 0.15
    p_lo  = int((min(all_l) - pad) // BUCKET) * BUCKET
    p_hi  = int((max(all_h) + pad) // BUCKET) * BUCKET + BUCKET
    prices = list(range(p_lo, p_hi + BUCKET, BUCKET))
    P      = len(prices)
    p2i    = {p: i for i, p in enumerate(prices)}

    # Build a lookup: 5m bar index by timestamp
    ts5_list = [int(k[0]) for k in klines_5m]

    def col_for_ts(ts_ms):
        """Return the 5m column index for a given timestamp.
           Returns 0 if the timestamp is before the 24h window (old positions
           are still open → bar starts at left edge)."""
        if ts_ms <= ts0:
            return 0
        # find first 5m bar that starts at or after ts_ms
        for j, t5 in enumerate(ts5_list):
            if t5 >= ts_ms:
                return j
        return N5 - 1

    # heat[price_idx, time_col] — horizontal-bar accumulation
    heat = np.zeros((P, N5), dtype=np.float64)

    current_oi_total = oi_map[oi_ts[-1]]   # for calibration

    for i in range(1, len(oi_ts)):
        t_curr = oi_ts[i]
        t_prev = oi_ts[i - 1]
        oi_curr = oi_map[t_curr]
        oi_prev = oi_map[t_prev]
        delta_usd = oi_curr - oi_prev

        # ── OI increased → new positions opened ──────────────────────────────
        if delta_usd >= 500:
            kts_near  = nearest(kl4_ts, t_curr)
            kl        = kl4_map[kts_near]
            entry     = (float(kl[2]) + float(kl[3])) / 2   # high+low midpoint
            ls_near   = nearest(ls_ts, t_curr)
            long_pct  = ls_map[ls_near]
            short_pct = 1.0 - long_pct
            col       = col_for_ts(t_curr)   # which 5m column this starts at

            for lev, wt in LEVERAGE_DIST.items():
                ll = int((entry * (1 - 1 / lev)) // BUCKET) * BUCKET
                sl = int((entry * (1 + 1 / lev)) // BUCKET) * BUCKET
                if ll in p2i:
                    heat[p2i[ll], col:] += delta_usd * long_pct  * wt
                if sl in p2i:
                    heat[p2i[sl], col:] += delta_usd * short_pct * wt

        # ── OI decreased → positions closing (scale down proportionally) ─────
        elif delta_usd < -500 and oi_prev > 0:
            close_frac = min(abs(delta_usd) / oi_prev, 0.95)
            col = col_for_ts(t_curr)
            heat[:, col:] *= (1.0 - close_frac)

    # ── Calibration: scale so the profile sum ≈ current total OI × typical
    #    liquidation reach (positions within ±15 % of price, across all levs)
    tracked = heat[:, -1].sum()
    if tracked > 0:
        # CoinGlass accounts for more exchanges; Binance share ≈ 60 % of global
        # Our model covers positive-delta OI; scale to full current OI
        scale = (current_oi_total * 0.55) / tracked
        heat  *= min(scale, 4.0)   # cap at 4× to avoid wild extrapolation

    # ── Sweep clearing: zero any level that price has already traded through ──
    # When price sweeps a bucket, all positions liquidated there are gone.
    # Find the first 5m candle whose range covers each price bucket and zero
    # heat from that column onward — so already-swept levels don't stay hot.
    for pi, p in enumerate(prices):
        mid = p + BUCKET * 0.5   # bucket midpoint
        for j, k in enumerate(klines_5m):
            lo, hi = float(k[3]), float(k[2])
            if lo <= mid <= hi:
                heat[pi, j:] = 0
                break   # first sweep clears; later sweeps find zeros anyway

    return heat, prices

# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(heat, prices, klines_5m):
    cp      = float(klines_5m[-1][4])
    ts_str  = datetime.fromtimestamp(int(klines_5m[-1][0]) / 1000).strftime("%d %b %Y, %H:%M")
    profile = heat[:, -1]
    total   = profile.sum()

    W = 72
    print("\n" + "=" * W)
    print(f"  Binance BTC/USDT  Liquidation Heatmap  [{ts_str}]")
    print(f"  Current Price : ${cp:,.2f}")
    print(f"  Total modeled : ${total / 1e9:.2f} B   (30-day OI history, calibrated)")
    print("=" * W)

    # Separate into above (short liq) and below (long liq) current price
    above = sorted([(prices[i], profile[i]) for i in range(len(prices))
                    if prices[i] > cp and profile[i] > 1e5],
                   key=lambda x: x[0])   # ascending by price
    below = sorted([(prices[i], profile[i]) for i in range(len(prices))
                    if prices[i] <= cp and profile[i] > 1e5],
                   key=lambda x: -x[0])  # descending by price (nearest first)

    peak = max(profile.max(), 1.0)
    BAR  = 20

    def _bar(usd):
        n = int(usd / peak * BAR)
        return "#" * n + "." * (BAR - n)

    # ── SHORT LIQ ABOVE (nearest 10 first) ───────────────────────────────────
    print(f"\n  SHORT LIQUIDATIONS (UP)  above ${cp:,.0f} -- nearest first")
    print(f"  {'#':<4}  {'Price':>12}  {'Amount':>12}  {'Dist':>7}  Bar")
    print(f"  {'-'*60}")
    for rank, (p, usd) in enumerate(above[:10], 1):
        dist = (p - cp) / cp * 100
        print(f"  {rank:<4}  ${p:>10,}  ${usd/1e6:>9.2f} M  {dist:>+6.1f}%  {_bar(usd)}")

    # ── LONG LIQ BELOW (nearest 10 first) ────────────────────────────────────
    print(f"\n  LONG LIQUIDATIONS (DOWN)  below ${cp:,.0f} -- nearest first")
    print(f"  {'#':<4}  {'Price':>12}  {'Amount':>12}  {'Dist':>7}  Bar")
    print(f"  {'-'*60}")
    for rank, (p, usd) in enumerate(below[:10], 1):
        dist = (p - cp) / cp * 100
        print(f"  {rank:<4}  ${p:>10,}  ${usd/1e6:>9.2f} M  {dist:>+6.1f}%  {_bar(usd)}")

    cp_bkt = int(cp // BUCKET) * BUCKET
    cp_idx = next((i for i, p in enumerate(prices) if p == cp_bkt), None)
    cp_usd = profile[cp_idx] if cp_idx is not None else 0
    print(f"\n  AT CURRENT PRICE ${cp_bkt:,} – ${cp_bkt + BUCKET:,}")
    print(f"  Liquidation Leverage : ${cp_usd / 1e6:.2f} M")
    print("=" * W + "\n")

    # ── Save JSON for liq_algo.py ─────────────────────────────────────────────
    data = {
        "generated_at": datetime.now().isoformat(),
        "current_price": cp,
        "bucket_usd": BUCKET,
        "short_liquidations": [
            {"price": float(p), "usd": round(usd, 0),
             "dist_pct": round((p - cp) / cp * 100, 3)}
            for p, usd in above
        ],
        "long_liquidations": [
            {"price": float(p), "usd": round(usd, 0),
             "dist_pct": round((p - cp) / cp * 100, 3)}
            for p, usd in below
        ],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Data saved -> {OUTPUT_JSON}")

# ── Colormap ──────────────────────────────────────────────────────────────────
def make_cmap():
    return mcolors.LinearSegmentedColormap.from_list("cg", [
        (0.00, "#0a0118"), (0.18, "#1f0a5c"), (0.35, "#1a3a8a"),
        (0.50, "#0a6e7b"), (0.64, "#128a50"), (0.76, "#5ab52a"),
        (0.87, "#b5cc1a"), (0.94, "#d4e100"), (1.00, "#f0f000"),
    ])

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(heat, prices, klines_5m):
    cmap  = make_cmap()
    N5    = heat.shape[1]
    P     = len(prices)
    BS    = BUCKET
    cp    = float(klines_5m[-1][4])
    ts5   = [int(k[0]) for k in klines_5m]

    # Threshold + clip
    h    = heat.copy()
    vmax = np.percentile(h[h > 0], 99.5) if (h > 0).any() else 1.0
    h[h < vmax * LIQUIDITY_THRESHOLD] = 0
    h    = np.clip(h, 0, vmax)

    # Visible price window ±9 %
    row_lo = max(0, int((cp * 0.91 - prices[0]) // BS))
    row_hi = min(P, int((cp * 1.09 - prices[0]) // BS) + 2)
    h_vis  = h[row_lo:row_hi, :]
    p_vis  = prices[row_lo:row_hi]

    fig   = plt.figure(figsize=(22, 11), facecolor="#080114")
    ax    = fig.add_axes([0.05, 0.09, 0.65, 0.84])
    ax_r  = fig.add_axes([0.72, 0.09, 0.21, 0.84])
    ax_cb = fig.add_axes([0.945, 0.09, 0.013, 0.84])

    # Heatmap
    ax.imshow(h_vis, aspect="auto", cmap=cmap, origin="lower",
              extent=[0, N5, p_vis[0], p_vis[-1] + BS],
              interpolation="nearest", vmin=0, vmax=vmax)

    # 5m Candlesticks
    for i, k in enumerate(klines_5m):
        o, hi, lo, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i + .5, i + .5], [lo, hi], color=col, lw=0.5, alpha=0.9, zorder=5)
        b_lo, b_hi = min(o, c), max(o, c)
        b_hi = max(b_hi, b_lo + BS * 0.12)
        ax.add_patch(mpatches.Rectangle((i + .08, b_lo), .84, b_hi - b_lo,
                                         color=col, alpha=0.95, zorder=6))

    # Current price line
    ax.axhline(cp, color="#ffffff", lw=1.0, ls="--", alpha=0.7, zorder=7)
    ax.text(N5 + .3, cp, f"${cp:,.0f}", color="white", fontsize=8,
            va="center", ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="#222244", ec="white", lw=0.6))

    # Annotate top 6 liq zones on the chart
    profile_vis = h_vis[:, -1]
    order_vis   = np.argsort(profile_vis)[::-1]
    labeled = 0
    for idx in order_vis:
        if labeled >= 6: break
        p, usd = p_vis[idx], profile_vis[idx]
        if usd < 5e5: break
        dist = (p - cp) / cp * 100
        side = "LNG" if p < cp else "SHT"
        ax.axhline(p + BS / 2, color="#ffff00", lw=0.4, ls=":", alpha=0.35, zorder=4)
        ax.text(N5 - 2, p + BS * 0.55,
                f"${usd / 1e6:.1f}M  {side} {dist:+.1f}%",
                color="#f0f000", fontsize=6.5, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.15", fc="#0a0118",
                          ec="#555500", lw=0.5, alpha=0.85), zorder=9)
        labeled += 1

    # Axes
    y_step = max(BS, round((p_vis[-1] - p_vis[0]) / 8 / BS) * BS)
    yticks = range(int(p_vis[0] // y_step) * y_step, p_vis[-1] + y_step, y_step)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"${y:,}" for y in yticks], color="#aaaacc", fontsize=8)
    step = max(1, N5 // 12)
    xticks = list(range(0, N5, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [datetime.fromtimestamp(ts5[i] / 1000).strftime("%d, %H:%M") for i in xticks],
        color="#aaaacc", fontsize=7, rotation=25, ha="right")
    ax.set_xlim(0, N5)
    ax.set_ylim(p_vis[0], p_vis[-1] + BS)
    ax.set_facecolor("#080114")
    for sp in ax.spines.values(): sp.set_color("#222244")
    ax.tick_params(colors="#aaaacc")

    # Right panel — labeled bars
    prof_full = heat[:, -1]
    pmax_full = prof_full.max() if prof_full.max() > 0 else 1.0
    prof_vis2 = prof_full[row_lo:row_hi]
    bar_cols  = [cmap(v / pmax_full) for v in prof_vis2]
    ax_r.barh(p_vis, prof_vis2 / 1e6, height=BS * 0.88, color=bar_cols, alpha=0.92)

    # Label bars ≥ $2M
    for p, usd in zip(p_vis, prof_vis2):
        if usd < 2e6: continue
        col  = "#f0f000" if usd / pmax_full > 0.65 else "#aaffcc"
        dist = (p - cp) / cp * 100
        side = "LNG" if p < cp else "SHT"
        ax_r.text(usd / 1e6 + 0.3, p + BS * 0.45,
                  f"${usd / 1e6:.1f}M  {dist:+.1f}%",
                  color=col, fontsize=6.5, va="center", ha="left")

    ax_r.axhline(cp, color="white", lw=1.0, ls="--", alpha=0.6)

    # Tooltip box
    cp_bkt = int(cp // BS) * BS
    cp_idx = next((i for i, p in enumerate(prices) if p == cp_bkt), None)
    cp_usd = prof_full[cp_idx] if cp_idx is not None else 0
    now_str = datetime.now().strftime("%d %b %Y, %H:%M")
    tip = f"{now_str}\nPrice          ${cp:,.2f}\nLiq Leverage  ${cp_usd / 1e6:.2f} M"
    ax_r.text(0.98, 0.98, tip, transform=ax_r.transAxes, color="white", fontsize=7.5,
              va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.5", fc="#111133", ec="#aaaacc", lw=0.8))

    ax_r.set_facecolor("#080114")
    ax_r.set_ylim(p_vis[0], p_vis[-1] + BS)
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

    fig.suptitle(
        f"Binance BTC/USDT Liquidation Heatmap  "
        f"[30-day model · 24h chart · {datetime.now().strftime('%Y-%m-%d %H:%M')}]",
        color="white", fontsize=13, fontweight="bold", x=0.47)
    lev_str = "  ".join(f"{l}×" for l in LEVERAGE_DIST)
    ax.text(0.01, 0.012, f"Leverages: {lev_str}",
            transform=ax.transAxes, color="#8888aa", fontsize=6.5)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="#080114")
    print(f"Chart saved → {OUTPUT_FILE}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    oi_4h, klines_4h, ls_4h, klines_5m = fetch_data()
    print(f"  4h bars: {len(oi_4h)}  |  5m bars: {len(klines_5m)}")

    heat, prices = build_heatmap(oi_4h, klines_4h, ls_4h, klines_5m)

    cp      = float(klines_5m[-1][4])
    profile = heat[:, -1]
    cp_bkt  = int(cp // BUCKET) * BUCKET
    cp_idx  = next((i for i, p in enumerate(prices) if p == cp_bkt), None)
    cp_usd  = profile[cp_idx] if cp_idx is not None else 0
    print(f"  Price range modeled : ${prices[0]:,} – ${prices[-1]:,}")
    print(f"  Peak liq level      : ${profile.max() / 1e6:.1f} M")
    print(f"  At current price    : ${cp_usd / 1e6:.2f} M")

    print_summary(heat, prices, klines_5m)
    print("Rendering chart …")
    plot(heat, prices, klines_5m)

if __name__ == "__main__":
    main()
