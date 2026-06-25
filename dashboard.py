"""
dashboard.py  —  LSCO Bloomberg Terminal
Three assets simultaneously. No tabs. Run: python dashboard.py → http://localhost:8050

Refresh loops:
  1.5s  — per-asset price/state, header ticker, live P&L, positions
  5s    — per-asset data strips (funding/OI/whale/zones), account, session stats
  20s   — per-asset charts (candles + liq zones + OB walls), trade log
"""

import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "asterdex_trade"))

from market_data import get_klines, get_price, get_ob

try:
    from account_data import get_open_orders, get_position_risk, get_balances
    LIVE = True
except Exception:
    LIVE = False

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOLS        = ["BTCUSDT", "ETHUSDT", "XAUUSDT"]
MIN_WALL_USD   = 1_000_000
WALL_RANGE_PCT = 1.5
TRADE_LOG      = ROOT / "trade_log.json"

SYM_ID  = {"BTCUSDT": "btc", "ETHUSDT": "eth", "XAUUSDT": "xau"}
SYM_CLR = {"BTCUSDT": "#ffc000", "ETHUSDT": "#00b8ff", "XAUUSDT": "#ff6820"}
SYM_LBL = {"BTCUSDT": "BTC/USDT ×20", "ETHUSDT": "ETH/USDT ×20", "XAUUSDT": "XAU/USDT ×10"}
SYM_UNIT = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "XAUUSDT": "oz"}

def heatmap_json(sym): return ROOT / "data_fetching" / f"binance_liq_heatmap_{sym}.json"
def whale_json(sym):   return ROOT / f"whale_{sym}.json"

def algo_state_file(sym):
    candidates = [
        ROOT / f"algo_state_v2_{sym}.json",
        ROOT / f"algo_state_v1_{sym}.json",
        ROOT / "algo_state_v2.json",
        ROOT / "algo_state.json",
    ]
    best, best_t = None, 0
    for p in candidates:
        if p.exists():
            t = p.stat().st_mtime
            if t > best_t:
                best, best_t = p, t
    return best

# ── Colour palette ─────────────────────────────────────────────────────────────
BG     = "#010810"
BG2    = "#030e1a"
BG3    = "#061420"
BORDER = "#0b2035"
SEP    = "#102840"
TEXT   = "#b8ccdc"
DIM    = "#1e3a52"
MUTED  = "#2c4e68"
LABEL  = "#4a7a9a"
GREEN  = "#00e888"
RED    = "#ff2040"
YELLOW = "#ffc000"
CYAN   = "#00b8ff"
ORANGE = "#ff6820"
PURPLE = "#b060ff"
WHITE  = "#e0eef8"
FONT   = "'Courier New', Consolas, 'Lucida Console', monospace"

TV_BG   = "#010d18"
TV_GRID = "rgba(11,32,53,0.9)"


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _live(fn, *args, default=None):
    if not LIVE:
        return default
    try:
        return fn(*args)
    except Exception:
        return default

def fetch_market_info(symbol):
    info = dict(mark=0., funding=0., next_fund="", vol24=0.,
                oi_unit=0., oi_usd=0., high24=0., low24=0., chg24=0.,
                bid=0., ask=0., spread=0.)
    try:
        d = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": symbol}, timeout=4).json()
        info["mark"]    = float(d.get("markPrice", 0))
        info["funding"] = float(d.get("lastFundingRate", 0)) * 100
        nft = d.get("nextFundingTime", 0)
        if nft:
            info["next_fund"] = datetime.fromtimestamp(int(nft)/1000).strftime("%H:%M")
    except Exception:
        pass
    try:
        d2 = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                          params={"symbol": symbol}, timeout=4).json()
        info["vol24"]  = float(d2.get("quoteVolume", 0))
        info["high24"] = float(d2.get("highPrice", 0))
        info["low24"]  = float(d2.get("lowPrice", 0))
        info["chg24"]  = float(d2.get("priceChangePercent", 0))
    except Exception:
        pass
    try:
        d3 = requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                          params={"symbol": symbol}, timeout=4).json()
        info["oi_unit"] = float(d3.get("openInterest", 0))
        info["oi_usd"]  = info["oi_unit"] * (info["mark"] or 1)
    except Exception:
        pass
    try:
        ob = get_ob(symbol)
        info["bid"]    = ob.get("best_bid", 0)
        info["ask"]    = ob.get("best_ask", 0)
        info["spread"] = round(info["ask"] - info["bid"], 2)
    except Exception:
        pass
    return info

def fetch_ob_walls(current_price, symbol):
    walls = []
    for label, url in [("SPOT", "https://api.binance.com/api/v3/depth"),
                       ("PERP", "https://fapi.binance.com/fapi/v1/depth")]:
        try:
            data = requests.get(url, params={"symbol": symbol, "limit": 500}, timeout=5).json()
            for side_key, side_name in [("bids","BID"),("asks","ASK")]:
                for row in data.get(side_key, []):
                    p, q = float(row[0]), float(row[1])
                    usd = p * q
                    if usd >= MIN_WALL_USD:
                        walls.append(dict(src=label, price=p, qty=q, usd=usd, side=side_name,
                                          dist=(p-current_price)/current_price*100))
        except Exception:
            pass
    walls.sort(key=lambda w: -w["usd"])
    return walls[:40]

def fmt_usd(v):
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:.2f}"

def fmt_pnl(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:.4f}" if abs(v) < 0.005 else f"{sign}${v:.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kv(label, value, vc=TEXT, lw="96px", size="10px"):
    return html.Div([
        html.Span(label, style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                 "display": "inline-block", "minWidth": lw}),
        html.Span(value, style={"color": vc, "fontSize": size, "fontFamily": FONT}),
    ], style={"marginBottom": "2px", "lineHeight": "1.4"})

def sep():
    return html.Hr(style={"border": "none", "borderTop": f"1px solid {SEP}", "margin": "3px 0"})

def panel_wrap(title, body_id, min_h="50px"):
    return html.Div([
        html.Div(title, style={"fontSize": "9px", "fontWeight": "bold", "color": CYAN,
                                "letterSpacing": "2px", "padding": "3px 8px",
                                "background": BG3, "borderBottom": f"1px solid {SEP}",
                                "fontFamily": FONT}),
        html.Div(id=body_id, style={"padding": "5px 8px", "minHeight": min_h}),
    ], style={"background": BG2, "border": f"1px solid {BORDER}",
               "borderTop": f"2px solid {SEP}"})

def build_chart(symbol, price, klines, algo, hm, whale):
    fig = go.Figure()
    if not klines:
        fig.update_layout(paper_bgcolor=TV_BG, plot_bgcolor=TV_BG,
                          margin=dict(l=0,r=60,t=2,b=2),
                          xaxis=dict(showgrid=False, color=LABEL),
                          yaxis=dict(showgrid=False, color=LABEL, side="right"))
        return fig

    times  = [datetime.fromtimestamp(int(k[0])/1000) for k in klines]
    opens  = [float(k[1]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    vols   = [float(k[5]) for k in klines]

    short_zones = sorted(hm["short_liquidations"], key=lambda z: z["price"])[:20] if hm else []
    long_zones  = sorted(hm["long_liquidations"],  key=lambda z:-z["price"])[:20] if hm else []

    t0, t1 = times[0], times[-1]
    all_usd = [z["usd"] for z in short_zones + long_zones]
    max_usd = max(all_usd) if all_usd else 1e6

    for z in short_zones:
        s = 0.12 + 0.65 * min(z["usd"]/max_usd, 1.0)
        fig.add_shape(type="line", layer="below",
            x0=t0, x1=t1, y0=z["price"], y1=z["price"],
            line=dict(color=f"rgba(255,30,60,{s:.2f})", width=1))
        if z["usd"] >= 30_000_000:
            fig.add_annotation(x=t1, y=z["price"],
                text=f"  ${z['usd']/1e6:.0f}M S", showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=7, color="rgba(255,120,140,0.9)", family=FONT))

    for z in long_zones:
        s = 0.12 + 0.65 * min(z["usd"]/max_usd, 1.0)
        fig.add_shape(type="line", layer="below",
            x0=t0, x1=t1, y0=z["price"], y1=z["price"],
            line=dict(color=f"rgba(0,220,100,{s:.2f})", width=1))
        if z["usd"] >= 30_000_000:
            fig.add_annotation(x=t1, y=z["price"],
                text=f"  ${z['usd']/1e6:.0f}M L", showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=7, color="rgba(80,255,160,0.9)", family=FONT))

    vol_colors = ["rgba(0,200,120,0.3)" if closes[i]>=opens[i] else "rgba(255,40,60,0.3)"
                  for i in range(len(times))]
    fig.add_trace(go.Bar(x=times, y=vols, marker_color=vol_colors,
                         showlegend=False, yaxis="y2",
                         hovertemplate="%{x|%H:%M}  Vol:%{y:.2f}<extra></extra>"))

    if whale:
        wts = whale.get("candle_ts", 0)
        wb  = whale.get("whale_buy_usd", 0)
        ws  = whale.get("whale_sell_usd", 0)
        if wts and (wb+ws) > 0:
            ws_ = datetime.fromtimestamp(wts)
            we_ = datetime.fromtimestamp(wts+60)
            bull = wb >= ws
            fig.add_vrect(x0=ws_, x1=we_,
                fillcolor="rgba(0,220,100,0.06)" if bull else "rgba(255,40,60,0.06)",
                line=dict(color="rgba(0,220,100,0.35)" if bull else "rgba(255,40,60,0.35)",
                          width=1), layer="below")

    fig.add_trace(go.Candlestick(
        x=times, open=opens, high=highs, low=lows, close=closes,
        increasing=dict(line=dict(color="#00c878",width=1), fillcolor="#00c878"),
        decreasing=dict(line=dict(color="#ff2040",width=1), fillcolor="#ff2040"),
        showlegend=False,
        hovertext=[f"{t.strftime('%H:%M')}  O:{o:,.1f}  H:{h:,.1f}  L:{l:,.1f}  C:{c:,.1f}"
                   for t,o,h,l,c in zip(times,opens,highs,lows,closes)],
    ))

    tlog = load_json(TRADE_LOG)
    if tlog:
        for tr in tlog[-40:]:
            if tr.get("symbol", symbol) != symbol:
                continue
            try:
                td = datetime.fromisoformat(tr["time"])
                ci = min(range(len(times)), key=lambda i: abs((times[i]-td).total_seconds()))
                if abs((times[ci]-td).total_seconds()) > 600:
                    continue
                is_win   = tr["result"] == "WIN"
                is_short = tr["direction"] == "SHORT"
                mc  = "#00e888" if is_win else "#ff2040"
                my  = lows[ci]*0.9994 if not is_short else highs[ci]*1.0006
                sym = "triangle-up" if not is_short else "triangle-down"
                fig.add_trace(go.Scatter(
                    x=[times[ci]], y=[my], mode="markers+text",
                    marker=dict(symbol=sym, size=9, color=mc,
                                line=dict(width=1,color="#010810")),
                    text=[f"{'W' if is_win else 'L'} {fmt_pnl(tr['pnl_usd'])}"],
                    textfont=dict(size=7, color=mc, family=FONT),
                    textposition="bottom center" if not is_short else "top center",
                    showlegend=False,
                    hovertemplate=(f"{tr['direction']} {tr['result']} "
                                   f"{fmt_pnl(tr['pnl_usd'])}<extra></extra>"),
                ))
            except Exception:
                pass

    if price:
        fig.add_hline(y=price,
            line=dict(color="rgba(200,215,230,0.4)", width=1, dash="dash"),
            annotation_text=f"  ${price:,.2f}",
            annotation_position="right",
            annotation_font=dict(size=8, color=WHITE, family=FONT))

    if algo and algo.get("state") == "IN_TRADE":
        tp    = algo.get("tp",0)
        sl    = algo.get("sl",0)
        ep    = algo.get("entry",0)
        trail = algo.get("trail_stage","none")
        if tp:
            fig.add_hline(y=tp, line=dict(color="rgba(0,232,136,0.8)",width=1.5,dash="dash"),
                annotation_text=f"  TP ${tp:,.2f}", annotation_position="right",
                annotation_font=dict(size=8,color=GREEN,family=FONT))
        if sl:
            slc = {"none":"rgba(255,32,64,0.8)","breakeven":"rgba(255,192,0,0.8)",
                   "trailing":"rgba(255,104,32,0.8)"}.get(trail,"rgba(255,32,64,0.8)")
            slcol = {"none":RED,"breakeven":YELLOW,"trailing":ORANGE}.get(trail,RED)
            sllbl = {"none":"SL","breakeven":"BE","trailing":"TRAIL"}.get(trail,"SL")
            fig.add_hline(y=sl, line=dict(color=slc,width=1.5,dash="dash"),
                annotation_text=f"  {sllbl} ${sl:,.2f}", annotation_position="right",
                annotation_font=dict(size=8,color=slcol,family=FONT))
        if ep:
            fig.add_hline(y=ep, line=dict(color="rgba(255,192,0,0.5)",width=1,dash="dot"),
                annotation_text=f"  ENTRY ${ep:,.2f}", annotation_position="right",
                annotation_font=dict(size=8,color=YELLOW,family=FONT))

    if price:
        ob_walls = fetch_ob_walls(price, symbol)
        for w in [x for x in ob_walls if abs(x["dist"]) <= WALL_RANGE_PCT]:
            is_bid = w["side"] == "BID"
            wc = "rgba(0,200,255,0.45)" if is_bid else "rgba(200,80,255,0.45)"
            wf = "#00c8ff" if is_bid else "#c050ff"
            fig.add_hline(y=w["price"], line=dict(color=wc,width=1,dash="dashdot"),
                annotation_text=f"  {w['side']} {w['src']} ${w['usd']/1e6:.1f}M",
                annotation_position="right",
                annotation_font=dict(size=7,color=wf,family=FONT))

    fig.update_layout(
        paper_bgcolor=TV_BG, plot_bgcolor=TV_BG,
        margin=dict(l=0, r=90, t=2, b=2),
        xaxis=dict(showgrid=True, gridcolor=TV_GRID, color=LABEL,
                   tickformat="%H:%M", rangeslider=dict(visible=False),
                   showline=False, zeroline=False,
                   tickfont=dict(size=8, family=FONT)),
        yaxis=dict(showgrid=True, gridcolor=TV_GRID, color=LABEL,
                   tickformat="$,.0f", side="right",
                   showline=False, zeroline=False,
                   tickfont=dict(size=8, family=FONT)),
        yaxis2=dict(overlaying="y", side="left",
                    range=[0, max(vols)*5] if vols else [0,1],
                    showgrid=False, showticklabels=False, zeroline=False),
        font=dict(family=FONT, size=8, color=TEXT),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#020d18", font_size=9, bordercolor=BORDER, font_family=FONT),
        bargap=0,
    )
    return fig

def build_strip(symbol, mi, algo, price):
    hm  = load_json(heatmap_json(symbol))
    whale = load_json(whale_json(symbol))
    clr = SYM_CLR[symbol]

    fund_col  = RED if mi["funding"] > 0 else GREEN
    unit_lbl  = SYM_UNIT[symbol]

    # Nearest zones
    above_txt, below_txt = "—", "—"
    if hm and price:
        sz = sorted(hm.get("short_liquidations",[]), key=lambda z: z["price"])
        lz = sorted(hm.get("long_liquidations",[]),  key=lambda z:-z["price"])
        above = [z for z in sz if z["price"] > price]
        below = [z for z in lz if z["price"] < price]
        if above:
            z = above[0]
            above_txt = f"${z['price']:,.1f}  {fmt_usd(z['usd'])}  (+{(z['price']-price)/price*100:.2f}%)"
        if below:
            z = below[0]
            below_txt = f"${z['price']:,.1f}  {fmt_usd(z['usd'])}  (-{(price-z['price'])/price*100:.2f}%)"

    # Whale summary
    whale_line = html.Span("OFFLINE", style={"color": RED, "fontSize": "9px", "fontFamily": FONT})
    if whale:
        b = whale.get("whale_buy_usd",0)
        s = whale.get("whale_sell_usd",0)
        d = b - s
        dc = GREEN if d >= 0 else RED
        age_s = int(time.time() - whale.get("candle_ts",0))
        albl = f"{age_s//60}m{age_s%60}s" if age_s < 3600 else "stale"
        stale = age_s > 120
        whale_line = html.Span([
            html.Span(f"BUY ${b/1e6:.2f}M", style={"color": GREEN, "fontSize": "9px", "fontFamily": FONT}),
            html.Span(f"  SELL ${s/1e6:.2f}M", style={"color": RED, "fontSize": "9px", "fontFamily": FONT}),
            html.Span(f"  NET ${d/1e6:+.2f}M", style={"color": dc, "fontSize": "9px", "fontFamily": FONT}),
            html.Span(f"  ({albl})", style={"color": RED if stale else MUTED,
                                              "fontSize": "9px", "fontFamily": FONT}),
        ])

    # Engine state
    state = "WATCHING"
    conf  = 0
    if algo:
        state = algo.get("state","WATCHING")
        conf  = algo.get("confidence",0)

    state_col = {"WATCHING": MUTED, "APPROACHING": YELLOW, "IN_TRADE": GREEN}.get(state, MUTED)

    rows = [
        html.Div([
            html.Span("●", style={"color": state_col, "fontFamily": FONT, "fontSize": "9px",
                                   "marginRight": "4px"}),
            html.Span(state, style={"color": state_col, "fontFamily": FONT, "fontSize": "9px",
                                     "fontWeight": "bold", "letterSpacing": "1px",
                                     "marginRight": "10px"}),
            html.Span(f"CONF {conf}/100", style={"color": clr, "fontFamily": FONT,
                                                   "fontSize": "9px"}),
        ], style={"marginBottom": "3px"}),
        html.Div([
            html.Span("MARK  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                        "minWidth": "52px", "display": "inline-block"}),
            html.Span(f"${mi['mark']:,.2f}" if mi["mark"] else "—",
                      style={"color": WHITE, "fontSize": "10px", "fontFamily": FONT,
                              "fontWeight": "bold", "marginRight": "14px"}),
            html.Span("FUND  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                        "minWidth": "52px", "display": "inline-block"}),
            html.Span(f"{mi['funding']:+.4f}%  @{mi['next_fund']}" if mi["next_fund"] else
                      (f"{mi['funding']:+.4f}%" if mi["funding"] else "—"),
                      style={"color": fund_col, "fontSize": "9px", "fontFamily": FONT,
                              "marginRight": "14px"}),
            html.Span("OI  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                      "minWidth": "36px", "display": "inline-block"}),
            html.Span(f"{mi['oi_unit']:,.0f} {unit_lbl}  ({fmt_usd(mi['oi_usd'])})"
                      if mi["oi_unit"] else "—",
                      style={"color": PURPLE, "fontSize": "9px", "fontFamily": FONT,
                              "marginRight": "14px"}),
            html.Span("SPREAD  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                          "minWidth": "60px", "display": "inline-block"}),
            html.Span(f"${mi['spread']:.2f}" if mi["spread"] else "—",
                      style={"color": YELLOW if mi["spread"]>1 else GREEN,
                              "fontSize": "9px", "fontFamily": FONT}),
        ], style={"marginBottom": "2px"}),
        html.Div([
            html.Span("ZONE↑  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                         "minWidth": "52px", "display": "inline-block"}),
            html.Span(above_txt, style={"color": RED, "fontSize": "9px", "fontFamily": FONT,
                                         "marginRight": "14px"}),
            html.Span("ZONE↓  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                         "minWidth": "52px", "display": "inline-block"}),
            html.Span(below_txt, style={"color": GREEN, "fontSize": "9px", "fontFamily": FONT}),
        ], style={"marginBottom": "2px"}),
        html.Div([
            html.Span("WHALE  ", style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                         "minWidth": "52px", "display": "inline-block"}),
            whale_line,
        ]),
    ]
    return html.Div(rows)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def asset_col(sym):
    sid = SYM_ID[sym]
    clr = SYM_CLR[sym]
    lbl = SYM_LBL[sym]
    return html.Div([
        html.Div([
            html.Span(lbl, style={"color": clr, "fontSize": "10px", "fontWeight": "bold",
                                   "fontFamily": FONT, "letterSpacing": "1.5px"}),
            html.Span(id=f"{sid}-price", style={"fontSize": "18px", "fontWeight": "bold",
                                                  "fontFamily": FONT, "color": WHITE,
                                                  "marginLeft": "10px"}),
            html.Span(id=f"{sid}-chg",   style={"fontSize": "10px", "marginLeft": "6px",
                                                  "fontFamily": FONT, "color": MUTED}),
            html.Span(id=f"{sid}-state", style={"fontSize": "9px", "fontFamily": FONT,
                                                  "marginLeft": "auto", "padding": "1px 6px",
                                                  "border": f"1px solid {clr}", "color": clr}),
        ], style={"display": "flex", "alignItems": "center", "padding": "5px 8px",
                   "background": BG3, "borderBottom": f"1px solid {BORDER}",
                   "gap": "2px"}),

        dcc.Graph(id=f"{sid}-chart", style={"height": "300px"},
                  config={"displayModeBar": False, "scrollZoom": True}),

        html.Div(id=f"{sid}-strip", style={"padding": "5px 8px", "background": BG,
                                             "borderTop": f"1px solid {DIM}",
                                             "minHeight": "72px"}),
    ], style={
        "flex": "1", "minWidth": 0,
        "background": BG2, "border": f"1px solid {BORDER}",
        "display": "flex", "flexDirection": "column",
    })


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="LSCO Terminal")

app.index_string = """<!DOCTYPE html><html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #010810; color: #b8ccdc;
           font-family: 'Courier New', Consolas, monospace; overflow-x: hidden; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #010810; }
    ::-webkit-scrollbar-thumb { background: #0b2035; border-radius: 0; }
    .blink { animation: blink 1.2s step-end infinite; }
    @keyframes blink { 50% { opacity: 0.3; } }
  </style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""

app.layout = html.Div([

    dcc.Interval(id="iv-fast", interval=1500,  n_intervals=0),
    dcc.Interval(id="iv-med",  interval=5000,  n_intervals=0),
    dcc.Interval(id="iv-slow", interval=20000, n_intervals=0),

    # ── Header ─────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("◈ ", style={"color": CYAN, "fontSize": "14px",
                                    "className": "blink"}),
            html.Span("LSCO v2", style={"color": WHITE, "fontSize": "12px",
                                         "fontWeight": "bold", "letterSpacing": "2px"}),
            html.Span("  MULTI-ASSET PERP", style={"color": MUTED, "fontSize": "9px",
                                                     "marginLeft": "6px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "0"}),

        html.Div(id="hdr-ticker", style={"display": "flex", "alignItems": "center",
                                          "gap": "18px", "flex": "1",
                                          "justifyContent": "center"}),

        html.Div([
            html.Span(id="hdr-pnl", style={"fontSize": "12px", "fontWeight": "bold",
                                            "fontFamily": FONT, "marginRight": "16px"}),
            html.Span(id="hdr-time", style={"fontSize": "10px", "color": MUTED,
                                             "fontFamily": FONT}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "7px 12px", "background": BG3,
        "borderBottom": f"2px solid {BORDER}",
        "fontFamily": FONT,
    }),

    # ── Three asset columns ────────────────────────────────────────────────────
    html.Div([
        asset_col("BTCUSDT"),
        asset_col("ETHUSDT"),
        asset_col("XAUUSDT"),
    ], style={"display": "flex", "gap": "5px", "padding": "5px",
               "flex": "none"}),

    # ── Bottom panels ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([panel_wrap("POSITIONS", "pos-panel", "110px")],
                 style={"flex": "1.2"}),
        html.Div([panel_wrap("ACCOUNT & ORDERS", "acct-panel", "110px")],
                 style={"flex": "1"}),
        html.Div([panel_wrap("SESSION STATS", "sess-panel", "110px")],
                 style={"flex": "0.7"}),
    ], style={"display": "flex", "gap": "5px", "padding": "0 5px 5px"}),

    # ── Trade log ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("TRADE LOG", style={"fontSize": "9px", "fontWeight": "bold",
                                           "letterSpacing": "2.5px", "color": CYAN,
                                           "fontFamily": FONT}),
            html.Button("⟳ SYNC", id="sync-btn", n_clicks=0, style={
                "fontSize": "9px", "fontFamily": FONT, "color": BG,
                "background": CYAN, "border": "none", "padding": "2px 8px",
                "cursor": "pointer", "marginLeft": "10px",
            }),
            html.Span(id="sync-msg", style={"fontSize": "9px", "color": MUTED,
                                             "fontFamily": FONT, "marginLeft": "6px"}),
        ], style={"padding": "4px 8px", "background": BG3,
                   "borderBottom": f"1px solid {SEP}",
                   "display": "flex", "alignItems": "center"}),
        html.Div(id="log-panel", style={"padding": "5px 8px", "minHeight": "60px"}),
    ], style={"background": BG2, "border": f"1px solid {BORDER}",
               "borderTop": f"2px solid {SEP}", "margin": "0 5px 5px"}),

], style={"background": BG, "color": TEXT, "minHeight": "100vh",
           "display": "flex", "flexDirection": "column"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK 1 — FAST (1.5s): prices, states, header, positions
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("btc-price",  "children"),
    Output("btc-chg",    "children"),
    Output("btc-chg",    "style"),
    Output("btc-state",  "children"),
    Output("eth-price",  "children"),
    Output("eth-chg",    "children"),
    Output("eth-chg",    "style"),
    Output("eth-state",  "children"),
    Output("xau-price",  "children"),
    Output("xau-chg",    "children"),
    Output("xau-chg",    "style"),
    Output("xau-state",  "children"),
    Output("hdr-ticker", "children"),
    Output("hdr-pnl",    "children"),
    Output("hdr-pnl",    "style"),
    Output("hdr-time",   "children"),
    Output("pos-panel",  "children"),
    Input("iv-fast", "n_intervals"),
)
def cb_fast(_):
    now      = datetime.now().strftime("%H:%M:%S")
    prices   = {}
    chg24s   = {}
    states   = {}
    col_out  = []   # (price_str, chg_str, chg_style, state_str) per symbol

    for sym in SYMBOLS:
        sid = SYM_ID[sym]
        clr = SYM_CLR[sym]

        try:
            p = get_price(sym)
        except Exception:
            p = 0.0
        prices[sym] = p

        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                             params={"symbol": sym}, timeout=3).json()
            chg = float(r.get("priceChangePercent", 0))
        except Exception:
            chg = 0.0
        chg24s[sym] = chg

        algo  = load_json(algo_state_file(sym))
        state = algo.get("state","WATCHING") if algo else "WATCHING"
        states[sym] = state
        sc = {"WATCHING":MUTED,"APPROACHING":YELLOW,"IN_TRADE":GREEN}.get(state, MUTED)

        price_str = f"${p:,.2f}" if p else "—"
        chg_str   = f"{chg:+.2f}%" if chg else ""
        chg_clr   = GREEN if chg >= 0 else RED
        state_str = f"● {state}"

        col_out += [
            price_str, chg_str,
            {"fontSize": "10px", "color": chg_clr, "fontFamily": FONT, "marginLeft": "6px"},
            state_str,
        ]

    # Header ticker
    ticker_items = []
    for sym in SYMBOLS:
        clr = SYM_CLR[sym]
        sid = SYM_ID[sym]
        p   = prices[sym]
        chg = chg24s[sym]
        chg_c = GREEN if chg >= 0 else RED
        ticker_items.append(html.Div([
            html.Span(sym.replace("USDT",""), style={"color": clr, "fontSize": "10px",
                                                      "fontFamily": FONT, "fontWeight": "bold",
                                                      "marginRight": "5px"}),
            html.Span(f"${p:,.2f}" if p else "—",
                      style={"color": WHITE, "fontSize": "13px", "fontWeight": "bold",
                              "fontFamily": FONT, "marginRight": "3px"}),
            html.Span(f"{chg:+.2f}%" if chg else "",
                      style={"color": chg_c, "fontSize": "9px", "fontFamily": FONT}),
        ], style={"display": "flex", "alignItems": "center"}))

    # Positions (all symbols)
    live_pnl = 0.0
    pos_out  = []

    if LIVE:
        for sym in SYMBOLS:
            p = prices[sym]
            pos_resp = _live(get_position_risk, sym, default=[])
            positions = pos_resp if isinstance(pos_resp, list) else (pos_resp or {}).get("data",[])
            active = [x for x in positions if abs(float(x.get("positionAmt",0))) > 0]

            for pos in active:
                side = pos.get("positionSide","")
                amt  = abs(float(pos.get("positionAmt",0)))
                ep   = float(pos.get("entryPrice",0))
                pnl  = float(pos.get("unRealizedProfit",0))
                liq  = float(pos.get("liquidationPrice",0))
                lev  = pos.get("leverage","20")
                live_pnl += pnl
                pc = GREEN if pnl >= 0 else RED
                sc = GREEN if side == "LONG" else RED

                notional = amt * ep if ep else 1
                pnl_pct  = pnl / (notional / float(lev or 20)) * 100
                dist     = (p - ep) / ep * 100 if ep and p else 0.0
                dist_c   = GREEN if ((side=="LONG" and dist>0) or (side=="SHORT" and dist<0)) else RED

                sym_algo = load_json(algo_state_file(sym))
                clr = SYM_CLR[sym]

                if pos_out:
                    pos_out.append(sep())
                pos_out += [
                    html.Div([
                        html.Span(sym, style={"color": clr, "fontSize": "9px",
                                              "fontFamily": FONT, "fontWeight": "bold",
                                              "marginRight": "6px"}),
                        html.Span("▲ LONG" if side=="LONG" else "▼ SHORT",
                                  style={"color": sc, "fontSize": "12px",
                                         "fontWeight": "bold", "fontFamily": FONT}),
                        html.Span(f"  {amt} ×{lev}",
                                  style={"color": MUTED, "fontSize": "9px",
                                         "fontFamily": FONT}),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("UNRL P&L  ", style={"color": LABEL, "fontSize": "9px",
                                                        "fontFamily": FONT}),
                        html.Span(fmt_pnl(pnl), style={"color": pc, "fontSize": "16px",
                                                        "fontWeight": "bold", "fontFamily": FONT}),
                        html.Span(f"  ({pnl_pct:+.2f}%)",
                                  style={"color": pc, "fontSize": "10px", "fontFamily": FONT}),
                    ], style={"marginBottom": "3px"}),
                    kv("ENTRY",   f"${ep:,.2f}"),
                    kv("CURRENT", f"${p:,.2f}  ({dist:+.2f}%)", dist_c),
                    kv("LIQ",     f"${liq:,.2f}", RED),
                ]
                if sym_algo and sym_algo.get("state") == "IN_TRADE":
                    tp = sym_algo.get("tp",0)
                    sl = sym_algo.get("sl",0)
                    if tp and p:
                        d = abs(tp - p)
                        pos_out.append(kv("TP", f"${tp:,.2f}  ({d/p*100:.2f}%)", GREEN))
                    if sl and p:
                        d = abs(sl - p)
                        trail = sym_algo.get("trail_stage","none")
                        slc = {"none":RED,"breakeven":YELLOW,"trailing":ORANGE}.get(trail,RED)
                        sll = {"none":"SL","breakeven":"BE","trailing":"TRAIL"}.get(trail,"SL")
                        pos_out.append(kv(sll, f"${sl:,.2f}  ({d/p*100:.2f}%)", slc))

        if not pos_out:
            pos_out = [html.Div("NO OPEN POSITIONS",
                                style={"color": MUTED, "fontSize": "10px",
                                       "fontFamily": FONT, "padding": "4px 0"})]
    else:
        pos_out = [html.Div("API UNAVAILABLE",
                            style={"color": RED, "fontSize": "10px", "fontFamily": FONT})]

    tlog = load_json(TRADE_LOG)
    if live_pnl != 0:
        pnl_str = f"LIVE {fmt_pnl(live_pnl)}"
        pnl_col = GREEN if live_pnl >= 0 else RED
    elif tlog:
        sp = sum(t.get("pnl_usd",0) for t in tlog)
        pnl_str = f"SESSION {fmt_pnl(sp)}"
        pnl_col = GREEN if sp >= 0 else RED
    else:
        pnl_str, pnl_col = "SESSION  $0.00", MUTED

    return (
        col_out[0],  col_out[1],  col_out[2],  col_out[3],   # BTC
        col_out[4],  col_out[5],  col_out[6],  col_out[7],   # ETH
        col_out[8],  col_out[9],  col_out[10], col_out[11],  # XAU
        ticker_items,
        pnl_str,
        {"fontSize": "12px", "fontWeight": "bold", "color": pnl_col, "fontFamily": FONT},
        now,
        html.Div(pos_out),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK 2 — MEDIUM (5s): per-asset strips, account, session
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("btc-strip",  "children"),
    Output("eth-strip",  "children"),
    Output("xau-strip",  "children"),
    Output("acct-panel", "children"),
    Output("sess-panel", "children"),
    Input("iv-med", "n_intervals"),
)
def cb_med(_):
    strips = []
    for sym in SYMBOLS:
        try:
            price = get_price(sym)
        except Exception:
            price = 0.0
        mi   = fetch_market_info(sym)
        algo = load_json(algo_state_file(sym))
        strips.append(build_strip(sym, mi, algo, price))

    # Account + orders panel
    acct_rows = []
    if LIVE:
        bal_resp = _live(get_balances, default=[])
        bals = bal_resp if isinstance(bal_resp, list) else (bal_resp or {}).get("data",[])
        for b in bals:
            asset  = b.get("asset","")
            wallet = float(b.get("balance",0) or b.get("walletBalance",0) or 0)
            avail  = float(b.get("availableBalance") or wallet)
            margin = wallet - avail
            if wallet > 0:
                acct_rows += [
                    kv(asset,      f"${wallet:.4f}", YELLOW, "56px"),
                    kv("  AVAIL",  f"${avail:.4f}",  GREEN,  "56px"),
                ]
                if margin > 0:
                    acct_rows.append(kv("  MARGIN", f"${margin:.4f}", ORANGE, "56px"))
        if acct_rows:
            acct_rows.append(sep())
        for sym in SYMBOLS:
            or_resp = _live(get_open_orders, sym, default=[])
            ords = or_resp if isinstance(or_resp, list) else (or_resp or {}).get("data",[])
            if ords:
                clr = SYM_CLR[sym]
                for o in ords[:3]:
                    side = o.get("side","")
                    qty  = o.get("origQty", o.get("qty","?"))
                    opx  = float(o.get("price",0))
                    sc   = GREEN if side == "BUY" else ORANGE
                    acct_rows.append(html.Div([
                        html.Span(sym.replace("USDT",""), style={"color": clr, "fontSize": "9px",
                                                                   "fontFamily": FONT,
                                                                   "minWidth": "36px",
                                                                   "display": "inline-block"}),
                        html.Span(f"{side:<4}",  style={"color": sc,   "fontSize": "9px",
                                                          "fontFamily": FONT, "minWidth": "32px",
                                                          "display": "inline-block"}),
                        html.Span(f"{qty}",       style={"color": TEXT, "fontSize": "9px",
                                                          "fontFamily": FONT, "minWidth": "56px",
                                                          "display": "inline-block"}),
                        html.Span(f"${opx:,.1f}", style={"color": YELLOW,"fontSize": "9px",
                                                           "fontFamily": FONT}),
                    ], style={"marginBottom": "1px"}))
        if not acct_rows:
            acct_rows = [html.Div("—", style={"color": MUTED, "fontSize": "10px",
                                               "fontFamily": FONT})]
    else:
        acct_rows = [html.Div("API UNAVAILABLE",
                              style={"color": RED, "fontSize": "10px", "fontFamily": FONT})]

    # Session stats
    tlog = load_json(TRADE_LOG)
    if tlog:
        wins   = sum(1 for t in tlog if t["result"] == "WIN")
        losses = sum(1 for t in tlog if t["result"] == "LOSS")
        total  = wins + losses
        wr     = wins/total*100 if total else 0
        pnl    = sum(t.get("pnl_usd",0) for t in tlog)
        avg_w  = (sum(t.get("pnl_usd",0) for t in tlog if t["result"]=="WIN") / wins
                  if wins else 0)
        avg_l  = (sum(abs(t.get("pnl_usd",0)) for t in tlog if t["result"]=="LOSS") / losses
                  if losses else 0)
        expect = wr/100 * avg_w - (1-wr/100) * avg_l
        sess_out = html.Div([
            kv("TRADES",   str(total),                WHITE, "72px"),
            kv("W / L",    f"{wins} / {losses}",      TEXT,  "72px"),
            kv("WIN RATE", f"{wr:.0f}%",
               GREEN if wr >= 50 else RED,             "72px"),
            sep(),
            kv("NET P&L",  fmt_pnl(pnl),
               GREEN if pnl >= 0 else RED,             "72px"),
            kv("AVG WIN",  f"+${avg_w:.3f}",           GREEN, "72px"),
            kv("AVG LOSS", f"-${avg_l:.3f}",           RED,   "72px"),
            kv("EXPECT",   fmt_pnl(expect),
               GREEN if expect >= 0 else RED,           "72px"),
        ])
    else:
        sess_out = html.Div("NO TRADES",
                            style={"color": MUTED, "fontSize": "10px", "fontFamily": FONT})

    return strips[0], strips[1], strips[2], html.Div(acct_rows), sess_out


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK 3 — SLOW (20s): charts + trade log
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("btc-chart", "figure"),
    Output("eth-chart", "figure"),
    Output("xau-chart", "figure"),
    Output("log-panel", "children"),
    Input("iv-slow", "n_intervals"),
)
def cb_slow(_):
    figs = []
    for sym in SYMBOLS:
        try:
            price  = get_price(sym)
            klines = get_klines(sym, "5m", 200)
        except Exception:
            price  = 0.0
            klines = []
        algo  = load_json(algo_state_file(sym))
        hm    = load_json(heatmap_json(sym))
        whale = load_json(whale_json(sym))
        figs.append(build_chart(sym, price, klines, algo, hm, whale))

    # Trade log
    tlog = load_json(TRADE_LOG)
    if tlog:
        wins      = sum(1 for t in tlog if t["result"] == "WIN")
        losses    = sum(1 for t in tlog if t["result"] == "LOSS")
        total_pnl = sum(t.get("pnl_usd",0) for t in tlog)
        wr = wins/len(tlog)*100 if tlog else 0

        summary = html.Div([
            html.Span("TRADES ", style={"color": LABEL, "fontSize": "10px", "fontFamily": FONT}),
            html.Span(f"{len(tlog)}", style={"color": WHITE, "fontSize": "10px",
                                              "fontFamily": FONT, "marginRight": "14px"}),
            html.Span("W/L ", style={"color": LABEL, "fontSize": "10px", "fontFamily": FONT}),
            html.Span(f"{wins}/{losses}", style={"color": TEXT, "fontSize": "10px",
                                                  "fontFamily": FONT, "marginRight": "14px"}),
            html.Span("WIN RATE ", style={"color": LABEL, "fontSize": "10px", "fontFamily": FONT}),
            html.Span(f"{wr:.0f}%", style={"color": GREEN if wr>=50 else RED,
                                            "fontSize": "10px", "fontFamily": FONT,
                                            "marginRight": "14px"}),
            html.Span("NET P&L ", style={"color": LABEL, "fontSize": "10px", "fontFamily": FONT}),
            html.Span(fmt_pnl(total_pnl), style={"color": GREEN if total_pnl>=0 else RED,
                                                   "fontSize": "10px", "fontWeight": "bold",
                                                   "fontFamily": FONT}),
        ], style={"marginBottom": "5px", "paddingBottom": "4px",
                   "borderBottom": f"1px solid {SEP}"})

        def col(txt, w):
            return html.Span(txt, style={"color": LABEL, "fontSize": "9px", "fontFamily": FONT,
                                          "display": "inline-block", "minWidth": w})

        hdr = html.Div([
            col("#", "24px"), col("TIME","100px"), col("SYM","64px"),
            col("DIR","44px"), col("ENTRY","78px"), col("CLOSE","78px"),
            col("QTY","56px"), col("NET P&L","78px"), col("RESULT","52px"),
        ], style={"marginBottom": "2px", "paddingBottom": "2px",
                   "borderBottom": f"1px solid {DIM}"})

        rows = [summary, hdr]
        for t in reversed(tlog[-20:]):
            pc = GREEN if t.get("pnl_usd",0) >= 0 else RED
            dc = RED if t["direction"] == "SHORT" else GREEN
            sym_clr = SYM_CLR.get(t.get("symbol","BTCUSDT"), CYAN)
            try:
                ts = datetime.fromisoformat(t["time"]).strftime("%m-%d %H:%M:%S")
            except Exception:
                ts = t.get("time","")[:19]
            entry = t.get("entry",0)
            close = t.get("actual_close",0)
            qty   = t.get("actual_qty",0)
            net   = t.get("pnl_usd",0)
            res   = t.get("result","—")
            sym_s = t.get("symbol","?").replace("USDT","")

            rows.append(html.Div([
                html.Span(str(t.get("id","?")),
                          style={"color": MUTED, "fontSize": "9px", "fontFamily": FONT,
                                  "display": "inline-block", "minWidth": "24px"}),
                html.Span(ts, style={"color": MUTED, "fontSize": "9px", "fontFamily": FONT,
                                      "display": "inline-block", "minWidth": "100px"}),
                html.Span(sym_s, style={"color": sym_clr, "fontSize": "9px", "fontFamily": FONT,
                                         "display": "inline-block", "minWidth": "64px",
                                         "fontWeight": "bold"}),
                html.Span(t["direction"], style={"color": dc, "fontSize": "10px",
                                                   "fontFamily": FONT, "fontWeight": "bold",
                                                   "display": "inline-block", "minWidth": "44px"}),
                html.Span(f"${entry:>9,.1f}" if entry else "—",
                          style={"color": TEXT, "fontSize": "10px", "fontFamily": FONT,
                                  "display": "inline-block", "minWidth": "78px"}),
                html.Span(f"${close:>9,.1f}" if close else "—",
                          style={"color": TEXT, "fontSize": "10px", "fontFamily": FONT,
                                  "display": "inline-block", "minWidth": "78px"}),
                html.Span(f"{qty}" if qty else "—",
                          style={"color": CYAN, "fontSize": "9px", "fontFamily": FONT,
                                  "display": "inline-block", "minWidth": "56px"}),
                html.Span(fmt_pnl(net),
                          style={"color": pc, "fontSize": "10px", "fontWeight": "bold",
                                  "fontFamily": FONT, "display": "inline-block",
                                  "minWidth": "78px"}),
                html.Span(res, style={"color": BG, "fontSize": "9px", "fontFamily": FONT,
                                       "fontWeight": "bold",
                                       "background": GREEN if res=="WIN" else RED,
                                       "padding": "1px 4px", "display": "inline-block"}),
            ], style={"marginBottom": "2px", "paddingBottom": "1px",
                       "borderBottom": f"1px solid {DIM}"}))

        log_out = html.Div(rows)
    else:
        log_out = html.Div("NO TRADES YET",
                           style={"color": MUTED, "fontSize": "10px", "fontFamily": FONT})

    return figs[0], figs[1], figs[2], log_out


# ══════════════════════════════════════════════════════════════════════════════
# SYNC CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("sync-msg", "children"),
    Output("sync-msg", "style"),
    Input("sync-btn", "n_clicks"),
    prevent_initial_call=True,
)
def sync_asterdex(_):
    try:
        from fetch_real_trades import fetch_all_fills, group_sessions, session_to_entry
        from datetime import timezone as _tz
        start_ms = int(datetime(2026, 5, 12, tzinfo=_tz.utc).timestamp() * 1000)
        log = []
        running = 0.0
        for sym in SYMBOLS:
            fills    = fetch_all_fills(sym, start_ms)
            sessions = group_sessions(fills)
            for i, sess in enumerate(sessions, 1):
                entry, running = session_to_entry(sess, i, running)
                log.append(entry)
        with open(TRADE_LOG, "w") as f:
            json.dump(log, f, indent=2)
        sign = "+" if running >= 0 else ""
        return (f"✓ {len(log)} trades synced  net {sign}${running:.4f}",
                {"fontSize": "9px", "color": GREEN, "fontFamily": FONT, "marginLeft": "6px"})
    except Exception as e:
        return (f"✗ {e}",
                {"fontSize": "9px", "color": RED, "fontFamily": FONT, "marginLeft": "6px"})


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 52)
    print("  LSCO Bloomberg Terminal  →  http://localhost:8050")
    print("=" * 52)
    app.run(debug=False, host="127.0.0.1", port=8050, threaded=True)
