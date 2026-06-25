#!/usr/bin/env python3
"""
live_trade_report.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetches ALL real trades from AsterDEX across ALL symbols
from 2026-06-07 onward, groups fills into complete sessions,
computes full performance metrics, and generates a professional
investor-grade HTML report.

Output: backtest_results/LSCO_Live_Report.html
"""

import sys, os, json, math, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "asterdex_trade"))
from account_data import (
    get_trade_history, get_income_history, get_balances, get_account
)

# ── Config ─────────────────────────────────────────────────────────────────────
START_DATE = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
START_MS   = int(START_DATE.timestamp() * 1000)
OUTPUT_DIR = Path(__file__).parent / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_HTML   = OUTPUT_DIR / "LSCO_Live_Report.html"

# All symbols to probe (covers live algos + known active markets)
PROBE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT",
    "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "INJUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "MATICUSDT", "BCHUSDT",
    "AAVEUSDT", "FILUSDT", "TRXUSDT", "RUNEUSDT", "SUIUSDT",
    "FETUSDT", "SEIUSDT", "LDOUSDT", "CFXUSDT", "ASTERUSDT",
    "XAUUSDT",
]


# ── Step 1: Fetch fills for one symbol ────────────────────────────────────────

def fetch_fills(symbol: str, start_ms: int, retries: int = 3) -> list:
    all_fills = []
    from_id   = None
    while True:
        for attempt in range(retries):
            try:
                resp = get_trade_history(
                    symbol     = symbol,
                    start_time = start_ms if from_id is None else None,
                    from_id    = from_id,
                    limit      = 1000,
                )
                break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)   # 1s, 2s backoff
                else:
                    return []                  # give up on this symbol
        if isinstance(resp, dict) and "code" in resp:
            return []
        fills = resp if isinstance(resp, list) else resp.get("data", [])
        if not fills:
            break
        if from_id is not None:
            fills = [f for f in fills if int(f.get("time", 0)) >= start_ms]
        all_fills.extend(fills)
        if len(fills) < 1000:
            break
        from_id = max(int(f.get("id", 0)) for f in fills) + 1
        time.sleep(0.3)   # small pause between pagination calls
    all_fills.sort(key=lambda f: int(f.get("time", 0)))
    return all_fills


# ── Step 2: Group fills → complete trade sessions ─────────────────────────────

def group_sessions(fills: list, symbol: str) -> list:
    sessions = []
    for pos_side, direction, open_side, close_side in [
        ("SHORT", "SHORT", "SELL", "BUY"),
        ("LONG",  "LONG",  "BUY",  "SELL"),
    ]:
        pos_fills = [f for f in fills
                     if f.get("positionSide", "").upper() == pos_side]
        if not pos_fills:
            continue
        open_buf, close_buf = [], []
        open_qty, close_qty = 0.0, 0.0
        for f in pos_fills:
            side = f.get("side", "").upper()
            qty  = float(f.get("qty", 0))
            if side == open_side:
                open_buf.append(f); open_qty += qty
            elif side == close_side:
                close_buf.append(f); close_qty += qty
                if open_buf and close_qty >= open_qty - 0.0005:
                    sessions.append({
                        "symbol":      symbol,
                        "direction":   direction,
                        "open_fills":  list(open_buf),
                        "close_fills": list(close_buf),
                    })
                    open_buf = []; close_buf = []
                    open_qty = close_qty = 0.0
    sessions.sort(key=lambda s: int(s["open_fills"][0].get("time", 0)))
    return sessions


# ── Step 3: Session → trade record ────────────────────────────────────────────

def session_to_trade(session: dict, trade_id: int) -> dict:
    open_fills  = session["open_fills"]
    close_fills = session["close_fills"]
    all_fills   = open_fills + close_fills

    def wavg(fills):
        tot_n = sum(float(f["qty"]) * float(f["price"]) for f in fills)
        tot_q = sum(float(f["qty"]) for f in fills)
        return tot_n / tot_q if tot_q else 0.0

    avg_entry  = wavg(open_fills)
    avg_close  = wavg(close_fills)
    open_qty   = sum(float(f["qty"]) for f in open_fills)
    close_qty  = sum(float(f["qty"]) for f in close_fills)
    notional   = avg_entry * close_qty

    gross_pnl  = sum(float(f.get("realizedPnl", 0)) for f in all_fills)
    commission = sum(float(f.get("commission",  0)) for f in all_fills)
    net_pnl    = gross_pnl - commission

    open_ts  = int(open_fills[0].get("time",  0)) / 1000
    close_ts = int(close_fills[-1].get("time", 0)) / 1000 if close_fills else open_ts
    duration_h = (close_ts - open_ts) / 3600

    result = "WIN" if net_pnl > 0 else "LOSS"

    return {
        "id":          trade_id,
        "open_time":   datetime.fromtimestamp(open_ts,  tz=timezone.utc),
        "close_time":  datetime.fromtimestamp(close_ts, tz=timezone.utc),
        "duration_h":  round(duration_h, 2),
        "symbol":      session["symbol"],
        "direction":   session["direction"],
        "entry":       round(avg_entry,  4),
        "exit":        round(avg_close,  4),
        "qty":         round(close_qty,  6),
        "notional":    round(notional,   2),
        "gross_pnl":   round(gross_pnl,  4),
        "commission":  round(commission, 4),
        "net_pnl":     round(net_pnl,    4),
        "result":      result,
        "n_fills":     len(all_fills),
    }


# ── Step 4: Compute statistics ────────────────────────────────────────────────

def compute_stats(trades: list, start_balance: float = 0.0) -> dict:
    if not trades:
        return {}

    wins  = [t for t in trades if t["result"] == "WIN"]
    losses= [t for t in trades if t["result"] == "LOSS"]

    net_pnls   = [t["net_pnl"]   for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]

    gross_wins  = sum(t["gross_pnl"] for t in wins)
    gross_loss  = abs(sum(t["gross_pnl"] for t in losses))
    net_wins    = sum(t["net_pnl"]   for t in wins)
    net_loss    = abs(sum(t["net_pnl"]   for t in losses))

    gross_pf = round(gross_wins / gross_loss, 3) if gross_loss > 0 else float("inf")
    net_pf   = round(net_wins   / net_loss,   3) if net_loss   > 0 else float("inf")
    win_rate = round(len(wins) / len(trades) * 100, 1)

    total_net   = sum(net_pnls)
    total_gross = sum(gross_pnls)
    total_comm  = sum(t["commission"] for t in trades)

    # Equity curve — real account balance per trade close
    equity = []
    running = start_balance
    for t in trades:
        running += t["net_pnl"]
        equity.append(round(running, 4))

    # Max drawdown on equity curve
    peak = -float("inf")
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = e - peak
        if dd < max_dd:
            max_dd = dd

    # Sharpe (per-trade, annualised by trade freq)
    if len(net_pnls) > 1:
        mean_r = sum(net_pnls) / len(net_pnls)
        std_r  = math.sqrt(sum((x - mean_r)**2 for x in net_pnls) / (len(net_pnls) - 1))
        # days in period
        days = (trades[-1]["close_time"] - trades[0]["open_time"]).total_seconds() / 86400
        tpy  = len(trades) / max(days / 365.25, 0.01)
        sharpe = round((mean_r / std_r * math.sqrt(max(tpy, 1))) if std_r > 0 else 0, 2)
    else:
        sharpe = 0.0

    # Streaks
    max_ws = max_ls = cur_ws = cur_ls = 0
    for t in trades:
        if t["result"] == "WIN":
            cur_ws += 1; cur_ls = 0
        else:
            cur_ls += 1; cur_ws = 0
        max_ws = max(max_ws, cur_ws)
        max_ls = max(max_ls, cur_ls)

    avg_win      = round(net_wins  / len(wins),   4) if wins   else 0
    avg_loss     = round(-net_loss / len(losses),  4) if losses else 0
    best_trade   = max(net_pnls)
    worst_trade  = min(net_pnls)
    avg_duration = round(sum(t["duration_h"] for t in trades) / len(trades), 2)

    # Per-symbol breakdown
    by_symbol = defaultdict(lambda: {"trades": 0, "wins": 0, "net_pnl": 0.0})
    for t in trades:
        sym = t["symbol"]
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["wins"]   += 1 if t["result"] == "WIN" else 0
        by_symbol[sym]["net_pnl"] = round(by_symbol[sym]["net_pnl"] + t["net_pnl"], 4)

    return {
        "total_trades": len(trades),
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     win_rate,
        "gross_pf":     gross_pf,
        "net_pf":       net_pf,
        "total_gross":  round(total_gross, 4),
        "total_comm":   round(total_comm,  4),
        "total_net":    round(total_net,   4),
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "best_trade":   round(best_trade,  4),
        "worst_trade":  round(worst_trade, 4),
        "max_dd":       round(max_dd,      4),
        "sharpe":       sharpe,
        "max_win_str":  max_ws,
        "max_los_str":  max_ls,
        "avg_duration": avg_duration,
        "equity":       equity,
        "by_symbol":    dict(by_symbol),
    }


# ── Step 5: Generate HTML ─────────────────────────────────────────────────────

def generate_html(trades: list, stats: dict, balance_info: dict) -> str:
    now_str    = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    start_str  = START_DATE.strftime("%Y-%m-%d")
    n          = stats.get("total_trades", 0)
    wr         = stats.get("win_rate", 0)
    net_pf     = stats.get("net_pf", 0)
    gross_pf   = stats.get("gross_pf", 0)
    total_net  = stats.get("total_net", 0)
    total_comm = stats.get("total_comm", 0)
    total_gross= stats.get("total_gross", 0)
    sharpe     = stats.get("sharpe", 0)
    max_dd     = stats.get("max_dd", 0)
    avg_win    = stats.get("avg_win", 0)
    avg_loss   = stats.get("avg_loss", 0)
    best       = stats.get("best_trade", 0)
    worst      = stats.get("worst_trade", 0)
    mws        = stats.get("max_win_str", 0)
    mls        = stats.get("max_los_str", 0)
    avg_dur    = stats.get("avg_duration", 0)
    equity     = stats.get("equity", [])
    by_symbol  = stats.get("by_symbol", {})

    usdt_bal    = balance_info.get("usdt_balance",   0.0)
    unreal_pnl  = balance_info.get("unrealized_pnl", 0.0)
    deposited   = balance_info.get("deposited",       0.0)
    inc_rpnl    = balance_info.get("realized_pnl",    0.0)
    inc_comm    = balance_info.get("commission",       0.0)
    inc_fund    = balance_info.get("funding",          0.0)
    return_pct  = balance_info.get("return_pct",       0.0)

    wallet_bal  = f"${usdt_bal:,.2f}"
    dep_disp    = f"${deposited:,.2f}"
    rpnl_disp   = f"+${inc_rpnl:,.2f}" if inc_rpnl >= 0 else f"-${abs(inc_rpnl):,.2f}"
    rpnl_color  = "#22c55e" if inc_rpnl >= 0 else "#ef4444"
    ret_disp    = f"{return_pct:+.2f}%"
    ret_color   = "#22c55e" if return_pct >= 0 else "#ef4444"
    unreal_disp = f"+${unreal_pnl:,.4f}" if unreal_pnl >= 0 else f"-${abs(unreal_pnl):,.4f}"

    # colour helpers
    pnl_cls  = lambda v: "pos" if float(v) >= 0 else "neg"
    pnl_fmt  = lambda v: f"+${v:,.4f}" if float(v) >= 0 else f"-${abs(float(v)):,.4f}"

    # Trade rows
    trade_rows = ""
    for t in reversed(trades):   # newest first
        cls   = "win-row" if t["result"] == "WIN" else "loss-row"
        badge = '<span class="badge-win">WIN</span>'  if t["result"] == "WIN" \
                else '<span class="badge-loss">LOSS</span>'
        net_c = "pos" if t["net_pnl"] >= 0 else "neg"
        net_s = f"+${t['net_pnl']:,.4f}" if t["net_pnl"] >= 0 else f"-${abs(t['net_pnl']):,.4f}"
        dir_badge = '<span class="long-badge">LONG</span>' if t["direction"] == "LONG" \
                    else '<span class="short-badge">SHORT</span>'
        trade_rows += f"""
        <tr class="{cls}">
          <td>#{t['id']}</td>
          <td>{t['open_time'].strftime('%m-%d %H:%M')}</td>
          <td>{t['close_time'].strftime('%m-%d %H:%M')}</td>
          <td>{t['duration_h']}h</td>
          <td><strong>{t['symbol']}</strong></td>
          <td>{dir_badge}</td>
          <td>${t['entry']:,.2f}</td>
          <td>${t['exit']:,.2f}</td>
          <td>{t['qty']}</td>
          <td>${t['notional']:,.2f}</td>
          <td class="pos">+${t['gross_pnl']:,.4f}</td>
          <td class="neg">-${t['commission']:,.4f}</td>
          <td class="{net_c}">{net_s}</td>
          <td>{badge}</td>
        </tr>"""

    # Symbol breakdown rows
    sym_rows = ""
    for sym, d in sorted(by_symbol.items(), key=lambda x: x[1]["net_pnl"], reverse=True):
        wr_s   = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0
        pnl_c  = "pos" if d["net_pnl"] >= 0 else "neg"
        pnl_s  = f"+${d['net_pnl']:,.4f}" if d["net_pnl"] >= 0 else f"-${abs(d['net_pnl']):,.4f}"
        sym_rows += f"""
        <tr>
          <td><strong>{sym}</strong></td>
          <td>{d['trades']}</td>
          <td>{d['wins']}</td>
          <td>{d['trades'] - d['wins']}</td>
          <td>{wr_s}%</td>
          <td class="{pnl_c}">{pnl_s}</td>
        </tr>"""

    # Equity curve data
    eq_labels = json.dumps([f"#{i+1}" for i in range(len(equity))])
    eq_data   = json.dumps(equity)
    eq_colors = json.dumps(["#22c55e" if v >= 0 else "#ef4444" for v in equity])

    net_color = "#22c55e" if total_net >= 0 else "#ef4444"
    net_disp  = f"+${total_net:,.4f}" if total_net >= 0 else f"-${abs(total_net):,.4f}"
    dd_disp   = f"-${abs(max_dd):,.4f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LSCO Live Trading Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0a0a0f;
    --card: #111118;
    --border: #1e1e2e;
    --text: #e2e8f0;
    --muted: #64748b;
    --gold: #f59e0b;
    --green: #22c55e;
    --red: #ef4444;
    --blue: #3b82f6;
    --purple: #a855f7;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; }}
  a {{ color:var(--gold); }}

  /* Header */
  .header {{ background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 100%);
             border-bottom:1px solid var(--border); padding:32px 40px; }}
  .header-top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
  .logo {{ font-size:28px; font-weight:800; color:var(--gold); letter-spacing:2px; }}
  .logo span {{ color:var(--text); }}
  .header-meta {{ text-align:right; color:var(--muted); font-size:12px; line-height:1.8; }}
  .header-meta strong {{ color:var(--text); }}
  .header-sub {{ margin-top:12px; color:var(--muted); font-size:13px; }}
  .live-badge {{ display:inline-flex; align-items:center; gap:6px; background:#16213e;
                 border:1px solid var(--green); border-radius:20px; padding:4px 12px;
                 font-size:11px; color:var(--green); font-weight:600; letter-spacing:1px; }}
  .live-dot {{ width:7px; height:7px; border-radius:50%; background:var(--green);
               animation:pulse 1.5s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}

  /* Layout */
  .container {{ max-width:1400px; margin:0 auto; padding:32px 40px; }}
  .section {{ margin-bottom:36px; }}
  .section-title {{ font-size:13px; font-weight:700; color:var(--muted); letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:16px; padding-bottom:8px;
                    border-bottom:1px solid var(--border); }}

  /* KPI Cards */
  .kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
  .kpi-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
               padding:20px 24px; position:relative; overflow:hidden; }}
  .kpi-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
  .kpi-card.gold::before {{ background:var(--gold); }}
  .kpi-card.green::before {{ background:var(--green); }}
  .kpi-card.blue::before {{ background:var(--blue); }}
  .kpi-card.purple::before {{ background:var(--purple); }}
  .kpi-label {{ font-size:11px; color:var(--muted); text-transform:uppercase;
                letter-spacing:1px; margin-bottom:8px; }}
  .kpi-value {{ font-size:32px; font-weight:800; line-height:1; }}
  .kpi-sub {{ font-size:11px; color:var(--muted); margin-top:6px; }}
  .pos {{ color:var(--green); }}
  .neg {{ color:var(--red); }}

  /* Stats grid */
  .stats-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .stat-card {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                padding:16px 20px; }}
  .stat-label {{ font-size:11px; color:var(--muted); margin-bottom:6px; }}
  .stat-value {{ font-size:20px; font-weight:700; }}
  .stat-sub {{ font-size:11px; color:var(--muted); margin-top:4px; }}

  /* Chart */
  .chart-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
                 padding:24px; }}
  .chart-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; }}
  .chart-title {{ font-size:15px; font-weight:600; }}
  .chart-wrap {{ position:relative; height:260px; }}

  /* Account bar */
  .account-bar {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                  padding:16px 24px; display:flex; gap:40px; align-items:center; }}
  .acc-item {{ display:flex; flex-direction:column; gap:4px; }}
  .acc-label {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }}
  .acc-value {{ font-size:18px; font-weight:700; }}

  /* Table */
  .table-wrap {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
                 overflow:hidden; }}
  table {{ width:100%; border-collapse:collapse; }}
  thead th {{ background:#0d0d1a; color:var(--muted); font-size:11px; text-transform:uppercase;
              letter-spacing:1px; padding:12px 14px; text-align:left; border-bottom:1px solid var(--border); }}
  tbody tr {{ border-bottom:1px solid var(--border); transition:background 0.15s; }}
  tbody tr:hover {{ background:#16162a; }}
  tbody tr:last-child {{ border-bottom:none; }}
  td {{ padding:10px 14px; font-size:13px; }}
  .win-row td:first-child {{ border-left:3px solid var(--green); }}
  .loss-row td:first-child {{ border-left:3px solid var(--red); }}
  .badge-win {{ background:#14532d; color:var(--green); font-size:10px; font-weight:700;
                padding:3px 8px; border-radius:4px; border:1px solid #16a34a; }}
  .badge-loss {{ background:#450a0a; color:var(--red); font-size:10px; font-weight:700;
                 padding:3px 8px; border-radius:4px; border:1px solid #b91c1c; }}
  .long-badge  {{ background:#1e3a5f; color:#60a5fa; font-size:10px; font-weight:700;
                  padding:3px 8px; border-radius:4px; }}
  .short-badge {{ background:#3d1a2e; color:#f472b6; font-size:10px; font-weight:700;
                  padding:3px 8px; border-radius:4px; }}

  /* Disclaimer */
  .disclaimer {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                 padding:16px 20px; font-size:11px; color:var(--muted); line-height:1.7; }}

  @media(max-width:900px) {{
    .kpi-grid {{ grid-template-columns:repeat(2,1fr); }}
    .stats-grid {{ grid-template-columns:repeat(2,1fr); }}
    .container {{ padding:16px; }}
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-top">
    <div>
      <div class="logo">LSCO<span> Trading</span></div>
      <div class="header-sub">Liquidation Zone Scalp — Live Performance Dashboard</div>
    </div>
    <div class="header-meta">
      <span class="live-badge"><span class="live-dot"></span>LIVE DATA</span><br><br>
      <strong>Exchange:</strong> AsterDEX Futures<br>
      <strong>Period:</strong> {start_str} → {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}<br>
      <strong>Generated:</strong> {now_str}
    </div>
  </div>
</div>

<div class="container">

  <!-- ACCOUNT STATUS -->
  <div class="section">
    <div class="section-title">Account Status (Real USDT Only)</div>
    <div class="account-bar">
      <div class="acc-item">
        <span class="acc-label">Current USDT Balance</span>
        <span class="acc-value" style="color:var(--gold)">{wallet_bal} USDT</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Capital Deployed</span>
        <span class="acc-value" style="color:var(--muted)">{dep_disp} USDT</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Net Trading Return</span>
        <span class="acc-value" style="color:{ret_color}">{ret_disp}</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Realized PnL (Exchange)</span>
        <span class="acc-value" style="color:{rpnl_color}">{rpnl_disp} USDT</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Unrealized PnL</span>
        <span class="acc-value">{unreal_disp} USDT</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Commission Paid</span>
        <span class="acc-value" style="color:var(--muted)">${abs(inc_comm):,.2f} USDT</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Funding Fees</span>
        <span class="acc-value" style="color:var(--muted)">{f'+${inc_fund:.2f}' if inc_fund >= 0 else f'-${abs(inc_fund):.2f}'} USDT</span>
      </div>
      <div class="acc-item">
        <span class="acc-label">Active Since</span>
        <span class="acc-value" style="font-size:14px;color:var(--muted)">{start_str}</span>
      </div>
    </div>
  </div>

  <!-- KPI CARDS -->
  <div class="section">
    <div class="section-title">Key Performance Indicators</div>
    <div class="kpi-grid">
      <div class="kpi-card gold">
        <div class="kpi-label">Total Net PnL</div>
        <div class="kpi-value" style="color:{net_color}">{net_disp}</div>
        <div class="kpi-sub">Gross: +${total_gross:,.4f} &nbsp;|&nbsp; Fees: -${total_comm:,.4f}</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value" style="color:var(--green)">{wr}%</div>
        <div class="kpi-sub">{stats.get('wins',0)} wins / {stats.get('losses',0)} losses / {n} total</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Net Profit Factor</div>
        <div class="kpi-value" style="color:var(--blue)">{net_pf}</div>
        <div class="kpi-sub">Gross PF: {gross_pf} &nbsp;|&nbsp; Net wins / Net losses</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value" style="color:var(--purple)">{sharpe}</div>
        <div class="kpi-sub">Trade-frequency annualised</div>
      </div>
    </div>
  </div>

  <!-- EQUITY CURVE -->
  <div class="section">
    <div class="section-title">Account Balance Curve (Real USDT)</div>
    <div class="chart-card">
      <div class="chart-header">
        <span class="chart-title">Realized PnL per Closed Trade (USDT)</span>
        <span style="color:var(--muted);font-size:12px">{n} trades · {start_str} onward</span>
      </div>
      <div class="chart-wrap">
        <canvas id="eqChart"></canvas>
      </div>
    </div>
  </div>

  <!-- DETAILED STATS -->
  <div class="section">
    <div class="section-title">Detailed Statistics</div>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Average Win</div>
        <div class="stat-value pos">+${avg_win:,.4f}</div>
        <div class="stat-sub">Net per winning trade</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Average Loss</div>
        <div class="stat-value neg">${avg_loss:,.4f}</div>
        <div class="stat-sub">Net per losing trade</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Best Trade</div>
        <div class="stat-value pos">+${best:,.4f}</div>
        <div class="stat-sub">Single trade max profit</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Worst Trade</div>
        <div class="stat-value neg">${worst:,.4f}</div>
        <div class="stat-sub">Single trade max loss</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Max Drawdown</div>
        <div class="stat-value neg">{dd_disp}</div>
        <div class="stat-sub">Peak-to-trough PnL drop</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Trade Duration</div>
        <div class="stat-value">{avg_dur}h</div>
        <div class="stat-sub">Open → close</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Max Win Streak</div>
        <div class="stat-value pos">{mws} trades</div>
        <div class="stat-sub">Consecutive winners</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Max Loss Streak</div>
        <div class="stat-value neg">{mls} trades</div>
        <div class="stat-sub">Consecutive losers</div>
      </div>
    </div>
  </div>

  <!-- PER-SYMBOL BREAKDOWN -->
  <div class="section">
    <div class="section-title">Per-Symbol Breakdown</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th><th>Trades</th><th>Wins</th><th>Losses</th>
            <th>Win Rate</th><th>Net PnL</th>
          </tr>
        </thead>
        <tbody>{sym_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- FULL TRADE LOG -->
  <div class="section">
    <div class="section-title">Complete Trade Log ({n} trades · newest first)</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Open</th><th>Close</th><th>Dur</th>
            <th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
            <th>Qty</th><th>Notional</th><th>Gross</th><th>Fee</th>
            <th>Net PnL</th><th>Result</th>
          </tr>
        </thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- DISCLAIMER -->
  <div class="section">
    <div class="disclaimer">
      <strong style="color:var(--text)">Disclaimer:</strong>
      This report shows real closed-trade data fetched directly from AsterDEX Futures via authenticated API.
      All PnL figures are realized and reflect actual exchange fills including commissions.
      Past performance does not guarantee future results.
      Futures trading involves substantial risk of loss and is not suitable for all investors.
      Generated automatically by LSCO Trading System · {now_str}
    </div>
  </div>

</div><!-- /container -->

<script>
const ctx = document.getElementById('eqChart').getContext('2d');
const labels = {eq_labels};
const data   = {eq_data};
const colors = {eq_colors};

new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{
      label: 'Net PnL per Trade',
      data: data,
      backgroundColor: colors,
      borderRadius: 3,
      borderSkipped: false,
    }},
    {{
      label: 'Account Balance (USDT)',
      data: data.map((_, i) => data.slice(0, i+1).reduce((a,b) => a+b, 0)),
      type: 'line',
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245,158,11,0.08)',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.3,
      fill: true,
      yAxisID: 'y',
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
      tooltip: {{
        backgroundColor: '#1e1e30',
        borderColor: '#334155',
        borderWidth: 1,
        callbacks: {{
          label: ctx => {{
            const v = ctx.parsed.y;
            return ` ${{ctx.dataset.label}}: ${{v >= 0 ? '+' : ''}}${{v.toFixed(4)}}`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#475569', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#475569', font: {{ size: 11 }},
                      callback: v => (v >= 0 ? '+' : '') + '$' + v.toFixed(2) }},
             grid: {{ color: '#1e293b' }},
             border: {{ color: '#334155' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  LSCO Live Trade Report Generator")
    print(f"  Fetching from: {START_DATE.strftime('%Y-%m-%d')} onward")
    print(f"  Exchange     : AsterDEX Futures")
    print("=" * 62)

    # ── Account balance + income breakdown ───────────────────────────────────
    print("\nFetching account balance & income history...")
    balance_info = {
        "usdt_balance":   0.0,
        "unrealized_pnl": 0.0,
        "deposited":      0.0,
        "realized_pnl":   0.0,
        "commission":     0.0,
        "funding":        0.0,
        "return_pct":     0.0,
    }
    try:
        # Real USDT wallet balance
        bal = get_balances()
        usdt_bal = 0.0
        unreal   = 0.0
        if isinstance(bal, list):
            for b in bal:
                if b.get("asset") == "USDT":
                    usdt_bal = float(b.get("balance", 0))
                    unreal   = float(b.get("crossUnPnl", 0))
                    break
        balance_info["usdt_balance"]   = usdt_bal
        balance_info["unrealized_pnl"] = unreal

        # Income history ALL TIME (use early start to capture deposits before period)
        EPOCH_MS = 1700000000000  # Nov 2023 — before any possible trades
        inc_all = get_income_history(symbol=None, income_type=None,
                                     start_time=EPOCH_MS, limit=1000)
        for r in (inc_all if isinstance(inc_all, list) else []):
            amt   = float(r.get("income", 0))
            itype = r.get("incomeType", "")
            if itype == "TRANSFER_SPOT_TO_FUTURE":
                balance_info["deposited"]    += max(amt, 0)
            elif itype == "REALIZED_PNL":
                balance_info["realized_pnl"] += amt
            elif itype == "COMMISSION":
                balance_info["commission"]   += amt
            elif itype == "FUNDING_FEE":
                balance_info["funding"]      += amt

        dep = balance_info["deposited"]
        balance_info["return_pct"] = (
            (usdt_bal - dep) / dep * 100 if dep > 0 else 0.0
        )
        print(f"  USDT Balance  : ${usdt_bal:,.2f}")
        print(f"  Deposited     : ${dep:,.2f}")
        print(f"  Realized PnL  : ${balance_info['realized_pnl']:+,.2f}")
        print(f"  Commission    : ${balance_info['commission']:,.2f}")
        print(f"  Funding       : ${balance_info['funding']:+,.2f}")
        print(f"  Return        : {balance_info['return_pct']:+.2f}%")
    except Exception as e:
        print(f"  Balance fetch failed: {e}")

    # ── Fetch fills per symbol ────────────────────────────────────────────────
    print("\nProbing symbols for trade history...")
    all_trades   = []
    active_syms  = []

    for sym in PROBE_SYMBOLS:
        fills = fetch_fills(sym, START_MS)
        time.sleep(0.5)   # rate-limit: 2 req/s max
        if not fills:
            continue
        active_syms.append(sym)
        sessions = group_sessions(fills, sym)
        print(f"  {sym:<12} {len(fills):>4} fills  ->  {len(sessions)} sessions")
        for i, sess in enumerate(sessions, 1):
            trade = session_to_trade(sess, len(all_trades) + 1)
            all_trades.append(trade)

    print(f"\nTotal: {len(all_trades)} closed trades across {len(active_syms)} symbols")

    if not all_trades:
        print("\nNo completed trades found since", START_DATE.strftime('%Y-%m-%d'))
        print("(Open positions or no activity yet)")
        # still generate an empty report
        stats = {"total_trades": 0, "wins": 0, "losses": 0,
                 "win_rate": 0, "gross_pf": 0, "net_pf": 0,
                 "total_gross": 0, "total_comm": 0, "total_net": 0,
                 "avg_win": 0, "avg_loss": 0, "best_trade": 0,
                 "worst_trade": 0, "max_dd": 0, "sharpe": 0,
                 "max_win_str": 0, "max_los_str": 0, "avg_duration": 0,
                 "equity": [], "by_symbol": {}}
    else:
        # Sort all trades by open time
        all_trades.sort(key=lambda t: t["open_time"])
        # Re-number after sort
        for i, t in enumerate(all_trades, 1):
            t["id"] = i

        stats = compute_stats(all_trades, start_balance=balance_info["deposited"])

        print(f"\n{'─'*40}")
        print(f"  Win Rate    : {stats['win_rate']}%  "
              f"({stats['wins']}W / {stats['losses']}L)")
        print(f"  Gross PF    : {stats['gross_pf']}")
        print(f"  Net PF      : {stats['net_pf']}")
        print(f"  Total Net   : ${stats['total_net']:+,.4f}")
        print(f"  Sharpe      : {stats['sharpe']}")
        print(f"  Max DD      : ${stats['max_dd']:,.4f}")
        print(f"  Best Trade  : ${stats['best_trade']:+,.4f}")
        print(f"  Worst Trade : ${stats['worst_trade']:+,.4f}")
        print(f"{'─'*40}")

    # ── Generate HTML ─────────────────────────────────────────────────────────
    print(f"\nGenerating HTML report...")
    html = generate_html(all_trades, stats, balance_info)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Saved → {OUT_HTML}")
    print(f"  Open in browser → right-click → Print → Save as PDF")
    print("=" * 62)


if __name__ == "__main__":
    main()
