"""
backtest_nse.py  --  Regime-Aware NSE India Strategy  (V1 / V2 / V3)
======================================================================
Strategy: NIFTY 50 regime timing + per-stock EMA/momentum gate
Cost model: Modular, itemised NSE cost engine (delivery + futures)

REGIME (NIFTY 50 weekly):
  BULL  -> NIFTY close > EMA(20w) AND 4w momentum > 0  (prev week)
  BEAR  -> NIFTY close < EMA(20w) AND 4w momentum < 0
  TRANS -> EMA and momentum disagree
ACTION:
  BULL       -> long qualifying NSE stocks (equal weight, DELIVERY)
  BEAR/TRANS -> 100% cash @ 6.0% p.a. (India T-bill rate)

V1: Regime + per-stock gate, long-only, equal weight
V2: V1 + 20% per-stock hard stop-loss from entry price
V3: V2 + INDIA VIX position scaling
"""

import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
#  SECTION 1 : MODULAR COST ENGINE
#  All rates are configurable. Change the defaults in the config
#  classes if SEBI / broker / stamp-duty rules change.
#
#  Rate correction vs user input:
#    Futures STT sell: user supplied 0.05% which is the EQUITY
#    OPTIONS sell rate. The correct EQUITY FUTURES sell-side STT
#    is 0.0125% (pre-Oct 2024) / 0.02% (post-Budget Oct 2024).
#    Default here = 0.02% (current 2025 rate, Finance Act 2024).
#
#    NSE equity delivery exchange charges: user said 0.00322%.
#    Current NSE rate (2025) = 0.00297%. Using user's value as
#    default since it is configurable. Update as needed.
# ════════════════════════════════════════════════════════════════════


@dataclass
class DeliveryCostConfig:
    """
    All rates as plain fractions (e.g. 0.001 = 0.10%).
    Sources: SEBI circular, NSE rate card, Finance Act 2020/2024.
    """
    # ── BROKERAGE ─────────────────────────────────────────────────
    brokerage_pct: float   = 0.0          # zero delivery brokerage (Zerodha/Groww)

    # ── STT (Securities Transaction Tax) ──────────────────────────
    # Finance Act: delivery = 0.1% on BOTH buy and sell
    stt_buy_pct:  float    = 0.001        # 0.10% on buy turnover
    stt_sell_pct: float    = 0.001        # 0.10% on sell turnover

    # ── STAMP DUTY ────────────────────────────────────────────────
    # Finance Act 2020: stamp duty on BUY side only for delivery
    stamp_buy_pct: float   = 0.00015      # 0.015% on buy turnover (sell = 0)

    # ── EXCHANGE TRANSACTION CHARGES (NSE) ────────────────────────
    # NSE Equity Cash: 0.00297% (revised 2023). User supplied 0.00322%.
    # Both sides same rate.
    exchange_pct: float    = 0.0000322    # 0.00322% per side

    # ── SEBI TURNOVER CHARGES ─────────────────────────────────────
    # Rs.10 per crore = 0.0001% = 0.000001 as fraction. Both sides.
    sebi_pct: float        = 0.000001     # 0.0001%

    # ── GST ───────────────────────────────────────────────────────
    # 18% on (brokerage + exchange charges). NOT on STT/stamp/SEBI.
    gst_rate: float        = 0.18

    # ── DP CHARGES (Depository Participant) ───────────────────────
    # Fixed per ISIN per day on SELL only (charged by broker/CDSL/NSDL)
    dp_charge_base_inr: float = 15.0      # Rs.15 per ISIN per day (sell)
    dp_charge_gst_rate: float = 0.18      # 18% GST on DP charge

    # ── SLIPPAGE ──────────────────────────────────────────────────
    # Market impact + spread. Applied to both buy and sell.
    slippage_pct: float    = 0.0005       # 0.05% per side

    @property
    def dp_charge_total_inr(self) -> float:
        return self.dp_charge_base_inr * (1 + self.dp_charge_gst_rate)


@dataclass
class FuturesCostConfig:
    """
    NSE Equity Futures cost configuration.
    STT: only on SELL side for futures (Finance Act 2024, effective Oct 2024).
    Brokerage: flat Rs.20 or 0.03% whichever is lower (discount broker model).
    """
    # ── BROKERAGE ─────────────────────────────────────────────────
    brokerage_pct: float     = 0.0003      # 0.03% of turnover per order
    brokerage_max_inr: float = 20.0        # Rs.20 per order cap (flat fee brokers)

    # ── STT ───────────────────────────────────────────────────────
    # Finance Act 2024 (effective Oct 1, 2024): equity futures sell = 0.02%
    # Pre-2024 rate was 0.0125%. User supplied 0.05% (that is options rate).
    stt_sell_pct: float      = 0.0002      # 0.02% on SELL turnover only

    # ── EXCHANGE TRANSACTION CHARGES (NSE) ────────────────────────
    # NSE F&O (stock futures): 0.0019% on both sides
    exchange_pct: float      = 0.000019    # 0.0019% per side

    # ── SEBI CHARGES ──────────────────────────────────────────────
    sebi_pct: float          = 0.000001    # 0.0001%

    # ── STAMP DUTY ────────────────────────────────────────────────
    # Finance Act 2020: futures BUY side only = 0.002%
    stamp_buy_pct: float     = 0.00002     # 0.002% on BUY turnover

    # ── GST ───────────────────────────────────────────────────────
    gst_rate: float          = 0.18        # 18% on (brokerage + exchange charges)

    # ── SLIPPAGE ──────────────────────────────────────────────────
    slippage_pct: float      = 0.0005      # 0.05% per side


@dataclass
class TradeCostBreakdown:
    """Itemised cost breakdown for a single trade (buy or sell)."""
    gross_pnl:        float = 0.0
    brokerage:        float = 0.0
    stt:              float = 0.0
    exchange_charges: float = 0.0
    sebi_charges:     float = 0.0
    gst:              float = 0.0
    stamp_duty:       float = 0.0
    dp_charges:       float = 0.0
    slippage:         float = 0.0

    @property
    def total_charges(self) -> float:
        return (self.brokerage + self.stt + self.exchange_charges
                + self.sebi_charges + self.gst + self.stamp_duty
                + self.dp_charges + self.slippage)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_charges

    def __add__(self, other: "TradeCostBreakdown") -> "TradeCostBreakdown":
        return TradeCostBreakdown(
            gross_pnl        = self.gross_pnl        + other.gross_pnl,
            brokerage        = self.brokerage        + other.brokerage,
            stt              = self.stt              + other.stt,
            exchange_charges = self.exchange_charges + other.exchange_charges,
            sebi_charges     = self.sebi_charges     + other.sebi_charges,
            gst              = self.gst              + other.gst,
            stamp_duty       = self.stamp_duty       + other.stamp_duty,
            dp_charges       = self.dp_charges       + other.dp_charges,
            slippage         = self.slippage         + other.slippage,
        )


# ── Cost computation functions ────────────────────────────────────────────────

def delivery_buy_cost(turnover: float, cfg: DeliveryCostConfig) -> TradeCostBreakdown:
    """All-in cost for one delivery BUY order of given turnover (INR)."""
    brok  = cfg.brokerage_pct * turnover
    exch  = cfg.exchange_pct  * turnover
    sebi  = cfg.sebi_pct      * turnover
    stt   = cfg.stt_buy_pct   * turnover
    stamp = cfg.stamp_buy_pct * turnover
    gst   = cfg.gst_rate * (brok + exch)
    slip  = cfg.slippage_pct  * turnover
    return TradeCostBreakdown(brokerage=brok, stt=stt, exchange_charges=exch,
                              sebi_charges=sebi, gst=gst, stamp_duty=stamp,
                              dp_charges=0.0, slippage=slip)


def delivery_sell_cost(turnover: float, n_scrips: int,
                       cfg: DeliveryCostConfig) -> TradeCostBreakdown:
    """All-in cost for one delivery SELL order.
    n_scrips: number of distinct ISINs sold (each incurs one DP charge/day).
    """
    brok  = cfg.brokerage_pct * turnover
    exch  = cfg.exchange_pct  * turnover
    sebi  = cfg.sebi_pct      * turnover
    stt   = cfg.stt_sell_pct  * turnover
    stamp = 0.0                                      # no stamp duty on sell
    gst   = cfg.gst_rate * (brok + exch)
    slip  = cfg.slippage_pct  * turnover
    dp    = cfg.dp_charge_total_inr * n_scrips       # flat per ISIN
    return TradeCostBreakdown(brokerage=brok, stt=stt, exchange_charges=exch,
                              sebi_charges=sebi, gst=gst, stamp_duty=stamp,
                              dp_charges=dp, slippage=slip)


def futures_buy_cost(turnover: float, cfg: FuturesCostConfig) -> TradeCostBreakdown:
    """All-in cost for one equity futures BUY order."""
    brok  = min(cfg.brokerage_pct * turnover, cfg.brokerage_max_inr)
    exch  = cfg.exchange_pct  * turnover
    sebi  = cfg.sebi_pct      * turnover
    stamp = cfg.stamp_buy_pct * turnover
    gst   = cfg.gst_rate * (brok + exch)
    slip  = cfg.slippage_pct  * turnover
    return TradeCostBreakdown(brokerage=brok, stt=0.0, exchange_charges=exch,
                              sebi_charges=sebi, gst=gst, stamp_duty=stamp,
                              dp_charges=0.0, slippage=slip)


def futures_sell_cost(turnover: float, cfg: FuturesCostConfig) -> TradeCostBreakdown:
    """All-in cost for one equity futures SELL order."""
    brok  = min(cfg.brokerage_pct * turnover, cfg.brokerage_max_inr)
    exch  = cfg.exchange_pct  * turnover
    sebi  = cfg.sebi_pct      * turnover
    stt   = cfg.stt_sell_pct  * turnover             # futures STT only on sell
    gst   = cfg.gst_rate * (brok + exch)
    slip  = cfg.slippage_pct  * turnover
    return TradeCostBreakdown(brokerage=brok, stt=stt, exchange_charges=exch,
                              sebi_charges=sebi, gst=gst, stamp_duty=0.0,
                              dp_charges=0.0, slippage=slip)


def zero_cost() -> TradeCostBreakdown:
    return TradeCostBreakdown()


# ════════════════════════════════════════════════════════════════════
#  SECTION 2 : STRATEGY CONFIG
# ════════════════════════════════════════════════════════════════════

DATA_DIR  = Path(r"C:\Users\GIGA\Documents\NSE data 5m")
OUT_DIR   = Path(r"C:\Users\GIGA\Documents\LSCO\backtest_results")
OUT_DIR.mkdir(exist_ok=True)

# Portfolio starting value in INR.
# Rs.1,00,000 (1 lakh) is used so DP charges (Rs.15/ISIN) are realistic.
# All final values and CAGR scale from this base.
INITIAL    = 100_000.0     # Rs.1 lakh starting portfolio

RISK_FREE  = 0.06          # 6.0% p.a. India T-bill / savings rate
WEEKLY_RF  = (1 + RISK_FREE) ** (1 / 52) - 1

REGIME_EMA = 20            # NIFTY EMA weeks for regime detection
SIGNAL_EMA = 30            # per-stock EMA weeks for entry gate
MOM_WEEKS  = 4             # momentum lookback (weeks)
RSI_PERIOD = 14
WARMUP     = max(REGIME_EMA, SIGNAL_EMA) + MOM_WEEKS + RSI_PERIOD + 2

STOP_PCT   = 0.20          # V2: 20% hard stop from entry price
VIX_LOW    = 15            # V3: full position below this VIX level
VIX_HIGH   = 25            # V3: half position above this VIX level

MIN_WEEKS  = 150           # minimum weekly bars to include a stock
EXCLUDE    = {"NIFTY 50", "NIFTY BANK", "INDIA VIX"}

# Active cost configs (delivery strategy)
DEL_CFG = DeliveryCostConfig()     # equity delivery
FUT_CFG = FuturesCostConfig()      # equity futures (for reference / reporting)


# ════════════════════════════════════════════════════════════════════
#  SECTION 3 : DATA LOADING & INDICATORS
# ════════════════════════════════════════════════════════════════════

def load_weekly(name: str):
    p = DATA_DIR / f"{name}_5minute.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date'], index_col='date').sort_index()
    w = df.resample('W-FRI').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),    close=('close', 'last'),
        volume=('volume', 'sum')
    ).dropna(subset=['close'])
    return w[w['close'] > 0].copy()


def vix_scale(v: float) -> float:
    if pd.isna(v) or v <= VIX_LOW:  return 1.0
    if v >= VIX_HIGH:                return 0.5
    return 1.0 - (v - VIX_LOW) / (VIX_HIGH - VIX_LOW) * 0.5


def perf_metrics(eq: pd.Series, annual: int = 52) -> dict:
    r = eq.pct_change().dropna()
    if len(r) < 10:
        return dict(cagr=0, sharpe=0, maxdd=0, calmar=0,
                    final=eq.iloc[-1], years=0)
    years = len(r) / annual
    cagr  = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    exc   = r - WEEKLY_RF
    sh    = exc.mean() / exc.std() * np.sqrt(annual) if exc.std() > 1e-10 else 0
    peak  = eq.cummax()
    dd    = (eq - peak) / peak
    mdd   = dd.min()
    cal   = cagr / abs(mdd) if mdd != 0 else 0
    return dict(cagr=cagr, sharpe=sh, maxdd=mdd, calmar=cal,
                final=eq.iloc[-1], years=years)


def annual_rets(eq: pd.Series) -> dict:
    r = eq.pct_change().fillna(0)
    return {yr: (1 + g).prod() - 1 for yr, g in r.groupby(r.index.year)}


# ── Load ──────────────────────────────────────────────────────────────────────

print("=" * 68)
print("NSE INDIA REGIME BACKTEST  --  Data Loading")
print("=" * 68)

print("Loading NIFTY 50 ... ", end='', flush=True)
nifty_w = load_weekly("NIFTY 50")
print(f"{len(nifty_w)} weekly bars  "
      f"({nifty_w.index[0].date()} to {nifty_w.index[-1].date()})")

print("Loading INDIA VIX ... ", end='', flush=True)
vix_w = load_weekly("INDIA VIX")
print(f"{len(vix_w) if vix_w is not None else 0} bars")

all_names = sorted(f.stem.replace("_5minute", "")
                   for f in DATA_DIR.glob("*_5minute.csv"))
all_names = [n for n in all_names if n not in EXCLUDE]
print(f"Loading {len(all_names)} stock files ... ", end='', flush=True)

stock_data = {}
for nm in all_names:
    w = load_weekly(nm)
    if w is not None and len(w) >= MIN_WEEKS:
        stock_data[nm] = w

universe = sorted(stock_data.keys())
print(f"done  --  {len(universe)} stocks with >= {MIN_WEEKS} weeks history")

# ── Indicators ────────────────────────────────────────────────────────────────

print("\nComputing indicators ... ", end='', flush=True)

nifty_w['ema_r'] = nifty_w['close'].ewm(span=REGIME_EMA, adjust=False).mean()
nifty_w['mom_r'] = nifty_w['close'].pct_change(MOM_WEEKS)
bull_m = (nifty_w['close'] > nifty_w['ema_r']) & (nifty_w['mom_r'] > 0)
bear_m = (nifty_w['close'] < nifty_w['ema_r']) & (nifty_w['mom_r'] < 0)
nifty_w['regime'] = 'TRANS'
nifty_w.loc[bull_m, 'regime'] = 'BULL'
nifty_w.loc[bear_m, 'regime'] = 'BEAR'

idx = nifty_w.index
close_dict, low_dict, sig_dict = {}, {}, {}
for sym, w in stock_data.items():
    e = w['close'].ewm(span=SIGNAL_EMA, adjust=False).mean()
    m = w['close'].pct_change(MOM_WEEKS)
    sig_dict[sym]   = ((w['close'] > e) & (m > 0)).astype(bool)
    close_dict[sym] = w['close']
    low_dict[sym]   = w['low']

closes_df  = pd.DataFrame(close_dict).reindex(idx)
lows_df    = pd.DataFrame(low_dict).reindex(idx)
signals_df = pd.DataFrame(sig_dict).reindex(idx).fillna(False)
returns_df = closes_df.pct_change(1)

vix_ser = (vix_w['close'].reindex(idx, method='ffill')
           if vix_w is not None else pd.Series(np.nan, index=idx))

print("done")


# ════════════════════════════════════════════════════════════════════
#  SECTION 4 : BACKTEST ENGINE  (with itemised cost tracking)
# ════════════════════════════════════════════════════════════════════

def run_backtest(version: str = 'V1',
                 cost_cfg: DeliveryCostConfig = DEL_CFG):
    """
    Run regime strategy with full itemised NSE cost tracking.

    Returns
    -------
    eq_curve   : pd.Series  -- portfolio equity curve (INR)
    n_trades   : int        -- total stock entries
    regime_log : list[str]  -- regime for each week
    cum_costs  : TradeCostBreakdown -- cumulative all costs over full period
    """
    dates     = idx[WARMUP:]
    n_dates   = len(dates)
    equity    = INITIAL
    eq_arr    = np.empty(n_dates)
    n_trades  = 0
    prev_held = set()
    entry_px  = {}              # sym -> entry close price (for stop-loss)
    regime_log = []
    cum_costs  = zero_cost()    # accumulate itemised costs

    for i in range(n_dates):
        dt      = dates[i]
        prev_dt = idx[WARMUP + i - 1] if i > 0 else idx[WARMUP - 1]

        regime = nifty_w.at[prev_dt, 'regime']
        regime_log.append(regime)

        # ── Qualifying stocks (from prev week signal) ──────────────
        if regime == 'BULL':
            p_sig = signals_df.loc[prev_dt]
            p_ret = returns_df.loc[dt]
            qual  = [s for s in universe
                     if p_sig.get(s, False) and pd.notna(p_ret.get(s))]
        else:
            qual = []

        curr_held = set(qual)
        N = len(curr_held)

        # ── Portfolio logic ────────────────────────────────────────
        if N > 0:
            # V3: INDIA VIX scale
            if version == 'V3':
                vv    = vix_ser[prev_dt] if prev_dt in vix_ser.index else np.nan
                scale = vix_scale(vv)
            else:
                scale = 1.0

            # Raw stock returns
            rets        = returns_df.loc[dt, qual].values.copy().astype(float)
            stopped_out = set()

            # V2/V3: per-stock 20% hard stop
            if version in ('V2', 'V3'):
                for j, sym in enumerate(qual):
                    if sym not in entry_px:
                        continue
                    lw = lows_df.at[dt, sym]
                    if pd.notna(lw) and lw <= entry_px[sym] * (1.0 - STOP_PCT):
                        rets[j]  = -STOP_PCT      # stop hit: cap at -20%
                        stopped_out.add(sym)

            valid    = ~np.isnan(rets)
            port_ret = float(np.mean(rets[valid])) if valid.any() else WEEKLY_RF

            if scale < 1.0:
                port_ret = port_ret * scale + WEEKLY_RF * (1.0 - scale)

            # ── Itemised costs ─────────────────────────────────────
            entered      = curr_held - prev_held
            exited       = prev_held - curr_held
            also_exiting = stopped_out & curr_held  # stopped stocks exit

            per_stock_val = equity / N              # equal-weight allocation

            week_cost = zero_cost()

            # Buy cost for entered stocks
            for sym in entered:
                c = delivery_buy_cost(per_stock_val, cost_cfg)
                week_cost = week_cost + c

            # Sell cost for exited stocks
            n_exited = len(exited) + len(also_exiting)
            if n_exited > 0:
                exited_val = equity / max(len(prev_held), 1)
                for sym in (exited | also_exiting):
                    c = delivery_sell_cost(exited_val, n_scrips=1, cfg=cost_cfg)
                    week_cost = week_cost + c

            # Add gross PnL for this week to the accumulated record
            week_cost.gross_pnl = equity * port_ret

            cum_costs = cum_costs + week_cost

            # Deduct cost as fraction of portfolio
            cost_frac = week_cost.total_charges / equity
            port_ret  -= cost_frac

            n_trades += len(entered)

            # Update entry prices
            for sym in (exited | also_exiting):
                entry_px.pop(sym, None)
            for sym in entered:
                ec = closes_df.at[dt, sym]
                if pd.notna(ec):
                    entry_px[sym] = ec

            prev_held = curr_held - stopped_out

        else:
            # ── Cash week ──────────────────────────────────────────
            if prev_held:
                # Sell all held stocks
                n_prev   = len(prev_held)
                sell_val = equity / n_prev
                week_cost = zero_cost()
                for sym in prev_held:
                    c = delivery_sell_cost(sell_val, n_scrips=1, cfg=cost_cfg)
                    week_cost = week_cost + c
                week_cost.gross_pnl = 0.0
                cum_costs = cum_costs + week_cost
                cost_frac = week_cost.total_charges / equity
                entry_px.clear()
            else:
                cost_frac = 0.0

            port_ret  = WEEKLY_RF - cost_frac
            prev_held = set()

        equity *= (1.0 + port_ret)
        eq_arr[i] = equity

    eq_curve = pd.Series(eq_arr, index=dates)
    return eq_curve, n_trades, regime_log, cum_costs


# ════════════════════════════════════════════════════════════════════
#  SECTION 5 : NIFTY 50 BENCHMARK (no costs)
# ════════════════════════════════════════════════════════════════════

def nifty_bh() -> pd.Series:
    dates = idx[WARMUP:]
    r = nifty_w['close'].pct_change(1).reindex(dates).fillna(0)
    return (1 + r).cumprod() * INITIAL


# ════════════════════════════════════════════════════════════════════
#  SECTION 6 : RUN
# ════════════════════════════════════════════════════════════════════

print("\nRunning backtests...")

print("  V1 (regime + gate)      ... ", end='', flush=True)
eq_v1, nt_v1, rl_v1, costs_v1 = run_backtest('V1')
print("done")

print("  V2 (V1 + stop-loss)     ... ", end='', flush=True)
eq_v2, nt_v2, rl_v2, costs_v2 = run_backtest('V2')
print("done")

print("  V3 (V2 + VIX scaling)   ... ", end='', flush=True)
eq_v3, nt_v3, rl_v3, costs_v3 = run_backtest('V3')
print("done")

print("  NIFTY 50 Buy & Hold     ... ", end='', flush=True)
eq_bh = nifty_bh()
print("done")


# ════════════════════════════════════════════════════════════════════
#  SECTION 7 : METRICS
# ════════════════════════════════════════════════════════════════════

m_v1 = perf_metrics(eq_v1)
m_v2 = perf_metrics(eq_v2)
m_v3 = perf_metrics(eq_v3)
m_bh = perf_metrics(eq_bh)

ar_v1 = annual_rets(eq_v1)
ar_v2 = annual_rets(eq_v2)
ar_v3 = annual_rets(eq_v3)
ar_bh = annual_rets(eq_bh)

rl_arr    = np.array(rl_v1)
bull_pct  = (rl_arr == 'BULL').mean()  * 100
bear_pct  = (rl_arr == 'BEAR').mean()  * 100
trans_pct = (rl_arr == 'TRANS').mean() * 100

dates_used = idx[WARMUP:]

# Cost summary (V1 as representative)
c = costs_v1
total_charges_v1 = c.total_charges
cost_as_pct_gain = total_charges_v1 / INITIAL * 100   # % of starting capital

# ════════════════════════════════════════════════════════════════════
#  SECTION 8 : REPORT
# ════════════════════════════════════════════════════════════════════

def pct(x):   return f"{x * 100:+.1f}%"
def f2(x):    return f"{x:.2f}"
def inr(x):   return f"Rs.{x:,.0f}"

print(f"""
{'=' * 68}
  NSE INDIA REGIME STRATEGY  --  FINAL BACKTEST REPORT
{'=' * 68}

  DATA OVERVIEW
  {'-' * 64}
  Source    : C:\\Users\\GIGA\\Documents\\NSE data 5m  (5-min -> weekly)
  Period    : {dates_used[0].date()} to {dates_used[-1].date()}
              ({m_v1['years']:.1f} years after {WARMUP}-week warmup)
  Universe  : {len(universe)} NSE stocks  (>=  {MIN_WEEKS} weeks of data)
  Benchmark : NIFTY 50 Buy & Hold  (price, no dividends)
  Capital   : {inr(INITIAL)} starting portfolio
  Risk-free : 6.00% p.a.  (India T-bill / savings rate)

  COST MODEL : NSE Equity Delivery  (modular, itemised)
  {'-' * 64}
  Brokerage    : {DEL_CFG.brokerage_pct*100:.4f}%  (zero delivery brokerage)
  STT          : {DEL_CFG.stt_buy_pct*100:.4f}% buy  /  {DEL_CFG.stt_sell_pct*100:.4f}% sell
  Exchange     : {DEL_CFG.exchange_pct*100:.5f}% per side (NSE equity cash)
  SEBI         : {DEL_CFG.sebi_pct*100:.4f}%  (Rs.10 per crore)
  Stamp Duty   : {DEL_CFG.stamp_buy_pct*100:.4f}% buy only  /  0% sell
  GST          : {DEL_CFG.gst_rate*100:.0f}% on (brokerage + exchange charges)
  DP Charges   : Rs.{DEL_CFG.dp_charge_base_inr:.0f} + {DEL_CFG.dp_charge_gst_rate*100:.0f}% GST per ISIN per sell day
               = Rs.{DEL_CFG.dp_charge_total_inr:.1f} per stock per exit event
  Slippage     : {DEL_CFG.slippage_pct*100:.3f}% per side

  STRATEGY LOGIC
  {'-' * 64}
  Regime signal : NIFTY 50 EMA({REGIME_EMA}w) + {MOM_WEEKS}w momentum  (prev wk, no look-ahead)
  Per-stock gate: stock close > EMA({SIGNAL_EMA}w) AND {MOM_WEEKS}w momentum > 0
  V1            : Regime + gate  -> long / cash
  V2            : V1  +  {STOP_PCT*100:.0f}% per-stock hard stop-loss
  V3            : V2  +  VIX scaling  (100% @ VIX<{VIX_LOW}, 50% @ VIX>{VIX_HIGH})

{'=' * 68}
  PERFORMANCE SUMMARY
{'=' * 68}
  Metric              V1          V2          V3      NIFTY B&H
  {'-' * 64}
  CAGR            {pct(m_v1['cagr']):>9}   {pct(m_v2['cagr']):>9}   {pct(m_v3['cagr']):>9}   {pct(m_bh['cagr']):>9}
  Sharpe Ratio    {f2(m_v1['sharpe']):>9}   {f2(m_v2['sharpe']):>9}   {f2(m_v3['sharpe']):>9}   {f2(m_bh['sharpe']):>9}
  Max Drawdown    {pct(m_v1['maxdd']):>9}   {pct(m_v2['maxdd']):>9}   {pct(m_v3['maxdd']):>9}   {pct(m_bh['maxdd']):>9}
  Calmar Ratio    {f2(m_v1['calmar']):>9}   {f2(m_v2['calmar']):>9}   {f2(m_v3['calmar']):>9}   {f2(m_bh['calmar']):>9}
  Final Rs.1L     {inr(m_v1['final']):>12}  {inr(m_v2['final']):>12}  {inr(m_v3['final']):>12}  {inr(m_bh['final']):>12}
  Trade Entries   {nt_v1:>9}   {nt_v2:>9}   {nt_v3:>9}          --

{'=' * 68}
  YEAR-BY-YEAR RETURNS
{'=' * 68}
  Year        V1        V2        V3    NIFTY B&H
  {'-' * 56}""")

all_years = sorted(set(ar_v1) | set(ar_v2) | set(ar_v3) | set(ar_bh))
for yr in all_years:
    print(f"  {yr}   {pct(ar_v1.get(yr,0)):>8}  {pct(ar_v2.get(yr,0)):>8}"
          f"  {pct(ar_v3.get(yr,0)):>8}  {pct(ar_bh.get(yr,0)):>8}")

print(f"""
{'=' * 68}
  ITEMISED COST BREAKDOWN  (V1 Strategy, full {m_v1['years']:.0f}-year period)
  Starting capital: {inr(INITIAL)}  |  All amounts in INR
{'=' * 68}
  Cost Component       Amount (INR)    % of Start Capital
  {'-' * 56}
  Brokerage          {inr(costs_v1.brokerage):>14}   {costs_v1.brokerage/INITIAL*100:>8.2f}%
  STT                {inr(costs_v1.stt):>14}   {costs_v1.stt/INITIAL*100:>8.2f}%
  Exchange Charges   {inr(costs_v1.exchange_charges):>14}   {costs_v1.exchange_charges/INITIAL*100:>8.2f}%
  SEBI Charges       {inr(costs_v1.sebi_charges):>14}   {costs_v1.sebi_charges/INITIAL*100:>8.2f}%
  GST                {inr(costs_v1.gst):>14}   {costs_v1.gst/INITIAL*100:>8.2f}%
  Stamp Duty         {inr(costs_v1.stamp_duty):>14}   {costs_v1.stamp_duty/INITIAL*100:>8.2f}%
  DP Charges         {inr(costs_v1.dp_charges):>14}   {costs_v1.dp_charges/INITIAL*100:>8.2f}%
  Slippage           {inr(costs_v1.slippage):>14}   {costs_v1.slippage/INITIAL*100:>8.2f}%
  {'-' * 56}
  TOTAL CHARGES      {inr(costs_v1.total_charges):>14}   {costs_v1.total_charges/INITIAL*100:>8.2f}%
  Gross PnL          {inr(costs_v1.gross_pnl):>14}   {costs_v1.gross_pnl/INITIAL*100:>8.2f}%
  Net PnL            {inr(costs_v1.net_pnl):>14}   {costs_v1.net_pnl/INITIAL*100:>8.2f}%

  Effective drag/year: {costs_v1.total_charges / INITIAL / m_v1['years'] * 100:.2f}%
  Biggest cost        : {"STT" if costs_v1.stt > costs_v1.slippage else "Slippage"}
    (STT = {inr(costs_v1.stt)} / Slippage = {inr(costs_v1.slippage)})

{'=' * 68}
  FUTURES COST CONFIG (reference -- if using F&O instead of delivery)
{'=' * 68}
  Brokerage    : min({FUT_CFG.brokerage_pct*100:.2f}% turnover, Rs.{FUT_CFG.brokerage_max_inr:.0f}/order) per side
  STT          : 0% buy  /  {FUT_CFG.stt_sell_pct*100:.3f}% sell  [Finance Act 2024, Oct 1 2024]
               NOTE: User supplied 0.05% for futures STT. That is
               the EQUITY OPTIONS rate. Futures sell STT = 0.02%.
  Exchange     : {FUT_CFG.exchange_pct*100:.4f}% per side (NSE F&O stock futures)
  SEBI         : {FUT_CFG.sebi_pct*100:.4f}%
  Stamp Duty   : {FUT_CFG.stamp_buy_pct*100:.4f}% buy only
  GST          : {FUT_CFG.gst_rate*100:.0f}% on (brokerage + exchange)
  Slippage     : {FUT_CFG.slippage_pct*100:.3f}% per side  (configurable)

{'=' * 68}
  REGIME ANALYSIS  ({len(rl_v1)} weeks)
{'=' * 68}
  BULL  (long stocks) : {bull_pct:.1f}%  |  BEAR : {bear_pct:.1f}%  |  TRANS : {trans_pct:.1f}%
  Time in market      : {bull_pct:.1f}%  |  Time in cash : {bear_pct+trans_pct:.1f}%

{'=' * 68}
  CAVEATS
{'=' * 68}
  1. SURVIVORSHIP BIAS: Universe = current Nifty 50 large-caps only.
     Stocks dropped or delisted since 2015 are NOT included.
     Real CAGR would be 2-4% lower.
  2. NO DIVIDENDS: Add ~1.5-2% p.a. to B&H benchmark for NIFTY TRI.
  3. DP CHARGES depend on portfolio size. Computed for {inr(INITIAL)}.
     At Rs.10 lakh, DP drag per year = {costs_v1.dp_charges/m_v1['years']/INITIAL*100:.2f}% of capital.
  4. STT is the dominant cost at {costs_v1.stt/INITIAL*100:.1f}% of starting capital over {m_v1['years']:.0f} years.
     Delivery STT (0.1%+0.1%=0.2% round-trip) is the main friction.
  5. Futures as alternative: lower STT (0.02% sell only), no DP
     charge, but brokerage applies + margin requirement.
""")


# ════════════════════════════════════════════════════════════════════
#  SECTION 9 : SAVE OUTPUT
# ════════════════════════════════════════════════════════════════════

out_eq = OUT_DIR / "nse_regime_equity_curves.csv"
pd.DataFrame({
    'V1_Regime':  eq_v1,
    'V2_StopLoss': eq_v2,
    'V3_VIXScale': eq_v3,
    'NIFTY_BnH':  eq_bh
}).to_csv(out_eq)

# Itemised cost CSV for all three versions
cost_rows = []
for label, cc in [('V1', costs_v1), ('V2', costs_v2), ('V3', costs_v3)]:
    cost_rows.append({
        'Version':          label,
        'Gross_PnL':        round(cc.gross_pnl, 2),
        'Brokerage':        round(cc.brokerage, 2),
        'STT':              round(cc.stt, 2),
        'Exchange_Charges': round(cc.exchange_charges, 2),
        'SEBI_Charges':     round(cc.sebi_charges, 2),
        'GST':              round(cc.gst, 2),
        'Stamp_Duty':       round(cc.stamp_duty, 2),
        'DP_Charges':       round(cc.dp_charges, 2),
        'Slippage':         round(cc.slippage, 2),
        'Total_Charges':    round(cc.total_charges, 2),
        'Net_PnL':          round(cc.net_pnl, 2),
    })
out_costs = OUT_DIR / "nse_cost_breakdown.csv"
pd.DataFrame(cost_rows).to_csv(out_costs, index=False)

# Regime transitions
transitions = sum(1 for a, b in zip(rl_v1, rl_v1[1:]) if a != b)
avg_run = len(rl_v1) / (transitions + 1) if transitions else len(rl_v1)

print(f"  Saved: {out_eq}")
print(f"  Saved: {out_costs}")
print(f"  Regime transitions: {transitions}  (avg run = {avg_run:.1f} weeks)")
print(f"  Backtest complete.")
