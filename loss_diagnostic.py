
import pandas as pd
import numpy as np

df = pd.read_csv('backtest_results/trade_analysis_v8_4h.csv')

print("=== WHY WE ARE LOSING - DEEP DIVE ===")
print()

losses = df[df['result']=='LOSS'].copy()
wins   = df[df['result']=='WIN'].copy()

# Deep vs shallow MAE
deep   = df[df['mae_atr'] > 0.6]
shallow = df[df['mae_atr'] <= 0.6]

print(f"Shallow (MAE <=0.6 ATR): {len(shallow)} trades -> WR {shallow['result'].eq('WIN').mean()*100:.0f}%")
print(f"Deep    (MAE > 0.6 ATR): {len(deep)} trades -> WR {deep['result'].eq('WIN').mean()*100:.0f}%")
print()

# What features predict deep vs shallow AT ENTRY TIME?
print("--- WHAT PREDICTS A DEEP LOSS? ---")
feat = ['vol_ratio','ema_slope_pct','atr_pct','zone_touch']
for f in feat:
    if f not in df.columns:
        continue
    sm = shallow[f].mean()
    dm = deep[f].mean()
    diff_pct = (dm - sm) / abs(sm) * 100 if sm != 0 else 0
    print(f"  {f:<22}  shallow {sm:>8.4f}  deep {dm:>8.4f}  diff {diff_pct:>+.0f}%")
print()

# Symbol deep loss stats
print("--- DEEP LOSSES BY SYMBOL ---")
for sym in sorted(df['symbol'].unique()):
    s = df[df['symbol']==sym]
    d = s[s['mae_atr'] > 0.6]
    n_deep_loss = len(d[d['result']=='LOSS'])
    print(f"  {sym}: {len(d)}/{len(s)} trades went deep ({len(d)/len(s)*100:.0f}%) -> {n_deep_loss}/{len(d)} losses")
print()

# EMA slope split for LONG trades
print("--- LONG trades: EMA slope vs outcome ---")
longs = df[df['dir']=='LONG']
pos_slope = longs[longs['ema_slope_pct'] > 0]
neg_slope = longs[longs['ema_slope_pct'] <= 0]
print(f"  EMA slope POSITIVE (rising): {len(pos_slope)} trades  WR {pos_slope['result'].eq('WIN').mean()*100:.0f}%")
print(f"  EMA slope NEGATIVE (falling): {len(neg_slope)} trades  WR {neg_slope['result'].eq('WIN').mean()*100:.0f}%")
print()

# Stop hunt survivors
stop_hunts = df[(df['mae_atr']>=0.6) & (df['mae_atr']<0.8) & (df['result']=='WIN')]
print(f"--- STOP HUNT SURVIVORS (MAE 0.6-0.8 ATR but WON): {len(stop_hunts)} trades ---")
if len(stop_hunts) > 0:
    print(f"  avg mfe_atr  {stop_hunts['mfe_atr'].mean():.2f}  (how far they ran after)")
    print(f"  avg dur_h    {stop_hunts['duration_h'].mean():.1f}h")
    print(f"  symbols: {stop_hunts['symbol'].value_counts().to_dict()}")
    print(f"  vol_ratio avg: {stop_hunts['vol_ratio'].mean():.2f}")
    print(f"  ema_slope avg: {stop_hunts['ema_slope_pct'].mean():.4f}")
print()

# Year-by-year
df['year'] = pd.to_datetime(df['ts']).dt.year
print("--- YEAR BY YEAR (loss clustering) ---")
for yr in sorted(df['year'].unique()):
    s = df[df['year']==yr]
    l = s[s['result']=='LOSS']
    yr_longs = s[s['dir']=='LONG']
    yr_shorts = s[s['dir']=='SHORT']
    lwr = yr_longs['result'].eq('WIN').mean()*100 if len(yr_longs)>0 else 0
    swr = yr_shorts['result'].eq('WIN').mean()*100 if len(yr_shorts)>0 else 0
    print(f"  {yr}: {len(s):>3} trades  {len(l):>2} losses  WR {(1-len(l)/len(s))*100:.0f}%  long WR {lwr:.0f}%  short WR {swr:.0f}%")
print()

# Summary: the 3 root causes
print("=== SUMMARY: ROOT CAUSES OF LOSSES ===")
print()
print("CAUSE 1 - BEAR MARKET / DOWNTREND (EMA slope negative)")
bear = df[df['ema_slope_pct'] < -0.005]
bull = df[df['ema_slope_pct'] >= -0.005]
print(f"  Strong downtrend (slope < -0.005%): {len(bear)} trades  WR {bear['result'].eq('WIN').mean()*100:.0f}%")
print(f"  Not downtrend   (slope >= -0.005%): {len(bull)} trades  WR {bull['result'].eq('WIN').mean()*100:.0f}%")
print()

print("CAUSE 2 - SHORT DIRECTION (entering short support zones)")
shorts = df[df['dir']=='SHORT']
longs2 = df[df['dir']=='LONG']
print(f"  SHORT: {len(shorts)} trades  WR {shorts['result'].eq('WIN').mean()*100:.0f}%")
print(f"  LONG:  {len(longs2)} trades  WR {longs2['result'].eq('WIN').mean()*100:.0f}%")
print()

print("CAUSE 3 - ZONE RE-TESTED TOO MANY TIMES")
t2 = df[df['zone_touch']==2]
t3 = df[df['zone_touch']>=3]
print(f"  Touch #2 (1st re-test): {len(t2)} trades  WR {t2['result'].eq('WIN').mean()*100:.0f}%")
print(f"  Touch #3+ (2nd+ re-test): {len(t3)} trades  WR {t3['result'].eq('WIN').mean()*100:.0f}%")
print()

print("CAUSE 4 - HIGH VOLATILITY at entry (vol_ratio)")
low_vol  = df[df['vol_ratio'] < 2.0]
high_vol = df[df['vol_ratio'] >= 3.5]
print(f"  Low vol  (<2.0x ATR spike): {len(low_vol)} trades  WR {low_vol['result'].eq('WIN').mean()*100:.0f}%")
print(f"  High vol (>3.5x ATR spike): {len(high_vol)} trades  WR {high_vol['result'].eq('WIN').mean()*100:.0f}%")
print()

# Winning vs losing entry profile
print("--- WIN vs LOSS entry profile ---")
for col in ['vol_ratio','ema_slope_pct','atr_pct','zone_touch']:
    if col in df.columns:
        wv = wins[col].mean()
        lv = losses[col].mean()
        print(f"  {col:<22}  WIN {wv:>8.4f}  LOSS {lv:>8.4f}")
print()

# ATR% analysis - is LOSS more common on high-volatility candles?
if 'atr_pct' in df.columns:
    print("--- atr_pct buckets (candle volatility at entry) ---")
    for lo, hi in [(0,0.5),(0.5,1.0),(1.0,2.0),(2.0,5.0)]:
        s = df[(df['atr_pct']>=lo) & (df['atr_pct']<hi)]
        if len(s) > 0:
            print(f"  ATR {lo:.1f}-{hi:.1f}%: {len(s):>3} trades  WR {s['result'].eq('WIN').mean()*100:.0f}%  "
                  f"avg_mae {s['mae_atr'].mean():.3f}")
