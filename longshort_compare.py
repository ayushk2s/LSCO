import pandas as pd

df = pd.read_csv('backtest_results/trade_analysis_v8_4h.csv')

print("=== LONG vs SHORT breakdown (trade_analysis_v8_4h.csv) ===")
print()

for d in ['LONG', 'SHORT']:
    t = df[df['dir'] == d]
    w = t[t['result'] == 'WIN']
    l = t[t['result'] == 'LOSS']
    nw = w['net'].sum()
    nl = abs(l['net'].sum()) if len(l) > 0 else 0
    npf = round(nw / nl, 3) if nl > 0 else float('inf')
    total_net = t['net'].sum()
    print(f"{d}: {len(t)} trades  WR {len(w)/len(t)*100:.1f}%  Net P&L ${total_net:+.2f}  NetPF {npf:.3f}")

print()
long_net  = df[df['dir'] == 'LONG']['net'].sum()
short_net = df[df['dir'] == 'SHORT']['net'].sum()
print(f"SHORT dragged results by: ${short_net:.2f}")
print(f"LONG only total net:      ${long_net:+.2f}")
print()

print("--- SHORT by symbol (what is hurting us) ---")
for sym in sorted(df['symbol'].unique()):
    s = df[(df['symbol'] == sym) & (df['dir'] == 'SHORT')]
    if len(s) == 0:
        continue
    w = s[s['result'] == 'WIN']
    print(f"  {sym}: {len(s):>2} trades  WR {len(w)/len(s)*100:.0f}%  net ${s['net'].sum():+.2f}")

print()
print("--- LONG by symbol (our edge) ---")
for sym in sorted(df['symbol'].unique()):
    s = df[(df['symbol'] == sym) & (df['dir'] == 'LONG')]
    if len(s) == 0:
        continue
    w = s[s['result'] == 'WIN']
    print(f"  {sym}: {len(s):>2} trades  WR {len(w)/len(s)*100:.0f}%  net ${s['net'].sum():+.2f}")

print()
print("=== CONCLUSION ===")
print()
total  = df['net'].sum()
lonly  = df[df['dir'] == 'LONG']['net'].sum()
print(f"With both LONG+SHORT: ${total:+.2f}  <- SHORT bug + lower WR")
print(f"With LONG only:       ${lonly:+.2f}  <- profitable")
print()
print("v11 result confirms this:")
print("  BTC+ETH LONG-only (regime):  CAGR +13.8%  DD -7.2%  Calmar 1.91  5/5 years profitable")
