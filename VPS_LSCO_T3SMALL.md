# LSCO VPS — T3.Small Reference Guide

## Server Details

| Field         | Value                              |
|---------------|------------------------------------|
| Provider      | AWS EC2                            |
| Instance      | t3.small                           |
| Region        | ap-northeast-1 (Tokyo)             |
| Public IP     | 13.112.47.16                       |
| OS            | Ubuntu 24.04.4 LTS                 |
| Kernel        | 6.17.0-1007-aws                    |
| CPU           | Intel Xeon Platinum 8259CL @ 2.50 GHz |
| vCPUs         | 2 (1 core × 2 threads)             |
| RAM           | 2 GB total                         |
| Disk          | 20 GB (3.8 GB used / 15 GB free)   |
| Swap          | 1 GB (/swapfile)                   |
| SSH User      | ubuntu                             |
| Key File      | AYUSH.LSCO.T3SMALL.ppk             |

---

## SSH Connection

### From Windows (PuTTY)
```
Host   : 13.112.47.16
Port   : 22
User   : ubuntu
Key    : AYUSH.LSCO.T3SMALL.ppk
```

### From Terminal / PowerShell (after converting PPK to PEM)
```bash
ssh -i AYUSH.LSCO.T3SMALL.pem ubuntu@13.112.47.16
```

---

## Directory Structure

```
/home/ubuntu/
└── LSCO/
    ├── asterdex_trade/
    │   ├── lsco.py              ← MAIN ENGINE (multi-asset, runs BTC+ETH+XAU)
    │   ├── account_data.py      ← AsterDEX API client (EIP-712 signing)
    │   ├── market_data.py       ← Price, klines, order book, ATR
    │   └── order_executor.py    ← SmartLimitOrder (partial fill tracking)
    ├── data_fetching/
    │   └── binance_liq_heatmap_2.py  ← Liquidation heatmap fetcher (--symbol arg)
    ├── run_lsco.sh              ← Manual run script (logs to lsco.log)
    ├── lsco.log                 ← Live bot output log
    ├── trade_log.json           ← All trades across all symbols (auto-created)
    ├── algo_state_v2_BTCUSDT.json   ← BTC engine state (auto-created)
    ├── algo_state_v2_ETHUSDT.json   ← ETH engine state (auto-created)
    ├── algo_state_v2_XAUUSDT.json   ← XAU engine state (auto-created)
    ├── whale_BTCUSDT.json       ← BTC whale data (written by whale_monitor)
    ├── whale_ETHUSDT.json       ← ETH whale data
    └── whale_XAUUSDT.json       ← XAU whale data
```

---

## Python Packages Installed

| Package       | How Installed   | Used By                          |
|---------------|-----------------|----------------------------------|
| requests      | apt             | market_data.py, account_data.py  |
| numpy         | apt             | binance_liq_heatmap_2.py         |
| matplotlib    | apt             | binance_liq_heatmap_2.py (Agg)   |
| eth-account   | pip3            | account_data.py (EIP-712 signing)|
| Python 3.12.3 | system          | runtime                          |

---

## LSCO Engine — What It Runs

Three parallel async engines, one per symbol:

| Symbol  | Leverage | Min Qty | Price Round | Min Zone   |
|---------|----------|---------|-------------|------------|
| BTCUSDT | 20x      | 0.001   | 1 decimal   | $10M       |
| ETHUSDT | 20x      | 0.001   | 2 decimals  | $2M        |
| XAUUSDT | 10x      | 0.01    | 2 decimals  | $500K      |

- **Sizing**: 1% of free balance, hard cap $2 USDT (minimum lots, testing phase)
- **Trigger**: Touch + rejection wick on liquidation zone
- **Exit**: TP = 1.2× ATR, SL = 0.60× ATR (2:1 R:R)
- **Heatmap**: Auto-refreshes every 5 min per symbol (staggered 30s apart)
- **State**: Saved to `algo_state_v2_{symbol}.json` after every tick
- **Recovery**: On restart, resumes open positions from state file

---

## Systemd Service Commands

### Start the bot
```bash
sudo systemctl start lsco
```

### Stop the bot
```bash
sudo systemctl stop lsco
```

### Restart the bot
```bash
sudo systemctl restart lsco
```

### Check status
```bash
sudo systemctl status lsco
```

### Enable auto-start on reboot (already done)
```bash
sudo systemctl enable lsco
```

### Disable auto-start
```bash
sudo systemctl disable lsco
```

---

## Logs

### Watch live log (most important)
```bash
tail -f ~/LSCO/lsco.log
```

### Last 100 lines
```bash
tail -100 ~/LSCO/lsco.log
```

### Search for trades only
```bash
grep -E "\[log\]|WIN|LOSS|TRIGGER" ~/LSCO/lsco.log
```

### Search for errors
```bash
grep -iE "error|traceback|failed|exception" ~/LSCO/lsco.log
```

### View trade history (JSON)
```bash
cat ~/LSCO/trade_log.json | python3 -m json.tool
```

### View current engine state
```bash
cat ~/LSCO/algo_state_v2_BTCUSDT.json
cat ~/LSCO/algo_state_v2_ETHUSDT.json
cat ~/LSCO/algo_state_v2_XAUUSDT.json
```

---

## Whale Monitor (Optional — run separately)

The whale monitor is NOT part of the systemd service. Run it in 3 separate terminal
sessions or add separate systemd units for each symbol.

```bash
# Terminal 1 — BTC
python3 ~/whale_monitor.py --symbol BTCUSDT

# Terminal 2 — ETH
python3 ~/whale_monitor.py --symbol ETHUSDT

# Terminal 3 — XAU
python3 ~/whale_monitor.py --symbol XAUUSDT
```

Without whale monitor running, the engine still works — it just scores confidence
slightly lower (no whale bonus). All trades still execute normally.

---

## System Monitoring

### Check RAM usage
```bash
free -h
```

### Check disk space
```bash
df -h /
```

### Check CPU load
```bash
uptime
```

### Check what's running
```bash
ps aux | grep python
```

### Check swap usage
```bash
swapon --show
```

### Full resource snapshot
```bash
free -h && df -h / && uptime && ps aux | grep python3
```

---

## Updating the Bot Code

### Push a single file from your Windows machine
```powershell
scp -i AYUSH.LSCO.T3SMALL.pem asterdex_trade\lsco.py ubuntu@13.112.47.16:~/LSCO/asterdex_trade/
```

### After pushing an update, restart the service
```bash
sudo systemctl restart lsco
```

### Push all core files at once
```powershell
scp -i AYUSH.LSCO.T3SMALL.pem `
    asterdex_trade\lsco.py `
    asterdex_trade\market_data.py `
    asterdex_trade\order_executor.py `
    ubuntu@13.112.47.16:~/LSCO/asterdex_trade/
```

---

## Useful One-Liners

### See last 5 trades
```bash
python3 -c "import json; log=json.load(open('/home/ubuntu/LSCO/trade_log.json')); [print(t['time'],t['symbol'],t['result'],t['pnl_usd']) for t in log[-5:]]"
```

### Check if bot is alive
```bash
sudo systemctl is-active lsco
```

### Hard restart (kills and restarts)
```bash
sudo systemctl kill lsco && sudo systemctl start lsco
```

### Clear log file (use carefully)
```bash
> ~/LSCO/lsco.log
```

---

## Heatmap Files

Auto-generated every 5 minutes by the running engine:

```
~/LSCO/data_fetching/binance_liq_heatmap_BTCUSDT.json
~/LSCO/data_fetching/binance_liq_heatmap_ETHUSDT.json
~/LSCO/data_fetching/binance_liq_heatmap_XAUUSDT.json
```

Run manually for one symbol:
```bash
python3 ~/LSCO/data_fetching/binance_liq_heatmap_2.py --symbol BTCUSDT
python3 ~/LSCO/data_fetching/binance_liq_heatmap_2.py --symbol ETHUSDT
python3 ~/LSCO/data_fetching/binance_liq_heatmap_2.py --symbol XAUUSDT
```

---

## Reboot

```bash
sudo reboot
```

After reboot, the `lsco` service starts automatically (systemd enabled).
Wait ~30 seconds then check: `sudo systemctl status lsco`

---

## Notes

- **No modifications to `account_data.py`** — contains EIP-712 wallet credentials, do not edit
- **Sizing is conservative** (1% balance, $2 cap) — testing phase showing investors
- **t3.small burst**: unlimited CPU burst (no credit cap like t2), safe for heatmap spikes
- **Swap**: 1 GB added at `/swapfile`, persists on reboot via `/etc/fstab`
- **Log rotation**: `lsco.log` is append-only and grows forever — clear it manually if needed
