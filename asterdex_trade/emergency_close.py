"""
emergency_close.py — Close ALL open positions and cancel all orders.
Run: python emergency_close.py  (from asterdex_trade/ directory)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from account_data import place_order, cancel_all_orders, get_position_risk
from market_data  import get_price

SYMBOLS     = ["BTCUSDT", "ETHUSDT", "XAUUSDT"]
PRICE_ROUND = {"BTCUSDT": 1, "ETHUSDT": 2, "XAUUSDT": 2}
SLIP        = 0.003   # 0.3% overshoot so IOC limit always fills


def close_symbol(symbol: str):
    pr = PRICE_ROUND[symbol]

    print(f"\n[{symbol}] Cancelling open orders...", end=" ", flush=True)
    try:
        cancel_all_orders(symbol)
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

    try:
        pd        = get_position_risk(symbol)
        positions = pd if isinstance(pd, list) else pd.get("data", [])
        active    = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
    except Exception as e:
        print(f"[{symbol}] Position fetch failed: {e}")
        return

    if not active:
        print(f"[{symbol}] No open positions.")
        return

    price = get_price(symbol)

    for pos in active:
        side = pos.get("positionSide", "")
        amt  = abs(float(pos.get("positionAmt", 0)))
        ep   = float(pos.get("entryPrice", 0))
        pnl  = float(pos.get("unrealizedProfit", 0))
        print(f"[{symbol}] {side}  qty={amt}  entry=${ep:,.{pr}f}  PnL=${pnl:+.4f}")

        close_side = "BUY" if side == "SHORT" else "SELL"
        lim_price  = round(
            price * (1 + SLIP) if close_side == "BUY" else price * (1 - SLIP), pr
        )

        closed = False
        for attempt in range(4):
            try:
                resp   = place_order(symbol, close_side, "LIMIT", amt, lim_price, "IOC", side)
                filled = float(resp.get("executedQty") or 0)
                status = resp.get("status", "?")
                print(f"  attempt {attempt+1}: {status}  filled={filled}")
                if filled > 0:
                    closed = True
                    break
                # refresh price for next attempt
                price     = get_price(symbol)
                lim_price = round(
                    price * (1 + SLIP) if close_side == "BUY" else price * (1 - SLIP), pr
                )
            except Exception as e:
                print(f"  attempt {attempt+1}: ERROR {e}")
            time.sleep(1.0)

        if not closed:
            print(f"  !! FAILED to close {symbol} {side} — close manually on exchange !!")


def main():
    print("=" * 58)
    print("  EMERGENCY CLOSE — All LSCO Positions")
    print("=" * 58)

    for sym in SYMBOLS:
        close_symbol(sym)

    print()
    print("=" * 58)
    print("  DONE.")
    print("  Check exchange to confirm all positions are closed.")
    print()
    print("  Then kill the VPS process:")
    print("    ps aux | grep python")
    print("    kill -9 <PID>")
    print("=" * 58)


if __name__ == "__main__":
    main()
