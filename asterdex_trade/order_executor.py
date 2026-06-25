"""
order_executor.py
SmartLimitOrder — places a limit order and keeps it at the BEST price in the
order book until it fills or is cancelled.

Two modes:
  track_ob = True  (default) → entry orders and emergency SL close
      Watches best bid/ask every 1.5 s. If the best price moved ≥ $0.50,
      cancels the old order and places a new one at the updated best price.
      Partial fills are tracked: replacement is always for REMAINING qty only.
      This way we are always FIRST in the queue without paying taker fees.

  track_ob = False → TP orders placed at a fixed calculated price
      Places once at the exact TP price and sits there. No replacement.
      These are resting limit orders that fill when price reaches the target.

Usage
-----
    order = SmartLimitOrder(
        symbol="BTCUSDT", side="SELL", qty=0.001,
        pos_side="SHORT", label="parent_entry"
    )
    await order.place()          # starts OB monitoring in background
    await order.wait_fill(300)   # wait up to 5 min for fill
    print(order.fill_price)
"""

import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from account_data import place_order, cancel_order, get_open_orders, get_order_history
from market_data import get_ob

OB_MOVE_THRESHOLD  = 0.50   # $0.50 — cancel+replace if best price moves this much
OB_POLL_INTERVAL   = 1.5    # seconds between OB checks
FILL_POLL_INTERVAL = 2.0    # seconds between fill-status polls


class SmartLimitOrder:
    """
    A limit order that automatically chases the best bid/ask.

    For a BUY  order → always stays at the best BID  (we are the maker buyer).
    For a SELL order → always stays at the best ASK  (we are the maker seller).

    When the best price moves ≥ OB_MOVE_THRESHOLD the old order is cancelled
    and a NEW order is placed at the updated best price — but only for the
    REMAINING (unfilled) quantity so partial fills are never re-ordered.
    """

    def __init__(self,
                 symbol:      str,
                 side:        str,        # "BUY" or "SELL"
                 qty:         float,
                 pos_side:    str,        # "LONG" or "SHORT"
                 label:       str   = "",
                 track_ob:    bool  = True,
                 fixed_price: float = None,
                 price_round: int   = 1):   # decimal places for price rounding
        self.symbol      = symbol
        self.side        = side.upper()
        self.qty         = qty
        self.pos_side    = pos_side.upper()
        self.label       = label
        self.track_ob    = track_ob
        self.fixed_price = fixed_price
        self.price_round = price_round

        # Public state — read by callers
        self.order_id     = None
        self.fill_price   = None    # weighted-average fill price across all partials
        self.is_filled    = False   # True once filled_qty >= qty
        self.is_cancelled = False
        self.filled_qty   = 0.0     # cumulative executed quantity

        # Internal
        self._placed_price       = None   # price of the currently resting order
        self._fill_value         = 0.0    # sum(exec_qty * avg_px) for weighted avg
        self._done_event         = asyncio.Event()
        self._ob_task            = None
        self._user_cancelled     = False  # True only when user calls cancel() — not exchange expiry
        self._place_fail_count   = 0      # consecutive placement failures (id=None)

    # ── Public interface ──────────────────────────────────────────────────────

    async def place(self):
        """Place the order and start OB monitoring. Returns self for chaining."""
        price = await self._best_price() if self.track_ob else self.fixed_price
        await self._place_at(price, self.qty)

        if self.track_ob:
            self._ob_task = asyncio.create_task(self._ob_loop())

        return self

    async def wait_fill(self, timeout: float = 300.0) -> bool:
        """
        Wait until fully filled OR timeout seconds.
        Returns True if filled, False if timed out.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._done_event.is_set():
                return self.is_filled

            filled = await self._check_fill()
            if filled:
                return True

            await asyncio.sleep(FILL_POLL_INTERVAL)

        return self.is_filled

    def cancel(self):
        """
        Cancel the order (safe to call from any context).
        Stops OB monitoring and sends cancel to exchange.
        """
        self._user_cancelled = True
        self.is_cancelled    = True
        self._done_event.set()

        if self._ob_task and not self._ob_task.done():
            self._ob_task.cancel()

        if self.order_id:
            try:
                cancel_order(self.symbol, order_id=self.order_id)
                print(f"  [executor] {self.label} cancelled (id={self.order_id})")
            except Exception as e:
                print(f"  [executor] cancel failed for {self.label}: {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _best_price(self) -> float:
        ob = await asyncio.to_thread(get_ob, self.symbol)
        return ob["best_ask"] if self.side == "SELL" else ob["best_bid"]

    async def _place_at(self, price: float, qty: float):
        """Send a GTX limit order for `qty` at `price`."""
        price = round(price, self.price_round)
        try:
            resp = await asyncio.to_thread(
                place_order,
                self.symbol,
                self.side,
                "LIMIT",
                qty,
                price,
                "GTX",        # Post-Only — maker only, never taker
                self.pos_side,
            )
            self.order_id      = resp.get("orderId")
            self._placed_price = price
            print(f"  [executor] {self.label:20s} {self.side} {qty} @ {price:,.1f}"
                  f"  id={self.order_id}")
            if self.order_id is None:
                self._place_fail_count += 1
                if self._place_fail_count >= 5:
                    # Exchange keeps rejecting — position likely gone, stop loop
                    print(f"  [executor] {self.label} 5 consecutive rejections"
                          f" — position likely closed, stopping")
                    self.is_cancelled = True
                    self._done_event.set()
            else:
                self._place_fail_count = 0
        except Exception as e:
            print(f"  [executor] place_at failed for {self.label}: {e}")

    async def _get_executed_qty(self, order_id) -> tuple:
        """
        Query order history for a specific order.
        Returns (executed_qty, avg_price).
        """
        try:
            hist = await asyncio.to_thread(get_order_history, self.symbol, order_id)
            hist_orders = hist if isinstance(hist, list) else hist.get("data", [])
            for o in hist_orders:
                if o.get("orderId") == order_id:
                    exec_qty = float(o.get("executedQty") or 0)
                    avg_px   = float(o.get("avgPrice") or o.get("price") or self._placed_price or 0)
                    return exec_qty, avg_px
        except Exception:
            pass
        return 0.0, 0.0

    def _accumulate_fill(self, exec_qty: float, avg_px: float):
        """
        Add a partial fill into the running totals.
        Only counts qty above what we've already recorded.
        """
        new_qty = exec_qty - self.filled_qty
        if new_qty <= 0:
            return
        self._fill_value  += new_qty * avg_px
        self.filled_qty    = exec_qty
        self.fill_price    = self._fill_value / self.filled_qty

    def _remaining(self) -> float:
        return max(round(self.qty - self.filled_qty, 3), 0.0)

    async def _ob_loop(self):
        """
        Background task: poll OB every OB_POLL_INTERVAL seconds.
        If best price moved ≥ OB_MOVE_THRESHOLD, cancel the current order,
        collect any partial fill from it, then re-place for the remaining qty.
        """
        try:
            while not self._done_event.is_set():
                await asyncio.sleep(OB_POLL_INTERVAL)

                if self.is_filled or self._user_cancelled:
                    break

                best = await self._best_price()

                if abs(best - self._placed_price) >= OB_MOVE_THRESHOLD:
                    print(f"  [executor] {self.label} OB moved"
                          f" {self._placed_price:,.1f} -> {best:,.1f}, replacing ...")

                    cancelled_id = self.order_id
                    try:
                        await asyncio.to_thread(
                            cancel_order, self.symbol, order_id=cancelled_id
                        )
                    except Exception:
                        pass   # might already be filled; check next iteration

                    await asyncio.sleep(0.3)

                    # Collect any partial fill from the cancelled order
                    exec_qty, avg_px = await self._get_executed_qty(cancelled_id)
                    if exec_qty > self.filled_qty:
                        self._accumulate_fill(exec_qty, avg_px)
                        print(f"  [executor] {self.label} partial fill:"
                              f" {exec_qty}/{self.qty}  avg={avg_px:,.1f}")

                    remaining = self._remaining()
                    if remaining < 0.001:
                        # Fully filled through partial fills
                        self.is_filled = True
                        self._done_event.set()
                        print(f"  [executor] {self.label} FULLY FILLED"
                              f" @ ~{self.fill_price:,.1f}")
                        break

                    # Re-place for remaining qty only
                    await self._place_at(best, remaining)

        except asyncio.CancelledError:
            pass

    async def _check_fill(self) -> bool:
        """
        Check if the order is filled.

        1. If order is still in open orders → check executedQty for partial fills.
           If remaining becomes 0 → mark fully filled.
        2. If order is gone from open orders → confirm via history:
           - FILLED / PARTIALLY_FILLED → accumulate fill, mark done.
           - CANCELED / EXPIRED / REJECTED → mark cancelled.
        """
        if self._user_cancelled or self.order_id is None:
            return False

        # Short-circuit: we may have already accumulated enough via _ob_loop
        if self.filled_qty >= self.qty - 0.0001:
            if not self.is_filled:
                self.is_filled = True
                if self.fill_price is None:
                    self.fill_price = self._placed_price
                self._done_event.set()
                if self._ob_task:
                    self._ob_task.cancel()
                print(f"  [executor] {self.label} FILLED @ ~{self.fill_price:,.1f}")
            return True

        try:
            resp   = await asyncio.to_thread(get_open_orders, self.symbol)
            orders = resp if isinstance(resp, list) else resp.get("data", [])
            open_map = {o.get("orderId"): o for o in orders}

            if self.order_id in open_map:
                # Order still resting — check for partial fill in the open order record
                o = open_map[self.order_id]
                exec_qty = float(o.get("executedQty") or 0)
                if exec_qty > self.filled_qty:
                    avg_px = float(o.get("avgPrice") or self._placed_price)
                    self._accumulate_fill(exec_qty, avg_px)
                    print(f"  [executor] {self.label} partial fill"
                          f" (resting): {exec_qty}/{self.qty}")
                    if self._remaining() < 0.001:
                        self.is_filled = True
                        self._done_event.set()
                        if self._ob_task: self._ob_task.cancel()
                        print(f"  [executor] {self.label} FILLED @ ~{self.fill_price:,.1f}")
                        return True
                return False

            # Order gone from open orders — confirm via history
            hist = await asyncio.to_thread(
                get_order_history, self.symbol, self.order_id
            )
            hist_orders = hist if isinstance(hist, list) else hist.get("data", [])
            status   = None
            exec_qty = 0.0
            avg_px   = 0.0
            for o in hist_orders:
                if o.get("orderId") == self.order_id:
                    status   = o.get("status", "").upper()
                    exec_qty = float(o.get("executedQty") or 0)
                    avg_px   = float(o.get("avgPrice") or o.get("price") or 0)
                    break

            if status in ("FILLED", "PARTIALLY_FILLED"):
                if exec_qty > self.filled_qty:
                    self._accumulate_fill(exec_qty, avg_px)
                remaining = self._remaining()
                if remaining < 0.001 or status == "FILLED":
                    self.is_filled = True
                    if self.fill_price is None:
                        self.fill_price = self._placed_price
                    self._done_event.set()
                    if self._ob_task: self._ob_task.cancel()
                    print(f"  [executor] {self.label} FILLED"
                          f" @ ~{self.fill_price:,.1f}"
                          f"  (filled={self.filled_qty}/{self.qty})")
                    return True
                # Partially filled but order ended — _ob_loop will re-place
                return False

            elif status in ("CANCELED", "EXPIRED", "REJECTED", "CANCELLED"):
                if exec_qty > self.filled_qty:
                    self._accumulate_fill(exec_qty, avg_px)
                if self._remaining() < 0.001:
                    self.is_filled = True
                    self._done_event.set()
                    if self._ob_task: self._ob_task.cancel()
                    print(f"  [executor] {self.label} FILLED @ ~{self.fill_price:,.1f}")
                    return True
                # Exchange expired/rejected — re-place immediately at new best price
                # (never give up unless the USER cancelled)
                if self._user_cancelled:
                    self.is_cancelled = True
                    self._done_event.set()
                    if self._ob_task: self._ob_task.cancel()
                    return False
                print(f"  [executor] {self.label} {status} by exchange — re-placing ...")
                best = await self._best_price() if self.track_ob else self.fixed_price
                await self._place_at(best, self._remaining())
                return False

            elif status is not None:
                print(f"  [executor] {self.label} unknown status={status}, skipping")
                return False
            else:
                return False   # API lag — try again next poll

        except Exception as e:
            print(f"  [executor] fill-check error for {self.label}: {e}")
        return False
