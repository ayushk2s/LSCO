import asyncio
import websockets
import json
import gzip
from datetime import datetime

response_data = []

OUTPUT_FILE = "coinglass_whale_order.jsonl"
URL = "wss://wss.coinglass.com/v2/ws"

HEADERS = {
    "Origin": "https://www.coinglass.com",
    "User-Agent": "Mozilla/5.0"
}

SYMBOL = "Binance_BTCUSDT"

CHANNELS = [
    "kline",
    "largeTakerOrder",
    "largeTakerTrade",
    "largeTakerOrderRevoke"
]


# =========================
# 🔥 DECODE FUNCTION
# =========================
def decode_message(msg):
    try:
        if isinstance(msg, bytes):
            try:
                decompressed = gzip.decompress(msg)
                text = decompressed.decode("utf-8")

                try:
                    return json.loads(text)
                except:
                    return text

            except Exception:
                return f"RAW_BINARY({len(msg)} bytes)"

        return msg

    except Exception as e:
        return f"decode_error: {e}"


# =========================
# 🔥 SUBSCRIBE / UNSUBSCRIBE
# =========================
def make_sub(channel, interval):
    return {
        "method": "subscribe",
        "params": [{
            "listenerGuid": f"{SYMBOL}#_{channel}_{interval}",
            "symbol": f"{SYMBOL}#{channel}" if channel == "kline" else SYMBOL,
            "interval": interval,
            "channel": channel
        }]
    }


def make_unsub(channel, interval):
    return {
        "method": "unsubscribe",
        "params": [{
            "listenerGuid": f"{SYMBOL}#_{channel}_{interval}",
            "symbol": f"{SYMBOL}#{channel}" if channel == "kline" else SYMBOL,
            "interval": interval,
            "channel": channel
        }]
    }


# =========================
# 🔥 MAIN CONNECTOR
# =========================
async def connect(interval="m1"):

    async with websockets.connect(
        URL,
        additional_headers=HEADERS,
        ping_interval=20,
        ping_timeout=10
    ) as ws:

        print(f"\n✅ Connected | Interval = {interval}\n")

        # =========================
        # Subscribe channels
        # =========================
        for ch in CHANNELS:
            sub = make_sub(ch, interval)
            await ws.send(json.dumps(sub))
            print(f"📡 Subscribed: {ch} ({interval})")

        print("\n🚀 Listening...\n")

        # =========================
        # Receive loop
        # =========================
        while True:
            try:
                msg = await ws.recv()

                decoded = decode_message(msg)

                save_data(decoded)
                # =========================
                # Pretty output
                # =========================
                if isinstance(decoded, dict):
                    channel = decoded.get("channel", "unknown")

                    print(f"\n🟢 CHANNEL: {channel}")
                    print(json.dumps(decoded, indent=2))

                else:
                    print(f"🟡 {decoded}")

            except Exception as e:
                print("❌ Connection error:", e)
                break

def save_data(data):
    response_data.append(data)

    with open(OUTPUT_FILE, "a", encoding='utf-8') as f:
        record = {
            "timestamp" : datetime.utcnow().isoformat(),
            "data": data    
        }

        f.write(json.dumps(record) + "\n")




# =========================
# 🔥 SWITCH HERE
# =========================

if __name__ == "__main__":

    # 👉 CHANGE THIS:
    INTERVAL = "m1"   # "m1" or "m5"

    asyncio.run(connect(INTERVAL))