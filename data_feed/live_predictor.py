import asyncio
import websockets
import json
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

BINANCE_WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@trade"

async def listen():
    try:
        async with websockets.connect(BINANCE_WS_URL) as ws:
            print("✅ Connected to Binance WebSocket")
            message_count = 0
            async for message in ws:
                message_count += 1
                print(f"🔁 Message #{message_count} received")

                try:
                    data = json.loads(message)
                    print("📦 Raw Message:", data)

                    if "p" in data:
                        price = float(data["p"])
                        print(f"💰 BTC Price: {price}")
                    else:
                        print("⚠️ 'p' not in message, structure may be different.")
                except Exception as parse_error:
                    print(f"❌ Error parsing message: {parse_error}")
    except Exception as ws_error:
        print(f"❌ WebSocket connection error: {ws_error}")

if __name__ == "__main__":
    asyncio.run(listen())
