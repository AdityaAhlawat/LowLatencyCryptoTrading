import asyncio
import websockets
import json
import uvloop
import numpy as np

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

BINANCE_WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@trade"
WINDOW_SIZE = 30
price_buffer = []

async def listen():
    async with websockets.connect(BINANCE_WS_URL) as ws:
        print("Connected to Binance WebSocket")
        async for message in ws:
            data = json.loads(message)

            try:
                price = float(data["p"])
                volume = float(data["q"])
                ts = data["T"]
            except KeyError:
                continue

            price_buffer.append(price)
            if len(price_buffer) > WINDOW_SIZE:
                price_buffer.pop(0)

            print(f"[{ts}] Price: {price:.2f} | Volume: {volume:.6f}")
            if len(price_buffer) == WINDOW_SIZE:
                print(f"Buffer ready for LSTM | Last 3: {price_buffer[-3:]}")

if __name__ == "__main__":
    asyncio.run(listen())
