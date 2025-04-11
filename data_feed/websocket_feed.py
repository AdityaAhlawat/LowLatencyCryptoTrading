import asyncio
import websockets
import json
import uvloop
import numpy as np
from datetime import datetime

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

BINANCE_WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@trade"
WINDOW_SIZE = 30
price_buffer = []

async def listen():
    async with websockets.connect(BINANCE_WS_URL) as ws:
        print("✅ Connected to Binance WebSocket")
        async for message in ws:
            data = json.loads(message)
            try:
                event_type = data["e"]
                event_time = int(data["E"]) / 1000  # seconds since epoch
                symbol = data["s"]
                trade_id = int(data["t"])
                price = float(data["p"])
                quantity = float(data["q"])
                buyer_order_id = int(data["b"])
                seller_order_id = int(data["a"])
                trade_time = int(data["T"]) / 1000  # seconds since epoch
                is_buyer_market_maker = data["m"]
            except KeyError:
                continue

            # Use np.float64 to maintain high precision (optional but safe)
            price = np.float64(price)
            quantity = np.float64(quantity)

            # Maintain rolling price buffer
            price_buffer.append(price)
            if len(price_buffer) > WINDOW_SIZE:
                price_buffer.pop(0)

            # Display clean numeric data
            print(
                f"[{trade_time:.3f}] Symbol: {symbol} | Trade ID: {trade_id} | Price: {price:.8f} | "
                f"Qty: {quantity:.8f} | Buyer ID: {buyer_order_id} | Seller ID: {seller_order_id} | "
                f"Maker: {is_buyer_market_maker}"
            )

            if len(price_buffer) == WINDOW_SIZE:
                print(f"📦 Buffer ready for LSTM | Last 3 Prices: {[float(x) for x in price_buffer[-3:]]}")

if __name__ == "__main__":
    asyncio.run(listen())
