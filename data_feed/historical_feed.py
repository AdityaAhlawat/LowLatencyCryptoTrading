import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import re
import os

def parse_lookback(lookback_str):
    """
    Parses strings like '1y 2mo 5d 3h 20m' into a timedelta object
    """
    units = {
        'y': 365 * 24 * 60,
        'mo': 30 * 24 * 60,
        'd': 24 * 60,
        'h': 60,
        'm': 1
    }

    total_minutes = 0
    pattern = r"(\d+)\s*(y|mo|d|h|m)"
    matches = re.findall(pattern, lookback_str.lower())

    if not matches:
        raise ValueError("Invalid lookback format. Use like: '1y 2mo 5d 3h 20m'")

    for value, unit in matches:
        if unit not in units:
            raise ValueError(f"Invalid time unit: {unit}")
        total_minutes += int(value) * units[unit]

    return timedelta(minutes=total_minutes)

def get_historical_klines(symbol="BTCUSDT", interval="1m", lookback="2h", save_path=None):
    url = "https://api.binance.us/api/v3/klines"
    interval_minutes = int(interval[:-1]) if interval.endswith("m") else 1
    lookback_duration = parse_lookback(lookback)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - lookback_duration

    df_all = pd.DataFrame()

    while start_time < end_time:
        batch_end = start_time + timedelta(minutes=interval_minutes * 1000)
        if batch_end > end_time:
            batch_end = end_time

        print(f"Requesting from {start_time} to {batch_end}... ", end="")

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(batch_end.timestamp() * 1000),
            "limit": 1000,
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not isinstance(data, list):
            print("❌ API Error:", data)
            break

        print(f"Received {len(data)} rows")

        if not data:
            print("⚠️ No rows returned, stopping.")
            break

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[["open_time", "open", "high", "low", "close", "volume"]]

        df_all = pd.concat([df_all, df], ignore_index=True)

        start_time = batch_end
        time.sleep(0.25)

    if not df_all.empty and save_path:
        df_all.to_csv(save_path, index=False)
        print(f"✅ Saved training data to {save_path}")

    return df_all

if __name__ == "__main__":
    # Set path to: ../data/trainingData.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root project directory
    output_path = os.path.join(base_dir, "data", "trainingData.csv")

    df = get_historical_klines(symbol="BTCUSDT", interval="1m", lookback="1y 2mo 2m", save_path=output_path)
    if df.empty:
        print("⚠️ No data was returned from the API.")
