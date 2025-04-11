import requests

import requests

def get_current_btc_price(symbol="BTCUSDT"):
    url = "https://api.binance.us/api/v3/ticker/price"
    params = {"symbol": symbol}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() 
        data = response.json()

        if "price" not in data:
            raise ValueError("Price data not found in the response.")

        price = float(data["price"])
        return price

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except ValueError as e:
        print(f"Data parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None  
