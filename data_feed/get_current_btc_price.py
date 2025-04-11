import requests

def get_current_btc_price(symbol="BTCUSDT"):
    url = "https://api.binance.us/api/v3/ticker/price"
    params = {"symbol": symbol}

    response = requests.get(url, params=params)
    data = response.json()

    price = float(data["price"])
    return price

btc_price = get_current_btc_price()
print(btc_price)





