import requests
import pandas as pd

import time
import requests
import pandas as pd

def load_binance_data(symbol="BTCUSDT", interval='1h',total_limit=5000):

    base_url = "https://api.binance.com/api/v3/klines"
    
    max_limit_per_request = 1000
    all_data = []
    now = int(time.time() * 1000)

    num_batches = (total_limit + max_limit_per_request - 1) // max_limit_per_request

    for i in range(num_batches):
        limit = min(max_limit_per_request, total_limit - len(all_data))
        end_time = now - i * max_limit_per_request * 60 * 60 * 1000
        start_time = end_time - limit * 60 * 60 * 1000

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print("request error:", response.text)
            break

        data = response.json()
        if not data:
            print("no data", i)
            break

        all_data = data + all_data

        time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    # Преобразование типов
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


def prepare_data(df, ma_window=5):

    df = df.copy()
    df["ma_close"] = df["close"].rolling(window=ma_window).mean()
    df["future_close"] = df["close"].shift(-1)
    df.dropna(inplace=True)
    return df

def load_and_preprocess(symbol="BTCUSDT", interval="1d", limit=1000, ma_window=5):
    df = load_binance_data(symbol, interval, limit)
    df = prepare_data(df, ma_window)
    return df

if __name__ == "__main__":
    df = load_binance_data("BTCUSDT", total_limit=5000)
    print("loaded:", len(df), "hours")
    print(df.head())
