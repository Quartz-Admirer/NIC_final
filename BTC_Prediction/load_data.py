import requests
import pandas as pd

import time
import requests
import pandas as pd

def load_binance_data(symbol="BTCUSDT", interval='1h',total_limit=5000):
    """
    Загружает исторические часовые данные Binance, начиная с текущего момента в прошлое.
    Работает с ограничением в 1000 записей за один запрос.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    max_limit_per_request = 1000
    all_data = []
    now = int(time.time() * 1000)  # текущий момент в миллисекундах

    # Сколько полных запросов нужно сделать
    num_batches = (total_limit + max_limit_per_request - 1) // max_limit_per_request

    for i in range(num_batches):
        limit = min(max_limit_per_request, total_limit - len(all_data))
        end_time = now - i * max_limit_per_request * 60 * 60 * 1000  # шаг назад на i * 1000 часов
        start_time = end_time - limit * 60 * 60 * 1000  # назад на limit часов

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print("Ошибка запроса:", response.text)
            break

        data = response.json()
        if not data:
            print("Нет данных, завершение на шаге", i)
            break

        all_data = data + all_data  # prepend, чтобы сохранить порядок от старого к новому

        time.sleep(0.1)  # ограничим скорость запросов (во избежание блокировок)

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
    """
    Простейший пример подготовки данных:
    1) Скользящая средняя close (ma_close)
    2) 'future_close' = цена закрытия на следующий шаг (shift -1)
    3) Удаляем строки с NaN.
    """
    df = df.copy()
    df["ma_close"] = df["close"].rolling(window=ma_window).mean()
    df["future_close"] = df["close"].shift(-1)
    df.dropna(inplace=True)
    return df

def load_and_preprocess(symbol="BTCUSDT", interval="1d", limit=1000, ma_window=5):
    """
    Сквозная функция: загружаем данные, делаем простую подготовку.
    Возвращаем DataFrame со всеми нужными столбцами.
    """
    df = load_binance_data(symbol, interval, limit)
    df = prepare_data(df, ma_window)
    return df

# Пример
if __name__ == "__main__":
    df = load_binance_data("BTCUSDT", total_limit=5000)
    print("Загружено:", len(df), "часовых свечей")
    print(df.head())
