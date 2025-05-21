from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import threading
import time

app = Flask(__name__)
trade_signals = {}

top_stocks = [
    "AAPL", "MSFT",
    "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "UNH",
    "V", "JNJ", "WMT", "JPM", "MA", "PG", "XOM", "HD", "LLY", "ABBV",
    "COST", "PEP", "MRK", "AVGO", "CVX", "KO", "TMO", "NVO", "BAC", "MCD",
    "ADBE", "PFE", "CRM", "ABT", "CSCO", "LIN", "NFLX", "ACN", "WFC", "INTC",
    "AMD", "DHR", "QCOM", "TMUS", "NEE", "TXN", "PM", "LOW", "BMY", "UPS",
    "UNP", "RTX", "AMAT", "MS", "SCHW", "ISRG", "AMGN", "HON", "MDT", "GILD",
    "SPGI", "INTU", "CAT", "T", "ZTS", "LMT", "DE", "CB", "BA", "IBM",
    "PLD", "NOW", "MU", "BKNG", "GE", "C", "ADI", "BLK", "SYK", "CI",
    "REGN", "MMC", "AXP", "MO", "ELV", "MDLZ", "ADI", "ETN", "PANW", "VRTX",
    "EW", "KLAC", "CSX", "FDX", "ADP", "APD", "ILMN", "GM", "ORLY", "ROST"
]

def fetch_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, threads=True)

    if df.empty:
        raise ValueError(f"No data fetched for {ticker}")

    # Check if MultiIndex (this happens only if multiple tickers are passed)
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.levels[1]:
            raise KeyError(f"Ticker {ticker} not found in DataFrame. Found: {df.columns}")
        df = pd.DataFrame({
            'Close': df['Close', ticker],
            'Volume': df['Volume', ticker]
        })
    else:
        # lowercase just in case
        df.columns = [col.lower() for col in df.columns]
        if 'close' not in df.columns or 'volume' not in df.columns:
            raise KeyError(f"Required columns ['close', 'volume'] not found in {ticker} data: {df.columns}")
        df = df[['close', 'volume']]
        df.columns = ['Close', 'Volume']

    return df

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Bollinger_High'] = df['SMA_20'] + 2 * df['Close'].rolling(20).std()
    df['Bollinger_Low'] = df['SMA_20'] - 2 * df['Close'].rolling(20).std()

    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['ATR'] = df['Close'].rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))))
    df['VP_Ratio'] = df['Volume'] / df['Close']
    return df

def label_data(df):
    df['future_return'] = df['Close'].pct_change(1).shift(-1)
    df['label'] = df['future_return'].apply(lambda x: 1 if x > 0.01 else -1 if x < -0.01 else 0)
    return df.dropna()

def train_ml_model(df):
    features = df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal',
                   'Bollinger_High', 'Bollinger_Low', 'Volume_MA_20',
                   'Return_1d', 'Return_5d', 'ATR', 'VP_Ratio']].fillna(0)
    labels = df['label'].replace({-1: 0, 0: 1, 1: 2})
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model, scaler

def generate_signals(model_scaler_tuple, df):
    model, scaler = model_scaler_tuple
    latest = df.iloc[-1]
    X_latest = latest[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal',
                       'Bollinger_High', 'Bollinger_Low', 'Volume_MA_20',
                       'Return_1d', 'Return_5d', 'ATR', 'VP_Ratio']].fillna(0).values.reshape(1, -1)
    X_latest = scaler.transform(X_latest)
    prediction = model.predict(X_latest)[0]
    proba = model.predict_proba(X_latest)[0]
    confidence = max(proba)
    if prediction == 1:
        signal = "BUY"
    elif prediction == -1:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, confidence

def predict_price(df):
    df = df.dropna()
    X = df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'Bollinger_High', 'Bollinger_Low', 'Volume_MA_20']].fillna(0)
    y = df['Close']
    model = RandomForestRegressor()
    model.fit(X, y)
    latest = X.iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(latest)[0]
    return predicted_price


@app.route("/predict/<ticker>")
def get_price_prediction(ticker):
    df = fetch_stock_data(ticker)
    df = calculate_indicators(df)
    predicted_price = predict_price(df)

    # Define predicted buy/sell prices as ±2% buffer
    predicted_buy_price = round(predicted_price * 0.98, 2)   # 2% below predicted close
    predicted_sell_price = round(predicted_price * 1.02, 2)  # 2% above predicted close

    return jsonify({
        "ticker": ticker,
        "predicted_close": round(predicted_price, 2),
        "predicted_buy_price": predicted_buy_price,
        "predicted_sell_price": predicted_sell_price
    })

def background_signal_checker():
    while True:
        for ticker in top_stocks:
            try:
                df = fetch_stock_data(ticker)
                df = calculate_indicators(df)
                df = label_data(df)
                model_scaler_tuple = train_ml_model(df)
                signal, confidence = generate_signals(model_scaler_tuple, df)
                predicted_price = predict_price(df)
                predicted_buy_price = round(predicted_price * 0.98, 2)
                predicted_sell_price = round(predicted_price * 1.02, 2)
                print(f"[{ticker}] Signal: {signal}, Confidence: {confidence:.2f}, Buy @ {predicted_buy_price}, Sell @ {predicted_sell_price}")
                # print(f"[{ticker}] Signal: {signal}, Confidence: {confidence:.2f}")
            except Exception as e:
                print(f"[{ticker}] Error: {e}")
        time.sleep(60)  # check every 60 seconds

if __name__ == "__main__":
    threading.Thread(target=background_signal_checker, daemon=True).start()
    app.run(debug=True, port=5002)
