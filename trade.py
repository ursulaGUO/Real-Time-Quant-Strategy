import socket
import json
import pickle
import pandas as pd

# Server details
HOST = "127.0.0.1"
PORT = 8080

# Load trained XGBoost model
model_filename = "xgboost_stock_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

print("XGBoost model loaded successfully.")

# Initialize trading variables
cash = 20000  # Starting capital
portfolio = {}  # Dictionary to track holdings (ticker -> number of shares)
trade_history = []  # Store trade records
price_history = {}  # Store historical price data for feature computation
latest_prices = {}  # Dictionary to store the latest market prices of stocks

def compute_features(ticker, current_data):
    """Compute necessary features for XGBoost model using historical data."""
    
    # Ensure historical data storage exists
    if ticker not in price_history:
        price_history[ticker] = pd.DataFrame(columns=["Date", "Close", "High", "Low", "Open", "Volume"])
    
    # Append new data point
    df = price_history[ticker]
    df = pd.concat([df, pd.DataFrame([current_data])], ignore_index=True)

    # Keep only the latest 30 days of data for rolling calculations
    df = df.tail(30)

    # Convert necessary columns to numeric
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute Features
    df["Prev_Close"] = df["Close"].shift(1)
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = rolling_mean + (rolling_std * 2)
    df["Bollinger_Lower"] = rolling_mean - (rolling_std * 2)
    df["Volatility"] = df["Close"].pct_change().rolling(5).std()

    # Save updated history
    price_history[ticker] = df

    # Extract latest computed row
    latest_features = df.iloc[-1]

    # Return as a dictionary for model input
    return {
        "Prev_Close": latest_features["Prev_Close"],
        "SMA_10": latest_features["SMA_10"],
        "SMA_20": latest_features["SMA_20"],
        "EMA_10": latest_features["EMA_10"],
        "EMA_20": latest_features["EMA_20"],
        "MACD": latest_features["MACD"],
        "Bollinger_Upper": latest_features["Bollinger_Upper"],
        "Bollinger_Lower": latest_features["Bollinger_Lower"],
        "Volatility": latest_features["Volatility"],
        "Volume": latest_features["Volume"]
    }



def predict_next_price(ticker, current_data):
    """Compute features and predict next day's closing price using XGBoost."""
    feature_data = compute_features(ticker, current_data)

    # Convert to DataFrame for model prediction
    df = pd.DataFrame([feature_data])

    # Ensure all expected features are present
    features = ["Prev_Close", "SMA_10", "SMA_20", "EMA_10", "EMA_20", 
                "MACD", "Bollinger_Upper", "Bollinger_Lower", "Volatility", "Volume"]
    
    for feature in features:
        if feature not in df or pd.isna(df[feature].values[0]):
            df[feature] = 0  # Default missing features to 0 to avoid errors

    # Predict next day's price
    predicted_price = model.predict(df)[0]
    return predicted_price

def execute_trade(ticker, current_price, predicted_price, trade_date):
    """Decide to buy, sell, or hold based on predictions."""
    global cash, portfolio

    if predicted_price > current_price:
        # Buy Signal (if cash is available)
        shares_to_buy = int(min(cash * 0.1, 500) / current_price)  # Spend as much as $500
        if shares_to_buy > 0:
            cost = shares_to_buy * current_price
            cash -= cost
            portfolio[ticker] = portfolio.get(ticker, 0) + shares_to_buy
            trade_history.append(f"{trade_date} - BUY {shares_to_buy} shares of {ticker} at ${current_price:.2f}")
            print(f"{trade_date} - BUY {shares_to_buy} shares of {ticker} at ${current_price:.2f} | Cash: ${cash:.2f}")

    elif predicted_price < current_price and ticker in portfolio:
        # Sell Signal if stock is held
        shares_to_sell = portfolio.get(ticker, 0)
        if shares_to_sell > 0:
            revenue = shares_to_sell * current_price
            cash += revenue
            del portfolio[ticker]
            trade_history.append(f"{trade_date} - SELL {shares_to_sell} shares of {ticker} at ${current_price:.2f}")
            print(f"{trade_date} - SELL {shares_to_sell} shares of {ticker} at ${current_price:.2f} | Cash: ${cash:.2f}")

    # Print summary after each trade
    print_portfolio_summary()

def print_portfolio_summary():
    """Prints the current cash balance, open positions, and total earnings."""
    total_stock_value = sum(portfolio[ticker] * latest_prices.get(ticker, 0) for ticker in portfolio)
    total_value = cash + total_stock_value
    total_earnings = total_value - 20000  # Assuming $20,000 as the starting capital

    print("\nCurrent Portfolio Summary:")
    print(f" - Cash: ${cash:.2f}")
    print(f" - Stock Holdings:")
    for ticker, shares in portfolio.items():
        stock_value = shares * latest_prices.get(ticker, 0)
        print(f"   {ticker}: {shares} shares, Market Value: ${stock_value:.2f}")
    print(f" - Total Portfolio Value: ${total_value:.2f}")
    print(f" - Total Earnings: ${total_earnings:.2f}\n")

def connect_to_server():
    """Connect to real-time stock data server and trade based on predictions."""
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")

        while True:
            data = client.recv(1024).decode()
            if not data:
                print("Server closed connection.")
                break

            try:
                # Parse incoming JSON stock data
                stock_data = json.loads(data)
                ticker = stock_data.get("Ticker")
                current_price = float(stock_data.get("Close", 0))
                trade_date = stock_data.get("Date", "Unknown Date")  # Extract date from data
                latest_prices[ticker] = current_price

                # Predict next day's price
                predicted_price = predict_next_price(ticker, stock_data)
                print(f"{trade_date} - {ticker}: Current Price = ${current_price:.2f}, Predicted = ${predicted_price:.2f}")

                # Execute trade decision with date
                execute_trade(ticker, current_price, predicted_price, trade_date)

            except json.JSONDecodeError:
                print("Received unexpected data format:", data)

    except ConnectionRefusedError:
        print("Failed to connect to server.")
    except KeyboardInterrupt:
        print("\nClient closed manually.")
    finally:
        client.close()

        # Print final summary
        print("\nFinal Trading Summary:")
        print_portfolio_summary()
        print("Trade History:")
        for trade in trade_history:
            print(" -", trade)



if __name__ == "__main__":
    connect_to_server()
