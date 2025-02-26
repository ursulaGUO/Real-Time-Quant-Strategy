import socket
import json
import pickle
import pandas as pd
import numpy as np
import argparse

# Server details
HOST = "127.0.0.1"
PORT = 8080

# Load model based on argument
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, required=True, help="Input model name e.g. 'xgboost' or 'Gradient_Boosting' or 'Random_Forest' or 'var'")
args = parser.parse_args()
model_name = args.model
model_filename = f"{model_name}_stock_model.pkl"
print(f"You've opted to use {model_name} model, which uses {model_filename}.")
with open(model_filename, "rb") as file:
    model = pickle.load(file)

print("XGBoost model loaded successfully.")

# Initialize trading variables
cash = 20000  # Starting capital
portfolio = {}  # Dictionary to track holdings (ticker -> number of shares)
trade_history = [] 
trade_records = [] 
price_history = {} 
latest_prices = {} 

############ VAR changes ################################
if model_filename[0:3] == "var":
    var_results = model

    k_ar = var_results.k_ar  # how many lags the model uses
    print(f"Loaded VAR model with k_ar={k_ar}")
    var_columns = ["AAPL", "AMZN", "BRK-B", "GOOGL", "JNJ", "META", "MSFT", "NVDA", "TSLA", "V"]  # example for multi-ticker
    hist_df = pd.DataFrame(columns=var_columns)

    def update_var_data(ticker, close_price):
        global hist_df
        
        new_row = pd.DataFrame({ticker: [close_price]})
        
        for col in var_columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        
        # Reorder columns
        new_row = new_row[var_columns]
        
        # Append to hist_df
        hist_df = pd.concat([hist_df, new_row], ignore_index=True)
        # Forward fill or something to fill missing columns for other tickers
        hist_df.fillna(method="ffill", inplace=True)
        
        # Keep only the last k_ar rows + 1 or 2 buffer
        hist_df = hist_df.tail(k_ar+2)


    def predict_next_price_var(ticker):

        global hist_df
        
        # We need at least k_ar rows to forecast
        if len(hist_df) < k_ar:
            print("Not enough history to forecast yet.")
            return None
        
        # The 'last_obs' are the final k_ar rows
        last_obs = hist_df.values[-k_ar:] 
        
        # Forecast 1 step
        forecast_array = var_results.forecast(last_obs, steps=1) 
        
        # Identify the column index for this ticker
        if ticker not in var_columns:
            print(f"Ticker {ticker} not in VAR columns. Cannot forecast.")
            return None
        
        col_idx = var_columns.index(ticker)
        predicted_value = forecast_array[0, col_idx]
        return predicted_value

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
    """Decide to buy, sell, or hold based on predictions and update portfolio records."""
    global cash, portfolio
    if predicted_price is None:
        return

    # Calculate total portfolio value for record keeping
    def get_portfolio_value():
        total_stock_value = sum(portfolio[t] * latest_prices.get(t, 0) for t in portfolio)
        return cash + total_stock_value

    if predicted_price > current_price:
        # Buy Signal (if cash is available)
        if current_price == 0:
            shares_to_buy = 0
        else:
            shares_to_buy = int(min(cash * 0.1, 500) / current_price)  # Spend as much as $500
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                cash -= cost
                portfolio[ticker] = portfolio.get(ticker, 0) + shares_to_buy
                message = f"{trade_date} - BUY {shares_to_buy} shares of {ticker} at ${current_price:.2f}"
                trade_history.append(message)
                # Save trade details as a record (dictionary)
                trade_records.append({
                    "Date": trade_date,
                    "Ticker": ticker,
                    "Action": "BUY",
                    "Shares": shares_to_buy,
                    "Trade Price": current_price,
                    "Cash Balance": cash,
                    "Portfolio Snapshot": str(portfolio),
                    "Total Portfolio Value": get_portfolio_value()
                })
                print(f"{message} | Cash: ${cash:.2f}")

    elif predicted_price < current_price and ticker in portfolio:
        # Sell Signal if stock is held
        shares_to_sell = portfolio.get(ticker, 0)
        if shares_to_sell > 0:
            revenue = shares_to_sell * current_price
            cash += revenue
            del portfolio[ticker]
            message = f"{trade_date} - SELL {shares_to_sell} shares of {ticker} at ${current_price:.2f}"
            trade_history.append(message)
            trade_records.append({
                "Date": trade_date,
                "Ticker": ticker,
                "Action": "SELL",
                "Shares": shares_to_sell,
                "Trade Price": current_price,
                "Cash Balance": cash,
                "Portfolio Snapshot": str(portfolio),
                "Total Portfolio Value": get_portfolio_value()
            })
            print(f"{message} | Cash: ${cash:.2f}")

    # Print summary after each trade
    print_portfolio_summary()

def print_portfolio_summary():
    """Prints the current cash balance, open positions, and total earnings."""
    total_stock_value = sum(portfolio[ticker] * latest_prices.get(ticker, 0) for ticker in portfolio)
    total_value = cash + total_stock_value
    total_earnings = total_value - 20000 

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

                # Exit condition: if current_price == 0, exit the loop.
                if current_price == 0:
                    print("Trading concluded.")
                    break

                # Predict next day's price
                if model_filename[0:3] != "var":
                    predicted_price = predict_next_price(ticker, stock_data)
                else:
                    update_var_data(ticker, current_price)
                    predicted_price = predict_next_price_var(ticker)
                if predicted_price is not None:
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
        # Print final summary and trade records
        print("Trade History:")
        for trade in trade_history:
            print(" -", trade)
        print("\nFinal Trading Summary:")
        print_portfolio_summary()

        # Convert trade records to a DataFrame and display/save
        trade_df = pd.DataFrame(trade_records)
        print("\nTrade Records DataFrame:")
        # Save trade records
        trade_df.to_csv(f"portfolios/{model_filename[:-16]}_trade_records.csv", index=False)

if __name__ == "__main__":

    connect_to_server()
