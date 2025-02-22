import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to get US stock tickers from a source (replace with an actual source)
def get_us_stock_tickers():
    # You may replace this list with a more comprehensive list from a CSV or API
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]  # Add more tickers

# Define date range
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')

# Get stock tickers
stock_tickers = get_us_stock_tickers()

# Create a folder to store the CSV files
import os
output_dir = "US_Stock_Data"
os.makedirs(output_dir, exist_ok=True)

# Download data for each stock and save as CSV
for ticker in stock_tickers:
    try:
        print(f"Downloading {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            file_path = os.path.join(output_dir, f"{ticker}_stock_data.csv")
            stock_data.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

print("Download complete.")


# Save to combined csv
# Path to folder containing stock data
data_folder = "US_Stock_Data"

# Read and merge all CSV files
all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
df_list = []

for file in all_files:
    try:
        # Read CSV with multi-row header
        data = pd.read_csv(file, header=[0, 1], index_col=0, parse_dates=True)
        
        # Extract the ticker symbol from the second row of the header
        ticker = data.columns[0][1]  # Extract the ticker symbol (e.g., 'AAPL')
        
        # Rename columns to remove multi-index structure
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        
        # Add a 'Ticker' column
        data["Ticker"] = ticker
        
        df_list.append(data)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all stock data into one DataFrame

data_folder = "US_Stock_Data"
all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
df_list = []

for file in all_files:
    try:
        # Read CSV with multi-row header
        data = pd.read_csv(file, header=[0, 1], index_col=0, parse_dates=True)
        
        # Extract the ticker symbol from the second row of the header
        ticker = data.columns[0][1]  # Extract the ticker symbol (e.g., 'AAPL')
        
        # Rename columns to remove multi-index structure
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        
        # Add a 'Ticker' column
        data["Ticker"] = ticker
        
        df_list.append(data)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all stock data into one DataFrame
df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
df.columns.name = None
df = df.reset_index(drop=True)
df.to_csv("finance/finance.csv")