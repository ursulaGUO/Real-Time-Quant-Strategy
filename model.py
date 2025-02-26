import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
import random
import matplotlib.pyplot as plt

random.seed(999)

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
df = pd.concat(df_list, ignore_index=True)
df.columns.name = None
df = df.reset_index(drop=True)
df = df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

# Ensure 'Date' column is datetime index
df.set_index("Date", inplace=True)

# Feature Engineering Functions
def compute_technical_indicators(df):
    """Computes various technical indicators for momentum trading strategies."""
    
    df["Prev_Close"] = df.groupby("Ticker")["Close"].shift(1)  # Previous day's close price

    # Simple Moving Average (SMA)
    df["SMA_10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=10).mean())
    df["SMA_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=20).mean())

    # Exponential Moving Average (EMA)
    df["EMA_10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df["EMA_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())

    # Moving Average Convergence Divergence (MACD)
    df["MACD"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean())

    # Bollinger Bands (20-day SMA ± 2*std)
    rolling_mean = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=20).mean())
    rolling_std = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=20).std())
    df["Bollinger_Upper"] = rolling_mean + (rolling_std * 2)
    df["Bollinger_Lower"] = rolling_mean - (rolling_std * 2)

    # Volatility (5-day rolling std)
    df["Volatility"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change().rolling(5).std())

    return df

# Apply feature engineering
df = compute_technical_indicators(df)

# Target Variable (Next Day's Close)
df["Next_Close"] = df.groupby("Ticker")["Close"].shift(-1)

# Drop NaN values after feature creation
df.dropna(inplace=True)

# Define Features and Target
features = ["Prev_Close", "SMA_10", "SMA_20", "EMA_10", "EMA_20", "MACD", 
            "Bollinger_Upper", "Bollinger_Lower", "Volatility", "Volume"]
X = df[features]
y = df["Next_Close"]

""" 
Trying a variety of ML methods
"""

# Train Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=999, n_jobs=-1)
model1.fit(X_train, y_train)

# Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model2 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05,random_state=999)
model2.fit(X_train, y_train)

# Train Gradient Boosting
model3 = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=800,
    subsample=0.6,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=6,
    min_impurity_decrease=0.0,
    init=None,
    random_state=999,
    max_features=0.8,
    alpha=0.9,
    verbose=1,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    tol=0.0001
)

model3.fit(X_train, y_train)


# Compute Actual and Predicted Movement Direction
def direction(X_test, y_test, y_pred, pred_df):
    pred_df["Actual_Direction"] = (y_test.values > X_test["Prev_Close"].values).astype(int)
    pred_df["Predicted_Direction"] = (y_pred > X_test["Prev_Close"].values).astype(int)
    direction_accuracy = (pred_df["Actual_Direction"] == pred_df["Predicted_Direction"]).mean()
    return direction_accuracy



# Check performance metrics
models_dict = {"Random_Forest": model1,
               "XGBoost": model2,
               "Gradient_Boosting": model3}
rmse_scores = {}
direction_accuracy_scores = {}
for key, model in models_dict.items():
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}, index=y_test.index)
    rmse = root_mean_squared_error(y_test, y_pred)
    direction_accuracy = direction(X_test, y_test, y_pred, pred_df)
    print(f"Model {key} \n RMSE: {rmse}; Direction Accuracy: {direction_accuracy}")
    rmse_scores[key] = rmse
    direction_accuracy_scores[key] = direction_accuracy

    # Save models
    model_filename = f"{key}_stock_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model,file)
    print(f"Model saved as {model_filename}")

plt.figure(figsize=(4,4))
for model, value in rmse_scores.items():
    plt.scatter(model, value, color="darkgreen")
    plt.text(model, value, f"{value:.2f}", fontsize=10, ha='center', va='bottom')
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE for Different Models")
plt.show()

plt.figure(figsize=(4,4))
for model, value in direction_accuracy_scores.items():
    plt.scatter(model, value, color="darkgreen")
    plt.text(model, value, f"{value:.2f}", fontsize=10, ha='center', va='bottom')
plt.xlabel("Model")
plt.ylabel("Directional Accuracy")
plt.title("Directional Accuracy for Different Models")
plt.show()

