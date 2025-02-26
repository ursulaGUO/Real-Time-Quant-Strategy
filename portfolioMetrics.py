import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dictionary mapping model names to their trade records CSV filenames.
models = {
    "xgboost": "portfolios/xgboost_trade_records.csv",
    "random_forest": "portfolios/random_forest_trade_records.csv",
    "gradient_boosting": "portfolios/gradient_boosting_trade_records.csv",
    "var": "portfolios/var_trade_records.csv"
}

# Load benchmark data from finance/marketIndex.csv
benchmark_df = pd.read_csv("finance/marketIndex.csv")
benchmark_df['observation_date'] = pd.to_datetime(benchmark_df['observation_date'])
benchmark_df = benchmark_df.sort_values('observation_date')
benchmark_df.set_index('observation_date', inplace=True)
# Compute daily returns for the SP500 index
benchmark_df['benchmark_return'] = benchmark_df['SP500'].pct_change(fill_method=None)

# Set constants
initial_capital = 20000
risk_free_rate_annual = 0.0425
risk_free_rate_daily = risk_free_rate_annual / 252

# Dictionary to store performance metrics for each model.
model_metrics = {}

# Process each model's trade records.
for model_name, csv_filename in models.items():
    print(f"\nProcessing model: {model_name}")

    # Load the trade records CSV for this model.
    trade_df = pd.read_csv(csv_filename)
    trade_df['Date'] = pd.to_datetime(trade_df['Date'])
    trade_df = trade_df.sort_values('Date')

    # Group by day: if multiple trades occur on the same day, take the last recorded Total Portfolio Value.
    daily_portfolio = trade_df.groupby(trade_df['Date'].dt.date).last()['Total Portfolio Value']

    # Convert index to datetime and resample daily (forward-fill missing days).
    daily_portfolio.index = pd.to_datetime(daily_portfolio.index)
    daily_portfolio = daily_portfolio.resample('D').ffill().dropna()

    # Compute daily portfolio returns.
    portfolio_returns = daily_portfolio.pct_change().dropna()

    # Align portfolio returns with benchmark returns on common dates.
    aligned_df = pd.concat([portfolio_returns, benchmark_df['benchmark_return']], axis=1, join='inner').dropna()
    aligned_portfolio_returns = aligned_df.iloc[:, 0]
    aligned_benchmark_returns = aligned_df.iloc[:, 1]

    # Calculate performance metrics.
    sharpe_ratio = ((aligned_portfolio_returns.mean() - risk_free_rate_daily) /
                    aligned_portfolio_returns.std()) * np.sqrt(252)
    volatility = aligned_portfolio_returns.std() * np.sqrt(252)
    if len(aligned_benchmark_returns) > 1:
        covariance = np.cov(aligned_portfolio_returns, aligned_benchmark_returns)[0, 1]
        benchmark_variance = np.var(aligned_benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
    else:
        beta = np.nan
    portfolio_annual_return = (1 + aligned_portfolio_returns.mean())**252 - 1
    benchmark_annual_return = (1 + aligned_benchmark_returns.mean())**252 - 1
    jensen_alpha = portfolio_annual_return - risk_free_rate_annual - beta * (benchmark_annual_return - risk_free_rate_annual)

    # Final portfolio value and profit & loss.
    final_value = daily_portfolio.iloc[-1]
    pnl = final_value - initial_capital

    # Store metrics in the dictionary.
    model_metrics[model_name] = {
        "Final Portfolio Value": final_value,
        "Total P&L": pnl,
        "Sharpe Ratio": sharpe_ratio,
        "Annualized Volatility": volatility,
        "Beta": beta,
        "Jensen's Alpha": jensen_alpha
    }

    # Optionally print metrics.
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total P&L: ${pnl:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Annualized Volatility: {volatility:.2f}")
    print(f"Beta: {beta:.2f}")
    print(f"Jensen's Alpha: {jensen_alpha:.2%}")

# --- Plotting with Scatter Plots ---

def plot_scatter_metric(metric_name, title, ylabel, color):
    plt.figure(figsize=(8, 6))
    # Gather model names and their metric values.
    models_list = list(model_metrics.keys())
    values = [model_metrics[m][metric_name] for m in models_list]
    
    # Create a scatter plot.
    plt.scatter(models_list, values, color=color, s=100)
    
    # Annotate each point with its value.
    for i, value in enumerate(values):
        plt.text(models_list[i], value, f"{value:.2f}", ha='center', va='bottom', fontsize=10)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Plot each performance metric using scatter plots.
plot_scatter_metric("Final Portfolio Value", "Final Portfolio Value by Model", "Final Value ($)", "skyblue")
plot_scatter_metric("Total P&L", "Total Profit & Loss by Model", "P&L ($)", "lightgreen")
plot_scatter_metric("Sharpe Ratio", "Sharpe Ratio by Model", "Sharpe Ratio", "salmon")
plot_scatter_metric("Annualized Volatility", "Annualized Volatility by Model", "Volatility", "orange")
plot_scatter_metric("Beta", "Beta by Model", "Beta", "purple")
plot_scatter_metric("Jensen's Alpha", "Jensen's Alpha by Model", "Jensen's Alpha", "brown")
