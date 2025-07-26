import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock_data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

# Display the first few rows of the data
stock_data.head()

# Data Preprocessing
stock_data.dropna(inplace=True)
stock_data['Daily Return'] = stock_data[('Close', 'AAPL')].pct_change()
print(stock_data[['Daily Return']].head())

# Plot Adjusted Close Price (using the 'Close' price for AAPL)
plt.figure(figsize=(6,3))
plt.plot(stock_data[('Close', 'AAPL')], label='Adjusted Close Price')
plt.title('AAPL Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.savefig("aapl_stock_price.png", dpi=300)
plt.show()

# Plot Daily Returns
plt.figure(figsize=(6,3))
plt.plot(stock_data['Daily Return'], label='Daily Return', color='orange')
plt.title('AAPL Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.savefig("aapl_daily_returns.png", dpi=300)
plt.show()

# Calculate 50-day and 200-day moving averages
stock_data['50-Day MA'] = stock_data[('Close', 'AAPL')].rolling(window=50).mean()
stock_data['200-Day MA'] = stock_data[('Close', 'AAPL')].rolling(window=200).mean()

# Plot Adjusted Close and Moving Averages
plt.figure(figsize=(6,3))
plt.plot(stock_data[('Close', 'AAPL')], label='Adjusted Close Price')
plt.plot(stock_data['50-Day MA'], label='50-Day Moving Average')
plt.plot(stock_data['200-Day MA'], label='200-Day Moving Average')
plt.title('AAPL Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig("aapl_moving_averages.png", dpi=300)
plt.show()

data = stock_data.dropna()

X = data[['50-Day MA', '200-Day MA']] 
y = data[('Close', 'AAPL')] 

split_index = int(0.8 * len(data)) 
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

X_train = np.c_[np.ones(len(X_train)), X_train] 
X_test = np.c_[np.ones(len(X_test)), X_test] 

# Compute the coefficients using the Normal Equation
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Make predictions on the test set
y_pred = X_test @ theta


mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")


# Visualize the Predictions

# Plot actual vs predicted stock prices
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual Stock Price')
plt.plot(y_test.index, y_pred, label='Predicted Stock Price', color='orange')
plt.title('AAPL Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.savefig("aapl_actual_vs_predicted.png", dpi=300)
plt.show()


# Calculate the daily price changes
delta = stock_data[('Close', 'AAPL')].diff()

# Calculate gains and losses
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

# Calculate the relative strength (RS)
rs = gain / loss

# Calculate the Relative Strength Index (RSI)
rsi = 100 - (100 / (1 + rs))

# Add RSI to the dataframe
stock_data['RSI'] = rsi

# Plot RSI
plt.figure(figsize=(10, 6))
plt.plot(stock_data['RSI'], label='RSI')
plt.title('AAPL RSI (Relative Strength Index)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.savefig("aapl_rsi.png", dpi=300)
plt.show()