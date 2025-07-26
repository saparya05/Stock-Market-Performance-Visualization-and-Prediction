# 📈 Stock Price Analysis & Prediction

This project performs historical stock data analysis and simple predictive modeling for **Apple Inc. (AAPL)** using Python libraries such as `yfinance`, `pandas`, `NumPy`, and `matplotlib`.

## 🔍 Overview

The objective of this project is to:

- Download and explore historical AAPL stock data (2015–2024)
- Calculate daily returns and moving averages
- Visualize stock prices, returns, and technical indicators like RSI
- Predict stock prices using a simple linear regression model (Normal Equation)
- Evaluate the model with Mean Squared Error (MSE)

---

## 🧰 Tools & Libraries

- **Python**
- `yfinance` - Fetch historical stock data
- `pandas` - Data manipulation
- `numpy` - Numerical computation
- `matplotlib` - Data visualization

---

## 📊 Features

- Download historical data using `yfinance`
- Calculate daily returns
- Plot adjusted close price and daily returns
- Calculate and visualize 50-day and 200-day moving averages
- Build a linear regression model to predict stock prices
- Visualize actual vs. predicted stock prices
- Calculate and plot Relative Strength Index (RSI)

---

## 📥 Installation

1. **Clone the repository**
```bash
pip install -r requirements.txt
```
2. **Clone the repository**
```bash
git clone https://github.com/saparya05/Employee-Management-System
cd aapl-stock-analysis
```
3. **Clone the repository**
```bash
python stock_analysis.py
```
## 📈 Sample Visualizations
- Adjusted Close Price Over Time
- Daily Return Fluctuations
- 50-Day and 200-Day Moving Averages
- Actual vs. Predicted Stock Prices
- RSI (Relative Strength Index)

## 📉 Model Evaluation
We used the Normal Equation for linear regression:

- Features: 50-day and 200-day moving averages
- Target: AAPL closing price
- Metric: Mean Squared Error (MSE)

## 📌 Notes
- The model is a basic linear regression and does not account for real market complexity.
- This is meant for educational and exploratory purposes.

## 📜 License
This project is open-source and free to use under the MIT License.
