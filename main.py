# Data manipulation & fetching
import yfinance as yf
import pandas as pd
import numpy as np

# Modeling & evaluation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Visualization
import matplotlib.pyplot as plt


# Fetch historical gold prices for the past 5 years
gold = yf.Ticker("GC=F")  # GC=F is the ticker for Gold Futures on Yahoo Finance
df = gold.history(period="5y")

# Reset index to make 'Date' a column
df.reset_index(inplace=True)


# Show basic info about the DataFrame
print(df.info())

# Summary statistics (mean, std, min, max, etc.)
print(df.describe())


# Count missing values in each column
print(df.isnull().sum())


# Cap/floor extreme values at 1st and 99th percentiles
q_low = df['Close'].quantile(0.01)
q_high = df['Close'].quantile(0.99)
df = df[(df['Close'] >= q_low) & (df['Close'] <= q_high)]


# Reset and copy relevant columns
df = df[['Date', 'Close']].copy()

# Drop any remaining NaNs
df.dropna(inplace=True)

# Convert dates to numerical format
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

# Final features and target
X = df[['Date_ordinal']].values
y = df['Close'].values


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], color='darkorange')
plt.title('Gold Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(df['Close'], bins=50, color='gold', edgecolor='black')
plt.title('Distribution of Gold Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


df['RollingMean_30'] = df['Close'].rolling(window=30).mean()
df['RollingStd_30'] = df['Close'].rolling(window=30).std()

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Actual Price', alpha=0.5)
plt.plot(df['Date'], df['RollingMean_30'], label='30-Day Rolling Mean', color='blue')
plt.plot(df['Date'], df['RollingStd_30'], label='30-Day Rolling Std Dev', color='red', linestyle='--')
plt.title('Gold Price Trend & Volatility')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# Keep only necessary columns and make a copy to avoid SettingWithCopyWarning
df = df[['Date', 'Close']].copy()

# Drop missing values
df.dropna(inplace=True)

# Convert dates to ordinal format for modeling
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

# Define features (X) and target variable (y)
X = df[['Date_ordinal']].values
y = df['Close'].values


# Create polynomial features (e.g., X, X^2, X^3 for degree=3)
degree = 3
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)


# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict on the training data
y_pred = model.predict(X_poly)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predict using the trained model
y_pred = model.predict(X_poly)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Calculate R² Score
r2 = r2_score(y, y_pred)

# Print the results
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], y, label='Actual Prices', color='gold', s=10)
plt.plot(df['Date'], y_pred, label=f'Polynomial Regression (Degree {degree})', color='blue')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.title('Gold Price Prediction using Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
