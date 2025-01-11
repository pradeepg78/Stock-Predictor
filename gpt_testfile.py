import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Define the stock symbol and the time period
stock_symbol = 'AAPL'  # Apple Inc.
start_date = '2015-01-01'
end_date = '2023-01-01'

# Download the stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Use the 'Close' price for prediction
data = stock_data['Close'].values
data = data.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the window size for the prediction
window_size = 60

# Prepare the data
X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Transform the predictions back to the original scale
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# Transform the y_test back to the original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Prepare the data for future prediction
recent_data = scaled_data[-window_size:]
recent_data = recent_data.reshape(1, -1)

# Make the prediction
future_prediction = model.predict(recent_data)
future_prediction = scaler.inverse_transform(future_prediction.reshape(-1, 1))

print(f'Predicted future stock price: {future_prediction[0][0]}')
