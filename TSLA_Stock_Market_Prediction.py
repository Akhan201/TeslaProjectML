import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt # for plotting
import seaborn as sb        # for plotting
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.preprocessing import StandardScaler # for standardizing the data
from sklearn.metrics import mean_squared_error as mse # for calculating the mean squared error
from sklearn.linear_model import LinearRegression # for linear regression
from sklearn.metrics import r2_score # for calculating the r2 score

import warnings
warnings.filterwarnings('ignore')

tesla = pd.read_csv(r'C:\Users\arzik\Documents\ML Project\TSLA.csv')
tesla.info() # to get the information of the data
tesla['Date'] = pd.to_datetime(tesla['Date']) # to convert the date column to datetime
print(f'Total days = {tesla.Date.max() - tesla.Date.min()} days') # to get the total days of the data
tesla.describe() # to get the statistical information of the data

tesla[['Open', 'High', 'Low', 'Close']].plot(kind='box', title='Tesla Stock Prices') # to plot the box plot of the data
plt.show() # to show the plot

# Plot the data using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(tesla['Date'], tesla['Close'], label='Close Price')
plt.title('Tesla Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Split the data into train and test sets
X = np.array(tesla.index).reshape(-1, 1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)  # Changed random_state to an integer
scaler = StandardScaler().fit(X_train)

# Standardize the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
lm = LinearRegression()
lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)

# Calculate the mean squared error and R2 score
mse_value = mse(Y_test, predictions)
r2 = r2_score(Y_test, predictions)
print(f'Mean Squared Error: {mse_value}')
print(f'R2 Score: {r2}')

# Plot the predictions using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(tesla['Date'], tesla['Close'], label='Actual Close Price')
plt.plot(tesla['Date'].iloc[X_test.flatten().astype(int)], predictions, label='Predicted Close Price', linestyle='dashed')
plt.title('Tesla Stock Prices - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()







'''
layout = go.Layout
(title='Tesla Stock Prices', xaxis=dict(title='Date'), yaxis=dict(title='Price'));
'''
"""
 initial_balance = 10000  # Starting balance 
balance = initial_balance
position = 0  # Number of shares

for i in range(len(X_test)):
    current_price = X_test.iloc[i]['Close']
    predicted_price = predictions[i]

    if predicted_price > current_price and balance >= current_price:
        # Buy stock
        shares_to_buy = int(balance // current_price)  # Buy whole shares only
        if shares_to_buy > 0:  # Ensure we are buying at least one share
            position += shares_to_buy
            balance -= shares_to_buy * current_price
            print(f"Buying {shares_to_buy} shares at {current_price:.2f}")

    elif predicted_price < current_price and position > 0:
        # Sell stock
        balance += position * current_price
        print(f"Selling {position} shares at {current_price:.2f}")
        position = 0

# Calculate final balance including the value of the remaining shares
final_balance = balance + (position * X_test.iloc[-1]['Close'])
profit = final_balance - initial_balance
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit: ${profit:.2f}")
"""





