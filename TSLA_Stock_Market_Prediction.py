import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb        
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error as mse, r2_score 
from xgboost import XGBRegressor 

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
tesla = pd.read_csv(r'C:\Users\arzik\Documents\ML Project\TSLA.csv')
tesla.info()
tesla['Date'] = pd.to_datetime(tesla['Date']) 
print(f'Total days = {tesla.Date.max() - tesla.Date.min()} days') 
print(tesla.describe())  


'''
# Box Plot of Stock Prices
tesla[['Open', 'High', 'Low', 'Close']].plot(kind='box', title='Tesla Stock Prices')  
plt.show() 
'''

# Line plot of Closing Prices
plt.figure(figsize=(10, 6))
plt.plot(tesla['Date'], tesla['Close'], label='Close Price')
plt.title('Tesla Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Split the data into train and test sets
X = np.array(tesla.index).reshape(-1, 1)  # Use index for X values
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) 

# Standardize the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train_scaled, Y_train)  
predictions = xgb_model.predict(X_test_scaled)  

# Calculate Errors
XGBRegressor_mse = mse(Y_test, predictions)  
XGBRegressor_r2 = r2_score(Y_test, predictions)  

print(f'Mean Squared Error: {XGBRegressor_mse:.4f}')  
print(f'R2 Score: {XGBRegressor_r2:.4f}')  

# **Fixed Prediction Plot**
plt.figure(figsize=(10, 6))

# Actual closing prices
plt.plot(tesla['Date'], tesla['Close'], label='Actual Close Price', color='blue')

# Ensure valid index values for predictions
test_dates = tesla['Date'].iloc[X_test.flatten().astype(int)]

# Predicted closing prices (corrected)
plt.scatter(test_dates, predictions, label='Predicted Close Price', color='orange')

plt.title('Tesla Stock Prices - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate a Simple Moving Average (SMA)
tesla['SMA'] = tesla['Close'].rolling(window=20).mean()

# **Simulated Trading Strategy with SMA**
initial_balance = 10000  
balance = initial_balance
position = 0  

for i in range(len(X_test)): 
    test_index = X_test.flatten()[i]

    # Avoid out-of-bounds errors
    if test_index >= len(tesla):
        continue  

    current_price = tesla.iloc[test_index]['Close']
    predicted_price = predictions[i]
    sma_price = tesla.iloc[test_index]['SMA']

    # Ensure SMA is not NaN
    if pd.isna(sma_price):
        continue

    if predicted_price > sma_price and balance >= current_price:
        shares_to_buy = int(balance // current_price)
        if shares_to_buy > 0:  
            position += shares_to_buy  
            balance -= shares_to_buy * current_price  
            print(f"Buying {shares_to_buy} shares at {current_price:.2f}")  

    elif predicted_price < sma_price and position > 0:
        balance += position * current_price
        print(f"Selling {position} shares at {current_price:.2f}")
        position = 0  

# **Final Balance Calculation**
if len(X_test) > 0:
    last_test_index = X_test.flatten()[-1]
    if last_test_index < len(tesla):
        final_balance = balance + (position * tesla.iloc[last_test_index]['Close'])
    else:
        final_balance = balance  
else:
    final_balance = balance

profit = final_balance - initial_balance

print(f"Final balance: ${final_balance:.2f}")
print(f"Profit: ${profit:.2f}")





