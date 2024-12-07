import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('/content/EGX 30_historical_data.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
columns_to_clean = ['Price', 'Open', 'High', 'Low']
for col in columns_to_clean:
    data[col] = data[col].replace(',', '', regex=True).astype(float)
data['Change %'] = data['Change %'].str.replace('%', '', regex=False).astype(float)
data['Prev_Price'] = data['Price'].shift(1)
data['Prev_Open'] = data['Open'].shift(1)
data['Prev_High'] = data['High'].shift(1)
data['Prev_Low'] = data['Low'].shift(1)
data['Prev_Change'] = data['Change %'].shift(1)
data = data.dropna()


features = ['Open', 'High', 'Low', 'Change %', 'Prev_Price', 'Prev_Open', 'Prev_High', 'Prev_Low', 'Prev_Change']
X = data[features]
y = data['Price']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False, random_state=42)

#Linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

#Random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

#Neural network regressor
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=65, batch_size=32, verbose=0)
nn_pred = nn_model.predict(X_test).flatten()
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return mse, rmse, r2

#Model evaluation
print("Evaluation Results:")
evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, rf_pred, "Random Forest Regressor")
evaluate_model(y_test, nn_pred, "Neural Network Regressor")

#Plotting results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Price', alpha=0.7)
plt.plot(lr_pred, label='Linear Regression Predictions', linestyle='--')
plt.plot(rf_pred, label='Random Forest Predictions', linestyle='--')
plt.plot(nn_pred, label='Neural Network Predictions', linestyle='--')
plt.title('Comparison of Actual vs Predicted Prices')
plt.xlabel('Test Set Index')
plt.ylabel('Price')
plt.legend()
plt.show()