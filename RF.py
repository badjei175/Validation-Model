import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_excel("surgery_data.xlsx", parse_dates=["Date"], index_col="Date")

# Display the first few rows
print(df.head())

# Ensure the data is sorted by date
df = df.sort_index()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df, label="Surgery Count")
plt.xlabel("Date")
plt.ylabel("Number of Surgeries")
plt.legend()
plt.show()

# Create lag features for ML models
for lag in range(1, 8):  # Using past 7 days
    df[f"lag_{lag}"] = df["Surgery_Count"].shift(lag)

df.dropna(inplace=True)

# Splitting data
train_size = int(0.8 * len(df))
train, test = df.iloc[:train_size], df.iloc[train_size:]

X_train, y_train = train.drop(columns=["Surgery_Count"]), train["Surgery_Count"]
X_test, y_test = test.drop(columns=["Surgery_Count"]), test["Surgery_Count"]

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest
print("Random Forest MAE:", mean_absolute_error(y_test, rf_predictions))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_predictions)))

forecast_horizon = 8  # Number of days to forecast
last_known_data = X_test.iloc[-1].values.reshape(1, -1)  # Last known input features
future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")

future_predictions = []  # Store predictions

for date in future_dates:
    # Predict next day's surgery count
    predicted_value = rf_model.predict(last_known_data)[0]
    future_predictions.append((date, predicted_value))

    # Update the input features for next iteration
    last_known_data = np.roll(last_known_data, shift=-1)
    last_known_data[0, -1] = predicted_value  # Replace oldest lag with the new prediction

# Convert to DataFrame
forecast_df = pd.DataFrame(future_predictions, columns=["Date", "Predicted_Surgery_Count"]).set_index("Date")

# Print Forecasted Values
print(forecast_df)

# Optional: Save to Excel
forecast_df.to_excel("surgery_forecast_rf.xlsx")

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test, label="Actual", color="blue")
plt.plot(test.index, rf_predictions, label="RF Forecast (Test)", color="green")
plt.plot(forecast_df.index, forecast_df["Predicted_Surgery_Count"], label="RF Forecast (Future)", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Number of Surgeries")
plt.legend()
plt.show()

