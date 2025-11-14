import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import tensorflow as tf
from finta import TA  


n_past = 60      
n_future = 1     
model_save_path = 'bitcoin_price_predictor.keras'


base_features = ['Close', 'High', 'Low', 'Volume']

derived_features = ['SMA_20', 'RSI_14']

features = base_features + derived_features
n_features = len(features)

data_filename = 'btc_1d_data_2018_to_2025.csv'
date_col = 'Open time'                        



print(f"Loading data from {data_filename}...")
df = pd.read_csv(data_filename)

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col)

df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Volume": "volume"}, inplace=True)

base_features = ['close', 'high', 'low', 'volume']
features = base_features + derived_features


df = df.set_index(date_col)


for col in base_features:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in the CSV.")
        exit()

print("Calculating technical indicators...")

df['SMA_20'] = TA.SMA(df, period=20)
df['RSI_14'] = TA.RSI(df, period=14)


df = df.dropna()

print("Feature calculation complete.")
print(df.head()) 


training_data = df[features].values
print(f"Data loaded. Shape: {training_data.shape}")

scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_data)


import pickle
scaler_filename = 'full_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved to {scaler_filename}")


X_train = []
y_train = []

print(f"Generating training sequences... (Using {n_features} features)")
for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i-n_past:i, 0:n_features])
    y_train.append(training_set_scaled[i:i + n_future, 0:n_features].flatten())
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
print(f"Created {len(X_train)} training samples. X_train shape: {X_train.shape}")



print("Building 'Trader' Bi-LSTM model...")
regressor = Sequential()

regressor.add(Bidirectional(LSTM(units=100, return_sequences=True), 
                            input_shape=(n_past, n_features)))
regressor.add(Dropout(0.3))
regressor.add(Bidirectional(LSTM(units=100, return_sequences=True)))
regressor.add(Dropout(0.3))
regressor.add(Bidirectional(LSTM(units=100)))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=n_features))

regressor.compile(optimizer='adam', loss='mean_squared_error')
print("Model built. Starting training...")


regressor.fit(X_train, y_train, epochs=100, batch_size=32)
print("Training complete.")


regressor.save(model_save_path)
print(f"Model saved as '{model_save_path}'.")