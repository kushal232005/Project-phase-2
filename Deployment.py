import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Generate synthetic training data (replace with real dataset)
def generate_data(n=1000):
    data = pd.DataFrame({
        'PM2.5': np.random.uniform(10, 200, n),
        'PM10': np.random.uniform(20, 300, n),
        'NO2': np.random.uniform(5, 100, n),
        'SO2': np.random.uniform(1, 50, n),
        'CO': np.random.uniform(0.2, 2.0, n),
        'O3': np.random.uniform(10, 120, n),
        'temperature': np.random.uniform(5, 40, n),
        'humidity': np.random.uniform(20, 90, n),
        'wind_speed': np.random.uniform(0.5, 10, n)
    })
    data['AQI'] = (
        0.4 * data['PM2.5'] +
        0.3 * data['PM10'] +
        0.1 * data['NO2'] +
        0.05 * data['SO2'] +
        0.05 * data['O3'] +
        np.random.normal(0, 10, n)
    ).round(2)
    return data

data = generate_data()
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'temperature', 'humidity', 'wind_speed']
X = data[features]
y = data['AQI']

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X, y)

joblib.dump(model, 'model/xgb_aqi_model.pkl')
print("Model saved!")
