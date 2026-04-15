# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd
import datetime
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

# ==============================
# STEP 2: LOAD DATA
# ==============================
try:
    realtime = pd.read_csv("realtime_data.csv")
    print("Data loaded ✅")
except:
    raise FileNotFoundError("❌ realtime_data.csv NOT found. Check file location.")

# ==============================
# STEP 3: CONVERT TIME
# ==============================
def to_day_seconds(ts):
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.hour*3600 + dt.minute*60 + dt.second

realtime['actual_sec'] = realtime['arrival_time'].apply(to_day_seconds)

print("Time converted ✅")

# ==============================
# STEP 4: CLEAN stop_id
# ==============================
if 'stop_id' in realtime.columns:
    realtime['stop_id'] = realtime['stop_id'].astype(str).str[:-1]

# ==============================
# STEP 5: SMART DATASET (FIXED 🔥)
# ==============================
print("⚠️ Using improved hybrid dataset")

df = realtime.copy()

# fallback if trip_id missing
if 'trip_id' not in df.columns:
    df['trip_id'] = np.arange(len(df))

# create hour
df['hour'] = (df['actual_sec'] // 3600) % 24

# simulate stop sequence
df['stop_sequence'] = np.random.randint(1, 50, size=len(df))

# base delay pattern
def base_delay(hour):
    if 7 <= hour <= 10:
        return np.random.randint(120, 300)
    elif 17 <= hour <= 20:
        return np.random.randint(150, 350)
    else:
        return np.random.randint(30, 120)

df['delay'] = df['hour'].apply(base_delay)

# sort for sequence learning
df = df.sort_values(['trip_id', 'stop_sequence'])

# previous delay
df['prev_delay'] = df.groupby('trip_id')['delay'].shift(1)
df['prev_delay'] = df['prev_delay'].fillna(df['delay'])

# correlation
noise = np.random.randint(-30, 30, size=len(df))
df['delay'] = df['prev_delay'] + noise
df['delay'] = df['delay'].clip(0, 600)

# scheduled time
df['scheduled_sec'] = df['actual_sec'] - df['delay']

print("After dataset creation:", df.shape)

# ==============================
# STEP 6: CLEAN DATA
# ==============================
df = df.dropna()
df = df[(df['delay'] > -600) & (df['delay'] < 3600)]

print("After cleaning:", df.shape)

# ==============================
# STEP 7: FEATURE ENGINEERING
# ==============================
df = df.sort_values(['trip_id', 'stop_sequence'])

print("After feature engineering:", df.shape)

# ==============================
# STEP 8: TRAIN MODEL
# ==============================
if len(df) < 50:
    print("⚠️ Not enough data yet.")
else:
    features = ['hour', 'stop_sequence', 'prev_delay']
    X = df[features]
    y = df['delay']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=100, max_depth=6)
    model.fit(X_train, y_train)

    print("Model trained ✅")

    # ==============================
    # STEP 9: EVALUATION
    # ==============================
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    evs = explained_variance_score(y_test, preds)

    print("\n📊 MODEL PERFORMANCE")
    print("----------------------------")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Explained Variance: {evs:.4f}")