# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ==============================
# STEP 2: LOAD DATA
# ==============================
stop_times = pd.read_csv("stop_times.txt")
trips = pd.read_csv("trips.txt")
routes = pd.read_csv("routes.txt")

print("Data Loaded ✅")

# ==============================
# STEP 3: CLEAN + CONVERT TIME
# ==============================
def to_seconds(t):
    try:
        h, m, s = map(int, t.split(":"))
        return h*3600 + m*60 + s
    except:
        return None

stop_times['scheduled_sec'] = stop_times['arrival_time'].apply(to_seconds)

# remove invalid rows
stop_times = stop_times.dropna(subset=['scheduled_sec'])

print("Time converted ✅")

# ==============================
# STEP 4: MERGE DATA
# ==============================
df = stop_times.merge(trips, on='trip_id')
df = df.merge(routes, on='route_id')

# keep only subway
df = df[df['route_type'] == 1]

print("Data merged ✅")

# ==============================
# STEP 5: FEATURE ENGINEERING
# ==============================

# time-based features
df['hour'] = (df['scheduled_sec'] // 3600) % 24

# sort BEFORE lag features
df = df.sort_values(['trip_id', 'stop_sequence'])

# simulate delay (temporary)
np.random.seed(42)
df['delay'] = np.random.randint(-120, 300, size=len(df))

# actual arrival
df['actual_sec'] = df['scheduled_sec'] + df['delay']

# previous delay feature
df['prev_delay'] = df.groupby('trip_id')['delay'].shift(1)

# encode route
df['route_id'] = df['route_id'].astype('category').cat.codes

# drop missing
df = df.dropna()

print("Features created ✅")

# ==============================
# STEP 6: PREPARE DATASET
# ==============================
features = ['hour', 'stop_sequence', 'prev_delay', 'route_id']

X = df[features]
y = df['delay']

print("Dataset ready:", X.shape)

# ==============================
# STEP 7: TRAIN MODEL
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(n_estimators=100, max_depth=6)

model.fit(X_train, y_train)

print("Model trained ✅")

# ==============================
# STEP 8: EVALUATE MODEL
# ==============================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("Mean Absolute Error (seconds):", mae)

# ==============================
# STEP 9: TEST PREDICTION
# ==============================
sample = X_test.iloc[0:1]

pred_delay = model.predict(sample)[0]

print("Sample prediction (delay in seconds):", pred_delay)