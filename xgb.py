import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

start_time = time.time()

print("Loading datasets...")

train = pd.read_csv("train.csv")
stores = pd.read_csv("stores.csv")

print("Datasets loaded.")

# -----------------------
# Date Features
# -----------------------

print("Creating date features...")

train['date'] = pd.to_datetime(train['date'])

train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['dayofweek'] = train['date'].dt.dayofweek

# -----------------------
# Merge
# -----------------------

print("Merging datasets...")

df = pd.merge(
    train,
    stores,
    on='store_nbr',
    how='left'
)

print("Merge complete.")
print("Shape:", df.shape)

# -----------------------
# Encoding
# -----------------------

print("Encoding categorical columns...")

df['family'] = LabelEncoder().fit_transform(df['family'])
df['city'] = LabelEncoder().fit_transform(df['city'])
df['state'] = LabelEncoder().fit_transform(df['state'])
df['type'] = LabelEncoder().fit_transform(df['type'])

print("Encoding complete.")

# -----------------------
# Sort
# -----------------------

print("Sorting data...")

df = df.sort_values(
    ['store_nbr', 'family', 'date']
)

print("Sorting complete.")

# -----------------------
# Lag Features
# -----------------------

print("Creating lag_7...")

df['lag_7'] = (
    df.groupby(['store_nbr', 'family'])['sales']
    .shift(7)
)

print("Creating lag_30...")

df['lag_30'] = (
    df.groupby(['store_nbr', 'family'])['sales']
    .shift(30)
)

print("Creating rolling_mean_7...")

df['rolling_mean_7'] = (
    df.groupby(['store_nbr', 'family'])['sales']
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

print("Lag features created.")

# -----------------------
# Remove Nulls
# -----------------------

print("Dropping null values...")

df = df.dropna()

print("New shape:", df.shape)

# -----------------------
# Features
# -----------------------

features = [
    'store_nbr',
    'family',
    'onpromotion',
    'year',
    'month',
    'day',
    'dayofweek',
    'city',
    'state',
    'type',
    'cluster',
    'lag_7',
    'lag_30',
    'rolling_mean_7'
]

X = df[features]
y = df['sales']

# -----------------------
# Time-Series Split
# -----------------------

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# -----------------------
# XGBoost
# -----------------------

print("Starting XGBoost training...")

xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

train_start = time.time()

xgb.fit(X_train, y_train)

train_end = time.time()

print("Training completed.")
print("Training Time:", round(train_end - train_start, 2), "seconds")

# -----------------------
# Prediction
# -----------------------

print("Predicting...")

pred = xgb.predict(X_test)

# -----------------------
# Evaluation
# -----------------------

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n===== RESULTS =====")
print("XGBoost RMSE:", rmse)
print("XGBoost R2:", r2)

print(
    "\nTotal Runtime:",
    round(time.time() - start_time, 2),
    "seconds"
)