import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
stores = pd.read_csv("stores.csv")
holidays = pd.read_csv("holidays_events.csv")
transactions = pd.read_csv("transactions.csv")


train['date'] = pd.to_datetime(train['date'])

#print(train['date'].min())
#print(train['date'].max())
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['dayofweek'] = train['date'].dt.dayofweek

df = pd.merge(
    train,
    stores,
    on='store_nbr',
    how='left'
)

from sklearn.preprocessing import LabelEncoder

le_family = LabelEncoder()
le_city = LabelEncoder()
le_state = LabelEncoder()
le_type = LabelEncoder()

df['family'] = le_family.fit_transform(df['family'])
df['city'] = le_city.fit_transform(df['city'])
df['state'] = le_state.fit_transform(df['state'])
df['type'] = le_type.fit_transform(df['type'])
y = df['sales']
X = df[
    [
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
        'cluster'
    ]
]
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
r2_lr = r2_score(y_test, pred_lr)

print("Linear Regression RMSE:", rmse_lr)
print("Linear Regression R2:", r2_lr)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf = r2_score(y_test, pred_rf)

print("Random Forest RMSE:", rmse_rf)
print("Random Forest R2:", r2_rf)