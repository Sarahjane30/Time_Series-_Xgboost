import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
stores = pd.read_csv("stores.csv")
holidays = pd.read_csv("holidays_events.csv")
transactions = pd.read_csv("transactions.csv")

print(train.head())
print(train.shape)
print("train",train.isnull().sum())
print("store",stores.isnull().sum())
print("holiday",holidays.isnull().sum())
print("tra",transactions.isnull().sum())