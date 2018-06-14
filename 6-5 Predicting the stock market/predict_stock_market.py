#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:32:13 2018

@author: Soo Hyeon Kim
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 50)
np.set_printoptions(suppress=True)

### Introduction ---------------------

df = pd.read_csv("sphist.csv")
## change 'Date' column dtype to datetime
df['Date'] = pd.to_datetime(df['Date'])
## sort in ascending order by 'Date'. i.e., old - recent
df.sort_values(by='Date', ascending=True, inplace=True)

### adding new features ---------------
## to use rolling based on datetime, we need to have 'Date' index
df.set_index('Date', inplace=True)

df['avg_price_5_days'] = df['Close'].resample('1D').sum().rolling(window=5).mean().shift(1)
df['avg_price_30_days'] = df['Close'].resample('1D').sum().rolling(window=30).mean().shift(1)
df['avg_price_365_days'] = df['Close'].resample('1D').sum().rolling(window=365).mean().shift(1)
df['ratio_price_5days_to_1yr'] = df['avg_price_5_days'] / df['avg_price_365_days']
df['std_price_5_days'] = df['Close'].resample('1D').sum().rolling(window=5).std().shift(1)
df['std_price_365_days'] = df['Close'].resample('1D').sum().rolling(window=365).std().shift(1)
                     
######   ------------ for improvement (later checked) --------------
df['avg_volume_5_days'] = df['Volume'].resample('1D').sum().rolling(window=5).mean().shift(1)
df['avg_volume_365_days'] = df['Volume'].resample('1D').sum().rolling(window=365).mean().shift(1)
df['ratio_volume_5days_to_1yr'] = df['avg_volume_5_days'] / df['avg_volume_365_days']
df['std_volume_5_days'] = df['Close'].resample('1D').sum().rolling(window=5).std().shift(1)
df['std_volume_365_days'] = df['Close'].resample('1D').sum().rolling(window=365).std().shift(1)
######   -----------------------------------------------------------

## reset index
df.reset_index(inplace=True, drop=False)

## check from where 'avg_365_days' value exists (i.e., not NaN)
#print(df['avg_365_days'].isnull().idxmin())
#print(df.iloc[df['avg_365_days'].isnull().idxmin()]['Date'])

## remove below index (before 1951-01-03)
df = df[df['Date'] > dt(year=1951, month=1, day=2)]
## drop any rows with NaN values
df.dropna(axis=0)

### Linear Regression ----------------
train = df[df['Date'] < dt(year=2013, month=1, day=1)]
test = df[df['Date'] >= dt(year=2013, month=1, day=1)]

### Train 
## leave out original columns. bc, these all contain knowledge of the future that you don't want to feed the model
features = df.columns.drop(['Close', 'High', 'Low', 'Open', 'Volume', \
                            'Adj Close', 'Date', 'avg_volume_5_days', \
                            'avg_volume_365_days', 'std_volume_5_days', \
                            'std_volume_365_days']).values
target = 'Close'

model = LinearRegression()
model.fit(train[features], train[target])
y_hat = model.predict(test[features])

mae = MAE(test[target], y_hat)
print('Mean Absolute Error:', mae)
train_score = model.score(train[features], train[target])
test_score = model.score(test[features], test[target])
print('Training data prediction score:', train_score)
print('Test data prediction score:', test_score)
print()

### Try improvement ---------------

features = df.columns.drop(['Close', 'High', 'Low', 'Open', 'Volume', \
                            'Adj Close', 'Date']).values
target = 'Close'

model = LinearRegression()
model.fit(train[features], train[target])
y_hat = model.predict(test[features])

mae = MAE(test[target], y_hat)
print("With Added features")
print('Mean Absolute Error:', mae)
train_score = model.score(train[features], train[target])
test_score = model.score(test[features], test[target])
print('Training data prediction score:', train_score)
print('Test data prediction score:', test_score)
 


