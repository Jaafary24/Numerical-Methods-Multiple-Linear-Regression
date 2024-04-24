# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('UBER.csv')


df = df.select_dtypes(exclude='object')
df = df.dropna()
drop_these= ['Close','Adj Close']
X = df.drop(columns = drop_these, axis=1).values.T
y = df['Close'].values

mat_X = np.zeros((X.shape[0]+1,X.shape[0]+1))
mat_y = np.zeros((X.shape[0]+1))

mat_X[0,0] = X.shape[1]
mat_y[0] = y.sum()
for i in range(X.shape[0]):
  x_sum =  X[i].sum()

  mat_X[i+1,0] = x_sum
  mat_X[0,i+1] = x_sum

  # X columns
  for j in range( X.shape[0]):

    x_sum = (X[i] * X[j]).sum()

    mat_X[i+1,j+1] = x_sum
    if i != j:
      mat_X[j+1,i+1] = x_sum
    # print(mat)

  # y column
  mat_y[i+1] = (y * X[i]).sum()

#Linear Regression
linear_w = np.linalg.solve(mat_X, mat_y).round(8)

y_preds_list = []

for i in range(X.shape[1]):
  y_pred = linear_w[0]
  for j in range(X.shape[0]):
    y_pred += linear_w[j+1] * X[j, i]
  y_preds_list.append(y_pred)

#standard deviation
stand_dev = ((y - y.mean()) ** 2) .sum()

#standard error
stand_err = ((y - y_preds_list) ** 2).sum()

#correlation coefficient
r_value = np.sqrt((stand_dev-stand_err)/stand_dev).round(4)
