from sklearn import linear_model
import pandas
import numpy as np
import os

data = pandas.read_csv(os.getcwd() + '/data/ex1data2.txt', header=None, names=['Square', 'Rooms', 'Price'])
data.head()
data.describe()
cols = data.shape[1]
yst = data.iloc[:, cols-1:cols].std()
ym = data.iloc[:, cols-1:cols].mean()
data = (data - data.mean()) / data.std()
data.head()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)

model = linear_model.LinearRegression()
model.fit(x, y)
f1 = model.predict(x)

err = 0
for i in range(len(x)):
    err += abs(y[i, 0] - f1[i, 0])
print('Значение средней ошибки: ', err/len(x))
print('Значение первой предсказанной цены: ', f1[0, 0] * yst + ym)

