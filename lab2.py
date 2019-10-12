import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = 'data/ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data.head()
data.describe()

data = (data - data.mean()) / data.std()
data.head()

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)


def computeCost(X, y, theta):
    inner = (X * theta - y).T * (X * theta - y)
    return inner[0, 0] / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)

    for i in range(iters):
        cost[i] = computeCost(X, y, theta)
        theta = theta - alpha / len(X) * X.T * (X*theta - y)
    return theta, cost

alpha = 0.05
iters = 1000
theta = np.array([[0], [0], [0]])
th, c = gradientDescent(X, y, theta, alpha, iters)
th2, c2 = gradientDescent(X, y, theta, 0.01, iters)
th3, c3 = gradientDescent(X, y, theta, 0.001, iters)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(iters), c, 'r', label='Cost')
ax.plot(np.arange(iters), c2, 'g', label='Cost')
ax.plot(np.arange(iters), c3, 'b', label='Cost')
ax.legend(loc=1)
plt.show()
