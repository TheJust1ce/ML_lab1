import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = 'data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
data.describe()

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


alpha = 0.02
iters = 1000
theta = np.array([[0], [0]])
th, c = gradientDescent(X, y, theta, alpha, iters)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = th[0, 0] + (th[1, 0] * x)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted profit vs. Population Size')

_, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(iters), c, 'g', label='target function')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

plt.show()
