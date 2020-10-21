import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LAMBDA = 0.001


# ======================

def preprocess(data):
    x1 = data[:, [0]]
    x2 = data[:, [1]]
    y = data[:, [2]]

    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    y_mean = np.mean(y)
    x1_std = np.std(x1)
    x2_std = np.std(x2)
    y_std = np.std(y)

    x1 = (x1 - x1_mean) / x1_std
    x2 = (x2 - x2_mean) / x2_std
    y = (y - y_mean) / y_std

    bias = np.expand_dims(np.ones([len(x1)]), axis=1)
    X = np.append(bias, x1, axis=1)
    X = np.append(X, x2, axis=1)


    return X, y


def rsquare(h_x,y):
    SST=0
    SSR=0
    y_bar=np.mean(y,axis=0)
    for i in range(len(y)):
        SST=SST+ (y[i]-y_bar)**2
        SSR=SSR+ (h_x[i]-y_bar)**2
    rsquare= SSR/SST
    return rsquare



def del_j(h_x, X, y, idx):
    del_j = 0

    for i in range(len(y)):
        del_j += (h_x[i] - y[i]) * X[i, idx]

    return del_j



def compute_J(h_x, y):
    m = np.shape(h_x)[0]
    temp = h_x - y
    J = (1 / (2 * m)) * (np.sum(temp ** 2))


    return J


def batch_grad(data, alpha=2e-8, epsilon=2e-10):
    X, y = preprocess(data)
    cost_value = []
    W = np.zeros([3, 1])
    w1 = []
    w2 = []
    eps = np.Inf

    epochs = 1000
    ep = 0
    h_x = np.dot(X, W)


    while (eps > epsilon) and (ep < epochs):
        print(ep)

        for j in range(len(W)):
            W[j] = W[j] - alpha * (del_j(h_x, X, y, j))

        w1.append(W[1, 0])
        w2.append(W[2, 0])

        h_x = np.dot(X, W)
        cost_value.append(compute_J(h_x, y))
        ep += 1

        if (len(cost_value) > 1):
            eps = np.abs(cost_value[-1] - cost_value[-2])

        print(cost_value)

    return cost_value, W, w1, w2,h_x


data_ = pd.read_csv('data.txt', header=None)
data_ = np.array(data_)
data=data_[:,1:]
# data=rand(data_)
target=data[:,2]
target=(target-np.mean(target))/np.std(target)
print(np.mean(target,axis=0))
np.random.shuffle(data)
ALPHA = 1e-7
X,y=preprocess(data)
values, w, w1, w2,h_x = batch_grad(data, alpha=ALPHA)
SE=compute_J(h_x,y)
RMSE=SE**0.5
rsquared=rsquare(h_x,y)
print('W values ', w)
# print(len(values))
print("rsquare:",rsquared)
print("RMSE",RMSE)
plt.plot(values)
plt.title('Cost vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

