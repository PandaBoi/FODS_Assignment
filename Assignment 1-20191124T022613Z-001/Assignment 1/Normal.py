import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# sys.path.append('../')




def preprocess(data):
    x1 = data[:, [0]]
    x2 = data[:, [1]]
    y = data[:, [2]]

    # x1_mean =np.mean(x1)
    # x2_mean =np.mean(x2)
    # y_mean =np.mean(y)
    # x1_std = np.std(x1)
    # x2_std = np.std(x2)
    # y_std = np.std(y)

    x1 = (x1 - x1.mean()) / x1.std()
    x2 = (x2 - x2.mean()) / x2.std()
    y = (y - y.mean()) / y.std()

    # print(max(x1))

    bias = np.expand_dims(np.ones([len(x1)]), axis=1)
    X = np.append(bias, x1, axis=1)
    X = np.append(X, x2, axis=1)
    # plt.scatter(x1, y)
    # plt.show()
    # print(np.shape(X),np.shape(y))

    return X, y


def vector_LR(data):
    X, y = preprocess(data)

    W = np.zeros([3, 1])
    inv_inside = np.linalg.inv(np.dot(np.transpose(X), X))
    W = np.dot(np.dot(inv_inside, np.transpose(X)), y)

    return W

def rsquare(h_x,y):
    SST=0
    SSR=0
    y_bar=np.mean(y,axis=0)
    for i in range(len(y)):
        SST=SST+ (y[i]-y_bar)**2
        SSR=SSR+ (h_x[i]-y_bar)**2
    rsquare= SSR/SST
    return rsquare

def cost_function(X,W,y):

        h_x = np.matmul(X, W)
        h_x = h_x.reshape(len(h_x), 1)
        error = np.subtract(h_x, y)
        cost = np.sum(error ** 2) / (2 * len(y))
        return cost,h_x


data_ = pd.read_csv('FODS.txt', header=None)
data_ = np.array(data_)
data=data_[:,1:]
X,y=preprocess(data)
W = vector_LR(data)
SE,h_x=cost_function(X,W,y)
RMSE=SE**0.5
rsquared=rsquare(h_x,y)
print("rsquare:",rsquared)
print("RMSE",RMSE,SE)
# data=rand(data_)
# target=data[:,2]
# target=(target-np.mean(target))/np.std(target)


print(W)
