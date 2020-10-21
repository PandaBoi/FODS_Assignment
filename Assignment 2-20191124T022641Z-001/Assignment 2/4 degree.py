import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# LAMBDA = 0.001


# ======================

def preprocess(data):
    y = data[:, [-1]]
    # tuple = [np.shape(data)]
    # out = list(itertools.chain(*tuple))
    # columns=out[1]
    # for i in range(columns):
    #     data[:,[i]]=(data[:,[i]]-np.mean(data[:,[i]]))/np.std(data[:,[i]])
    bias = np.expand_dims(np.ones([len(y)]), axis=1)
    X=data[:,:-1]
    X=np.append(bias,X,axis=1)
    y = data[:, [-1]]






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

    # for i in range(len(y)):
    del_j = np.dot((h_x - y).T,X[:, idx])
    return del_j




def compute_J(h_x, y):
    m = np.shape(h_x)[0]
    temp = h_x - y
    J = (1 / (2 * m)) * (np.sum(temp ** 2))


    return J


def batch_grad(data, alpha=2e-8, epsilon=2e-8):
    X, y = preprocess(data)
    cost_value = []
    W = np.zeros([15, 1])
    w1 = []
    w2 = []
    eps = np.Inf

    epochs = 2000
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

        print(compute_J(h_x,y))

    return cost_value, W, w1, w2,h_x


data = pd.read_csv('FODS4.csv', header=None)
data = np.array(data)
data = data[:, 2:]
# print(np.shape(data[1]))
# data=rand(data_)
np.random.shuffle(data)
ALPHA = 7e-8
X,y=preprocess(data)
values, w, w1, w2,h_x = batch_grad(data, alpha=ALPHA)
SE=compute_J(h_x,y)
RMSE=SE**0.5
rsquared=rsquare(h_x,y)
print('W values ', w)
# print(len(values))
print("rsquare:",rsquared)
print("SE",SE)
print("RMSE",RMSE)
plt.plot(values)
plt.title('Cost vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()

