import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#==================================

# LAMBDA = 0.002
BATCH = 1

def preprocess(data):

	x1 = data[:,[0]]
	x2 = data[:,[1]]
	y = data[:,[2]]
	
	x1_mean =np.mean(x1)
	x2_mean =np.mean(x2)
	y_mean =np.mean(y)
	x1_std = np.std(x1)
	x2_std = np.std(x2)
	y_std = np.std(y)


	x1 = (x1 - x1_mean)/x1_std
	x2 = (x2 - x2_mean)/x2_std
	y = (y - y_mean)/y_std

	bias = np.expand_dims(np.ones([len(x1)]),axis = 1)
	X = np.append(bias,x1,axis = 1)
	X = np.append(X,x2,axis = 1)
	print(np.shape(X),np.shape(y))

	return X, y


#==========================================================================


def compute_J(h_x,y,lambdA,W):

	m = np.shape(h_x)[0]
	# print(m)
	temp = h_x - y
	# print(temp**2)
	J = (1/(2*m)) * (np.sum(temp**2)) + lambdA * np.sum(abs(W))
	# print(J)

	return J

def del_j (h_x,X,y,W,lambdA,idx):

	del_j = 0
	# print(X)
	del_j = np.sum(np.multiply((h_x - y),np.expand_dims(X[:,idx],1)) + lambdA*np.sign(W[idx]))
		
	return del_j

def batch_grad(X,y,alpha = 2e-8,lambdA = 0,epsilon = 2e-10):

	# X ,y = preprocess(data)
	cost_value = []
	W = np.zeros([3,1])
	w1 = []
	w2 = []
	eps = np.Inf
	# X = data[:,[0,1]]
	# bias = np.expand_dims(np.ones([len(X)]),axis = 1)
	
	# y = data[:,[2]]
	# X = np.append(bias,X,axis =1)
	epochs = 100
	ep =0
	h_x = np.dot(X,W)
	# print(np.shape(X),np.shape(W))
	

	
	while(eps>epsilon) and (ep < epochs):
		if ep%10 ==0:
			print('epoch',ep)

		for j in range(len(W)):

			W[j] = W[j] - alpha*(del_j(h_x,X,y,W,lambdA,j))
		
		w1.append(W[1,0])
		w2.append(W[2,0])

		h_x = np.dot(X,W)
		cost_value.append(compute_J(h_x,y,lambdA,W))
		ep +=1
		
		if(len(cost_value)>1):
			eps = np.abs(cost_value[-1] - cost_value[-2])
			# print(eps)
	# plt.plot(cost_value)
	# plt.show()

	return W



def test_train_split(X, y,p = 0.6):
	
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	slice_portion = int(p*len(y)-1) 
	train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
	X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
	# print(X_test,y_test)
	return X_train, X_test, y_train, y_test




def plotting_stuff(X_train, X_test, y_train, y_test,ALPHA ,params):

	val_loss = []
	for l in params:
		print('l',l)
		# W = stoch_grad(X_train,y_train,alpha = 0.01,lambdA = l)
		W = batch_grad(X_train,y_train,alpha = 2e-7,lambdA = l)
		h_x = np.dot(X_test,W)

		err = (1/2*len(h_x))* np.sum((h_x - y_test)**2)
		val_loss.append(err)
	# print(params)
	plt.plot(params,val_loss,'-o')
	plt.title('Cost Vs Lambda')
	plt.xlabel('Lambda')
	plt.ylabel('Cost')
	plt.show()







data = np.loadtxt('data.txt',delimiter = ',')
data = np.array(data)
X, y = preprocess(data)
X_train, X_test, y_train, y_test = test_train_split(X, y,p = 0.6)
ALPHA = 1e-6
params = np.linspace(1e-6,1,25) # keep changing this
# print(params)
# exit(0)
plotting_stuff(X_train, X_test, y_train, y_test,ALPHA,params)


# values,w,w1, w2 = batch_grad(data,alpha = ALPHA)
# values,w,w1, w2 = stoch_grad(data,alpha = ALPHA)


































# def del_j (h_x,X,y,W,lambdA,idx):

# 	del_j = 0
# 	# print(X)
# 	for i in range(len(y)):
# 		del_j +=(h_x[i] - y[i])*X[i,idx] + LAMBDA * W[idx]
		
# 	return del_j


# def batch_grad(data,alpha = 2e-8,epsilon = 2e-10):

# 	X ,y = preprocess(data)
# 	cost_value = []
# 	W = np.zeros([3,1])
# 	w1 = []
# 	w2 = []
# 	eps = np.Inf
# 	# X = data[:,[0,1]]
# 	# bias = np.expand_dims(np.ones([len(X)]),axis = 1)
	
# 	# y = data[:,[2]]
# 	# X = np.append(bias,X,axis =1)
# 	epochs = 1000
# 	ep =0
# 	h_x = np.dot(X,W)
# 	print(np.shape(X),np.shape(W))
	

	
# 	while(eps>epsilon) and (ep < epochs):
		
# 		for j in range(len(W)):

# 			W[j] = W[j] - alpha*(del_j(h_x,X,y,W,j))
		
# 		w1.append(W[1,0])
# 		w2.append(W[2,0])

# 		h_x = np.dot(X,W)
# 		cost_value.append(compute_J(h_x,y,W))
# 		ep +=1
		
# 		if(len(cost_value)>1):
# 			eps = np.abs(cost_value[-1] - cost_value[-2])
# 			# print(eps)

# 	return cost_value,W,w1,w2