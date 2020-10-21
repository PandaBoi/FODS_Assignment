import numpy as np
from scipy.stats import beta, gamma, bernoulli
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#=======================================================


fig = plt.figure()
#setting beta dist graph with mean 0.4
a =2
b = 3
mean, var, skew, kurt = beta.stats(a,b ,moments = 'mvsk')
print(mean)
rv = beta(a,b)


p = np.linspace(0, 1.0, 100) #prob of getting head
prior = beta.pdf(p, a, b) 
posterior = prior#init
data = bernoulli.rvs(p = 0.7, size = 160) 
plt.plot(p,prior,'*')
# plt.show()

SIZE = 160
likes = []

zeros = np.shape(np.where(data==0))[1]
ones = np.shape(np.where(data==1))[1]

# posterior = np.multiply(likes,prior)

a += ones
b += zeros
posterior = beta.pdf(p,a,b)
plt.plot(p,posterior,'--')
plt.legend(('prior','posterior'))
plt.show()