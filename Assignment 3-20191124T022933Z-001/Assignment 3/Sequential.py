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

SIZE = 160
ims = []
# print(bernoulli.pmf(0,[0.4,0.2]))
# plt.figure()
for i in range(SIZE):

	

	if i ==0:
		
		plt.get_current_fig_manager().full_screen_toggle()
		plt.plot(p,posterior)
		plt.title('Iteration {}'.format(i+1))
		plt.xlabel('p')
		plt.show(block=False)
		plt.pause(0.00003)
		plt.clf()

	if i != SIZE -1:

		plt.plot(p,posterior)
		plt.title('Iteration {}'.format(i+1))
		plt.xlabel('p')
		plt.show(block=False)
		plt.pause(0.00003)
		
		plt.clf()
	else:
		plt.plot(p,posterior)
		plt.title('Iteration {}'.format(i+1))
		plt.xlabel('p')
		plt.show()
		# plt.get_current_fig_manager().full_screen_toggle()
		
	# plt.close()

	# if i ==0:
	# 	posterior = np.multiply(bernoulli.pmf(data[i],p),prior)

	# else:
	# 	posterior = np.multiply(bernoulli.pmf(data[i],p),posterior/(np.max(posterior)))


	if data[i] ==1:
		a +=1
	elif data[i] ==0:
		b +=1
	posterior = beta.pdf(p,a,b)
	




