import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DEGREE = 6

data = pd.read_csv('FODS.txt', header=None)
data = np.array(data)[:, 1:]
print(data)

output = data[:, -1]
output=np.reshape(output,(len(output),1))
x1 = data[:, 0]
x2 = data[:, 1]



# x1 = data[:, [0]]
# x2 = data[:, [1]]
# y = data[:, [2]]

x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
output_mean = np.mean(output)
x1_std = np.std(x1)
x2_std = np.std(x2)
output_std = np.std(output)

x1 = (x1 - x1_mean) / x1_std
x2 = (x2 - x2_mean) / x2_std
output = (output - output_mean) / output_std






new_data = np.ones([np.shape(x1)[0], 1])

z = 0

for i in range(DEGREE + 1):

    for j in range(DEGREE + 1):

        if i + j <= DEGREE:
            # continue
            # print(np.power(x1,i),np.power(x2,j))
            prod = np.multiply(np.power(x1, i), np.power(x2, j))
            prod = np.expand_dims(prod, axis=1)
            new_data = np.append(new_data, prod, axis=1)
            print(i, j)
        z += 1

new_data = new_data[:, :]
print(np.shape(new_data))
print(np.shape(output))
new_data = np.append(new_data, output, axis=1)

# save this new_data- cross check once

print(new_data[1])
np.savetxt('FODS6.csv', new_data,delimiter=',', fmt='%f')
