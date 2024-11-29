#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np



def sigmoid(data):
    return 1/(1+np.exp(-data))




x = np.linspace(-1, 1, 26)
y = np.linspace(-1, 1, 26)
z = np.linspace(-1, 1, 26)

X, Y, Z = np.meshgrid(x, y, z)
values = X + sigmoid(Y) + Z**3 - 10

list_val = np.array(values).flatten()
distinct = np.unique(list_val)

print("Num of values: {}".format(len(list_val)))
print("Density of values: {}".format(len(distinct)/len(list_val)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X, Y, Z, c=values, cmap='jet')
fig.colorbar(scatter, ax=ax)

plt.show()
