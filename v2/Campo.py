import numpy as np
import matplotlib.pyplot as plt

data = np.load("campo_temperatura.npz")
T_all = data["T_all"]
x = data["x"]
y = data["y"]
t = data["t"]

print(np.shape(T_all))
Tempx = np.zeros(len(t)-2)
for i in range(1,1000):
    Tempx[i-1] = T_all[i,39,10]
plt.plot(Tempx)
plt.show()

