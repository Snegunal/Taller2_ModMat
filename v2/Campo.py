import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 

data = np.load("campo_ac_al_1000s_hc50.npz")
T_all = data["T_all"]
x = data["x"]
y = data["y"]
t = data["t"]

print(np.shape(T_all))

Tempt = np.zeros(len(t)-2)

# para x = 0.25 y, y = 0.25
for i in range(1,1000): # Para graficar un punto en el tiempo
    Tempt[i-1] = T_all[i,4,24]
plt.plot(t[1:1000],Tempt)
plt.grid(True)
plt.xlabel('Tiempo [s]',fontsize = 16)
plt.ylabel('Temperatura [K]',fontsize = 16)
plt.tight_layout()
plt.show()

# para x = 0.5 y, y = 0.25
for i in range(1,1000): # Para graficar un punto en el tiempo
    Tempt[i-1] = T_all[i,4,74]
plt.plot(t[1:1000],Tempt)
ax = plt.gca()
formatter_y = mticker.FormatStrFormatter('%.2f')
ax.yaxis.set_major_formatter(formatter_y)
plt.grid(True)
plt.xlabel('Tiempo [s]',fontsize = 16)
plt.ylabel('Temperatura [K]',fontsize = 16)
plt.tight_layout()
plt.show()

# para x = 0.25 y, y = 0.5
for i in range(1,1000): # Para graficar un punto en el tiempo
    Tempt[i-1] = T_all[i,14,24]
plt.plot(t[1:1000],Tempt)
plt.grid(True)
plt.xlabel('Tiempo [s]',fontsize = 16)
plt.ylabel('Temperatura [K]',fontsize = 16)
plt.tight_layout()
plt.show()

# para x = 0.5 y, y = 0.5
for i in range(1,1000): # Para graficar un punto en el tiempo
    Tempt[i-1] = T_all[i,14,74]
plt.plot(t[1:1000],Tempt)
ax = plt.gca()
formatter_y = mticker.FormatStrFormatter('%.2f')
ax.yaxis.set_major_formatter(formatter_y)
plt.grid(True)
plt.xlabel('Tiempo [s]',fontsize = 16)
plt.ylabel('Temperatura [K]',fontsize = 16)
plt.tight_layout()
plt.show()

Tempx = np.zeros(len(x))
xi = np.arange(0,1,0.01)
for i in range(100): # Para graficar todo un eje a un tiempo particular
    Tempx[i] = T_all[999,10,i]

print(max(Tempx)-min(Tempx))
plt.plot(xi,Tempx)
plt.xlabel("x")
plt.ylabel("Temperatura")
plt.grid(True)
plt.tight_layout()
plt.show()
