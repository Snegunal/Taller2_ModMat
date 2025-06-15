import numpy as np
import matplotlib.pyplot as plt

# Parámetros de materiales
k1 = 15.3 #acero 398  # Cobre 
k2 = 222 #aluminio 85.1   # Latón   
a = 50   # control de suavidad

Nx = 200
x = np.linspace(0, 1, Nx)

# Función sigmoide y su derivada
def kx(x,Nx_in, k1, k2, a = 50):
    return k2 + (k1 - k2) / (1 + np.exp(-a*(x - Nx_in/2)))

def dkdx_func(x,Nx_in, k1, k2, a=50):
    exp_term = np.exp(-a*(x - Nx_in/2))
    return (k1 - k2) * a * exp_term / (1 + exp_term)**2

# Evaluación
k = kx(x,1, k1, k2)
dkdx = dkdx_func(x,1, k1, k2)

# Graficar k(x) y su derivada
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, k, label="k(x)")
plt.title("Conductividad térmica sigmoidal")
plt.xlabel("x")
plt.ylabel("k")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, dkdx, label="dk/dx", color="orange")
plt.title("Derivada de k(x)")
plt.xlabel("x")
plt.ylabel("dk/dx")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
