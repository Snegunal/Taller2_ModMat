import numpy as np

# Constantes

# Material derecho 'acero' (1)
rho1 = 7850 # [kg/m3]
Cp1 = 502 # [J/(kg K)]
k1 = 210 # [W/(m K)] 
epsilon = 0.75 # Emisividad

# Material izquierdo 'aluminio' (2)
rho2 = 2600 # [kg/m3]
Cp2 = 903 # [J/(kg K)]
k2 = 50 # [W/(m K)] 
epsilon = 0.05 # Emisividad

delta = 0.02 # [m]
hc = 10 # [w/(m2 K)]
Tamb = 298.15 # [K]
Boltsman =  5.67e-8 # [W/(m2 K4)]
dt = 0.0001

L = 1
H = 1

# Paso 02: definir número de nodos en el eje x
x0 = 0.0; dx = 0.25
x = np.arange(x0, L, dx)
NodosX = len(x)

# Paso 03: definir número de nodos en el eje y
y0 = 0.0; dy = 0.25
y = np.arange(y0, H, dy)
NodosY = len(y)

# Paso 04: definir el número de pasos temporales
t0 = 0.0; tf = 10.0
t = np.arange(t0, tf + dt, dt)
NodosT =  len(t)

# Paso 05: crear matrices donde almacenar la temperatura actual, y la del futuro
TPre =  np.zeros((NodosY,NodosX))
TFut =  np.zeros((NodosY,NodosX))

# Paso 06: crear matrices donde de posición nodal. Servirá para graficar
X,Y = np.meshgrid(x,y)

# Sustitucion para ahorrar codigo
Hx = delta * dt /(dx**2)
Hy = delta * dt /(dy**2)

def A(i,Nx_in): # Coeficiente que acompaña a Tn1_ij
    if i < Nx_in/2:
        A = (rho2 * Cp2 * delta - k2 * Hx - k2 * Hy + hc )
    else:
        A = (rho1 * Cp1 * delta - k1 * Hx - k1 * Hy + hc )
    return A

def B(i,Nx_in): # Coeficiente que acompaña a Tn1_i_1j
    if i < Nx_in/2:
        B = (-k2*Hx/2)
    else:
        B = (-k1*Hx/2)
    return B

def C(i,Nx_in): # Coeficiente que acompaña a Tn1_i1j
    if i < Nx_in/2:
        C = (-k2*Hx/2)
    else:
        C = (-k1*Hx/2)
    return C

def D(i,Nx_in): # Coeficiente que acompaña a Tn1_i1j
    if i < Nx_in/2:
        D = (-k2*Hy/2)
    else:
        D = (-k1*Hy/2)
    return D

# Paso 07: llenar la matriz TPre con las condiciones iniciales
for j in range(0,NodosY):
    for i in range(0,NodosX):
            TPre[j,i] = Tamb

incog = NodosY*NodosX

A_mat = np.zeros((incog,incog))
b_vec = np.zeros(incog)

# Paso 09: llenar la matriz A con los
m = lambda j,i: j*NodosX + i


# Llenamos para todas las filas internas (las que es están entre la primera y la última)
for j in range(0, NodosY):
    for i in range(0,NodosX):
        p = m(j,i)

        # --- Nodos centrales
        if i != 0 and i != NodosX-1 and j != 0 and j != NodosY-1:
            A_mat[p,m(j,i-1)] = B(i,NodosX)
            A_mat[p,m(j,i+1)] = C(i,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,m(j+1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        # --- Nodos casí de frontera
        if i == 0 and j != NodosY-1: # Nodos izquierdos
            A_mat[p,m(j,i+1)] = B(i,NodosX) + C(i,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,m(j+1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i != NodosX-1 and j == 0: # Nodos inferiores
            A_mat[p,m(j,i-1)] = B(i,NodosX)
            A_mat[p,m(j,i+1)] = C(i,NodosX)
            A_mat[p,m(j+1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i == NodosX -1 and j != NodosY-1: # Nodos Derechos
            A_mat[p,m(j,i-1)] = B(i,NodosX) + C(i,NodosX)
            A_mat[p,m(j+1,i)] = D(i,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        if i >= NodosX/2 and i != NodosX-1 and j == NodosY-1: # Nodos superior derecha
            A_mat[p,m(j,i-1)] = B(i,NodosX) 
            A_mat[p,m(j,i+1)] = C(i,NodosX)
            A_mat[p,m(j-1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i < NodosX/2 and j == NodosY-1: # Nodos superior derecha
            A_mat[p,m(j,i-1)] = B(i,NodosX) 
            A_mat[p,m(j,i+1)] = C(i,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        # ---- Esquinas
        if i == 0 and j == 0: # Esquina inferior izquierda
            A_mat[p,m(j,i+1)] = B(i,NodosX) + C(i,NodosX)
            A_mat[p,m(j+1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        if i == NodosX-1 and j == 0: # Esquina inferior Derecha
            A_mat[p,m(j,i-1)] = B(i,NodosX) + C(i,NodosX)
            A_mat[p,m(j+1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i == NodosX-1 and j == NodosY-1: # Esquina superior Derecha
            A_mat[p,m(j,i-1)] = B(i,NodosX) + C(i,NodosX)
            A_mat[p,m(j-1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i == 0 and j == NodosY-1: # Esquina superior Izquierda
            A_mat[p,m(j,i+1)] = B(i,NodosX) + C(i,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

np.set_printoptions(precision=2, linewidth=200, suppress=False)
print(A_mat)