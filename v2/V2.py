import numpy as np
import matplotlib.pyplot as plt
import os

# Constantes

# Material derecho 'acero' (1)
rho1 = 7850 # [kg/m3]
Cp1 = 502 # [J/(kg K)]
k1 = 210 # [W/(m K)] 
epsilon1 = 0.75 # Emisividad

# Material izquierdo 'aluminio' (2)
rho2 = 2600 # [kg/m3]
Cp2 = 903 # [J/(kg K)]
k2 = 210 # [W/(m K)] 
epsilon2 = 0.05 # Emisividad

delta = 0.02 # [m]
hc = 10 # [w/(m2 K)]
Tamb = 298.15 # [K]
Boltsman =  5.67e-8 # [W/(m2 K4)]
dt = 0.5
flux = 500 # [K/(m)]
dirichlet = 350 # [K]

L = 0.2
H = 0.2

# Paso 02: definir número de nodos en el eje x
x0 = 0.0; dx = 0.005
x = np.arange(x0, L, dx)
NodosX = len(x)

# Paso 03: definir número de nodos en el eje y
y0 = 0.0; dy = 0.005
y = np.arange(y0, H, dy)
NodosY = len(y)

# Paso 04: definir el número de pasos temporales
t0 = 0.0; tf = 50
t = np.arange(t0, tf + dt, dt)
NodosT =  len(t)

# Crear carpeta para las imágenes
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Guardar 100 imágenes aproximadamente
step_interval = NodosT // 100  # por ejemplo, nt=1000 → interval=10

# Paso 05: crear matrices donde almacenar la temperatura actual, y la del futuro
TPre =  np.zeros((NodosY,NodosX))
TFut =  np.zeros((NodosY,NodosX))

# Paso 06: crear matrices donde almacenar la posición nodal. Servirá para graficar
X,Y = np.meshgrid(x,y)

# Sustitucion para ahorrar codigo
Hx = delta * dt /(dx**2)
Hy = delta * dt /(dy**2)

def A(i,Nx_in): # Coeficiente que acompaña a Tn1_ij
    if i < Nx_in/2:
        A = (rho2 * Cp2 * delta + k2 * Hx + k2 * Hy + hc * dt )
    else:
        A = (rho1 * Cp1 * delta + k1 * Hx + k1 * Hy + hc * dt)
    return A

def B(i,Nx_in): # Coeficiente que acompaña a Tn1_i_1j
    if i < Nx_in/2:
        B = (-k2*Hx/2)
    elif i == Nx_in/2-1:
        B = (-2*k1*k2*Hx/(2*(k1+k2)))
    else:
        B = (-k1*Hx/2)
    return B

def C(i,Nx_in): # Coeficiente que acompaña a Tn1_i1j
    if i < Nx_in/2:
        C = (-k2*Hx/2)
    elif i == Nx_in/2:
        C = (-2*k1*k2*Hx/(2*(k1+k2)))    
    else:
        C = (-k1*Hx/2)
    return C

def D(i,Nx_in): # Coeficiente que acompaña a Tn1_i1j
    if i < Nx_in/2:
        D = (-k2*Hy/2)
    else:
        D = (-k1*Hy/2)
    return D

def E(j,i,Nx_in):
    if i < Nx_in/2:
        E = dt*hc*Tamb + dt*2*epsilon1*Boltsman*Tamb**4 \
            - dt*2*epsilon1*Boltsman*TPre[j,i]**4
    else:
         E = dt*hc*Tamb + dt*2*epsilon2*Boltsman*Tamb**4 \
            - dt*2*epsilon2*Boltsman*TPre[j,i]**4
    return E

def An(i,Nx_in): # Coeficiente que acompaña a Tn1_ij
    if i < Nx_in/2:
        An = (rho2 * Cp2 * delta - k2 * Hx - k2 * Hy - hc * dt )
    else:
        An = (rho1 * Cp1 * delta - k1 * Hx - k1 * Hy - hc * dt)
    return An

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
            A_mat[p,m(j,i-1)] = B(i-1,NodosX)
            A_mat[p,m(j,i+1)] = C(i+1,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,m(j+1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        # --- Nodos casí de frontera
        if i == 0 and j != NodosY-1 and j != 0: # Nodos izquierdos
            A_mat[p,m(j,i+1)] = B(i+1,NodosX) + C(i+1,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,m(j+1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i != NodosX-1 and j == 0 and i != 0: # Nodos inferiores
            A_mat[p,m(j,i-1)] = B(i-1,NodosX)
            A_mat[p,m(j,i+1)] = C(i+1,NodosX)
            A_mat[p,m(j+1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i == NodosX -1 and j != NodosY-1 and j != 0: # Nodos Derechos
            A_mat[p,m(j,i-1)] = B(i-1,NodosX) + C(i-1,NodosX)
            A_mat[p,m(j+1,i)] = D(i,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        if i >= NodosX/2 and i != NodosX-1 and j == NodosY-1: # Nodos superior derecha
            A_mat[p,m(j,i-1)] = B(i-1,NodosX) 
            A_mat[p,m(j,i+1)] = C(i+1,NodosX)
            A_mat[p,m(j-1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i < NodosX/2 and j == NodosY-1 and i != 0: # Nodos superior Izquierda
            A_mat[p,m(j,i-1)] = B(i-1,NodosX) 
            A_mat[p,m(j,i+1)] = C(i+1,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        # ---- Esquinas
        if i == 0 and j == 0: # Esquina inferior izquierda
            A_mat[p,m(j,i+1)] = B(i+1,NodosX) + C(i+1,NodosX)
            A_mat[p,m(j+1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

        if i == NodosX-1 and j == 0: # Esquina inferior Derecha
            A_mat[p,m(j,i-1)] = B(i-1,NodosX) + C(i-1,NodosX)
            A_mat[p,m(j+1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i == NodosX-1 and j == NodosY-1: # Esquina superior Derecha
            A_mat[p,m(j,i-1)] = B(i-1,NodosX) + C(i-1,NodosX)
            A_mat[p,m(j-1,i)] = 2*D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)
        
        if i == 0 and j == NodosY-1: # Esquina superior Izquierda
            A_mat[p,m(j,i+1)] = B(i+1,NodosX) + C(i+1,NodosX)
            A_mat[p,m(j-1,i)] = D(i,NodosX)
            A_mat[p,p] = A(i,NodosX)

# Paso 10: resolver en el tiempo. A medida que se avanzan en el tiempo, se debe actualizar el vector b
Tc = 350
Tf = 250
levels = np.arange(Tf,Tc+3,3)

for n in range(1,NodosT):
    for j in range(0,NodosY):
        for i in range(0,NodosX):
            p = m(j,i)

            # Nodos del centro
            if i != 0 and i != NodosX-1 and j != 0 and j != NodosY-1:      
                b_vec[p] = An(i,NodosX)*TPre[j,i] - B(i-1,NodosX)*TPre[j,i-1] \
                            - C(i+1,NodosX)*TPre[j,i+1] - D(i,NodosX)*TPre[j-1,i]\
                            - D(i,NodosX)*TPre[j+1,i] + E(j,i,NodosX)
                
            # --- Nodos casí de frontera
            if i == 0 and j != NodosY-1 and j !=0: # Nodos izquierdos
                b_vec[p] = An(i,NodosX)*TPre[j,i] \
                            + (-C(i+1,NodosX) - B(i+1,NodosX))*TPre[j,i+1] - D(i,NodosX)*TPre[j-1,i]\
                            - D(i,NodosX)*TPre[j+1,i] + E(j,i,NodosX)
                
            if i != NodosX-1 and j == 0 and i != 0: # Nodos inferiores
                b_vec[p] = An(i,NodosX)*TPre[j,i] - B(i-1,NodosX)*TPre[j,i-1] \
                            - C(i+1,NodosX)*TPre[j,i+1] \
                            - 2*D(i,NodosX)*TPre[j+1,i] + E(j,i,NodosX)
                
            if i == NodosX -1 and j != NodosY-1 and j != 0: # Nodos Derechos
                b_vec[p] = An(i,NodosX)*TPre[j,i] \
                            + (-C(i-1,NodosX) - B(i-1,NodosX))*TPre[j,i-1] - D(i,NodosX)*TPre[j-1,i]\
                            - D(i,NodosX)*TPre[j+1,i] + E(j,i,NodosX) -4*C(i,NodosX)*flux*dx
                
            if i >= NodosX/2 and i != NodosX-1 and j == NodosY-1: # Nodos superior derecha
                b_vec[p] = An(i,NodosX)*TPre[j,i] \
                            + (-C(i-1,NodosX) - B(i-1,NodosX))*TPre[j,i-1] - 2*D(i,NodosX)*TPre[j-1,i]\
                            + E(j,i,NodosX) 

            if i < NodosX/2 and j == NodosY-1 and i != 0: # Nodos superior Izquierda
                b_vec[p] = An(i,NodosX)*TPre[j,i] - B(i-1,NodosX)*TPre[j,i-1] \
                            - C(i+1,NodosX)*TPre[j,i+1] - D(i,NodosX)*TPre[j-1,i]\
                            + E(j,i,NodosX) -2*D(i,NodosX)*dirichlet

            # --- Nodos de las esquinas
            if i == 0 and j == 0: # Esquina inferior izquierda
                b_vec[p] = An(i,NodosX)*TPre[j,i] \
                            + (-C(i+1,NodosX) - B(i+1,NodosX))*TPre[j,i+1] \
                            - 2*D(i,NodosX)*TPre[j+1,i] + E(j,i,NodosX)
            
            if i == NodosX-1 and j == 0: # Esquina inferior Derecha
                b_vec[p] = An(i,NodosX)*TPre[j,i] \
                            + (-C(i-1,NodosX) - B(i-1,NodosX))*TPre[j,i-1] \
                            - 2*D(i,NodosX)*TPre[j+1,i] + E(j,i,NodosX) -4*C(i,NodosX)*flux*dx
                
            if i == NodosX-1 and j == NodosY-1: # Esquina superior Derecha        
                b_vec[p] = An(i,NodosX)*TPre[j,i] \
                            + (-C(i-1,NodosX) - B(i-1,NodosX))*TPre[j,i-1] \
                            - 2*D(i,NodosX)*TPre[j-1,i] + E(j,i,NodosX) -4*C(i,NodosX)*flux*dx
            if i == 0 and j == NodosY-1: # Esquina superior Izquierda
                b_vec[p] = An(i,NodosX)*TPre[j,i] + (-B(i+1,NodosX) \
                            - C(i+1,NodosX))*TPre[j,i+1] - D(i,NodosX)*TPre[j-1,i]\
                            + E(j,i,NodosX) -2*D(i,NodosX)*dirichlet
    # Resolver
    c = np.linalg.solve(A_mat,b_vec)

    # Extraer información
    for j in range(0,NodosY):
        for i in range(0,NodosX):
            TFut[j,i] = c[m(j,i)]
    TPre = TFut.copy()

    
    if n % step_interval == 0:
        fig, ax = plt.subplots()
        scalarField = ax.contourf(X, Y, TFut, levels=levels, cmap="magma")
        plt.colorbar(scalarField, ax=ax)
        ax.set_title(f"Tiempo {n*dt:.2f} s")
        filename = os.path.join(output_folder, f"frame_{n//1:04d}.png")
        plt.savefig(filename)
        plt.close()

# Paso 11: hacemos unas buenas obras de arte


fig, ax = plt.subplots(1,1,figsize = (8,5))
scalarField = ax.contourf(X,Y,TFut, levels = levels, cmap = "magma")
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.set_title("Campo temperatura tf = {:.2f} s".format(tf))

cbar = fig.colorbar(scalarField,orientation = 'vertical')
cbar.set_label("Temperatura / C")
plt.show()


#np.set_printoptions(precision=2, linewidth=200, suppress=False)
#print(A_mat)
#print(b_vec)