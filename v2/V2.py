import numpy as np
import matplotlib.pyplot as plt
import os

# Constantes

# Material derecho 'acero' (1)
rho2 = 7850 # [kg/m3]
Cp2 = 502 # [J/(kg K)]
k2 = 50 # [W/(m K)] 
epsilon1 = 0.75 # Emisividad

# Material izquierdo 'aluminio' (2)
rho1 = 2600 # [kg/m3]
Cp1 = 903 # [J/(kg K)]
k1 = 237 # [W/(m K)] 
epsilon2 = 0.75 # Emisividad

delta = 0.02 # [m]
hc = 25 # [w/(m2 K)]
Tamb = 298.15 # [K]
Boltsman =  5.67e-8 # [W/(m2 K4)]
dt = 0.5
flux = 500 # [K/(m)]
dirichlet = 350 # [K]

L = 0.2
H = 0.2

# Definir número de nodos en el eje x
x0 = 0.0; dx = 0.005
x = np.arange(x0, L, dx)
NodosX = len(x)

# Definir número de nodos en el eje y
y0 = 0.0; dy = 0.005
y = np.arange(y0, H, dy)
NodosY = len(y)

# Definir el número de pasos temporales
t0 = 0.0; tf = 500
t = np.arange(t0, tf + dt, dt)
NodosT =  len(t)

# Crear carpeta para las imágenes
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Guardar 100 imágenes aproximadamente
step_interval = NodosT // 100 

# Matrices de temperaturas presente y futuro
TPre =  np.zeros((NodosY,NodosX))
TFut =  np.zeros((NodosY,NodosX))
T_all = np.zeros((NodosT, NodosY, NodosX))  # Almacena T(x,y,t)
T_all[0,:,:] = TPre.copy()

# Paso 06: crear matrices donde almacenar la posición nodal. Servirá para graficar
X,Y = np.meshgrid(x,y)

# Sustitucion para ahorrar codigo
Hx = delta * dt /(dx**2)
Hy = delta * dt /(dy**2)

# Función sigmoide y su derivada
def kx(x,Nx_in, k1, k2, a = 100):
    return k2 + (k1 - k2) / (1 + np.exp(-a*(x - Nx_in/2)))

def dkdx_func(x,Nx_in, k1, k2, a=100):
    exp_term = np.exp(-a*(x - Nx_in/2))
    return (k1 - k2) * a * exp_term / (1 + exp_term)**2



def A(i,Nx_in):
    k = kx(i*dx, L, k1, k2)
    rho = rho1 if i >= Nx_in/2 else rho2
    cp = Cp1 if i >= Nx_in/2 else Cp2
    return rho * cp * delta + k * Hx + k * Hy + hc * dt

def B(i,Nx_in):
    k = kx(i*dx, L, k1, k2)
    dk = dkdx_func(i*dx, L, k1, k2)
    return -k * Hx / 2 - delta*dk*dt/(4*dx)

def C(i,Nx_in):
    k = kx(i*dx, L, k1, k2)
    dk = dkdx_func(i*dx, L, k1, k2)
    return -k * Hx / 2 + delta*dk*dt/(4*dx)

def D(i,Nx_in):
    k = kx(i*dx, L, k1, k2)
    return -k * Hy / 2

def An(i,Nx_in):
    k = kx(i*dx, L, k1, k2)
    rho = rho1 if i >= Nx_in/2 else rho2
    cp = Cp1 if i >= Nx_in/2 else Cp2
    return rho * cp * delta - k * Hx - k * Hy - hc * dt


def E(j,i,Nx_in):
    if i < Nx_in/2:
        E = dt*hc*Tamb + dt*2*epsilon1*Boltsman*Tamb**4 \
            - dt*2*epsilon1*Boltsman*TPre[j,i]**4
    else:
         E = dt*hc*Tamb + dt*2*epsilon2*Boltsman*Tamb**4 \
            - dt*2*epsilon2*Boltsman*TPre[j,i]**4
    return E


# se llena la matriz TPre con las condiciones iniciales
for j in range(0,NodosY):
    for i in range(0,NodosX):
            TPre[j,i] = Tamb

incog = NodosY*NodosX

A_mat = np.zeros((incog,incog))
b_vec = np.zeros(incog)

# ------- Matriz A -------------
m = lambda j,i: j*NodosX + i

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

# resolver en el tiempo. A medida que se avanzan en el tiempo, se debe actualizar el vector b
Tc = 480
Tf = 280
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
                            + -C(i+1,NodosX)*TPre[j,i+1] - B(i-1,NodosX)*TPre[j,i-1] - 2*D(i,NodosX)*TPre[j-1,i]\
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
                            - C(i+1,NodosX))*TPre[j,i+1] - D(i,NodosX)*TPre[j-1,i] \
                            + E(j,i,NodosX) -2*D(i,NodosX)*dirichlet
    # Resolver
    c = np.linalg.solve(A_mat,b_vec)

    # Extraer información
    for j in range(0,NodosY):
        for i in range(0,NodosX):
            TFut[j,i] = c[m(j,i)]   
    TPre = TFut.copy()

    # Guardar el campo de temperatura en T_all
    T_all[n-1,:,:] = TFut.copy()

    np.savez_compressed("campo_temperatura.npz", T_all=T_all, x=x, y=y, t=t)

    
    if n % step_interval == 0:
        fig, ax = plt.subplots()
        scalarField = ax.contourf(X, Y, TFut, levels=levels, cmap="magma")
        plt.colorbar(scalarField, ax=ax)
        ax.set_title(f"Tiempo {n*dt:.2f} s")
        filename = os.path.join(output_folder, f"frame_{n//20:04d}.png")
        plt.savefig(filename)
        plt.close()
    
    
# Se gráfica

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