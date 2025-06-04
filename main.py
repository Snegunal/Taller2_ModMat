import sympy as sy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
import os




start_time = time.time()

k1 = 237#45
k2 = 237
rho1 = 7890#2700
rho2 = 7870
flux = 200000
Cp1= 1903#502#
Cp2= 1903#

# Definimos las variables simbólicas

rho,Cp,delta,k,dt,dx,dy,h,Ta,epsilon,sigma,q = sy.symbols('rho,Cp,delta,k,dt,dx,dy,h,Ta,epsilon,sigma,q')
# Temperaturas futuras (n+1) y presentes (n)
temps = sy.symbols('Tn1_ij Tn1_i_1j Tn1_i1j Tn1_ij_1 Tn1_ij1 Tn_ij Tn_i_1j Tn_i1j Tn_ij_1 Tn_ij1')

# Asignamos nombres individuales
Tn1_ij, Tn1_i_1j, Tn1_i1j, Tn1_ij_1, Tn1_ij1, Tn_ij, Tn_i_1j, Tn_i1j, Tn_ij_1, Tn_ij1 = temps

# Ecuación despejada de crank nicolson con diferencias finitas centradas
f = -rho*Cp*delta*(Tn1_ij - Tn_ij)/dt \
    + 0.5*(k*delta*((Tn1_i_1j - 2*Tn1_ij + Tn1_i1j)/dx**2 + (Tn1_ij_1 - 2*Tn1_ij + Tn1_ij1)/dy**2) \
    - 2*h*(Tn1_ij - Ta) \
    - 2*epsilon*sigma*(Tn_ij**4 - Ta**4)) \
    + 0.5*(k*delta*((Tn_i_1j - 2*Tn_ij + Tn_i1j)/dx**2 + (Tn_ij_1 - 2*Tn_ij + Tn_ij1)/dy**2) \
    - 2*h*(Tn_ij - Ta) \
    - 2*epsilon*sigma*(Tn_ij**4 - Ta**4)) \
    
b_expr = rho*Cp*delta*Tn_ij/dt \
    + 0.5*k*delta*((Tn_i_1j - 2*Tn_ij + Tn_i1j)/dx**2 + (Tn_ij_1 - 2*Tn_ij + Tn_ij1)/dy**2) \
    - h*(Tn_ij - Ta) \
    - epsilon*sigma*(Tn_ij**4 - Ta**4)

from sympy import symbols, collect, Add

# 1. Lista de símbolos del futuro
fut_vars = [Tn1_ij, Tn1_i1j, Tn1_i_1j, Tn1_ij1, Tn1_ij_1]

# 2. Expandir la expresión para que todos los términos estén separados
f_expanded = f.expand()

# 3. Filtrar solo los términos que contienen alguna variable del futuro
f_fut_terms = []
for term in Add.make_args(f_expanded):  # descompone f en sumandos
    if any(v in term.free_symbols for v in fut_vars):
        f_fut_terms.append(-term)  # cambiar el signo porque vienen del lado derecho

# 4. Sumar todos los términos del futuro con signo cambiado
f_fut = sum(f_fut_terms)

# Expandimos y agrupamos por todas las temperaturas
f_fut = sy.expand(f_fut)
f_collected = sy.collect(f_fut, temps)

# Coeficientes de cada temperatura
#print("\nCoeficientes:")
#for temp in temps:
#    coef = f_collected.coeff(temp)
#    print(f"Coeficiente de {temp}:")
#    sy.pprint(coef)

# Parámetros físicos y espaciales
# Estos parametros cambian en función de donde se este iterando
param_values = {
    sy.symbols('rho'): rho1,
    sy.symbols('Cp'): 502,
    sy.symbols('delta'): 0.01,
    sy.symbols('k'): k1,
    sy.symbols('dt'): 0.005,
    sy.symbols('dx'): 0.01,
    sy.symbols('dy'): 0.01,
    sy.symbols('h'): 10,
    sy.symbols('Ta'): 300,
    sy.symbols('epsilon'): 0.9,
    sy.symbols('sigma'): 5.67e-8,
    sy.symbols('Tn_ij'): 300,  # solo necesario para calcular b
    #sy.symbols('q'): 2
}

Nx, Ny = 60,60
N_in = (Nx - 2) * (Ny - 2)
Nx_in = Nx - 2
Ny_in = Ny - 2
# Para acceder a los coeficientes para la matriz izquierda
coef_expressions = {
    'center': f_collected.coeff(Tn1_ij),
    'x_neg': f_collected.coeff(Tn1_i_1j),
    'x_pos': f_collected.coeff(Tn1_i1j),
    'y_neg': f_collected.coeff(Tn1_ij_1),
    'y_pos': f_collected.coeff(Tn1_ij1)
}

vars_needed = [rho, Cp, delta, k, dt, dx, dy, h, Ta, epsilon, sigma, Tn_ij]

coef_funcs = {
    key: sy.lambdify(vars_needed, expr, modules='numpy')
    for key, expr in coef_expressions.items()
}


#----------------------------- Matriz A -------------------------------

main_diag     = np.zeros(N_in, dtype=np.float64)
x_neg_diag    = np.zeros(N_in, dtype=np.float64)
x_pos_diag    = np.zeros(N_in, dtype=np.float64)
y_neg_diag    = np.zeros(N_in, dtype=np.float64)
y_pos_diag    = np.zeros(N_in, dtype=np.float64)

temps_fut = [Tn1_ij, Tn1_i1j, Tn1_i_1j, Tn1_ij1, Tn1_ij_1]

from sympy import Add

def neumann(fun, nodo_fantasma, nodo_interno, q, desp, k):
    sustitucion = nodo_interno + q * desp / k
    fmod = fun.subs(nodo_fantasma, sustitucion)
    fmod = sy.expand(fmod)

    # Coeficiente cambiado del interno
    coef = fmod.coeff(nodo_interno)

    # Se separan los terminos uno a uno
    indep_terms = []
    for term in Add.make_args(fmod):
        # Si no depende de ninguna temperatura futura, lo guardamos como independiente
        if all(not term.has(temp) for temp in temps_fut):
            indep_terms.append(term)

    indep = sy.Add(*indep_terms)

    # Debug prints opcionales
    #print("\n[DEBUG] fmod:")
    #sy.pprint(fmod)
    #print("\n[DEBUG] Término independiente (calculado manualmente):")
    #sy.pprint(indep)

    return coef, indep

#def neumann(fun, nodo_fantasma, nodo_interno, qsi, desp, k): # Esta esta pendiente por ahora

    q = sy.symbols('q')
    sustitucion = nodo_interno + q * desp / k
    fmod = fun.subs(nodo_fantasma, sustitucion)
    fcolm = sy.expand(fmod)
    fcolm = sy.collect(fmod,nodo_interno)
    coef = fcolm.coeff(nodo_interno)
    indep, _ = fcolm.as_independent(Tn1_ij, Tn1_i1j, Tn1_i_1j, Tn1_ij1, Tn1_ij_1)
    
   # sy.pprint(fmod)
    sy.pprint(indep)
    return coef,indep

Tvec_n = 300*np.ones(N_in)

for j in range(Ny-2): # Queda pendiente las condiciones de frontera
    for i in range(Nx-2):
        p = i + j * (Nx-2)  # índice global
        param_values[Tn_ij] = Tvec_n[p]

        # Propiedades de aluminio y acero
        if i < Nx_in/2:
            param_values[rho] = rho2
            param_values[k] = k2
            param_values[Cp] = Cp2
        if i >= Nx_in/2:
            param_values[rho] = rho1
            param_values[k] = k1
            param_values[Cp] = Cp1
        
        # Evaluar y asignar coeficientes en la diagonal
        vals = [param_values[sym] for sym in vars_needed]
        #main_diag[p]  = float(coef_expressions['center'].evalf(subs=param_values))
        main_diag[p] = coef_funcs['center'](*vals)

        # Dirección x: izquierda y derecha
        if i > 0:
            if i >= Nx_in/2 and i-1 < Nx_in/2: # Si se esta en la frontera de materiales xneg se evalua en el otro material
                param_values[rho] = rho2
                param_values[k] = k2
                param_values[Cp] = Cp2
                vals = [param_values[sym] for sym in vars_needed]
                #x_neg_diag[p] = float(coef_expressions['x_neg'].evalf(subs=param_values))
                x_neg_diag[p] = coef_funcs['x_neg'](*vals)
                param_values[rho] = rho1
                param_values[k] = k1
                param_values[Cp] = Cp1
                vals = [param_values[sym] for sym in vars_needed]
            else:
                x_neg_diag[p] = coef_funcs['x_neg'](*vals)
                #x_neg_diag[p] = float(coef_expressions['x_neg'].evalf(subs=param_values))
        else: # frontera izquierda
            x_neg_diag[p] = 0

        if i < Nx_in - 1:
            if i < Nx_in/2 and i+1 >= Nx_in/2: # Si se esta en la frontera de materiales xpos se evalua en el otro material
                param_values[rho] = rho1
                param_values[k] = k1
                param_values[Cp] = Cp1
                vals = [param_values[sym] for sym in vars_needed]
                #x_pos_diag[p] = float(coef_expressions['x_pos'].evalf(subs=param_values))
                x_pos_diag[p] = coef_funcs['x_pos'](*vals)
                param_values[rho] = rho2
                param_values[k] = k2
                param_values[Cp] = Cp2
                vals = [param_values[sym] for sym in vars_needed]
            else:
                x_pos_diag[p] = coef_funcs['x_pos'](*vals)
                #x_pos_diag[p] = float(coef_expressions['x_pos'].evalf(subs=param_values))
        else:
            x_pos_diag[p] = 0  # frontera derecha

        if j > 0:
            y_neg_diag[p] = coef_funcs['y_neg'](*vals)
           # y_neg_diag[p] = float(coef_expressions['y_neg'].evalf(subs=param_values))
        else:
            y_neg_diag[p] = 0  # frontera inferior

        if j < Ny_in - 1:
            y_pos_diag[p] = coef_funcs['y_pos'](*vals)
           # y_pos_diag[p] = float(coef_expressions['y_pos'].evalf(subs=param_values))
        else:
            y_pos_diag[p] = 0  # frontera superior

        ## Condiciones de frontera
        if i == 0:
            coef_interno,_ = neumann(f_fut, Tn1_i_1j, Tn1_i1j, 0, dx, k) # Frontera Izquierda
            x_pos_diag[p] = coef_interno.evalf(subs = param_values)

        if j == 0:
            coef_interno,_ = neumann(f_fut, Tn1_ij_1, Tn1_ij1, 0, dy, k) # Frontera Inferior
            y_pos_diag[p] = coef_interno.evalf(subs = param_values)

        if i == Nx-3:
            coef_interno,_ = neumann(f_fut, Tn1_i1j, Tn1_i_1j, flux, dx, k) # Frontera Derecha
            x_neg_diag[p] = coef_interno.evalf(subs = param_values)

        if j == Ny-3 and i >= Nx_in/2:
            coef_interno,_ = neumann(f_fut, Tn1_ij1, Tn1_ij_1, 0, dy, k) # Frontera superior derecha
            y_neg_diag[p] = coef_interno.evalf(subs = param_values)

#print((main_diag))
#print((x_neg_diag[1:]))
# Desplazamientos de diagonales:
# main: 0, x_neg: -1, x_pos: +1, y_neg: -Nx_in, y_pos: +Nx_in


offsets = [0, -1, 1, -Nx_in, Nx_in]
diagonals = [main_diag, x_neg_diag[1:], x_pos_diag, y_neg_diag[(Nx_in):], y_pos_diag]

# Creamos la matriz dispersa
A = sp.sparse.diags(diagonals, offsets, shape=(N_in, N_in), format='csr')
#A_dense = A.toarray()
#np.set_printoptions(precision=2, linewidth=150, suppress=True)
#print(A_dense)


# ------------------------- Vector b -----------------------------------

# Orden importante: variables primero, luego parámetros
b_vars = [Tn_ij, Tn_i_1j, Tn_i1j, Tn_ij_1, Tn_ij1,
          rho, Cp, delta, k, dt, dx, dy, h, Ta, epsilon, sigma]

b_func = sy.lambdify(b_vars, b_expr, modules='numpy')


def construir_vector_b(Tvec_n, Nx, Ny, param_values):
    Nx_in, Ny_in = Nx - 2, Ny - 2
    N_in = Nx_in * Ny_in
    b_vector = np.zeros(N_in)

    const_params = [param_values[s] for s in [rho, Cp, delta, k, dt, dx, dy, h, Ta, epsilon, sigma]]

    for j in range(Ny_in):
        for i in range(Nx_in):
            p = i + j * Nx_in  # índice global

            def idx(ii, jj):
                return ii + jj * Nx_in

            def T_at(ii, jj):
                if 0 <= ii < Nx_in and 0 <= jj < Ny_in:
                    return Tvec_n[idx(ii, jj)]
                elif 0 <= ii < Nx_in/2 and jj == Ny_in:
                    return 9000  # Dirichlet superior izquierdo
                else:
                    return Tvec_n[p] 

            Tij     = T_at(i, j)
            Ti_1j   = T_at(i-1, j)
            Ti1j    = T_at(i+1, j)
            Tij_1   = T_at(i, j-1)
            Tij1    = T_at(i, j+1)

            # Ajustar propiedades según la posición (aluminio/acero)
            if i < Nx_in / 2:
                const_params_local = [rho2,Cp2, param_values[delta], k2] + const_params[4:]  # cambiar rho, k
            else:
                const_params_local = [rho1,Cp1, param_values[delta], k1] + const_params[4:]

            input_vals = [Tij, Ti_1j, Ti1j, Tij_1, Tij1] + const_params_local

            b_vector[p] = b_func(*input_vals)

            if i == Nx_in - 1:
                param_values[Tn_ij] = Tvec_n[p]
                if i < Nx_in / 2:
                    param_values[rho] = rho2
                    param_values[k] = k2
                    param_values[Cp] = Cp2
                else:
                    param_values[rho] = rho1
                    param_values[k] = k1
                    param_values[Cp] = Cp1
                _,indep = neumann(f_fut, Tn1_i1j, Tn1_i_1j, flux, dx, k)
                #print(indep)
                indep_func = sy.lambdify(vars_needed, indep, modules='numpy')
            
                b_vector[p] += indep_func(*[param_values[s] for s in vars_needed])

    return b_vector

# Parámetros del tiempo
def bucle_temporal(n_steps, save_folder='frames', n_frames=10):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    T_current = Tvec_n.copy()

    # Pasos donde se guardarán imágenes
    save_steps = np.linspace(0, n_steps - 1, n_frames, dtype=int)

    for n in range(n_steps):
        b = construir_vector_b(T_current, Nx, Ny, param_values)
        T_next = sp.sparse.linalg.spsolve(A, b)
        T_current = T_next

        if n in save_steps:
            T_plot = T_current.reshape((Ny - 2, Nx - 2))
            plt.figure(figsize=(6, 5))
            plt.imshow(T_plot, origin='lower', cmap='hot', extent=[0, Nx-2, 0, Ny-2])
            plt.colorbar(label='Temperatura (K)')
            plt.title(f'Temperatura en paso {n}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.savefig(f'{save_folder}/frame_{n:03d}.png', dpi=150)
            plt.close()

    # Mostrar la última
    T_final = T_current.reshape((Ny - 2, Nx - 2))
    plt.imshow(T_final, origin='lower', cmap='hot', extent=[0, Nx-2, 0, Ny-2])
    plt.colorbar(label='Temperatura (K)')
    plt.title('Distribución de temperatura al paso final')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
bucle_temporal(2000)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo transcurrido (time.time()): {elapsed_time} segundos")