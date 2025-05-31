import sympy as sy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

k1 = 45
k2 = 237
# Definimos las variables simbólicas

rho,Cp,delta,k,dt,dx,dy,h,Ta,epsilon,sigma = sy.symbols('rho,Cp,delta,k,dt,dx,dy,h,Ta,epsilon,sigma')
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
        

# Expandimos y agrupamos por todas las temperaturas
f = sy.expand(f)
f_collected = sy.collect(f, temps)

# Mostramos la expresión agrupada
print("\nExpresión agrupada:")
sy.pprint(f_collected)

# Mostramos coeficientes de cada temperatura
print("\nCoeficientes:")
for temp in temps:
    coef = f_collected.coeff(temp)
    print(f"Coeficiente de {temp}:")
    sy.pprint(coef)
    print()

# Parámetros físicos y espaciales
param_values = {
    sy.symbols('rho'): 1000,
    sy.symbols('Cp'): 4200,
    sy.symbols('delta'): 0.01,
    sy.symbols('k'): k1,
    sy.symbols('dt'): 1,
    sy.symbols('dx'): 0.01,
    sy.symbols('dy'): 0.01,
    sy.symbols('h'): 10,
    sy.symbols('Ta'): 300,
    sy.symbols('epsilon'): 0.9,
    sy.symbols('sigma'): 5.67e-8,
    sy.symbols('Tn_ij'): 400  # solo necesario para calcular b
}

Nx, Ny = 6,6
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

main_diag     = np.zeros(N_in, dtype=np.float64)
x_neg_diag    = np.zeros(N_in, dtype=np.float64)
x_pos_diag    = np.zeros(N_in, dtype=np.float64)
y_neg_diag    = np.zeros(N_in, dtype=np.float64)
y_pos_diag    = np.zeros(N_in, dtype=np.float64)


Tvec_n = 300*np.ones(N_in)

for j in range(Ny-2): # Queda pendiente las condiciones de frontera
    for i in range(Nx-2):
        p = i + j * (Nx-2)  # índice global
        param_values[Tn_ij] = Tvec_n[p]

        if i < Nx_in/2:
            param_values[k] = k2
        if i >= Nx_in/2:
            param_values[k] = k1
        
        # Evaluar y asignar coeficientes
        main_diag[p]  = float(coef_expressions['center'].evalf(subs=param_values))

        # Dirección x: izquierda y derecha
        if i > 0:
            x_neg_diag[p] = float(coef_expressions['x_neg'].evalf(subs=param_values))
        else:
            x_neg_diag[p] = 0  # frontera izquierda

        if i < Nx_in - 1:
            x_pos_diag[p] = float(coef_expressions['x_pos'].evalf(subs=param_values))
        else:
            x_pos_diag[p] = 0  # frontera derecha

        if j > 0:
            y_neg_diag[p] = float(coef_expressions['y_neg'].evalf(subs=param_values))
        else:
            y_neg_diag[p] = 0  # frontera inferior

        if j < Ny_in - 1:
            y_pos_diag[p] = float(coef_expressions['y_pos'].evalf(subs=param_values))
        else:
            y_pos_diag[p] = 0  # frontera superior
        
print((main_diag))
print((x_neg_diag[1:]))
# Desplazamientos de diagonales:
# main: 0, x_neg: -1, x_pos: +1, y_neg: -Nx_in, y_pos: +Nx_in


offsets = [0, -1, 1, -Nx_in, Nx_in]
diagonals = [main_diag, x_neg_diag[1:], x_pos_diag, y_neg_diag[(Nx_in):], y_pos_diag]

# Creamos la matriz dispersa
A = sp.sparse.diags(diagonals, offsets, shape=(N_in, N_in), format='csr')
A_dense = A.toarray()
np.set_printoptions(precision=2, linewidth=150, suppress=True)
print(A_dense)

def construir_vector_b(Tvec_n, Nx, Ny, param_values): # Queda pendiente condiciones de neumann
    Nx_in, Ny_in = Nx - 2, Ny - 2
    N_in = Nx_in * Ny_in
    b_vector = np.zeros(N_in)

    for j in range(Ny_in):
        for i in range(Nx_in):
            p = i + j * Nx_in  # índice global

            # Índices vecinos
            def idx(ii, jj):
                return ii + jj * Nx_in

            # Extraer vecinos y dirichlet
            def T_at(ii, jj):
                if 0 <= ii < Nx_in and 0 <= jj < Ny_in:
                    return Tvec_n[idx(ii, jj)]
                elif 0 <= ii < Nx_in/2 and jj == Ny_in:        
                    return 350 # Dirichlet 350 K
                
            
            def aplicar_condicion_neumann(f, nodo_fantasma, nodo_interno, q, dx, k): # Esta esta pendiente por ahora
                sustitucion = nodo_interno + q * dx / k
                return f.subs(nodo_fantasma, sustitucion)


            vals = {
                Tn_ij:    T_at(i, j),
                Tn_i_1j:  T_at(i-1, j),
                Tn_i1j:   T_at(i+1, j),
                Tn_ij_1:  T_at(i, j-1),
                Tn_ij1:   T_at(i, j+1),
            }

            vals.update(param_values)  # parametros

            b_vector[p] = float(b_expr.evalf(subs=vals))

    return b_vector

