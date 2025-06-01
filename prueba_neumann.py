import sympy as sy
import numpy as np

k1 = 45
k2 = 237
rho1 = 2700
rho2 = 7870
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

f = sy.expand(f)
f_collected = sy.collect(f, temps)

param_values = {
    sy.symbols('rho'): rho1,
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
    sy.symbols('Tn_ij'): 400,  # solo necesario para calcular b
    sy.symbols('q'): 2
}

def neumann(f, nodo_fantasma, nodo_interno, q, dx, k): # Esta esta pendiente por ahora
    sustitucion = nodo_interno + q * dx / k
    fmod = f.subs(nodo_fantasma, sustitucion)
    fmod = sy.expand(fmod)
    fcol = sy.collect(fmod,nodo_interno)
    coef = fcol.coeff(nodo_interno)
    indep, _ = fmod.as_independent(*temps)
    return coef,indep

coef_expressions = {
    'center': f_collected.coeff(Tn1_ij),
    'x_neg': f_collected.coeff(Tn1_i_1j),
    'x_pos': f_collected.coeff(Tn1_i1j),
    'y_neg': f_collected.coeff(Tn1_ij_1),
    'y_pos': f_collected.coeff(Tn1_ij1)
}
independiente, _ = f.as_independent(*temps)
_,i=neumann(f,Tn1_i_1j,Tn1_i1j,1,dx,3)
sy.pprint(independiente)
sy.pprint(i)