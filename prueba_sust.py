import sympy as sy
import time
from sympy.utilities.lambdify import lambdify
import numpy as np

start_time = time.time()

rho = sy.symbols('rho')
k = sy.symbols('k')
temps = sy.symbols('A,B,C')
A,B,C = temps
f = rho*A*C/B + B*C/A+2*A**4*2 +rho*(A+ rho)/k**2
def neuman(NodoFront,NodoInverso):
    sust = NodoInverso**2
    fmod = f.subs(NodoFront,sust)
    fmod = sy.expand(fmod)
    fcol = sy.collect(fmod,NodoInverso)
    coef = fcol.coeff(NodoInverso)
    sy.pprint(fmod)
    indep, _ = fmod.as_independent(*temps)
    return coef,indep
J,I = neuman(C,A)
sy.pprint(I)

rho1 = 2
rho2 = 3
Cp1 = 4
Cp2 = 5



end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo transcurrido (time.time()): {elapsed_time} segundos")