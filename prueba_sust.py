import sympy as sy
temps = sy.symbols('A,B,C')
A,B,C = temps
f = A*C/B + B*C/A+2*A**4*2
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