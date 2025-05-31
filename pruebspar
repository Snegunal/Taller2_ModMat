from scipy.sparse import diags
import numpy as np

n = 5
main_diag = 2 * np.ones(n)
upper_diag = -1 * np.ones(n - 1)  # longitud correcta
lower_diag = -1 * np.ones(n - 1)

lower_diag[0] = 1
upper_diag[0] = 1

A = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csr')
print(A.toarray())
