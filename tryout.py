from itertools import combinations_with_replacement
import sympy as sp
import numpy as np
from time import perf_counter
# symbols

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
x3 = sp.Symbol('x3')
x4 = sp.Symbol('x4')
x5 = sp.Symbol('x5')
variables = [x1, x2, x3, x4, x5]

    # all degree-5 monomials (126 of them)
gh = list(combinations_with_replacement(variables, 5))

    # products for each 5-tuple
some_list = []
for j in range(len(gh)):
    y = gh[j]
    prod = y[0]*y[1]*y[2]*y[3]*y[4]
    some_list.append(prod)

rows = len(some_list)   # 126
cols = len(variables)   # 5

    # derivative matrix A (symbolic)
A = np.empty((rows, cols), dtype=object)
for j in range(rows):
    for i in range(cols):
        A[j, i] = sp.diff(some_list[j], variables[i])

    # substitute x1=1, x2=2, x3=3, x4=4, x5=5  -> numeric array
A_num = np.empty((rows, cols), dtype=float)
for j in range(rows):
    for i in range(cols):
        expr = A[j, i]
        value = (
            expr
            .subs(x1, 1)
            .subs(x2, 2)
            .subs(x3, 3)
            .subs(x4, 4)
            .subs(x5, 5)
            )
        A_num[j, i] = float(value)


B = np.empty((126,5,5), dtype=object)
B_num = np.empty((rows, cols, cols), dtype=float)
for j in range(rows):
    for i in range(cols):
        for k in range(cols):
            B[j, i, k] = sp.diff(A[j, i], variables[k])
            expr = B[j,i,k]
            value = (
                expr
                .subs(x1, 1)
                .subs(x2, 2)
                .subs(x3, 3)
                .subs(x4, 4)
                .subs(x5, 5)
                )
            B_num[j, i, k] = float(value)


# quick sanity checks
print(B_num.shape)      # (126, 5)
print(B_num)         # first row should be [5., 0., 0., 0., 0.]



dJ = np.arange(75).reshape(5,3,5)
dg = np.arange(125).reshape(5,5,5)
ddg = np.arange(625).reshape(5,5,5,5)
J = np.arange(15).reshape(3,5)
g = np.eye(5)+1j*np.eye(5)
cool = np.matrix.conj(g)
Tr = np.trace(g, dtype=complex)
i = np.einsum("mn,mn", g,g)

P = np.zeros((15,5,5), dtype=complex)
for i in range(15):
    for j in range(5):
        for k in range(5):
            P[i ,j, k] = 1

print(P.shape)

print(np.arange(15).reshape(3,5))

