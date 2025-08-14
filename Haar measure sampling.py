from itertools import combinations_with_replacement
import sympy as sp
import numpy as np

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

B = np.empty((126,5,5), dtype=object)
for j in range(rows):
    for i in range(cols):
        for k in range(cols):
            B[j,i,k] = sp.diff(A[j,i], variables[k])
print(B.shape)

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

B_num = np.empty((rows, cols, cols), dtype=float)
for j in range(rows):
    for i in range(cols):
        for k in range(cols):
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

