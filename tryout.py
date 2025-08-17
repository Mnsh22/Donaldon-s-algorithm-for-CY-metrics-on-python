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


w = [5,6,7,8,9]
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

#print(w)

#print(np.argmax(w))

def combinations():
    z0 = sp.Symbol('z0')
    z1 = sp.Symbol('z1')
    z2 = sp.Symbol('z2')
    z3 = sp.Symbol('z3')
    z4 = sp.Symbol('z4')
    gigachad = [z0, z1, z2, z3, z4]
    variables = [0,1,2,3,4]
    combo = combinations_with_replacement(variables, 5)  # change 612 line too
    gh = list(combo)
    print(gh)
    vector_list = []
    for x in gh:
        gigacombo = np.zeros(5, dtype=object)
        for i in range(5):
            b = x[i]
            gigacombo[i] = gigachad[b]
        vector_list.append(gigacombo)

    return vector_list

print(combinations())

def derivative_section_matrix_builder():
    z0 = sp.Symbol('z0')
    z1 = sp.Symbol('z1')
    z2 = sp.Symbol('z2')
    z3 = sp.Symbol('z3')
    z4 = sp.Symbol('z4')
    variables = [z0, z1, z2, z3, z4]

    # list of all degree-5 monomials (126 of them)
    combo = combinations_with_replacement(variables, 5)  # change 612 line too
    gh = list(combo)

    some_list = [] # S_alpha basically
    for j in range(len(gh)):
        y = gh[j]
        prod = sp.prod(y) #each s_alpha element (for each higher k add another product)
        some_list.append(prod)

    return some_list

print(derivative_section_matrix_builder())



def T_map_function(h, ff, n_iter):

    wml = w_M_list
    secmatrixgen = sfm

    T_map = ff * sohsf

    for _ in range(10): # Input here how many times to iterate the T_map
        T_map =  ff * sum( (secmatrixgen[i] * wml[i])/ (np.einsum("mn,mn",np.transpose(np.linalg.inv(T_map)),secmatrixgen[i])) for i in range(len(sample)) )
        print(T_map)
    return T_map

T_map = T_map_function(ff, sohsf)

#print(T_map)
print(T_map.shape)

h_new = np.transpose(np.linalg.inv(T_map))

print(h_new)




