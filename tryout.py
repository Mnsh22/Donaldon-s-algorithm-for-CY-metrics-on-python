import numpy as np
import sympy as sp
from itertools import combinations_with_replacement
def section_sympy_builder(K): #Using sympy for derivatives (I used it before and this is the best version I have so far)

    z = sp.symbols('z0 z1 z2 z3 z4')

    tuples = list(combinations_with_replacement(range(5), K))

    s_list = [sp.prod(z[idx] for idx in t) for t in tuples]  # N_k elements
    N_k = len(s_list)

    A_sym = sp.Matrix([[sp.diff(s_list[j], z[i]) for j in range(N_k)] for i in range(5)])
    A_aid = sp.lambdify(z, A_sym, 'numpy')  # returns 5xN_k array

    B_sym = sp.Array([[[sp.diff(A_sym[i, j], z[k]) for j in range(N_k)] for i in range(5)] for k in range(5)])
    B_aid = sp.lambdify(z, B_sym, 'numpy')  # returns 5x5xN_k

    return A_aid, B_aid

coordinate_list=[[1,2,3,4,5],[6,7,8,9,10]]
def derivative_section_matrix_builder(coords, K):

    A_aid, B_aid = section_sympy_builder(K)
    ds_list = []
    dds_list = []
    for coord in coords:  # coord is length-5 complex vector
        A_num = np.asarray(A_aid(*coord), dtype=np.complex128)   # (5, N_k)
        ds_list.append(A_num)
        # If not needed, skip the next two lines entirely:
        B_num = np.asarray(B_aid(*coord), dtype=np.complex128)   # (5, 5, N_k)
        dds_list.append(B_num)
    return ds_list, dds_list

ds_list, dds_list = derivative_section_matrix_builder(coordinate_list, 2)

print(ds_list[0].shape, dds_list[0].shape)

v = [1,2,3]
r = [4,5,6]

G = np.einsum('i,j->ij', v,r)
H = np.outer(v,r)
print(G)
print(H)