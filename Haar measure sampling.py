from itertools import combinations_with_replacement
from sympy import *
import sympy as sp
import numpy as np


x1 = Symbol('x1')
x2 = Symbol('x2')
x3 = Symbol('x3')
x4 = Symbol('x4')
x5 = Symbol('x5')

variables = [x1,x2,x3,x4,x5]
combo = combinations_with_replacement(variables, 5)
gh = list(combo)

some_list = []
for j in range(126):
    y = gh[j]
    prod = y[0]*y[1]*y[2]*y[3]*y[4]
    some_list.append(prod)

#print(some_list)

A = np.empty((126,5), dtype= object)

for i in range(5):
    for j in range(126):
        A[j,i]= sp.diff(some_list[j], variables[i])

#print(A)

A_sym = sp.Matrix(some_list).jacobian(variables)
A_num = A_sym.subs([(x1,1), (x2,2), (x3,3), (x4,4), (x5,5)])
print(A_sym)          # stays exact integers
print(np.array(A_num, dtype=float))  # if you need a NumPy array

## we can finally do derivatives. Yippieeee. Computationally much faster as well cause now I can just substitute the
## values at the end and then gg.


'''
yprime = y.diff(x)
print(yprime)

def Monomial_list_coord_value():

    Monomial_list = []
    for i in range(len(sample)):
        x = coordinates_for_every_p_M[i]
        variables = [x[0],x[1],x[2],x[3],x[4]]
        combo = combinations_with_replacement(variables, 5)
        gh = list(combo)
        Monomial_list.append(gh)

    return Monomial_list

every_single_monomial_combination_tuple = Monomial_list_coord_value()


def section_vector_list():

    section_vec_list = []

    for i in range(len(sample)):
        aid_list = []
        x = every_single_monomial_combination_tuple[i]
        s = np.zeros(N_k, dtype=complex)

        for j in range(N_k):
            y = list(x[j])
            prod = y[0]#*y[1]#*y[2]#*y[3]#*y[4]
            # cause y a tuple and not int
            aid_list.append(prod)


        for r in range(N_k):
            s[r] = aid_list[r] # s_alpha vector

        section_vec_list.append(s)

    return section_vec_list

svl = section_vector_list()'''
