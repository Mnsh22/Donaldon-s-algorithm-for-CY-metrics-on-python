import numpy as np
import sympy as sp
import math
import time
from itertools import combinations_with_replacement

S = np.array([1j, 2], dtype=complex)
Sbar = np.array([-1j, 2], dtype=complex)

A = np.einsum('m,n->mn', S, Sbar)
B = np.einsum('n,m->mn', S, Sbar)
C = (np.conj(B).T + A)/2
#print(C)

D = np.einsum('ab, ab', A, np.outer(S, Sbar) )

print(D)