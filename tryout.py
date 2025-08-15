import numpy as np
import sympy as sp
import math
from itertools import combinations_with_replacement
from time import perf_counter

t1 = perf_counter()
def sample_point_C5_on_unit_sphere():

    v = np.random.randn(5) + 1j*np.random.randn(5)
    vec = v / np.linalg.vector_norm(v)
    return vec

v = sample_point_C5_on_unit_sphere()

def projected_S9_point_onto_coord_of_CP4(v):
    w = np.exp(- 1j * np.angle(v[4])) * v
    return w

def find_quintic_roots():
    v1 = sample_point_C5_on_unit_sphere()
    v2 = sample_point_C5_on_unit_sphere()
    p = projected_S9_point_onto_coord_of_CP4(v1)
    q = projected_S9_point_onto_coord_of_CP4(v2)


    polynomial = np.zeros(6, dtype=complex)

    for k in range(6):
        polynomial[k] = sum(math.comb(5, k) * (p[i] ** (5 - k)) * (q[i] ** k) for i in range(5))

    roots = np.roots(polynomial[::-1])

    return roots, p, q

quintic_root_builder = find_quintic_roots()

p_M_points = 10100
def generate_quintic_points(p_M_points):
    points = []
    local_find_q_roots = quintic_root_builder
    while len(points) < p_M_points:
        roots, p, q = local_find_q_roots

        for t in roots:
            z = p + t * q
            z /= np.linalg.norm(z)

            Qz = np.sum(z ** 5)
            if np.abs(Qz) > 1e-16:  # threshold
               continue

            points.append(z)

        if len(points) >= p_M_points:
            break

    return np.array(points)

sample = generate_quintic_points(p_M_points)

t2 = perf_counter()
print(t2-t1)
print(sample)

a = np.array([1,2,3])
b = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
c = np.array([[1,0,0],
              [0,1,0]])

q = np.einsum('i,ij,kj->k', a,b,c)
print(q)