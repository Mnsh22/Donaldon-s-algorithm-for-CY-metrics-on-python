# benchmark_monomial_jacobian.py
from itertools import combinations_with_replacement
from time import perf_counter
import numpy as np
import sympy as sp
import csv

# ----- parameters you can tweak -----
DEGREE = 5         # total degree (your case)
NVARS  = 5         # number of variables (your case)
N_SAMPLES = 1000   # how many random points to test
RNG_SEED = 0       # make runs reproducible
# ------------------------------------

# 1) symbols and variable list
x1, x2, x3, x4, x5 = sp.symbols('x1:6')  # x1..x5
variables = [x1, x2, x3, x4, x5]         # keep it simple

# 2) build all degree-5 monomials in 5 vars (126 of them)
def build_monomials_degree5(vars_list):
    gh = list(combinations_with_replacement(vars_list, DEGREE))
    out = []
    for t in gh:
        prod = t[0]*t[1]*t[2]*t[3]*t[4]
        out.append(prod)
    return out

some_list = build_monomials_degree5(variables)  # length 126

# 3) SymPy Jacobian (126 x 5) and lambdify -> fast numeric evaluator
F = sp.Matrix(some_list)
J = F.jacobian(variables)
f_sym = sp.lambdify(variables, J, 'numpy')  # call as f_sym(x1, x2, x3, x4, x5)

# 4) Fast numeric path using exponent counts
# For each monomial row r and variable i, derivative is: e[r,i] * prod_j x_j^{e[r,j]} / x_i, when e[r,i] > 0.
# We'll avoid zeros in random x to dodge division by zero.
def build_exponent_counts(nvars):
    # same order as monomials above, but on indices (0..nvars-1)
    gh_idx = list(combinations_with_replacement(range(nvars), DEGREE))
    E = np.zeros((len(gh_idx), nvars), dtype=int)
    for r in range(len(gh_idx)):
        tpl = gh_idx[r]
        for i in range(nvars):
            # how many times index i appears in the tuple -> exponent
            E[r, i] = tpl.count(i)
    return E

E = build_exponent_counts(NVARS)  # shape (126, 5)

def eval_jacobian_fast(E, x):
    # x is shape (5,), no zeros please
    # base = prod_j x_j^{e_j} for each row
    Xpow = np.ones_like(E, dtype=float)
    for j in range(E.shape[1]):
        Xpow[:, j] = x[j] ** E[:, j]
    base = np.prod(Xpow, axis=1)  # shape (rows,)

    Jnum = np.zeros((E.shape[0], E.shape[1]), dtype=float)
    for i in range(E.shape[1]):
        e = E[:, i]
        # only rows with e>0 contribute
        mask = e > 0
        # derivative: e * base / x[i]
        Jnum[mask, i] = e[mask] * base[mask] / x[i]
        # rows with e==0 already zero
    return Jnum

# 5) generate random test points (avoid zeros)
rng = np.random.default_rng(RNG_SEED)
X = rng.uniform(0.5, 1.5, size=(N_SAMPLES, NVARS))  # keep away from 0 to avoid division

# 6) benchmark timings
# warm-up
_ = f_sym(X[0,0], X[0,1], X[0,2], X[0,3], X[0,4])
_ = eval_jacobian_fast(E, X[0])

# SymPy (lambdified) timing
t0 = perf_counter()
for k in range(N_SAMPLES):
    _ = f_sym(X[k,0], X[k,1], X[k,2], X[k,3], X[k,4])
t1 = perf_counter()
time_sym = t1 - t0

# Fast exponent method timing
t2 = perf_counter()
for k in range(N_SAMPLES):
    _ = eval_jacobian_fast(E, X[k])
t3 = perf_counter()
time_fast = t3 - t2

# 7) accuracy check (compare a few samples)
max_abs_diff = 0.0
for k in range(min(10, N_SAMPLES)):
    A1 = np.array(f_sym(X[k,0], X[k,1], X[k,2], X[k,3], X[k,4]), dtype=float)
    A2 = eval_jacobian_fast(E, X[k])
    diff = np.max(np.abs(A1 - A2))
    if diff > max_abs_diff:
        max_abs_diff = diff

print(f"SymPy lambdified eval time over {N_SAMPLES} points: {time_sym:.4f} s")
print(f"Fast exponent method time over {N_SAMPLES} points: {time_fast:.4f} s")
print(f"Max |difference| (should be ~0): {max_abs_diff:.3e}")

# 8) store results so you can compare different runs/parameters later
# Append to a CSV (creates it if missing)
with open("bench_results.csv", "a", newline="") as f:
    w = csv.writer(f)
    # header once (you can comment this out after first run)
    # w.writerow(["degree","nvars","n_samples","time_sympy","time_fast","max_abs_diff"])
    w.writerow([DEGREE, NVARS, N_SAMPLES, time_sym, time_fast, max_abs_diff])
print("Appended results to bench_results.csv")
