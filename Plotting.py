import numpy as np
import sympy as sp
import math
import time
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt


sigma_TO = [0.74-0.3, 0.63-0.3, 0.56-0.3, 0.51-0.3]
sigma_TO2 =[0.735-0.3, 0.626-0.3, 0.561-0.3, 0.516-0.3]
sigma_paper = [0.37, 0.27, 0.19, 0.13, 0.091, 0.066, 0.051, 0.04, 0.032, 0.027, 0.023, 0.02]

R_TO = [278.42, 248.89417654887467, 213.93, 179.14066463458934]
M_paper = [24, 18.1, 13.2, 9.21, 6.63, 4.95, 3.83, 3.06, 2.51, 2.13, 1.86, 1.69]



k0 = [1,2,3,4]
k2 = list(range(1,13))

plt.plot(k0, sigma_TO, marker='o', linestyle='-', color='b', label='σ code')
plt.plot(k0, sigma_TO2, marker='o', linestyle='-', color='g', label='σ code')
plt.plot(k2, sigma_paper, marker='o', linestyle='-', color='r', label='σ paper')


# Add labels and title
plt.xlabel('k')
plt.ylabel('σ')
plt.title('Sigma vs k')
plt.legend()

plt.show()

plt.plot(k0, R_TO, marker='o', linestyle='-', color='b', label='σ code')
plt.plot(k2, M_paper, marker='o', linestyle='-', color='r', label='R paper')

# Add labels and title
plt.xlabel('k')
plt.ylabel('||M||')
plt.title('||M|| vs k')
plt.legend()

plt.show()
