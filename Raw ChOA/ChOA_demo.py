import matplotlib.pyplot as plt
import numpy as np
from ChOA import CHOA


fobj = lambda x: -x[:,1] ** 2 - x[:,0] ** 2 if x.ndim == 2 else -x[1] ** 2 - x[0] ** 2


lb = np.array([-1,-1])
ub = np.array([1,1])

dim = 2
N = 50
Max_iter = 3000

best_x, best_y = CHOA(fobj, lb, ub, dim, N, Max_iter)

print(best_x, best_y)
