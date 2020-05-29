import numpy as np
import cvxpy as cp

# y_vec = np.random.choice([0, 1], size=(728,), p=[9./10, 1./10])
# M_mat = np.random.choice([0, 1], size=(728, 801), p=[9./10, 1./10])
# beta = cp.Variable(M_mat.shape[0])
# objective = 0
# for i in range(400):
#     objective += y_vec[i] * M_mat[:, i].T @ beta - cp.logistic(M_mat[:, i].T @ beta)

# prob = cp.Problem(cp.Maximize(objective))
# prob.solve(verbose=True)
# print("Optimal var reached",  beta.value)
# print()

a = np.array([1,2,3,4])
a = np.atleast_2d(a)
print(a.shape) # --> (
