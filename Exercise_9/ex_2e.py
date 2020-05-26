import numpy as np
import math
import cvxpy as cp


def caltulate_matrix(phi, t):

    # Input phy in rad/s and t in s
    H = t.dot(phi.T)
    H = np.cos(H)
    return H


# 2e)
y2_measured = np.genfromtxt('mathias_distress_call_2.csv', delimiter=',')
print(y2_measured.size)
y2_measured = y2_measured[:, np.newaxis]
t2 = np.array([np.arange(0, y2_measured.size/8192, 1/8192)]).T
reg_param = [10, 100, 1000, 10000]
print(cp.installed_solvers())

# exercise i
phi_i = np.array([[104, 111, 116, 122, 126, 131, 136, 142, 147, 158, 164, 169,
                   174, 181, 183, 193, 199, 205, 208, 214, 220, 226, 231, 237, 243, 249, 254]]).T
H_i = caltulate_matrix(phi_i, t2)
X_collector_i = np.zeros((phi_i.size, len(reg_param)))
for k in range(len(reg_param)):
    print('start optimization', k)
    x_i = cp.Variable(phi_i.size)
    # objective = cp.Minimize(cp.sum_squares(y2_measured - H_iii @ x_iii) + reg_param[k]* cp.norm(x_iii, 1)) # this approach worked but was really slow
    objective = cp.Minimize(cp.quad_form(x_i, H_i.T @ H_i) - 2*y2_measured.T @
                            H_i @ x_i + y2_measured.T@y2_measured + reg_param[k]*cp.norm(x_i, 1))
    constraint = [x_i >= 0]
    prob = cp.Problem(objective, constraint)
    prob.solve()
    print("Optimal var reached", k)
    X_collector_i[:, k] = x_i.value
print(X_collector_i)
np.savetxt('output_2e_i.csv', X_collector_i, delimiter=',')


# exercise ii
phi_ii = np.array([np.arange(311.127, 622.254, 311.127/27)]).T
H_ii = caltulate_matrix(phi_ii, t2)
X_collector_ii = np.zeros((phi_ii.size, len(reg_param)))
for k in range(len(reg_param)):
    print('start optimization', k)
    x_ii = cp.Variable(phi_ii.size)
    objective = cp.Minimize(cp.quad_form(x_ii, H_ii.T @ H_ii) - 2*y2_measured.T @
                            H_ii @ x_ii + y2_measured.T@y2_measured + reg_param[k]*cp.norm(x_ii, 1))
    constraint = [x_ii >= 0]
    prob = cp.Problem(objective, constraint)
    prob.solve()
    print("Optimal var reached", k)
    X_collector_ii[:, k] = x_ii.value
print(X_collector_ii)
np.savetxt('output_2e_ii.csv', X_collector_ii, delimiter=',')


# exercise iii
phi_iii = np.zeros((27, 1))
phi_iii[0] = 150
phi_iii[1] = 175
for i in range(2, phi_ii.size):
    phi_iii[i] = math.ceil(0.5*phi_iii[i-1]) + math.ceil(0.8*phi_iii[i-2])
H_iii = caltulate_matrix(phi_iii, t2)
X_collector_iii = np.zeros((phi_iii.size, len(reg_param)))
for k in range(len(reg_param)):
    print('start optimization', k)
    x_iii = cp.Variable(phi_iii.size)
    objective = cp.Minimize(cp.quad_form(x_iii, H_iii.T @ H_iii) - 2*y2_measured.T @
                            H_iii @ x_iii + y2_measured.T@y2_measured + reg_param[k]*cp.norm(x_iii, 1))
    constraint = [x_iii >= 0]
    prob = cp.Problem(objective, constraint)
    prob.solve()
    print("Optimal var reached", k)
    X_collector_iii[:, k] = x_iii.value
print(X_collector_iii)
np.savetxt('output_2e_iii.csv', X_collector_iii, delimiter=',')
