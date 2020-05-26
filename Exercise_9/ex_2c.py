'''Advanced Topic in Controls: Largew Scale Convex Optimization. Programming Exercise 2c '''

__author__ = 'Manuel Galliker'
__license__ = 'GPL'

import numpy as np
import math

def caltulate_matrix(phi, t):

    # Input phy in rad/s and t in s
    H = t.dot(phi.T)
    H = np.cos(H)
    return H

def solve_least_squares(phi, t, y1_measured):
    H = caltulate_matrix(phi, t)
    return (np.linalg.inv(H.T @ H) @ H.T @ y1_measured)
    
# 2c)
y1_measured = np.genfromtxt('mathias_distress_call_1.csv', delimiter=',')
y1_measured = y1_measured[:, np.newaxis]
t1 = np.array([np.arange(0, y1_measured.size/8192, 1/8192)]).T

# exercise i
phi_i = np.array([[104, 111, 116, 122, 126, 131, 136, 142, 147, 158, 164, 169,
                   174, 181, 183, 193, 199, 205, 208, 214, 220, 226, 231, 237, 243, 249, 254]]).T
x_i = solve_least_squares(phi_i, t1, y1_measured)
print('exercise i: x_i = ', x_i)

# exercise ii
phi_ii = np.array([np.arange(311.127, 622.254, 311.127/27)]).T
x_ii = solve_least_squares(phi_ii, t1, y1_measured)
print('exercise ii: x_ii = ', x_ii)

# exercise iii
phi_iii = np.zeros((27, 1))
phi_iii[0] = 150
phi_iii[1] = 175
for i in range(2, phi_ii.size):
    phi_iii[i] = math.ceil(0.5*phi_iii[i-1]) + math.ceil(0.8*phi_iii[i-2])
    print(phi_iii)
x_iii = solve_least_squares(phi_iii, t1, y1_measured)
print('exercise iii: x_iii = ', x_iii)
