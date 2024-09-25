import numpy as np
import matplotlib.pyplot as plt
from stationary_solver import StationarySolver
from thomas_solver import ThomasAlgorithm

N = 1000
h = 1.0 / (N - 1)
h_sqr = h * h
gamma_1 = 0.0
gamma_2 = 0.1

use_direct_solver = False

def f(x):
    return 250.0 * x * x * np.sin(10 * np.pi * x)

# SET UP
u = [0] * (N + 1)
b = [0] * (N + 1)
A = [[0] * (N + 1) for _ in range(N + 1)]
# Left
u[0] = 0
b[0] = gamma_1
A[0][0] = 1.0
# Interior points
for i in range(1, N):
    u[i] = 0
    b[i] = h_sqr * f(i * h)
    A[i][i-1] = -1.0
    A[i][i] = 2.0
    A[i][i+1] = -1.0
# Right
u[N] = 0
b[N] = gamma_2
A[N][N] = 1.0

# SOLVING
if use_direct_solver:
    solver = ThomasAlgorithm(np.array(A), np.array(b))
    x = [0] * (N + 1)  # Initialize solution vector
    solver.solve(x, verbose=True)
    # print("Solution:", x)

else:
    solver = StationarySolver(np.array(A), np.array(b))
    solver.set_max_iterations(100)
    solver.set_convergence_residue(1.0e-6)

    # method = "richardson"
    # method = "jacobi"
    # method = "gauss-seidel"
    method = "sor"

    x = solver.solve(method,False)
    
x_values = np.linspace(0, 1, N+1)

if use_direct_solver:
    plt.plot(x_values, x)
else:
    for i, iteration_solution in enumerate(x):
        if i % 10 == 0:  # Plot every 10th iteration
            plt.plot(x_values, iteration_solution, alpha=0.3, label=f'Iteration {i}')

plt.title('Solution of Poisson Equation (1D)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid(True)
plt.show()

