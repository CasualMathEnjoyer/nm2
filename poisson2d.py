# poisson2d.py
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from stationary_solver import StationarySolver
from find_optimal_relax import Optimiser

perform_grid_search = False

N = 50
h = 1.0 / (N - 1)
h_sqr = h * h

# Set-up the linear system
dofs = (N + 1) * (N + 1)
u = np.zeros(dofs)

def generate_Ab(n, rhs_name, boundary_name="const"):
    A = np.zeros((n * n, n * n))
    b = np.zeros(n * n)
    # Top, bot, Left and right Dirichlet boundary conditions
    for j in range(n):
        I_top = j + 0*n
        I_bot = j + (n-1)*n
        I_left = 0 + j*n
        I_right = (n-1) + j*n
        A[I_top, I_top] = 1
        A[I_bot, I_bot] = 1
        A[I_left, I_left] = 1
        A[I_right, I_right] = 1
        boundary = 0
        if boundary_name == "linear":
            boundary = 100 * j * h
        b[I_top] = boundary
        b[I_bot] = -boundary
        b[I_left] = boundary
        b[I_right] = -boundary

    # Rest of the matrix
    for j in range(1, n-1):
        for i in range(1, n-1):
            I = j + i*n
            A[I, I-n] = 1
            A[I, I-1] = 1
            A[I, I] = -4
            A[I, I+1] = 1
            A[I, I+n] = 1

    # Right hand side
    for j in range(1, n-1):
        for i in range(1, n-1):
            x_val = j * h
            y_val = i * h
            val = 0
            if rhs_name == "const":
                val = 10
            elif rhs_name == "linear":
                val = 10 * x_val * y_val
            elif rhs_name == "quadratic":
                val = 10 * (1.0 - 2*x_val*x_val - 2*y_val*y_val)
            elif rhs_name == "sin":
                val = 250 * x_val * x_val * math.sin(10 * math.pi * (x_val*x_val + y_val*y_val))
            b[j + i*n] = val
    return A, b

A, b = generate_Ab(N+1, "linear", "const")

start = time.time()
method = "gauss-seidel"  # "sor", "jacobi", "richardson", "gauss-seidel"
# Solve the linear system using StationarySolver
solver = StationarySolver(A, b)
solver.set_max_iterations(1000)
solver.set_convergence_residue(1.0e-4)
if perform_grid_search:
    relaxation = Optimiser(A, b, solver, method).grid_search(np.linspace(1, 2, 10))
    print(f"optimal relaxation for {method}:", relaxation)
else:
    relaxation = 1.87  # sor, richardson
    # relaxation = 1.89  # gauss-seidel
solver.set_relaxation(relaxation)
solution = solver.solve(method=method)
stop = time.time()
print(f"Solve time: {stop - start:0.2f} seconds")

# Write the solution to the output file
with open("poisson-2d.txt", "w") as f:
    for j in range(N + 1):
        for i in range(N + 1):
            index = j * (N + 1) + i
            f.write(f"{i * h} {j * h} {solution[index]}\n")

def plot_solution_3d(solution, N, model):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    Z = solution.reshape((N + 1, N + 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Solution')
    plt.savefig(model + "_3d" + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_solution_2d(solution, N, model):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    Z = solution.reshape((N + 1, N + 1))

    plt.figure()
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(label='Solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(model + "_2d" + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

plot_solution_2d(solution, N, "cc")  # c, l, q, s
plot_solution_3d(solution, N, "cc")