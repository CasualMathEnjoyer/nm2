import numpy as np
import math
import matplotlib.pyplot as plt

# Define parameters
n = 5
h = 0.1
boundary_name = "linear"
rhs_name = "const"

def generate_Ab(n, rhs_name):
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

A, b = generate_Ab(n, rhs_name)

# Function to plot the values of nonzero elements in a matrix
def plot_matrix_values(matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Value')
    # plt.title(title)
    # plt.xlabel('Column Index')
    # plt.ylabel('Row Index')
    plt.savefig("A.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# Subset of the matrix A for visualization (e.g., 20x20)
subset_A = A[:, :]

# Plot the values of nonzero elements in matrix A
plot_matrix_values(subset_A, 'Values of Nonzero Elements in Matrix A (Subset)')