import numpy as np
from scipy.linalg import lu_factor, lu_solve
from typing import Optional

def lu_solver(file_name: str, method: str = "gem", verbose: int = 0) -> Optional[np.ndarray]:
    # Read matrix from file
    try:
        matrix = np.loadtxt(file_name)
    except FileNotFoundError:
        print(f"Cannot open file {file_name}.")
        return None

    # Check if matrix is square
    n, m = matrix.shape
    if n != m:
        print("Only square matrices are allowed for LU decomposition.")
        return None

    print(f"Matrix dimensions are {n}x{n}.")

    # Perform LU decomposition
    if method == "gem":
        # Perform LU decomposition using Gaussian elimination method
        lu, piv = lu_factor(matrix)
    elif method == "crout":
        # Perform LU decomposition using Crout's method
        lu, piv = lu_factor(matrix, overwrite_a=True)
    else:
        print("Invalid method specified.")
        return None

    # Generate a random vector for testing
    x = np.random.rand(n)

    # Multiply matrix by vector
    print("Multiplying matrix-vector...")
    b = np.dot(matrix, x)

    # Solve the system
    print("Solving the system...")
    solution = lu_solve((lu, piv), b)

    return solution

if __name__ == "__main__":
    file_name = "your_file.txt"
    method = "gem"  # or "crout"
    verbose = 1  # Set to 0 for no verbosity, 1 for basic, 2 for detailed

    solution = lu_solver(file_name, method, verbose)
    if solution is not None:
        print("Solution:", solution)
