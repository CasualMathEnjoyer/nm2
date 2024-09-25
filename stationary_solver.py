# stationary_solver.py

import numpy as np
class StationarySolver:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        assert A.shape[0] == A.shape[1] == b.shape[0], "Matrix A and vector b must have compatible dimensions."
        self.A = A
        self.b = b
        self.max_iterations = 10
        self.convergence_residue = 1.0e-8
        self.iteration_results = []  # To store solutions at each iteration
        self.relaxation = 1.0
        self.max_reached_iterations = float('inf')
        self.residue = float('inf')

    def set_max_iterations(self, max_iterations: int):
        self.max_iterations = max_iterations

    def set_convergence_residue(self, convergence_residue: float):
        self.convergence_residue = convergence_residue

    def set_relaxation(self, relaxation: float):
        self.relaxation = relaxation

    def set_max_reached_iterations(self, max_reached_iterations: int):
        self.max_reached_iterations = max_reached_iterations

    def solve(self, method: str = "sor", verbose: bool = False):
        n = self.b.shape[0]
        x = np.zeros(n)
        iteration = 0
        if method == "jacobi":
            D = np.diag(self.A)
            D_inv = 1.0 / D
            R = self.A - np.diagflat(D)
        while iteration < self.max_iterations:
            if method == "richardson":
                for i in range(n):
                    x[i] = x[i] + self.relaxation * (self.b[i] - np.dot(self.A[i, :], x)) / self.A[i, i]
            elif method == "jacobi":
                x_old = x.copy()
                x = x_old + self.relaxation * D_inv * (self.b - np.dot(R, x_old))
            elif method == "gauss-seidel":
                for i in range(n):
                    sum_ = self.b[i] - np.dot(self.A[i, :i], x[:i]) - np.dot(self.A[i, i + 1:], x[i + 1:])
                    x[i] = (1 - self.relaxation) * x[i] + self.relaxation * (sum_ / self.A[i, i])
            elif method == "sor":
                for row in range(self.A.shape[0]):
                    a_ii = self.A[row, row]
                    # edit for sparse matrices
                    if type(self.A[row]) == dict:
                        A_row = self.A[row]
                        sigma = 0
                        for i in A_row:
                            if x[i] == 0:
                                pass
                            else:
                                sigma += A_row[i] * x[i]
                    elif type(self.A[row]) == np.ndarray:
                        sigma = np.dot(self.A[row], x)
                    else:
                        raise ValueError(f"Unexpected type:{type(self.A[row])}")
                    if a_ii == 0.0:
                        raise ValueError(f"a_ii = 0 for i = {row}, unable to continue.")
                    x[row] += self.relaxation * (self.b[row] - sigma) / a_ii

            self.iteration_results.append(np.copy(x))

            if iteration % 10 == 0:
                residue = np.linalg.norm(self.A @ x - self.b)
                if verbose:
                    print(f"ITER. {iteration}, L2-RES. {residue}")
                if residue <= self.convergence_residue:
                    self.residue = residue
                    return x

            iteration += 1
        self.set_max_reached_iterations(iteration)

        return self.iteration_results