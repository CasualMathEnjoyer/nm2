import numpy as np
from time import time

class ThomasAlgorithm:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.n = len(b)
        self.rho = np.zeros(self.n)
        self.mu = np.zeros(self.n)

    def solve(self, x, verbose=0):
        print(self.A.shape)
        # First phase: Eliminate elements under diagonal
        self.mu[0] = self.A[0, 1] / self.A[0, 0]
        self.rho[0] = self.b[0] / self.A[0, 0]
        for k in range(1, self.n):
            self.rho[k] = (self.b[k] - self.A[k, k - 1] * self.rho[k - 1]) / (self.A[k, k] - self.A[k, k - 1] * self.mu[k - 1])
            if k < self.n - 1:
                self.mu[k] = self.A[k, k + 1] / (self.A[k, k] - self.A[k, k - 1] * self.mu[k - 1])
        if verbose:
            # self.print()  # Too lazy to implement
            pass

        # Second phase: Backward substitution
        x[self.n - 1] = self.rho[self.n - 1]
        for k in range(self.n - 2, -1, -1):
            x[k] = self.rho[k] - self.mu[k] * x[k + 1]
        return True

if __name__ == "__main__":
    n = 100
    dense_matrix = np.zeros((n, n))
    tridiagonal_matrix = np.zeros((n, n))
    for k in range(n):
        dense_matrix[k, k] = 2.5
        tridiagonal_matrix[k, k] = 2.5
        if k > 1:
            dense_matrix[k, k - 1] = -1
            tridiagonal_matrix[k, k - 1] = -1
        if k < n - 1:
            dense_matrix[k, k + 1] = -1
            tridiagonal_matrix[k, k + 1] = -1

    x = np.ones(n)
    b = np.dot(tridiagonal_matrix, x)
    x.fill(0)

    print("Multiplying matrix-vector...")
    gem = np.linalg.inv(dense_matrix)
    b = np.dot(gem, b)

    print("Solving the system using GEM...")
    start = time()
    x_gem = np.linalg.solve(dense_matrix, b)
    end = time()
    print("Computation took", end - start, "seconds.")

    thomas = ThomasAlgorithm(tridiagonal_matrix, b)
    print("Solving the system using the Thomas Algorithm...")
    start = time()
    thomas.solve(x)
    end = time()
    print("Computation took", end - start, "seconds.")
