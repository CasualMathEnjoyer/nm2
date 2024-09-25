# find_optimal_relax.py

import numpy as np
from tqdm import tqdm

class Optimiser:
    def __init__(self, A, b, solver, method):
        self.A = A
        self.b = b
        self.solver = solver
        self.method = method

    def grid_search(self, relaxation_values):
        min_iterations = float('inf')
        min_residual = float('inf')
        optimal_relaxation = None
        for relaxation in tqdm(relaxation_values):  # tqdm added here
            self.solver.set_relaxation(relaxation)
            x = self.solver.solve(method=self.method)
            iterations = self.solver.max_reached_iterations
            residual = self.solver.residue
            print(iterations)
            if iterations < min_iterations:
                min_iterations = iterations
                optimal_relaxation = relaxation
                min_residual = residual
            elif iterations == min_iterations:
                if residual < min_residual:
                    optimal_relaxation = relaxation
                    min_residual = residual
        return optimal_relaxation

if __name__ == "__main__":
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
    b = np.array([5, -10, 5])

    solver = Optimiser(A, b)
    relaxation_values = np.linspace(0, 2, 1000)
    optimal_relaxation = solver.grid_search(relaxation_values)
    print("Optimal relaxation parameter:", optimal_relaxation)
