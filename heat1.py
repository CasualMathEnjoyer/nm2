import numpy as np
import math
import os
import numpy as np
from stationary_solver import StationarySolver
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class HeatEquationProblem1D:
    def __init__(self, size):
        self.size = size
        self.h = 1.0 / (size - 1)

    def get_degrees_of_freedom(self):
        return self.size

    def set_parameters(self):
        pass

    def set_initial_condition(self, u):
        for i in range(self.size):
            x = i * self.h
            u[i] = 1.0 if 0.4 < x < 0.6 else 0.0

    def get_right_hand_side(self, t, u, fu):
        # Zero Dirichlet
        u[0] = 0.0
        u[self.size - 1] = 0.0
        fu[0] = 0.0
        fu[self.size - 1] = 0.0

        h_sqr = self.h * self.h
        for i in range(1, self.size - 1):
            fu[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / h_sqr

    def write_solution(self, t, step, u):
        filename = f"heat-equation-{step:05d}.txt"
        with open(filename, 'w') as file:
            for i in range(self.size):
                file.write(f"{i * self.h} {u[i]}\n")
        return True

    @staticmethod
    def plot_solution(steps, size):
        x = np.linspace(0, 1, size)

        for step in steps:
            filename = f"heat-equation-{step:05d}.txt"
            u = np.zeros(size)

            with open(filename, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    u[i] = float(line.split()[1])

            plt.plot(x, u, label=f'Step {step}')

        plt.xlabel('Position')
        plt.ylabel('Temperature')
        plt.title('Heat Equation 1D Solution')
        plt.legend()
        plt.grid(True)
        plt.show()



initial_time = 0.0
final_time = 0.1
time_step = 0.01
integration_time_step = 0.01
size = 100


if __name__ == "__main__":
    problem = HeatEquationProblem1D(size)

    u = np.zeros(size)
    problem.set_initial_condition(u)
    problem.write_solution(0.0, 0, u)

    A = np.zeros((size, size))
    b = np.zeros(size)

    solver = StationarySolver(A, b)
    solver.set_max_iterations(100000)

    # Set Dirichlet boundary conditions
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[size - 1, size - 2] = 0.0
    A[size - 1, size - 1] = 1.0

    time = initial_time
    last_tau = -1.0
    h = 1.0 / size
    h_sqr = h * h
    step = 0
    steps_to_plot = [0]

    while time < final_time:
        stop_time = min(time + time_step, final_time)
        print(f"Time = {time} step = {step}")
        while time < stop_time:
            current_tau = min(integration_time_step, stop_time - time)
            if current_tau != last_tau:
                # Set-up lin sys
                lambda_ = current_tau / h_sqr
                for i in range(1, size - 1):
                    A[i, i - 1] = -lambda_
                    A[i, i] = 1.0 + 2.0 * lambda_
                    A[i, i + 1] = -lambda_

            # right-hand side b
            for i in range(size):
                b[i] = u[i]

            # solve with sor
            solver.solve(method="sor")
            u = solver.iteration_results[-1]

            time += current_tau
            last_tau = current_tau

        step += 1
        steps_to_plot.append(step)
        problem.write_solution(time, step, u)

    problem.plot_solution(steps_to_plot, size)
