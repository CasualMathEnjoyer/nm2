import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

import integrators
from odr import solve_loop

class HyperbolicProblem:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.filename = "hyperbolic.txt"
    def get_degrees_of_freedom(self):
        return 2
    def function_f(self, time, u, fu=None):  # harmonic oscillator
        u1, u2 = u
        return np.array([u2, -u1 - self.epsilon * u1 ** 2 * u2])
    def write_solution(self, t, step, u):
        mode = 'w' if step == 0 else 'a'
        with open(self.filename, mode) as file:
            file.write(f"{t} {u[0]} {u[1]}\n")
        return True
    def plot_solution(self):
        t_values, u1_values, u2_values = [], [], []
        with open(self.filename, 'r') as file:
            for line in file:
                t, u1, u2 = map(float, line.split())
                t_values.append(t)
                u1_values.append(u1)
                u2_values.append(u2)

        plt.plot(t_values, u1_values, label='position')
        # plt.plot(t_values, u2_values, label='speed')
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.title('Hyperbolic Problem Solution')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    problem = HyperbolicProblem()

    # integrator = integrators.Euler()
    integrator = integrators.RK_second_order()
    # integrator = integrators.Merson()

    problem.epsilon = 0.1
    initial_time = 0.0
    final_time = 100.0
    time_step = 0.1
    integration_time_step = 0.1
    initial_conditions = [0.0, 10.0]

    solve_loop(initial_time, final_time, time_step, integration_time_step, problem, integrator, initial_conditions)
    problem.plot_solution()

# ukol: je v prezentaci
# kralici - tak tri obrazky a trochu popsat

# ukol2 : numericka studie Lorenzovych rovnic