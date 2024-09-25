import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import integrators
from odr import solve_loop


class SpeciesProblem():
    def __init__(self):
        self.a, self.b, self.c, self.d, self.e, self.f = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        self.filename = "species.txt"
    def get_degrees_of_freedom(self):
        return 2
    def setParameters(self, _a, _b, _c, _d, _e, _f):
        self.a, self.b, self.c, self.d, self.e, self.f = _a, _b, _c, _d, _e, _f
    def function_f(self, t, _u, params=None):
        u1, u2 = _u
        fu = np.zeros(2)
        fu[0] = self.a * u1 - self.c * u1 * u2 - self.e * u1 * u1
        fu[1] = self.b * u2 - self.d * u1 * u2 - self.f * u2 * u2
        return fu
    def write_solution(self, t, step, u):
        mode = "w" if step == 0 else "a"
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

        plt.plot(t_values, u1_values, label='species1')
        plt.plot(t_values, u2_values, label='species2')
        plt.grid(True)
        plt.savefig("species1.pdf", format="pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    initialTime = 0.0
    finalTime = 200.0
    timeStep = 0.1
    integrationTimeStep = 1

    problem = SpeciesProblem()
    problem.setParameters(1, -3.0, 0.25, -0.25, 0.0, 0.0)
    integrator = integrators.Merson()

    u = np.array([5, 0.5])
    if not solve_loop(initialTime, finalTime, timeStep, integrationTimeStep, problem, integrator, u):
        print("SOLUTION FAILED")
    problem.plot_solution()
    print("Solved")