import math
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class RK_second_order:
    def __init__(self):
        self.dof = 0
        self.k1 = None
        self.k2 = None
    def setup(self, degrees_of_freedom):
        self.k1 = np.zeros(degrees_of_freedom)
        self.k2 = np.zeros(degrees_of_freedom)
    def solve(self, integration_time_step, stop_time, time, problem, x):
        iteration = 0
        while time < stop_time:
            tau = min(integration_time_step, stop_time - time)

            self.k1 = tau * problem.function_f(time, x, self.k1)
            self.k2 = tau * problem.function_f(time + tau, x + self.k1, self.k2)
            x += 0.5 * (self.k1 + self.k2)

            time += tau
            iteration += 1
        return time, x, True
class Euler:
    def __init__(self):
        self.k = None
    def setup(self, degrees_of_freedom):
        self.k = np.zeros(degrees_of_freedom)
    def solve(self, integration_time_step, stop_time, time, problem, x):
        iteration = 0
        while time < stop_time:
            tau = min(integration_time_step, stop_time - time)

            self.k = problem.function_f(time, x, self.k)
            x += tau * self.k

            time += tau
            iteration += 1
        return time, x, True

class Merson:
    def __init__(self):
        self.adaptivity = 1.0e-6
        self.k1, self.k2, self.k3, self.k4, self.k5, self.aux = None, None, None, None, None, None
        self.max_iterations = 100000

    def setup(self, degrees_of_freedom):
        self.k1 = np.zeros(degrees_of_freedom)
        self.k2 = np.zeros(degrees_of_freedom)
        self.k3 = np.zeros(degrees_of_freedom)
        self.k4 = np.zeros(degrees_of_freedom)
        self.k5 = np.zeros(degrees_of_freedom)
        self.aux = np.zeros(degrees_of_freedom)
        return True

    def solve(self, integration_time_step, stop_time, time, problem, x):
        dofs = len(x)
        tau = min(integration_time_step, stop_time - time)
        iteration = 0
        while time < stop_time:
            print(time)
            # Compute k1
            self.k1 = problem.function_f(time, x)
            # Compute k2
            self.aux[:] = x + tau * (1.0 / 3.0) * self.k1
            self.k2 = problem.function_f(time + 1.0 / 3.0 * tau, self.aux)
            # Compute k3
            self.aux[:] = x + tau * 1.0 / 6.0 * (self.k1 + self.k2)
            self.k3 = problem.function_f(time + 1.0 / 3.0 * tau, self.aux)
            # Compute k4
            self.aux[:] = x + tau * (0.125 * self.k1 + 0.375 * self.k3)
            self.k4 = problem.function_f(time + 1.0 / 2.0 * tau, self.aux)
            # Compute k5
            self.aux[:] = x + tau * (0.5 * self.k1 - 1.5 * self.k3 + 2.0 * self.k4)
            self.k5 = problem.function_f(time + tau, self.aux)

            # Compute error
            eps = 0.0
            for i in range(dofs):
                err = (tau / 3.0 * abs(0.2 * self.k1[i] + -0.9 * self.k3[i] + 0.8 * self.k4[i] + -0.1 * self.k5[i]))
                eps = max(eps, err)

            if self.adaptivity == 0.0 or eps < self.adaptivity:
                x += tau / 6.0 * (self.k1 + 4.0 * self.k4 + self.k5)
                time += tau
                iteration += 1
                if iteration > self.max_iterations:
                    print("The solver has reached the maximum number of iterations.", file=sys.stderr)
                    return None, None, False
                
            if self.adaptivity:
                tau *= 0.8 * (self.adaptivity / eps) ** 0.2
            tau = min(tau, stop_time - time)
        return time, x, True
