import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import integrators
from odr import solve_loop

class LorenzProblem():
    def __init__(self):
        self.sigma = 1.0
        self.rho = 1.0
        self.beta = 1.0
        self.filename = "lorenz.txt"
    def setParameters(self, _sigma, _rho, _beta):
        self.sigma = _sigma
        self.rho = _rho
        self.beta = _beta
    def get_degrees_of_freedom(self):
        return 3
    def function_f(self, t, _u):
        x, y, z = _u
        fu = np.zeros(3)
        fu[0] = self.sigma * (y - x)
        fu[1] = x * (self.rho - z) - y
        fu[2] =  x * y -self.beta * z
        return fu
    def write_solution(self, t, step, u):
        mode = "w" if step == 0 else "a"
        with open(self.filename, mode) as file:
            file.write(f"{u[0]} {u[1]} {u[2]}\n")
        return True
    def plot_solution(self):
        data = np.loadtxt(self.filename)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[0, 0], data[0, 1], data[0, 2], color='r', label='Starting Point')
        ax.plot(data[:, 0], data[:, 1], data[:, 2])
        ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2], color='g', label='End Point')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.title('Lorenz Attractor')
        plt.savefig("lorenz4.pdf", format="pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    initialTime = 0.0
    finalTime = 100.0
    timeStep = 1.0e-2
    integrationTimeStep = 1.0e-4

    problem = LorenzProblem()
    problem.setParameters(10.0, 28.0, 8.0 / 3.0)

    integrator = integrators.Merson()

    u = np.array([1.0, 1.0, 1.0])

    try:
        solve_loop(initialTime, finalTime, timeStep, integrationTimeStep, problem, integrator, u)
    except Exception as e:
        print(e)
        print("EXIT")

    problem.plot_solution()