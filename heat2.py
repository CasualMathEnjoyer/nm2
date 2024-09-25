from stationary_solver import StationarySolver
from integrators import Merson
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sparseMatrix import SparseMatrix

matplotlib.use('TkAgg')

class HeatEquationProblem2D:
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.hx = 1.0 / (sizeX - 1)
        self.hy = 1.0 / (sizeY - 1)

    def get_degrees_of_freedom(self):
        return self.sizeX * self.sizeY

    def set_initial_condition(self, u, kind="abs", r=0.3):
        if kind not in {"abs", "circle"}:
            raise ValueError("Invalid initial condition type")
        cx = np.linspace(-self.hx * self.sizeX / 2, self.hx * self.sizeX / 2, self.sizeX)
        cy = np.linspace(-self.hy * self.sizeY / 2, self.hy * self.sizeY / 2, self.sizeY)
        for y in range(self.sizeY):
            for x in range(self.sizeX):
                if kind == "abs":
                    u[y, x] = 1 if abs(cx[x]) + abs(cy[y]) - r < 0 else 0
                elif kind == "circle":
                    u[y, x] = 1 if cx[x] ** 2 + cy[y] ** 2 - r < 0 else 0

    def set_initial_condition_from_pgm(self, pgm_file):
        u = np.loadtxt(pgm_file, skiprows=4)  # Skip the first 4 lines of the PGM file
        u = u / np.max(u)  # Normalize the image data
        u = np.flipud(u)   # Flip the image vertically to match the orientation of the grid
        return u

    def function_f(self, time, u, k=None):
        u = u.reshape((self.sizeY, self.sizeX))
        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1] = (
                (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / self.hx ** 2 +
                (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / self.hy ** 2
        )
        return laplacian.flatten()

    def write_solution(self, t, step, u):
        filename = f"heat-equation-2d-{step:05d}.txt"
        np.savetxt(filename, u)
        return True

    @staticmethod
    def plot_solution(steps, sizeX, sizeY):
        for step in steps:
            filename = f"heat-equation-2d-{step:05d}.txt"
            u = np.loadtxt(filename).reshape((sizeY, sizeX))
            plt.imshow(u, extent=[0, 1, 0, 1], origin='lower')
            plt.savefig(f"butterfly-{step:05d}.pdf", bbox_inches='tight', format='pdf')
            plt.show()

initial_time = 0.0
final_time = 0.0001
time_step = 0.00001
# final_time = 0.1
# time_step = 0.01
integration_time_step = 0.01
# sizeX = 40
# sizeY = 40
sizeX = 434
sizeY = 606

if __name__ == "__main__":
    problem = HeatEquationProblem2D(sizeX, sizeY)

    # u = np.zeros((sizeY, sizeX))
    # problem.set_initial_condition(u, kind="abs", r=0.3)
    pgm_file = "motyl.txt"
    u = problem.set_initial_condition_from_pgm(pgm_file)

    problem.write_solution(0.0, 0, u)

    stationary = False

    # A = np.zeros((sizeX * sizeY, sizeX * sizeY))
    A = SparseMatrix(sizeX * sizeY, sizeX * sizeY)
    b = np.zeros(sizeX * sizeY)

    # Set boundary conditions
    for i in range(sizeX):
        A[i, i] = 1.0  # Top boundary
        A[sizeX * (sizeY - 1) + i, sizeX * (sizeY - 1) + i] = 1.0  # Bottom boundary
    for j in range(sizeY):
        A[j * sizeX, j * sizeX] = 1.0  # Left boundary
        A[j * sizeX + sizeX - 1, j * sizeX + sizeX - 1] = 1.0  # Right boundary

    import time as t
    start = t.time()
    if stationary:
        solver = StationarySolver(A, b)
        solver.set_max_iterations(10000)
        solver.relaxation = 1.6
    else:
        solver = Merson()
        solver.setup(problem.get_degrees_of_freedom())

    time = initial_time
    last_tau = -1.0
    hx_sqr = (1.0 / (sizeX - 1)) ** 2
    hy_sqr = (1.0 / (sizeY - 1)) ** 2
    step = 0
    steps_to_plot = [0]

    while time < final_time:
        stop_time = min(time + time_step, final_time)
        print(f"Time = {time} step = {step}")
        if stationary:
            while time < stop_time:
                current_tau = min(integration_time_step, stop_time - time)
                if current_tau != last_tau:
                    # Set-up lin sys
                    lambda_x = current_tau / hx_sqr
                    lambda_y = current_tau / hy_sqr
                    for j in range(1, sizeY - 1):
                        for i in range(1, sizeX - 1):
                            index = j * sizeX + i
                            A[index, index - sizeX] = -lambda_y
                            A[index, index - 1] = -lambda_x
                            A[index, index] = 1.0 + 2.0 * lambda_x + 2.0 * lambda_y
                            A[index, index + 1] = -lambda_x
                            A[index, index + sizeX] = -lambda_y
                # right-hand side b
                b[:] = u.flatten()
                # Solve the lin system using SOR
                print("sor")
                success = solver.solve(method="sor")
                if success is False:
                    exit("Solver failed!")
                u = solver.iteration_results[-1].reshape((sizeY, sizeX))
                time += current_tau
                last_tau = current_tau
        else:
            # merson
            time, u_flat, success = solver.solve(integration_time_step, stop_time, time, problem, u.flatten())
            u = u_flat.reshape((sizeY, sizeX))
        step += 1
        steps_to_plot.append(step)
        problem.write_solution(time, step, u)

    stop = t.time()
    print(f"The time: {stop - start:.2f} seconds")
    
    problem.plot_solution(steps_to_plot, sizeX, sizeY)
