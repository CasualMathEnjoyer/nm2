import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')

import integrators

def plot_solution(file_path, text, second=None):
    time_values, time_values2, solution_values, second_values = [], [], [], []

    # Read data from file
    with open(file_path, 'r') as file:
        for line in file:
            time, solution = map(float, line.split())
            time_values.append(time)
            solution_values.append(solution)
    if second is not None:
        with open(second, 'r') as file:
            for line in file:
                time, solution = map(float, line.split())
                time_values2.append(time)
                second_values.append(solution)

    # Plot the data in a separate window
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, solution_values, label=f'Exact Solution', color='r')
    if second is not None:
        plt.plot(time_values2, second_values, label=f'Numerical Solution', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.title(f'{text} Solution of Riccati problem')
    plt.legend()
    plt.grid(True)
    # plt.savefig("riccati_exact.pdf", format="pdf", bbox_inches="tight")
    plt.show()

class Ricatti():
    def __init__(self):
        self.l1Error = 0.0
        self.l2Error = 0.0
        self.maxError = 0.0
        self.filename = "ricatti.txt"
    def get_L1_error(self, time_step):
        return time_step * self.l1Error
    def get_L2_error(self, time_step):
        return math.sqrt(time_step * self.l2Error)
    def get_max_error(self):
        return self.maxError
    def get_degrees_of_freedom(self):
        return 1
    def function_f(self, t, x_, parameter=[0]):
        x = x_[0]
        parameter[0] = t ** (-4.0) * math.exp(t) + x + 2.0 * math.exp(-t) * x ** 2
        return np.array(parameter)
    def get_exact_solution(self, t, c=1):
        sqrt_2 = math.sqrt(2.0)
        return math.exp(t) * (1.0 / (sqrt_2 * t * t) * math.tan(sqrt_2 * (c - 1.0 / t)) - 1.0 / (2.0 * t))
    def write_exact_solution(self, file_name, initial_time, final_time, time_step, c=1.0):
        with open(file_name, 'w') as file:
            t = initial_time
            while t < final_time:  # gnuplot - lze predelat pro matplotlib
                file.write(f"{t} {self.get_exact_solution(t, c)}\n")
                t = min(t + time_step, final_time)
        return True

    def write_solution(self, t, step, _x):
        x = _x[0]
        file_mode = 'w' if step == 0 else 'a'
        problem.update_errors(t, x)
        with open(self.filename, file_mode) as file:
            file.write(f"{t} {x}\n")
        return True

    def update_errors(self, t, u):
        diff = abs(self.get_exact_solution(t) - u)
        self.l1Error += diff
        self.l2Error += diff * diff
        self.maxError = max(self.maxError, diff)

def solve_loop(initial_time, final_time, time_step, integration_time_step, problem, solver, x):
    solver.setup(problem.get_degrees_of_freedom())
    time_steps_count = math.ceil(max(0.0, final_time - initial_time) / time_step)
    time = initial_time
    problem.write_solution(time, 0, x)

    for step in tqdm(range(1, time_steps_count + 1), desc="Getting numerical solution"):
        current_stop_time = time + min(time_step, final_time - time)
        time, x, success = solver.solve(integration_time_step, current_stop_time, time, problem, x)
        if not success:
            return False
        problem.write_solution(time, step, x)
    print("Done.")
    return True

# Function to save data to CSV
def save_to_csv(data):
    with open('integration_errors.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Integration Timestep', 'L1 Error', 'L2 Error', 'Max Error'])
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":  # 0.05 - 0.15, e-4
    data_to_save = []
    initial_time = 0.5
    final_time = 10
    time_step = 1  # keep fixed
    integration_time_step = 1.0e-2

    problem = Ricatti()

    integrator = integrators.Euler()
    # integrator = integrators.RK_second_order()
    # integrator = integrators.Merson()

    start_x = problem.get_exact_solution(initial_time)
    print(start_x)
    problem.write_exact_solution("exact-riccati.txt", initial_time, final_time, time_step)

    if not solve_loop(initial_time, final_time, time_step, integration_time_step, problem, integrator, [start_x]):
        print("Error: Solution failed.")
        exit("EXIT_FAILURE")

    l1_error = problem.get_L1_error(time_step)
    l2_error = problem.get_L2_error(time_step)
    max_error = problem.get_max_error()

    print("L1 error:", l1_error)
    print("L2 error:", l2_error)
    print("Max error:", max_error)

    # plot_solution("exact-riccati.txt", "Exact vs Numerical")
    plot_solution("exact-riccati.txt", "Exact vs Numerical", "ricatti.txt",)