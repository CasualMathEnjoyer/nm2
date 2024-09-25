import numpy as np
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, a):
        self.a = a
    def function_f(self, time, x):
        y, v = x
        dydx = v
        dvdx = self.a * np.cos(time) ** 2 - self.a * y
        return np.array([dydx, dvdx])

class RK_second_order:
    def __init__(self):
        self.k1 = None
        self.k2 = None

    def setup(self, degrees_of_freedom):
        self.k1 = np.zeros(degrees_of_freedom)
        self.k2 = np.zeros(degrees_of_freedom)

    def solve(self, integration_time_step, stop_time, time, problem, x):
        self.setup(len(x))
        iteration = 0
        x_values = [x.copy()]
        t_values = [time]
        while time < stop_time:
            tau = min(integration_time_step, stop_time - time)

            self.k1 = tau * problem.function_f(time, x)
            self.k2 = tau * problem.function_f(time + tau, x + self.k1)
            x += 0.5 * (self.k1 + self.k2)

            time += tau
            iteration += 1
            x_values.append(x.copy())
            t_values.append(time)
        return np.array(t_values), np.array(x_values)

def shooting_method(guess, v_bool, plot_trials=False):
    """
    :param guess: the gues for the parameter we are trying to find
    :param v_bool: whether the given boundary condition at x1 is y(x1) or y'(x1)=v(x1)
    :param plot_trials: bool to include the all shooting attempts to the final graph
    :return: the value of boundary condition at x1 after applying the guess
    """

    initial_conditions = np.array([y0, guess])

    # Solve the differential equation using the Runge-Kutta method
    rk_solver = RK_second_order()
    stop_time = x1  # solving as an initial value problem
    t_values, x_values = rk_solver.solve(0.01, stop_time, x0, problem, initial_conditions)

    if plot_trials:
        plt.plot(t_values, x_values[:, 0]) #, label=f'v0_guess = {v0_guess:.1f}')

    if not v_bool:
        return x_values[-1][0]  #y(x1)
    else:
        return x_values[-1][1]  #v(x1)

def find_correct_v0(v_bool=True):
    """
    for a specified range of parameters v0_low, v0_high we perform the shooting method and then proceed with the method
    of taking a half of the appropriate interval
    :param v_bool: whether the boundary condition at x1 is y(x1) or y'(x1)=v(x1)'
    :return: the approximate value of v0
    """
    v0_low, v0_high = -100, 100  # Initial guess range for v0
    tol = 1e-6
    while v0_high - v0_low > tol:
        v0_guess = (v0_low + v0_high) / 2
        shot = shooting_method(v0_guess, v_bool, plot_trials=False)
        difference = shot - beta
        # if not v_bool:  # the final point when shooting for position, optional
        #     plt.scatter(x1, shot, color="blue")
        if difference > 0:
            v0_high = v0_guess
        else:
            v0_low = v0_guess
    return (v0_low + v0_high) / 2

def save_solution_to_file(t_values, x_values, filename):
    data = np.column_stack((t_values, x_values[:, 0]))  # Only save time and y values
    np.savetxt(filename, data, header='Time y', comments='')

def print_solution_gnuplot(t_values, x_values):
    data = np.column_stack((t_values, x_values[:, 0]))
    for line in data:
        print(line[0], line[1])


if __name__ == '__main__':
    # Boundary conditions
    x0, x1 = 0, 1

    # a = 2
    # y0 = 1          # y(x0) = y0 = alfa
    # beta = 3        # y'(x1) = v(x1) or y(x1)

    print("Solving equation: ddy/dxdx = a * cos(x) ** 2 - a * y")
    print("with boundary conditions: y(x0) = alfa, y'(x1) = beta")
    print("for x0 = 0, x1 = 1")
    print("Please select values")
    a = float(input("a:"))
    y0 = float(input("alfa:"))
    beta = float(input("beta:"))
    print("Thank you.")

    if a <= 0:
        raise ValueError("a must be positive")

    problem = Problem(a=a)

    find_V = True   # false means we are finding y(x1) = beta, true is v(x1) = beta
    plt.scatter(x0, y0, color="orange")  # plotting the start point

    # Find the correct initial condition v0
    v0_correct = find_correct_v0(v_bool=find_V)

    # Solve with the initial v0
    initial_conditions_correct = np.array([y0, v0_correct])
    rk_solver = RK_second_order()
    stop_time = x1
    t_values, x_values = rk_solver.solve(0.01, stop_time, x0, problem, initial_conditions_correct)

    y1 = x_values[-1][0]
    v1 = x_values[-1][1]
    if find_V:
        print(f'beta = {beta}, v1 = {v1:.3f}, difference: {round(abs(v1 - beta), 5)}')
    else:
        print(f'beta = {beta}, y1 = {y1:.3f}')

    # filename = f'solution.txt'
    # save_solution_to_file(t_values, x_values, filename)

    print()
    print("Solution:")
    print_solution_gnuplot(t_values, x_values)

    # ---------------------- PLOTS USING MATPLOTLIB -----------------------
    # exit(0)  # direct plotting disabled

    import matplotlib
    matplotlib.use('TkAgg')

    # Plot the line with slope v0 (guessed slope)
    x_points = np.linspace(x0, x1, 100)
    y_points = v0_correct * (x_points - x0) + y0
    plt.plot(x_points, y_points, color='red', linestyle='--', label=f'v0={v0_correct:.1f}')

    # Plot the blue line with slope v1
    x_points = np.linspace(x0, x1 + 1, 100)
    y_points = v1 * (x_points - x1) + y1
    plt.plot(x_points, y_points, color='blue', linestyle='-', label=f'v1={v1:.1f}')

    if find_V:
        # Plot blue line
        x_points = np.linspace(x0, x1 + 1, 100)
        y_points = beta * (x_points - x1) + y1
        plt.plot(x_points, y_points, color='green', linestyle='--', label=f'beta=v1')
    else:
        plt.axhline(y=beta, color='green', linestyle='--', label=f'beta=y1')

    # Plot final
    plt.plot(t_values, x_values[:, 0], label=f'Final Solution, beta = {beta}', linewidth=2, color='black')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.grid(True)
    plt.title(f'dv = a*cos(x)^2-a*y, x0={x0}, x1={x1}, y0={y0}, beta={beta}')
    plt.legend()
    plt.show()
