import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import integrators
from odr import *

class SIRModel:
    def __init__(self, N, n, f, b, c, m, mI):
        self.N = N   # total population
        self.n = n   # rate of new susceptible individuals (birth rate)
        self.f = f   # fraction of individuals becoming immune without infection
        self.b = b   # transmission rate
        self.c = c   # recovery rate
        self.m = m   # mortality rate
        self.mI = mI # mortality rate for infected people
        self.filename = "sir_solution.txt"

    def get_degrees_of_freedom(self):
        return 3

    def function_f(self, t, u, params=None):
        S, I, R = u
        dS_dt = self.n * (1 - self.f) * self.N - (self.b * I * S) / self.N - self.m * S
        dI_dt = (self.b * I * S) / self.N - self.c * I - self.mI * I
        dR_dt = self.c * I + self.n * self.f * self.N - self.m * R
        return np.array([dS_dt, dI_dt, dR_dt])

    def write_solution(self, t, step, u):
        mode = "w" if step == 0 else "a"
        with open(self.filename, mode) as file:
            file.write(f"{t} {u[0]} {u[1]} {u[2]}\n")
        return True

    def plot(self, name=None):
        t, S, I, R = [], [], [], []
        with open(self.filename, 'r') as file:
            for line in file:
                t_, S_, I_, R_ = map(float, line.split())
                t.append(t_)
                S.append(S_)
                I.append(I_)
                R.append(R_)
        I = np.array(I)
        S = np.array(S)
        R = np.array(R)
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, I, label='Infected')
        end_of_infection = np.where(I <= 0.0001)[0]
        if end_of_infection.size > 0:
            plt.scatter(t[end_of_infection[0]], 0, color='r', label='Infection End')
            print("infection ended")
        plt.plot(t, R, label='Recovered')
        plt.plot(t, S+I+R, label="Population")
        plt.xlabel('Time')
        plt.ylabel('Population')
        model_name = 'SIR Model'
        if name != None:
            model_name = 'SIR Model '+name
        plt.title(model_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(model_name+".pdf", format="pdf", bbox_inches="tight")
        plt.show()

# Parameters of the model
n = 0.1 # birth rate
f = 0.2   # fraction of individuals becoming immune without infection
b = 0.5   # transmission rate   # alfa = 0.208  # covid Iran
c = 0.3   # recovery rate  # beta = 0.085
m = 0.06  # mortality rate
mi = 0.065 # mortality rate for infected individuals
t_min = 0.0    # initial time
t_max = 200.0  # end time

time_step = 0.1  # time step
integration_time_step = 0.1
integrator = integrators.Euler()
# Initial conditions
S = 0.993
I = 0.007
R = 0
N = S + I + R  # population
# b = b * N
initial_conditions = (S, I, R)  # SIR

sir_model = SIRModel(N, n, f, b, c, m, mi)
if not solve_loop(t_min, t_max, time_step, integration_time_step, sir_model, integrator, initial_conditions):
    print("SOLUTION FAILED")
name = "n=" + str(n) + " f=" + str(f) + " b=" + str(b) + " c=" + str(c) + " m=" + str(m) + " mi=" + str(mi)
sir_model.plot(name)