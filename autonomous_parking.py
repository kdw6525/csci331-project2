# autonomous_parking.py
#
# This program will perform a genetic learning algorithm using reinforcement learning.
#
import math

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

import scipy.special

# GLOBAL PARAMETERS
# Used for testing to keep results consistent
RANDOM_SEED = 0
MAX_POPULATION_SIZE = 500
MAX_GENERATIONS = 1200

# HYPERPARAMETERS
# These are hyperparameters set by recommendation and project constraints
POPULATION_SIZE = 201
GENE_SIZE = 7
MAX_GENE_VAL = np.power(2, GENE_SIZE) - 1
MUTATION_RATE = 0.005
INFEASIBILITY = 200
GAMMA_UB = 0.524
GAMMA_LB = -0.524
BETA_UB = 5
BETA_LB = -5
T_START = 0
T_END = 10
STEPS = 100
STEP_DELTA = 0.1

# FEASIBLE REGION
# Regions the agents cannot touch or pass into
X_BEFORE_SPOT = -4
Y_BEFORE_SPOT = 3
Y_BEFORE_SPOT_SQ = Y_BEFORE_SPOT ** 2
Y_SPOT = -1
Y_SPOT_SQ = -1 ** 2
X_AFTER_SPOT = 4
Y_AFTER_SPOT = 3
Y_AFTER_SPOT_SQ = 3 ** 2

# START and TARGET defined by the project
# [x, y, angle, velocity]
START = np.array([0, 8, 0, 0])
TARGET = np.array([0, 0, 0, 0])


def create_population(random_generator):
    # generates a starting population
    return random_generator.randint(0, MAX_GENE_VAL, [POPULATION_SIZE, 20]), np.arange(0, POPULATION_SIZE)


def interpolate_individual(individual):
    # given an individual with 1 set of data points per second for 10 seconds (0 - 9)
    # interpolate
    convert_range = np.vectorize(convert_to_range)

    t = range(0, 10)
    gamma = convert_range(individual[0::2], GAMMA_LB, GAMMA_UB - GAMMA_LB)
    beta = convert_range(individual[1::2], BETA_LB, BETA_UB - BETA_LB)

    # interpolate using cubic spline
    f_gamma = sp.interpolate.CubicSpline(t, gamma, bc_type='natural')
    f_beta = sp.interpolate.CubicSpline(t, beta, bc_type='natural')

    # 100 steps between 0 and 10 seconds, giving .1 time per step
    t_new = np.linspace(T_START, T_END, STEPS)
    gamma_new = f_gamma(t_new)
    beta_new = f_beta(t_new)

    return gamma_new, beta_new, t_new


def calculate_step(step, gamma, beta):
    # calculates the next step
    x, y, a, v = step
    x_new = x + (v * math.cos(a) * STEP_DELTA)
    y_new = y + (v * math.sin(a) * STEP_DELTA)
    a_new = a + (gamma * STEP_DELTA)
    v_new = v + (beta * STEP_DELTA)
    return np.array([x_new, y_new, a_new, v_new])


def individual_fitness(current, end, gamma_beta):
    # calculates the fitness using euler's method
    gamma = gamma_beta[0]
    beta = gamma_beta[1]

    infeasibility = 0
    for g, b in zip(gamma, beta):
        current = calculate_step(current, g, b)
        # check if feasible spot and add infeasibility costs
        infeasibility += calc_infeasibility(current)

    dist = scipy.spatial.distance.euclidean(current, end)
    return dist + infeasibility


def individual_history(current, gamma_beta):
    # generates the history of an individual using euler's method
    gamma = gamma_beta[0]
    beta = gamma_beta[1]

    history = np.array([current])
    for g, b in zip(gamma, beta):
        current = calculate_step(current, g, b)
        # record current in history
        history = np.vstack((history, current))

    return history


def calculate_fitness(population_steps):
    # performs the euler method STEPS times
    start = np.copy(START)


def choose_parents(population, labels, probabilities, random_generator: np.random):
    # chooses 2 parents based on the fitness array
    # population:   POPULATION_SIZE x 20 array
    # labels:       1 x POPULATION_SIZE array
    # fitness:      1 x POPULATION_SIZE array
    # returns:      2 x 20 array
    parents = random_generator.choice(labels, 2, p=probabilities)
    return np.array([population[parents[0]], population[parents[1]]])


def evaluate_best_individual(individual):
    # perform analysis on the best fit individual
    # analysis ranges from exporting control variables to graphing various properties
    # TODO add exporting and additional graphs. Perhaps save the graphs as pdfs

    gamma_new, beta_new, t_new = interpolate_individual(individual)

    history = individual_history(np.copy(START), [gamma_new, beta_new])
    # fist column is x histories
    x_new = history[:, 0]
    # second column is y histories
    y_new = history[:, 1]

    # the path taken is related to x and y
    plt.figure(figsize=(10, 8))
    plt.plot(x_new, y_new, 'b')
    plt.title('Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def convert_to_range(val, lb, r):
    # converts val to a value between lb and ub.
    return (val / MAX_GENE_VAL) * r + lb


def gray_encode(n):
    # encodes n to gray value
    # Source: https://www.codespeedy.com/convert-binary-to-gray-code-in-python/
    return n ^ (n >> 1)


def gray_decode(n):
    # decodes n to binary integer
    # Source: https://stackoverflow.com/questions/72027920/python-graycode-saved-as-string-directly-to-decimal
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n


def calc_infeasibility(current):
    # Checks if the current step is in the feasible region.
    x = current[0]
    y = current[1]
    if x <= X_BEFORE_SPOT and y <= Y_BEFORE_SPOT:
        return INFEASIBILITY + (Y_BEFORE_SPOT_SQ - (y ** 2))
    elif X_BEFORE_SPOT < x < X_AFTER_SPOT and y <= Y_SPOT:
        return INFEASIBILITY + (Y_SPOT_SQ - (y ** 2))
    elif x >= X_AFTER_SPOT and y <= Y_AFTER_SPOT:
        return INFEASIBILITY + (Y_AFTER_SPOT_SQ - (y ** 2))
    else:
        return 0


def main(args):
    random_generator = np.random
    if len(args) > 1 and args[1] == '-s':
        random_generator.seed(RANDOM_SEED)

    population, labels = create_population(random_generator)

    # 3D array, POPULATION_SIZE, 2, 100
    # Takes each individual and calculates their beta and gamma values
    population_steps = np.apply_along_axis(interpolate_individual, 1, population)

    print(population_steps.shape)
    evaluate_best_individual(population[0])

    # probs = scipy.special.softmax(labels)
    # parents = choose_parents(population, labels, probs, random_generator)

    return


if __name__ == "__main__":
    main(sys.argv)
