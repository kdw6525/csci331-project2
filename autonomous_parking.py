# autonomous_parking.py
#
# This program will perform a genetic learning algorithm using reinforcement learning.
#

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
Y_SPOT = -1
X_AFTER_SPOT = 4
Y_AFTER_SPOT = 3

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
    gamma = convert_range(individual[0::2], GAMMA_LB, GAMMA_UB)
    beta = convert_range(individual[1::2], BETA_LB, BETA_UB)

    # interpolate using cubic spline
    f_gamma = sp.interpolate.CubicSpline(t, gamma, bc_type='natural')
    f_beta = sp.interpolate.CubicSpline(t, beta, bc_type='natural')

    # 100 steps between 0 and 10 seconds, giving .1 time per step
    t_new = np.linspace(T_START, T_END, STEPS)
    gamma_new = f_gamma(t_new)
    beta_new = f_beta(t_new)

    return gamma_new, beta_new


def calculate_step(step, gamma, beta):
    # calculates the next step
    x, y, a, v = step
    x_new = 0
    return


def euler_method(population_steps):
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


def convert_to_range(val, lb, ub):
    # converts val to a value between lb and ub.
    r = ub - lb
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


def is_feasible(current):
    # Checks if the current step is in the feasible region.
    x = current[0]
    y = current[1]
    return ((x <= X_BEFORE_SPOT and y > Y_BEFORE_SPOT) or
            (X_BEFORE_SPOT < x < X_AFTER_SPOT and y > Y_SPOT) or
            (x >= X_AFTER_SPOT and y > Y_AFTER_SPOT))


def main(args):
    random_generator = np.random
    if len(args) > 1 and args[1] == '-s':
        random_generator.seed(RANDOM_SEED)

    population, labels = create_population(random_generator)

    # 3D array, POPULATION_SIZE, 2, 100
    # Takes each individual and calculates their beta and gamma values
    population_steps = np.apply_along_axis(interpolate_individual, 1, population)

    print(population_steps.shape)

    # probs = scipy.special.softmax(labels)
    # parents = choose_parents(population, labels, probs, random_generator)

    return


if __name__ == "__main__":
    main(sys.argv)
