# parking.py
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
MUTATION_RATE_DEFAULT = 0.005
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
J_TOLERANCE = 0.01

# FEASIBLE REGION
# Regions the agents cannot touch or pass into
X_BEFORE_SPOT = -4
Y_BEFORE_SPOT = 3
Y_BEFORE_SPOT_SQ = Y_BEFORE_SPOT ** 2
Y_SPOT = -1
Y_SPOT_SQ = Y_SPOT ** 2
X_AFTER_SPOT = 4
Y_AFTER_SPOT = 3
Y_AFTER_SPOT_SQ = Y_AFTER_SPOT ** 2

# START and TARGET defined by the project
# [x, y, angle, velocity]
START = np.array([0, 8, 0, 0])
TARGET = np.array([0, 0, 0, 0])


def create_population(random_generator):
    # generates a starting population
    return random_generator.randint(0, MAX_GENE_VAL, [POPULATION_SIZE, 20]), np.arange(0, POPULATION_SIZE)


def create_next_generation(population, elite, labels, probs, random_generator):
    # create a new population based on the previous
    # population:       POPULATION_SIZE x 20 array as the potential parents
    # probs:            POPULATION_SIZE x 1 array as the probability distribution
    # random_generator: a potentially seeded random generator
    # returns:          POPULATION_SIZE x 20 array of the next generation

    parents = choose_parents(labels, probs, random_generator)
    children = create_children(population, elite, parents, random_generator)

    return children


def choose_parents(labels, probabilities, random_generator: np.random):
    # chooses 2 parents based on the fitness array
    # population:   POPULATION_SIZE x 20 array
    # labels:       1 x POPULATION_SIZE array
    # fitness:      1 x POPULATION_SIZE array
    # returns:      2 x 20 array
    return random_generator.choice(labels, [POPULATION_SIZE // 2, 2], p=probabilities)


def create_children(population, elite, parents, random_generator):
    # creates the children of the next generation
    # population:   POPULATION_SIZE x 20 array containing the real values of the parents
    # parents:      POPULATION_SIZE // 2 x 2 array containing the indices of the parents in population
    # returns:      POPULATION_SIZE x 20 array containing the new generation

    children = [elite]
    for parent_pair in parents:
        # both p's are 1 x 20 arrays
        p1 = population[parent_pair[0]]
        p2 = population[parent_pair[1]]

        # c1 uses left p1 (v1) at cross over, c2 uses left of p2 (v2) at cross over
        c1, c2 = cross_and_mutate(p1, p2, random_generator)
        children.append(c1)
        children.append(c2)

    return np.array(children)


def cross_and_mutate(p1, p2, random_generator):
    # performs a crossover and mutation
    # c1 uses left p1 (v1) at cross over, c2 uses left of p2 (v2) at cross over
    c1 = []
    c2 = []
    for v1, v2 in zip(p1, p2):
        cross_over = random_generator.randint(1, GENE_SIZE-2)
        v1_gray = bin(gray_encode(v1))[2:]
        v2_gray = bin(gray_encode(v2))[2:]

        # mutate and convert back to ints for conversion
        c1_val = int(mutate(v1_gray[:cross_over] + v2_gray[cross_over:], random_generator), 2)
        c2_val = int(mutate(v2_gray[:cross_over] + v1_gray[cross_over:], random_generator), 2)

        # decode the gray encoding and then append the values
        c1.append(gray_decode(c1_val))
        c2.append(gray_decode(c2_val))

    return np.array(c1), np.array(c2)


def flip(bit):
    # flips a '0' to a '1'
    # and a '1' to a '0'
    if bit == '0':
        return '1'
    else:
        return '0'


def mutate(binary_string, random_generator):
    # randomly flip bits in the binary string
    new_binary_string = ''
    for bit in binary_string:
        p = random_generator.random()
        if p <= MUTATION_RATE:
            new_binary_string += flip(bit)
        else:
            new_binary_string += bit
    return binary_string


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

    return gamma_new, beta_new


def calculate_step(step, gamma, beta):
    # calculates the next step
    x, y, a, v = step
    x_new = x + (v * math.cos(a) * STEP_DELTA)
    y_new = y + (v * math.sin(a) * STEP_DELTA)
    a_new = a + (gamma * STEP_DELTA)
    v_new = v + (beta * STEP_DELTA)
    return np.array([x_new, y_new, a_new, v_new])


def individual_fitness(gamma_beta, current, end):
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


def individual_history(gamma_beta, current):
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
    # gather all individual fitness scores and return the score
    # population_steps: POPULATION_SIZE x 2 x 100 array
    # return:           POPULATION_SIZE x 1 array

    fitness = np.zeros([POPULATION_SIZE])
    for i, individual_steps in zip(range(0, POPULATION_SIZE), population_steps):
        fitness[i] = 1 / (1 + individual_fitness(individual_steps, START, TARGET))
    return fitness


def evaluate_best_individual(individual):
    # perform analysis on the best fit individual
    # analysis ranges from exporting control variables to graphing various properties
    # TODO add exporting and additional graphs. Perhaps save the graphs as pdfs

    gamma_new, beta_new = interpolate_individual(individual)
    t_new = np.linspace(T_START, T_END, STEPS)

    history = individual_history([gamma_new, beta_new], np.copy(START))
    # fist column is x histories, second column is y histories
    x_new = history[:, 0]
    y_new = history[:, 1]

    print(history[-1])

    # obstacles
    x1_obs = np.linspace(-15, -4, 100)
    y1_obs = [3] * 100

    x2_obs = [-4] * 100
    y2_obs = np.linspace(3, -1, 100)

    x3_obs = np.linspace(-4, 4, 100)
    y3_obs = [-1] * 100

    x4_obs = [4] * 100
    y4_obs = np.linspace(-1, 3, 100)

    x5_obs = np.linspace(4, 15, 100)
    y5_obs = [3] * 100

    # the path taken is related to x and y
    plt.figure(figsize=(10, 8))
    plt.plot(x_new, y_new, 'b', label="Path")
    plt.plot(x1_obs, y1_obs, 'black', label="obstacle")
    plt.plot(x2_obs, y2_obs, 'black', label="obstacle")
    plt.plot(x3_obs, y3_obs, 'black', label="obstacle")
    plt.plot(x4_obs, y4_obs, 'black', label="obstacle")
    plt.plot(x5_obs, y5_obs, 'black', label="obstacle")
    plt.axis('square')

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
        return INFEASIBILITY + abs(Y_BEFORE_SPOT_SQ - (y ** 2))
    elif X_BEFORE_SPOT < x < X_AFTER_SPOT and y <= Y_SPOT:
        return INFEASIBILITY + abs(Y_SPOT_SQ - (y ** 2))
    elif x >= X_AFTER_SPOT and y <= Y_AFTER_SPOT:
        return INFEASIBILITY + abs(Y_AFTER_SPOT_SQ - (y ** 2))
    else:
        return 0


def calc_probabilities(fitness):
    # calculates the proportional probability of the fitness for each individual
    total = np.sum(fitness)
    probs = np.divide(fitness, total)
    return probs


def main(args):
    global MUTATION_RATE
    random_generator = np.random
    if len(args) > 1 and args[1] == '-s':
        random_generator.seed(RANDOM_SEED)

    population, labels = create_population(random_generator)

    generation = 0
    elite = None
    elite_score = -1
    elite_j = 10000
    exit_condition = False
    while not exit_condition:
        # 3D array, POPULATION_SIZE x 2 x 100
        # Takes each individual and calculates their beta and gamma values
        population_steps = np.apply_along_axis(interpolate_individual, 1, population)

        fitness = calculate_fitness(population_steps)
        best_index = np.argmax(fitness)

        if elite_score < fitness[best_index]:
            elite = population[best_index]
            elite_score = fitness[best_index]
            elite_j = (1 / elite_score) - 1
            print(elite)
            # evaluate_best_individual(elite)

        print(f"Generation {generation:<4} : J = {elite_j}")

        # using softmax to create a probability distribution
        probs = scipy.special.softmax(fitness)
        population = create_next_generation(population, elite, labels, probs, random_generator)
        exit_condition = (generation >= MAX_GENERATIONS) or (elite_j < J_TOLERANCE)
        generation += 1

    evaluate_best_individual(elite)
    return


if __name__ == "__main__":
    main(sys.argv)