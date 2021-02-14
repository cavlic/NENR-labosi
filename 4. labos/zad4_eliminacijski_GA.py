import numpy as np
import math
import random

PROBABILITY_OF_MUTATION = 0.1
NUMBER_OF_PARAMETERS = 5
SOL_PER_POPULATION = 24
NUMBER_OF_TOURNAMENTS = 10000

K = 3
crossover_point = np.uint8(NUMBER_OF_PARAMETERS / 2)
population_shape = (SOL_PER_POPULATION, NUMBER_OF_PARAMETERS)
fitness = np.empty(shape=(SOL_PER_POPULATION, 1))
population = np.random.uniform(low=-4, high=4, size=population_shape)


def func(x, y, sol):
    return math.sin(sol[0] + x * sol[1]) + (sol[2] * math.cos(x * (sol[3] + y)) *
                                            (1 / (1 + pow(math.e, pow((x - sol[4]), 2)))))


def calculate_fitness(sol):
    file_object = open("zad4-dataset1.txt", "rt")
    punishment = 0.0
    lines = 0
    for line in file_object.readlines():
        lines += 1
        x, y, real_output = list(map(float, line.split()))
        punishment += pow((func(x, y, sol) - real_output), 2)

    punishment /= lines

    return 1 / punishment


def mutation(offspring_crossover):
    mutation_probabilities = [1 - PROBABILITY_OF_MUTATION, PROBABILITY_OF_MUTATION]
    mutation_flags = np.random.choice([0, 1], NUMBER_OF_PARAMETERS, p=mutation_probabilities)
    #print("Mutating gene (NO/YES) == (0/1)", mutation_flags)

    for i in range(0, NUMBER_OF_PARAMETERS):
        if mutation_flags[i] == 1:
            offspring_crossover[0][i] += random.gauss(0, 1)

    return offspring_crossover


def crossover(parent1, parent2):
    offspring = np.empty(shape=(1, NUMBER_OF_PARAMETERS))

    offspring[0][0:crossover_point] = parent1[0:crossover_point]
    offspring[0][crossover_point:] = parent2[crossover_point:]

    return offspring


def select_worst_sol(fitness):
    sol_for_tournament = random.sample(range(0, SOL_PER_POPULATION), K)  # INDEXI jedinki odabranih za turnir
    min_fitness = 99999999
    worst_sol_index = None
    for sol in sol_for_tournament:
        if fitness[sol] < min_fitness:
            min_fitness = fitness[sol]
            worst_sol_index = sol

    sol_for_tournament.remove(worst_sol_index)

    return worst_sol_index, sol_for_tournament


for i in range(SOL_PER_POPULATION):
    fitness[i] = calculate_fitness(population[i])

count = 0

best_sol_fitness = 0.0
for i in range(SOL_PER_POPULATION):
    fitness[i] = calculate_fitness(population[i])
    if fitness[i] > best_sol_fitness:  # trazim najbolju jedinku
        best_sol_fitness = fitness[i]
        best_sol = population[i]

print("Dobrota najbolje jedinke je:", best_sol_fitness)


while count < NUMBER_OF_TOURNAMENTS:
    count += 1
    worst_sol_index, solos_for_crossover = select_worst_sol(fitness)

    # STVARANJE NOVE JEDINKE
    offspring_crossover = crossover(population[solos_for_crossover[0]], population[solos_for_crossover[1]])
    offspring_mutation = mutation(offspring_crossover)
    population[worst_sol_index] = offspring_mutation

    # OSVJEŽVANJE DOBROTA JEDINKI
    fitness[worst_sol_index] = calculate_fitness(population[worst_sol_index])

    max_fitness = max(fitness)
    if max_fitness > best_sol_fitness:
        best_sol_fitness = max_fitness
        print("Turnir", count, "Dobrota najbolje jedinke je:", best_sol_fitness)


best_sol_index = np.argmax(fitness)
print("Greska najbolje jedinke:", 1/fitness[best_sol_index])


