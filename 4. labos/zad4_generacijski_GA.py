import numpy as np
import math

PROBABILITY_OF_MUTATION = 0.05 #elitizam i veća vjerojatnost mutacije je dobra
NUMBER_OF_PARAMETERS = 5
SOL_PER_POPULATION = 12 #mora biti parna
NUMBER_OF_GENERATIONS = 2000

ELITISM = True

crossover_point = np.uint8(NUMBER_OF_PARAMETERS / 2)
population_shape = (SOL_PER_POPULATION, NUMBER_OF_PARAMETERS)
fitness = np.empty(shape=(SOL_PER_POPULATION, 1))
best_population = np.empty(shape=population_shape)
population = np.random.uniform(low=-4, high=4, size=population_shape)
selected_population = np.empty(shape=population_shape)
kids = np.empty(shape=population_shape)


def func(x, y, sol):
    return math.sin(sol[0] + x * sol[1]) + (sol[2] * math.cos(x * (sol[3] + y)) *
        (1/(1 + pow(math.e, pow((x - sol[4]), 2)))))


def calculate_fitness(sol):
    file_object = open("zad4-dataset1.txt", "rt")
    punishment = 0.0
    lines = 0
    for line in file_object.readlines():
        lines += 1
        x, y, real_output = list(map(float, line.split()))
        punishment += pow((func(x, y, sol) - real_output), 2)

    punishment /= lines

    return 1/punishment


def mutation(kids):
    mutation_probabilities = [1 - PROBABILITY_OF_MUTATION, PROBABILITY_OF_MUTATION]

    for kid in kids:
        mutation_flags = np.random.choice([0, 1], NUMBER_OF_PARAMETERS, p=mutation_probabilities)
        #print("Mutating gene (NO/YES) == (0/1)", mutation_flags)
        for i in range(0, NUMBER_OF_PARAMETERS):
            if mutation_flags[i] == 1:
                kid[i] += np.random.uniform(-1.0, 1.0, 1)# mogao sam stavit i gaussa

    return kids

def crossover(selected_population):
    kids = np.empty(shape=(population_shape))
    reproduction_indexes = np.random.randint(0, SOL_PER_POPULATION-1, SOL_PER_POPULATION)
    #print("Indexi roditelja koji daju djecu:\n", reproduction_indexes)
    
    for i in range(0, len(selected_population), 2):
        kids[i][0:crossover_point] = selected_population[reproduction_indexes[i]][0:crossover_point]
        kids[i][crossover_point:] = selected_population[reproduction_indexes[i+1]][crossover_point:]
        
        kids[i+1][0:crossover_point] = selected_population[reproduction_indexes[i+1]][0:crossover_point]
        kids[i+1][crossover_point:] = selected_population[reproduction_indexes[i]][crossover_point:]

    return kids

max_fitness = 0.0
best_generation_index = 0
best_sol = np.empty(shape=(1, NUMBER_OF_PARAMETERS))
best_sol_fitness = 0.0

for generation_index in range(NUMBER_OF_GENERATIONS):
    if (generation_index - best_generation_index > 1000):
        break

    for i in range(SOL_PER_POPULATION):
        fitness[i] = calculate_fitness(population[i])
        if fitness[i] > best_sol_fitness: #trazim najbolju jedinku (za elitizam)
            best_sol_fitness = fitness[i]
            best_sol = population[i]

    fitness_sum = np.sum(fitness)
    if fitness_sum > max_fitness:
        max_fitness = fitness_sum
        best_generation_index = generation_index
        best_population = population
        print("Generacija: ", generation_index)
        print("Ukupna dobrota ove generacije je: ", fitness_sum)

    parents_probabilities = [fitness[i][0]/fitness_sum for i in range(SOL_PER_POPULATION)]
    parents_indexes = np.random.choice(np.linspace(0, SOL_PER_POPULATION-1, SOL_PER_POPULATION, dtype=int), SOL_PER_POPULATION, p=parents_probabilities)

    for i in range(SOL_PER_POPULATION):
        selected_population[i] = population[parents_indexes[i]]

    kids = crossover(selected_population)
    kids = mutation(kids)

    population = kids

    if ELITISM:
        population[0] = best_sol


average_mistake = 1/calculate_fitness(best_population[0])

print("Zaustavljeno na iteraciji:\n", generation_index)
print("Najveća dobrota je: ", max_fitness, "i desila se u generaciji: ", best_generation_index, "koja izgleda:\n", best_population[0])
print("Prosjecna greska je:\n", average_mistake)
