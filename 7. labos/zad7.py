import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from copy import deepcopy
import random
import numpy as np
import time as t


f = lambda x, w, s: 1 / (1 + abs(x - w) / abs(s))
f_suma = lambda x, w, s: abs(x - w) / abs(s)
f2 = lambda suma: 1 / (1 + suma)
sigma = lambda x: 1 / (1 + math.exp(-x))
f_fitness = lambda x: 1 / x


def load_dataset(location):
    dataset = []

    f = open(location, 'r')
    for line in f.readlines():
        line = line.split()
        inputs, labels = list(map(float, line[:2])), list(map(int, line[2:]))
        example = [inputs, labels]
        dataset.append(example)

    f.close()

    return dataset

def get_test_data(dataset):
    X, y_ = [], []

    for example in dataset:
        x, y = example
        X.append(x)
        y_.append(y)

    return X, y_

def net(ar1, ar2):
    bias = ar1[0] # w0
    ar1 = ar1[1:] # w1, w2, ... , wn

    if (len(ar1) != len(ar2)):
        print("Dimenzije vektora nisu jednake!")
        return

    suma = 0.0
    for i in range(len(ar1)):
        suma += ar1[i] * ar2[i]

    return suma + bias

class NeuralNet():


    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_of_params = None
        self.outputs = None
        self.parameters = None # used for output calculation
        self.original_parameters = None # used for gen alg optimisation
        self.init_outputs()
        self.init_parameters()

    def init_outputs(self): # izlazi skrivenih i izlaznog sloja
        self.outputs = []
        for i in range(len(self.hidden_layer_sizes)):
            out = [0.0] * self.hidden_layer_sizes[i]
            self.outputs.append(out)

        out = [0.0] * 3
        self.outputs.append(out)

    def init_parameters(self):
        self.num_of_params = self.num_of_parameters()
        self.parameters = [random.uniform(-1.0, 1.0) for _ in range(self.num_of_params)]
        self.original_parameters = deepcopy(self.parameters)

    def num_of_parameters(self): # uključuje parametre skrivenih i izlaznog sloja
        num = 4 * self.hidden_layer_sizes[0]
        for i in range(1, len(self.hidden_layer_sizes)):
            num += self.hidden_layer_sizes[i] * (self.hidden_layer_sizes[i - 1] + 1)

        num += 3 * (self.hidden_layer_sizes[-1] + 1)

        return num
    
    def calc_1st_hidden_layer_outputs(self, inputs):
        offset = 4

        for j_neuron in range(len(self.outputs[0])):
            suma = 0
            k = 0
            for i in range(0, len(inputs)*2, 2):
                suma += f_suma(inputs[k], self.parameters[i + j_neuron * offset], self.parameters[i + 1 + j_neuron * offset])
                k += 1

            self.outputs[0][j_neuron] = f2(suma)

        cut_parameters = 4 * self.hidden_layer_sizes[0]
        self.parameters = self.parameters[cut_parameters:]

        return self.outputs[0]

    def calc_other_layers_outputs(self, inputs):
        for layer in range(1, len(self.outputs)):
            offset = len(inputs)

            for j_neuron in range(len(self.outputs[layer])):
                self.outputs[layer][j_neuron] = sigma(net(self.parameters[:offset + 1], inputs))
                self.parameters = self.parameters[offset + 1:]
            inputs = self.outputs[layer]


    def calc_outputs(self, inputs):
        inputs = self.calc_1st_hidden_layer_outputs(inputs)
        self.calc_other_layers_outputs(inputs)
        self.parameters = deepcopy(self.original_parameters) # refreshing parameters for next example

        return self.outputs[-1]

    def calc_error(self, dataset):
        error = 0

        for example in dataset:
            inputs, labels = example
            y = self.calc_outputs(inputs)
            for i in range(len(y)):
                error += (y[i] - labels[i]) ** 2

        return error / len(dataset)

    def test(self, inputs):
        predictions = []

        for x in inputs:
            outputs = [0 if y < 0.5 else 1 for y in self.calc_outputs(x)]
            predictions.append(outputs)

        return predictions

def select_worst_sol(fitnesses):
    sol_for_tournament = random.sample(range(0, VEL_POP), K)  # INDEXI jedinki odabranih za turnir
    min_fitness = 999999
    worst_sol_index = None
    for sol in sol_for_tournament:
        if fitnesses[sol] < min_fitness:
            min_fitness = fitnesses[sol]
            worst_sol_index = sol

    sol_for_tournament.remove(worst_sol_index)

    return worst_sol_index, sol_for_tournament

def mutation(offspring_crossover, vs, probs, sigmas):
    chosen_operator_index = np.random.choice([0, 1, 2], 1, p=vs)[0]
    chosen_probability = probs[chosen_operator_index]
    chosen_sigma = sigmas[chosen_operator_index]

    num_of_params = len(offspring_crossover)
    mutation_probabilities = [1 - chosen_probability, chosen_probability]
    mutation_flags = np.random.choice([0, 1], num_of_params, p=mutation_probabilities)

    for i in range(0, num_of_params):
        if mutation_flags[i] == 1:
            offspring_crossover[i] += random.gauss(0, chosen_sigma)

    return offspring_crossover

def crossover(parent1, parent2, better_parent_index):
    offspring = []
    num_genes = len(parent1)
    ctype = random.sample(range(0, 3), 1)[0]  # tip križanja (randomizirano)
    if ctype == 0: # discrete uniform
        discrete_uniform = np.random.choice([0, 1], num_genes, p=[0.5, 0.5])
        offspring = [parent1[i] if discrete_uniform[i] == 0 else parent2[i] for i in range(num_genes)]

    elif ctype == 1: # local aritmetic crx
        for i in range(num_genes):
            alfa = np.random.random()
            new_gene = alfa * parent1[i] + (1 - alfa) * parent2[i]
            offspring.append(new_gene)


    else: # heuristic
        for i in range(num_genes):
            alfa = np.random.random()
            if better_parent_index == 1:
                new_gene = alfa * parent1[i] + (1 - alfa) * parent2[i]
            else:
                new_gene = alfa * parent2[i] + (1 - alfa) * parent1[i]

            offspring.append(new_gene)


    return offspring

def get_best_sol_index(fitnesses):
    index = [i for i, value in enumerate(fitnesses) if value == max(fitnesses)]
    return index[0]

def zad2():
    f = open("zad7-dataset.txt", 'r')
    cnt_A, cnt_B, cnt_C = 0, 0, 0
    x_A, y_A = [], []
    x_B, y_B = [], []
    x_C, y_C = [], []

    for line in f.readlines():
        line = line.split()
        x, y, A, B, C = float(line[0]), float(line[1]), int(line[2]), int(line[3]), int(line[4])

        if A:
            cnt_A += 1
            x_A.append(x)
            y_A.append(y)

        elif B:
            cnt_B += 1
            x_B.append(x)
            y_B.append(y)

        else:
            cnt_C += 1
            x_C.append(x)
            y_C.append(y)

    f.close()

    return x_A, y_A, x_B, y_B, x_C, y_C


if __name__ == "__main__":
    NUMBER_OF_TOURNAMENTS = 200000
    VEL_POP = 50
    K = 3

    ts = input("Unesite poželjnosti određenih operatora mutacija u formatu: x y z\n") # npr 4 2 1
    ts = list(map(float, ts.rstrip().split()))

    vs = [t/sum(ts) for t in ts]# probabilities for choosing each mutation operator
    sigmas = [0.2, 1, 2]#sigmas for each mutation operator
    probs = [0.05, 0.05, 0.05]#probabilities of mutation for each mutation operator


    dataset = load_dataset("zad7-dataset.txt")
    X, y_ = get_test_data(dataset)

    population = [NeuralNet([8]) for _ in range(VEL_POP)]
    population_parameters = [sol.original_parameters for sol in population]




    errors = [sol.calc_error(dataset) for sol in population]
    error = sum(errors)
    print("Prosječni error populacije prije GA", error / len(population))
    fitnesses = [f_fitness(error) for error in errors]

    start_time = t.time()

    count = 0
    while (count < NUMBER_OF_TOURNAMENTS):
        worst_sol_index, solos_for_crossover = select_worst_sol(fitnesses)
        if fitnesses[solos_for_crossover[0]] > fitnesses[solos_for_crossover[1]]:
            better_parent_index = 1
        else:
            better_parent_index = 2

        # STVARANJE NOVE JEDINKE
        offspring_crossover = crossover(population_parameters[solos_for_crossover[0]],
                                        population_parameters[solos_for_crossover[1]],
                                        better_parent_index)

        offspring_mutation = mutation(offspring_crossover, vs, probs, sigmas)

        population[worst_sol_index].original_parameters = offspring_mutation
        population[worst_sol_index].parameters = offspring_mutation
        population_parameters[worst_sol_index] = offspring_mutation

        # OSVJEŽVANJE DOBROTA JEDINKI
        fitnesses[worst_sol_index] = f_fitness(population[worst_sol_index].calc_error(dataset))
        if count % 1000 == 0:
            best_sol_index = get_best_sol_index(fitnesses)
            print("tournament = ", count, " best_sol_error = ", population[best_sol_index].calc_error(dataset))

        count += 1

    print("postupak učenja trajanje: ", t.time() - start_time)
    suma = 0.0
    for i in range(len(population)):
        suma += population[i].calc_error(dataset)

    print("Prosječni error populacije nakon GA", suma / len(population))


    best_sol_index = get_best_sol_index(fitnesses)
    predictions = population[best_sol_index].test(X)

    tocnih, pogresnih = 0, 0
    for i in range(len(predictions)):
        #print("Y_ = ", y_[i], "predictions = ", predictions[i])
        if y_[i] == predictions[i]:
            tocnih += 1
        else:
            pogresnih += 1

    print("Tocno klasificiranih ima:", tocnih)
    print("Pogresno klasificiranih ima:", pogresnih)


    print("error najbolje jednike", population[best_sol_index].calc_error(dataset))

    ### 4 zad ###

    neurons_type_1_num = population[0].hidden_layer_sizes[0]
    neuron_type_1_weights = population[best_sol_index].original_parameters[:4 * neurons_type_1_num]
    print("neuron type 1 weights:\nlen:", len(neuron_type_1_weights))
    for w in neuron_type_1_weights:
        print(w)

    first_hidden_layer_parameters = population[best_sol_index].original_parameters[:4 * neurons_type_1_num]
    neurons_type_1_weights = []
    neurons_type_1_scales_x = []
    neurons_type_1_scales_y = []


    for i in range(0, neurons_type_1_num):
        neuron_weights = []
        neuron_weights.append(first_hidden_layer_parameters[i*4])
        neuron_weights.append(first_hidden_layer_parameters[i*4 + 2])
        neurons_type_1_weights.append(neuron_weights)

        neurons_type_1_scales_x.append(first_hidden_layer_parameters[i*4 + 1])
        neurons_type_1_scales_y.append(first_hidden_layer_parameters[i*4 + 3])

    x_A, y_A, x_B, y_B, x_C, y_C = zad2()
    plt.figure(1)
    plt.title("Clusters and their representative neurons")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x_A, y_A, c='b', marker='^')
    plt.scatter(x_B, y_B, c='r', marker='P')
    plt.scatter(x_C, y_C, c='g', marker='o')

    neuron_num = 1
    for w in neurons_type_1_weights:
        w1, w2 = w
        plt.scatter(w1, w2, c='y', label=str(neuron_num), marker='D')
        plt.annotate(neuron_num, (w1, w2))
        neuron_num += 1

    plt.figure(2)
    plt.title("Scales")
    plt.xlabel("scales for x")
    plt.ylabel("scales for y")
    plt.scatter(neurons_type_1_scales_x, neurons_type_1_scales_y)

    neuron_type_2_weights = population[best_sol_index].original_parameters[4 * neurons_type_1_num:]
    print("neuron type 2 weights:\nlen:", len(neuron_type_2_weights))
    for w in neuron_type_2_weights:
        print(w)

    plt.show()





