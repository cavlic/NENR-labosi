import math
from random import random
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import zad5_data_generator as dt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot_product(ar1, ar2):
    if (len(ar1) != len(ar2)):
        print("Dimenzije vektora nisu jednake!")
        return

    suma = 0
    for i in range(len(ar1)):
        suma += ar1[i] * ar2[i]

    return suma

def load_data(file_name):
    f = open(file_name, 'r')

    data = []
    for line in f.readlines():
        line = list(map(float, line.split()))
        X, Y = line[:2*M], line[2*M:]
        example = [X, Y]
        data.append(example)

    return data

def batch_data(data, group_size):
    examples_per_class = int(group_size / output_layer_size)
    batch_groups = int(len(data) / group_size)
    batches = [[] for _ in range(batch_groups)]

    for i in range(batch_groups):
        for c in range(output_layer_size):
            for j in range(examples_per_class):
                batches[i].append(data[j + i * 2 + c * 20])

    return batches


class Neuron():
    def __init__(self, weights, bias=0):
        self.weights = weights
        self.bias = bias
        self.epoch_mistakes = []
        self.epoch_outputs = []

    def reset_neuron(self):
        self.epoch_mistakes.clear()
        self.epoch_outputs.clear()


    def net(self, inputs):
        if len(inputs) != len(self.weights):
            print("Dimenzije vektora u netu nisu jednake!")
            return

        #print(inputs, "dot", self.weights)
        net = 0
        for i in range(len(inputs)):
            net += inputs[i] * self.weights[i]
        return net + self.bias

    def calculate_output(self, inputs):
        return sigmoid(self.net(inputs))

    def inspect(self):
        print(self.weights)


class Layer():

    # weights and biases as a LIST per a Layer!
    def __init__(self, size, weights, bias):
        self.size = size
        self.neurons = [Neuron(weights[i], bias) for i in range(self.size)]
        self.outputs = [0 for _ in range(self.size)]
        self.mistakes = [0 for _ in range(self.size)]

    def inspect(self):
        for neuron in self.neurons:
            neuron.inspect()
        print("--------------------------------------------------------------------")


    def get_affected_weights(self, i):
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weights[i])

        return weights

    def output_layer_mistakes(self, labels):
        example_error = 0
        for j in range(self.size):
            example_error += ((labels[j] - self.outputs[j]) ** 2)
            self.mistakes[j] = self.outputs[j] * (1 - self.outputs[j]) * (labels[j] - self.outputs[j])
            self.neurons[j].epoch_mistakes.append(self.mistakes[j])

        return example_error


    def hidden_layer_mistakes(self, next_layer):
        for j in range(self.size):
            weights = next_layer.get_affected_weights(j)
            self.mistakes[j] = self.outputs[j] * (1 - self.outputs[j]) * dot_product(weights, next_layer.mistakes)
            self.neurons[j].epoch_mistakes.append(self.mistakes[j])



class NeuralNet():

    def __init__(self, input_size, hidden_sizes, output_size, hidden_weights, output_weights, hidden_biases=[0], output_bias=0):
        self.input_size = input_size
        self.ins = [[] for _ in range(self.input_size)]
        self.layers = []
        self.hidden_sizes = hidden_sizes
        self.output_layer_size = output_size

        self.hidden_layer_weights = hidden_weights
        self.output_layer_weights = output_weights

        self.hidden_layer_biases = hidden_biases
        self.output_layer_bias = output_bias
        self.error = 0
        self.init_net()


    def reset(self):
        self.ins = [[] for _ in range(self.input_size)]

        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.reset_neuron()

    # hidden_layer_weights as = [[x, x, x ..]] if only one hidden layer
    # output_layer_biases as [x,x,x, ..]
    def init_net(self):
        for i in range(len(self.hidden_sizes)):
            self.layers.append(Layer(self.hidden_sizes[i], self.hidden_layer_weights[i], self.hidden_layer_biases[i]))

        self.layers.append(Layer(self.output_layer_size, self.output_layer_weights, self.output_layer_bias))


    def train(self, data, online):
        for example in data:
            inputs, labels = example[0], example[1]
            for i in range(len(inputs)):
                self.ins[i].append(inputs[i])
            self.feedfoward(inputs)
            self.update_mistakes(labels)
            if online:
                self.update_weights(online, inputs)


        if not online:
            self.update_weights(online, ["doesnt matter"])
            self.reset()


    # fills up sample_outputs for batch learning
    # calculates layer outputs for error checking
    def feedfoward(self, inputs):
        for layer in self.layers:
            for i in range(len(layer.neurons)):
                layer.outputs[i] = layer.neurons[i].calculate_output(inputs)
                layer.neurons[i].epoch_outputs.append(layer.outputs[i])

            inputs = layer.outputs # DEEPCOPY!?

    # calculates mistakes for each layer
    # sums up error for each example
    def update_mistakes(self, labels):
        for i in range(len(self.layers) - 1, -1, -1): # goes backwards in layers
            if i == (len(self.layers) - 1):
                example_error = self.layers[i].output_layer_mistakes(labels)
                self.error += example_error

            else:
                self.layers[i].hidden_layer_mistakes(self.layers[i+1])

    def update_last_hidden_layer(self, layer, inputs, online):
        if online:
            for i in range(len(inputs)):
                for j in range(layer.size):
                    layer.neurons[j].weights[i] += (LEARNING_RATE * layer.mistakes[j] * inputs[i])

            # bias update
            for j in range(layer.size):
                layer.neurons[j].bias += LEARNING_RATE * layer.mistakes[j]

        else:
            for i in range(len(inputs)):
                for j in range(layer.size):
                    suma = 0
                    for k in range(len(layer.neurons[0].epoch_mistakes)):
                        suma += (layer.neurons[j].epoch_mistakes[k] * self.ins[i][k])

                    layer.neurons[j].weights[i] += (LEARNING_RATE * suma)

            # bias update
            for j in range(layer.size):
                layer.neurons[j].bias += LEARNING_RATE * sum(layer.neurons[j].epoch_mistakes)


    def update_weights(self, online, inputs):
        for k in range(len(self.layers)-1, 0, -1): # goes backwards in layers
            prev_layer = self.layers[k-1]
            layer = self.layers[k]


            if online:
                for i in range(prev_layer.size):
                    for j in range(layer.size):
                        layer.neurons[j].weights[i] += LEARNING_RATE * layer.mistakes[j] * prev_layer.outputs[i]

                    # bias update
                    for j in range(layer.size):
                        layer.neurons[j].bias += LEARNING_RATE * layer.mistakes[j]

            else:
                for i in range(prev_layer.size):
                    for j in range(layer.size):
                        suma = 0
                        for k in range(len(layer.neurons[0].epoch_mistakes)):
                            suma += layer.neurons[j].epoch_mistakes[k] * prev_layer.neurons[i].epoch_outputs[k]

                        layer.neurons[j].weights[i] += LEARNING_RATE * suma

                # bias update
                for j in range(layer.size):
                    layer.neurons[j].bias += LEARNING_RATE * sum(layer.neurons[j].epoch_mistakes)


        # UPDATE LAST HIDDEN! MUST DO!
        self.update_last_hidden_layer(self.layers[0], inputs, online)


    def predict(self, data):
        outputs = [0 for _ in range(len(data))]
        for i in range(len(data)):
            self.feedfoward(data[i])
            outputs[i] = self.layers[-1].outputs[0]
            print(outputs)
        return outputs

    def predict2(self, inputs):
        self.feedfoward(inputs)
        return self.layers[-1].outputs

if __name__ == "__main__":

    # 2.5 for online, 10000 EPOCHS, error 0.00 ... works good!
    # 0.05 for group = batch = 100, 15000 EPOCHS, error 5.72
    # 0.1 for group = 20, 10000 EPOCHS, error = 2.38

    LEARNING_RATE = 2.5
    M = 10

    input_layer_size = 2*M
    hidden_layer_sizes = [8, 4]
    output_layer_size = 5
    hidden_weights = [[[random() for _ in range(input_layer_size)] for _ in range(hidden_layer_sizes[0])],
                      [[random() for _ in range(hidden_layer_sizes[0])] for _ in range(hidden_layer_sizes[1])]]
    output_weights = [[random() for _ in range(hidden_layer_sizes[1])] for _ in range(output_layer_size)]
    hidden_biases = [random() for _ in range(len(hidden_layer_sizes))]
    output_bias = random()


    nn = NeuralNet(input_layer_size, hidden_layer_sizes, output_layer_size, hidden_weights, output_weights, hidden_biases, output_bias)

    # load data from filename
    file_name = "train_data.txt"
    data = load_data(file_name)

    # which type of learning do you want?
    decision = input("Batch (y/n)?\n")
    while not (decision == "y" or decision == "n"):
        decision = input("Batch (y/n)?\n")

    batch = True if decision == "y" else False

    if batch:
        online = False
        group_size = int(input("Group size? Type number...\n"))
        groups = batch_data(data, group_size)
        for g in groups:
            print(g)

    else:
        decision = input("Online (y/n)?\n")
        while not (decision == "y" or decision == "n"):
            decision = input("Online (y/n)?\n")

        online = True if decision == "y" else False



    EPOCHS = 10000
    for i in range(EPOCHS):
        print("epoch no", i)
        if batch:
            for g in groups:
                nn.train(g, online)

        else:
            nn.train(data, online)

        print(nn.error / 2*len(data))
        nn.error = 0

    test_examples = 15

    for _ in range(test_examples):
        g = dt.GUI()
        g.mainloop()
        test_data = g.get_dataset(M)
        print(nn.predict2(test_data[0]))



