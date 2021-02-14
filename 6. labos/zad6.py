import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# PREPARING DATA FOR TRAINING
f = lambda x, y: ((x - 1) ** 2 + (y + 2) ** 2 - 5*x*y + 3) * (math.cos(x / 5) ** 2) # labels
f_A = lambda x, a, b : 1.0 / (1 + math.exp(b * (x - a))) # antecedent fuzzy_set look
f_z = lambda x, y, p, q, r: p * x + q * y + r # consequent

def create_dataset():
    dataset = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            dataset.append([[x, y], [f(x, y)]])

    return dataset

class FuzzySet():

    def __init__(self):
        self.a = random.random()
        self.b = random.random()
        self.batch_a = 0
        self.batch_b = 0
     

    def calc_alpha(self, x):
        return f_A(x, self.a, self.b)

    def update_parameters_stochastic(self, alphas, zs, i, xy_k, y_, o, alpha, beta, learning_rate):
        a, b = self.a, self.b # saving previous a and b

        big_sum = 0
        for j in range(len(alphas)):
            if j != i:
                big_sum += alphas[j] * (zs[i] - zs[j])

        self.a += (learning_rate * (y_ - o) * big_sum * b * beta * alpha * (1 - alpha)) / (sum(alphas) ** 2)
        self.b += (learning_rate * (y_ - o) * big_sum * (a - xy_k) * beta * alpha * (1 - alpha)) / (sum(alphas) ** 2)
        
    def update_parameters_batch(self, learning_rate):
        self.a += learning_rate * self.batch_a
        self.b += learning_rate * self.batch_b
        
        self.batch_a = 0
        self.batch_b = 0
        
class Consequent():

    def __init__(self):
        self.p = random.random()
        self.q = random.random()
        self.r = random.random()
        self.batch_p = 0
        self.batch_q = 0
        self.batch_r = 0
        

    def calc_z(self, x, y):
        return f_z(x, y, self.p, self.q, self.r)

    def update_parameters_stochastic(self, alphas, i, xy, y_, o, learning_rate):
        self.p += (learning_rate * (y_ - o) * alphas[i] * xy[0]) / sum(alphas)
        self.q += (learning_rate * (y_ - o) * alphas[i] * xy[1]) / sum(alphas)
        self.r += (learning_rate * (y_ - o) * alphas[i]) / sum(alphas)
        
    def update_parameters_batch(self, learning_rate):
        self.p += learning_rate * self.batch_p
        self.q += learning_rate * self.batch_q
        self.r += learning_rate * self.batch_r

        self.batch_p = 0
        self.batch_q = 0
        self.batch_r = 0


class Rule():

    def __init__(self):
        self.antecedent = [FuzzySet(), FuzzySet()]
        self.consequent = Consequent()

    def update_antecedent_stochastic(self, alphas, zs, i, xy, y_, o, learning_rate):
        alpha = f_A(xy[0], self.antecedent[0].a, self.antecedent[0].b)
        beta = f_A(xy[1], self.antecedent[1].a, self.antecedent[1].b)

        for k in range(len(self.antecedent)):
            self.antecedent[k].update_parameters_stochastic(alphas, zs, i, xy[k], y_, o, alpha, beta, learning_rate)
            alpha, beta = beta, alpha

    def update_consequent_stochastic(self, alphas, i, xy, y_, o, learning_rate):
        self.consequent.update_parameters_stochastic(alphas, i, xy, y_, o, learning_rate)
        
    def update_antecedent_batch(self, learning_rate):
        for k in range(len(self.antecedent)):
            self.antecedent[k].update_parameters_batch(learning_rate)
            
    def update_consequent_batch(self, learning_rate):
        self.consequent.update_parameters_batch(learning_rate)

    def get_conclusion(self, xy):
        '''
        xy is a list: stores x under xy[0] and y under xy[1]
        '''

        alpha = 1
        for i in range(len(self.antecedent)):
            alpha *= self.antecedent[i].calc_alpha(xy[i])

        z = self.consequent.calc_z(xy[0], xy[1])

        return alpha, z

class ANFIS():

    def __init__(self, no_of_rules, learning_rate):
        self.no_of_rules = no_of_rules
        self.learning_rate = learning_rate
        self.rules = [Rule() for _ in range(self.no_of_rules)]
        self.alphas = []
        self.zs = []

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def reset(self):
        self.alphas.clear()
        self.zs.clear()

    def calc_output(self, xy):
        brojnik = 0
        nazivnik = 0

        for rule in self.rules:
            alpha, z = rule.get_conclusion(xy)
            self.alphas.append(alpha)
            self.zs.append(z)
            brojnik += alpha * z
            nazivnik += alpha

        return brojnik / nazivnik

    def calc_error(self, dataset):
        error = 0

        for example in dataset:
            xy, y_ = example
            o = self.calc_output(xy)
            error += (y_[0] - o) ** 2

        return error / (2 * len(dataset))


    def train_stochastic(self, dataset, maxIter):
        iteration = 0
        epoch = 0
        epochs = []
        epochs_errors = []
        dataset_len = len(dataset)

        f = open("stochastic_error.txt", 'w')
        txt = '{:<10s}    {:<10s} \n'.format("EPOCH", "ERROR")
        f.write(txt)

        while True:
            for example in dataset:
                if (iteration % dataset_len == 0):
                    epoch += 1
                    epochs.append(epoch)
                    error = self.calc_error(dataset)
                    epochs_errors.append(error)
                    txt = '{:<10d}    {:<f} \n'.format(epoch, error)
                    f.write(txt)
                    print("epoch =", epoch, "error =", error)

                xy, y_ = example
                o = self.calc_output(xy)

                for i in range(self.no_of_rules):
                    self.rules[i].update_antecedent_stochastic(self.alphas, self.zs, i, xy, y_[0], o, self.learning_rate)
                    self.rules[i].update_consequent_stochastic(self.alphas, i, xy, y_[0], o, self.learning_rate)

                iteration += 1
                if iteration >= maxIter:
                    print("Dosegnut maxIter!")
                    f.close()
                    return epochs, epochs_errors

                self.reset()  # reset after each calculation of output o


    def train_batch(self, dataset, num_epochs):
        epochs = [i for i in range(1, num_epochs + 1)]
        epochs_error = []
        f = open("batch_error.txt", 'w')
        txt = '{:<10s}    {:<10s} \n'.format("EPOCH", "ERROR")
        f.write(txt)

        for epoch in range(1, num_epochs + 1):
            error = self.calc_error(dataset)
            epochs_error.append(error)
            txt = '{:<10d}    {:<f} \n'.format(epoch, error)
            f.write(txt)
            print("epoch =", epoch, "error =", error)

            for example in dataset:
                xy, y_ = example
                o = self.calc_output(xy)

                for i in range(self.no_of_rules):
                    suma = sum(self.alphas)
                    self.rules[i].consequent.batch_p += (y_[0] - o) * self.alphas[i] * xy[0] / suma
                    self.rules[i].consequent.batch_q += (y_[0] - o) * self.alphas[i] * xy[1] / suma
                    self.rules[i].consequent.batch_r += (y_[0] - o) * self.alphas[i] / suma

                    big_sum = 0
                    for j in range(len(self.alphas)):
                        if j != i:
                            big_sum += self.alphas[j] * (self.zs[i] - self.zs[j])

                    big_sum /= (suma ** 2)

                    alpha = f_A(xy[0], self.rules[i].antecedent[0].a, self.rules[i].antecedent[0].b)
                    beta = f_A(xy[1], self.rules[i].antecedent[1].a, self.rules[i].antecedent[1].b)

                    for k in range(len(self.rules[i].antecedent)):
                        a, b = self.rules[i].antecedent[k].a, self.rules[i].antecedent[k].b  # saving previous a and b
                        self.rules[i].antecedent[k].batch_a += ((y_[0] - o) * big_sum * b * beta * alpha * (1 - alpha) / (suma ** 2))
                        self.rules[i].antecedent[k].batch_b += ((y_[0] - o) * big_sum * (a - xy[k]) * beta * alpha * (1 - alpha) / (suma ** 2))
                        alpha, beta = beta, alpha

                self.reset()  # reset after each calculation of output o


            # update
            for i in range(self.no_of_rules):
                self.rules[i].update_antecedent_batch(self.learning_rate)
                self.rules[i].update_consequent_batch(self.learning_rate)

        f.close()

        return epochs, epochs_error

    def test(self, test_data):
        predictions = []
        for xy in test_data:
            o = self.calc_output(xy)
            predictions.append(o)

        return predictions



if __name__ == "__main__":

    parameter_1 = input("Unesite broj pravila: ")
    NUM_RULES = int(parameter_1.rstrip())


    # preparing data for training
    dataset = create_dataset()
    random.shuffle(dataset)
    
    # preparing data for testing
    test_data = create_dataset()
    X, Y_ = [], []
    for example in test_data:
        x, y = example
        X.append(x)
        Y_.append(y)



    # drawing results
    domain = [_ for _ in range(-4, 5)]
    x_domain = [x for x in domain for y in domain]
    y_domain = [y for x in domain for y in domain]
    y_ = [f(x, y) for x in domain for y in domain]


    #### 3.zadatak ####
    fig = plt.figure(25)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_domain, y_domain, y_, c='b', marker='o')
    ax.set_title("Train dataset")

    # creating ANFIS and setting up parameters such as learning rate, number of rules, max iterations...

    ############### STOCHASTIC #############
    # stochastic training and testing the model
    # 0.01 na granici s prevelikim, 0.005 taman fino pada ali i dalje alternira,
    # 0.001 fino pada bez alternacije (maxiter > 100000)
    stochastic = ANFIS(no_of_rules=NUM_RULES, learning_rate=0.01)
    stochastic_epochs, stochastic_errors = stochastic.train_stochastic(dataset, maxIter=243000)

    stochastic_predictions = stochastic.test(X)
    stochastic_mistakes = [stochastic_predictions[i] - y_[i] for i in range(len(y_))]

    ############### BATCH #############

    # batch training and testing the model and ploting error
    # 0.001 good, 0.005 better, 0.01 even better (mistake < 0.007) num_epochs = 1500 or more
    batch = ANFIS(no_of_rules=NUM_RULES, learning_rate=0.01)
    batch_epochs, batch_errors = batch.train_batch(dataset, num_epochs=3000)

    ############# PLOT ERRORS 2D ############### 7. zadatak
    plt.figure(0)
    plt.plot(stochastic_epochs, stochastic_errors, label='stochastic error')
    plt.legend(loc='best')
    plt.plot(batch_epochs, batch_errors, label='batch error')
    plt.legend(loc='best')

    batch_predictions = batch.test(X)
    batch_mistakes = [batch_predictions[i] - y_[i] for i in range(len(y_))]

    
    ######### PLOT PREDICTIONS ######## 4. a)
    fig = plt.figure(21)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_domain, y_domain, stochastic_predictions, c='b', marker='o')
    ax.set_title("Stochastic predictions")
    
    fig = plt.figure(22)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_domain, y_domain, batch_predictions, c='r', marker='o')
    ax.set_title("Batch predictions")
    
    ######### PLOT ERRORS 3D ######## 4. b) je ujedno i 6. zadatak
    fig = plt.figure(23)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_domain, y_domain, stochastic_mistakes, c='b', marker='o')
    ax.set_title("Stochastic mistakes")
    
    fig = plt.figure(24)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_domain, y_domain, batch_mistakes, c='r', marker='o')
    ax.set_title("Batch mistakes")


    ### 5. zadatak ###
    # ploting learned fuzzy sets (set stochastic or batch)
    for i in range(len(batch.rules)):
        plt.figure(i+1)
        for k in range(len(batch.rules[i].antecedent)):
            A = [f_A(x, batch.rules[i].antecedent[k].a, batch.rules[i].antecedent[k].b) for x in domain]
            plt.plot(domain, A, label="rule " + str(i+1) + " func " + str(k+1))
            plt.legend()

    '''

    ##### 8. zadatak #####
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("STOCHASTIC")
    axs[0].set(xlabel='epoch', ylabel='MSE') # mean squared error

    axs[1].set_title("BATCH")
    axs[1].set(xlabel='epoch', ylabel='MSE')
    
    # 0.001 good, 0.005 better, 0.01 even better (mistake < 0.007)
    etas = [0.001, 0.005, 0.01]
    # 0.01 na granici s prevelikim, 0.005 taman fino pada ali i dalje alternira,
    # 0.001 fino pada bez alternacije (maxiter > 100000)

    for eta in etas:
        anfis_stochastic = ANFIS(no_of_rules=NUM_RULES, learning_rate=eta)
        epochs, epochs_errors = anfis_stochastic.train_stochastic(dataset, maxIter=243000)
        axs[0].plot(epochs, epochs_errors, label="eta = " + str(eta))
        axs[0].legend(loc='best')

    for eta in etas:
        anfis_batch = ANFIS(no_of_rules=NUM_RULES, learning_rate=eta)
        epochs, epochs_errors = anfis_batch.train_batch(dataset, num_epochs=3000)
        axs[1].plot(epochs, epochs_errors, label="eta = " + str(eta))
        axs[1].legend(loc='best')

'''
    plt.tight_layout()
    plt.show()
