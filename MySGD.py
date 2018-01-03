import numpy as np
import random
import gzip
import pickle

import matplotlib.pyplot as plt



def load_data():
    f = gzip.open('d:\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = [(x, y) for x, y in zip(training_inputs, training_results)]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = [(x, y) for x, y in zip(validation_inputs, va_d[1])]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = [(x, y) for x, y in zip(test_inputs, te_d[1])]
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class NetWork(object):
    def __init__(self, sizes,a):
        self.n_layer = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, inputed_weights):
        return 1.0/(1.0 + np.exp(-inputed_weights))

    def sigmoid_prime(self, inputed_weights):
        return self.sigmoid(inputed_weights) * (1 - self.sigmoid(inputed_weights))

    def feedforward(self, activation):
        for b, w in zip(self.biases, self.weights):
            activation = self.sigmoid(np.dot(w, activation) + b)
        return activation

    def error(self,activation,y):
        activation = self.feedforward(activation)
        #b = 0
        z = activation - y
        b = z*z
        b = sum(b)
        #for i in range(len(z)):
        #   b = b+ z[i]*z[i]

        return b


    def cost_derivative(self, a, y):
        return (a - y)

    def backpropagation(self, x, y):

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])



        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2,self.n_layer):
            z = zs[-l]
            #delta = (self.weights[-l+1].transpose()*delta)#*self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmoid_prime(z)
            delta_nabla_b[-l] = delta
            #delta_nabla_w[-l]=delta*activations[-l-1].transpose()
            delta_nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (delta_nabla_b, delta_nabla_w)

    def update_mini_batch(self, mini_batch, eta):

        sum_delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        sum_delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            sum_delta_nabla_b = [db + sdb for db, sdb in zip(delta_nabla_b, sum_delta_nabla_b)]
            sum_delta_nabla_w = [dw + sdw for dw, sdw in zip(delta_nabla_w, sum_delta_nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, sum_delta_nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, sum_delta_nabla_w)]

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

    def SGD(self, training_data, epchos, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n_data = len(training_data)
        y=[]
        x=[]
        r=[]
        for j in range(epchos):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                y_result = self.evaluate(test_data)
                print("Epoches: {0}:--> {1}/{2}".format(j, y_result, n_test))
                y.append(y_result)
                x.append(j)

            else:
                print("Epoches:{0}:-->".format(j))
            k=[]
            for (a,b) in mini_batch:
                c=self.error(a,b)
                k.append(c)
            u = sum(k)
            r.append(u)
        w=[]
        for i in range(epchos):
            w.append(r[i][0])

        plt.plot(x,w)
        #plt.plot(x,y)
        print("w={0}".format(w))
        print("y={0}".format(y))
        plt.show()


training_data, validation_data, test_data1 = load_data_wrapper()
net = NetWork([784, 50,30,20, 10])
net.SGD(training_data, 20, 10, 3.0, test_data=test_data1)
