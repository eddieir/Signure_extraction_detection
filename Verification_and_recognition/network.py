import numpy as np
import random


class NeuralNetwork():

    def __init__(self, sizes):
        # sizes is an array with the number of units in each layer
        # [2,3,1] means w neurons of input, 3 in the hidden layer and 1 as output
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def separate_batches(self, training_data, batch_size):
        random.shuffle(training_data)
        n = len(training_data)
        # extracts chunks of data from the training set
        # the xrange function will return indices starting with 0 untill n, with a step size o batch_size
        # batches, then, will have several chunks of the main set, each defined by the batch_size_variable
        return [training_data[i:i + batch_size] for i in range(0, n, batch_size)]

    def update_batches(self, batches, alpha):
        for batch in batches:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            m = len(batch)

            # x is a array of length 901
            # y is a single value indicating the digit represented by the 901 elements
            for x, y in batch:
                delta_b, delta_w = self.backpropagation(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

            self.weights = [w - (alpha / m) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (alpha / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            # layer-bound b and w
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def sgd(self, training_data, epochs, batch_size, alpha, test_data):
        n_test = len(test_data)

        for epoch in range(epochs):
            batches = self.separate_batches(training_data, batch_size)
            self.update_batches(batches, alpha)

            print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))

    def evaluate(self, test_data):
        #r = [self.feedforward(x) for (x, y) in test_data]
        #for a in r:
        #    print("{0}, {1}".format(format(a[0][0], 'f'), format(a[1][0], 'f')))
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
