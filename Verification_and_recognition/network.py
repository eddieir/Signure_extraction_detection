import random
import numpy as np



class NeuralNetwork():

    def __init__(self,sizes):
        # sizes is an array with the number of units in each layer
        # [2,3,1] means w neurons of input, 3 in the hidden layer and 1 as output
        self.num_layers = len(sizes)
        self.sizes = sizes

        # biases will have one vector per layer
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # zip returns a tuple in which x is the element of the first array and y the element of the second
        # sizes[:-1] returns all the elements till the second to last
        # sizes[1:] returns all the elements from the second and on]
        # [2,3,1] means:
        # * matrix of 3 rows and 2 columns -- will be multiplied by the inputs
        # * matrix of 1 row and 3 columns -- will multiply the hidden layer and produce the output

        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]


    def feedforward(self,a):
        for p,i in zip(self.biases, self.weights):
            a = sigmoid(np.dot(i,a) + p)
            return a

    def seperate_batches(self,training_data,batch_size):
        random.shuffle(training_data)
        n = len(training_data)
        # extracts chunks of data from the training set
        # the xrange function will return indices starting with 0 untill n, with a step size o batch_size
        # batches, then, will have several chunks of the main set, each defined by the batch_size_variable
        return [training_data[i:i + batch_size] for i in range(0, n, batch_size)]

    def update_batches(self,batches,alpha):
        for batch in batches:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            m =len(batch)

            # x is a array of length 901
            # y is a single value indicating the digit represented by the 901 elements
            for x,y in batch:
                delta_b, delta_w = self.backpropagation(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

            self.weights = [w - (alpha / m) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (alpha / m) * nb for b, nb in zip(self.biases, nabla_b)]

            


def sigmoid(z):
    return  1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)
