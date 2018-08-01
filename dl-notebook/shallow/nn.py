"""
Shallow (one hidden layer) neural network for building intuition.
"""

import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z.

    Arguments:
        x -- A scalar or numpy array of any size.

    Return:
        s -- sigmoid(z)
    """
    return (np.exp(z) / (1 + np.exp(z)))


class ShallowNeuralNetwork(object):

    def __init__(self, layer_size, num_iterations, learning_rate=0.01, verbose=False):
        """
        Arguments:
            layer_size -- size of the hidden layer
            num_iterations -- number of iterations of the optimization loop
            learning_rate -- learning rate of the gradient descent update rule
            verbose -- print some innards of the training process?
        """
        self.layer_size = layer_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.is_fit = False


    def fit(self, X, Y, layer_size=None):
        """
        Arguments:
            X -- dataset of shape (2, number of examples)
            Y -- labels of shape (1, number of examples)
            n_h -- size of the hidden layer
        
        Modifies:
            parameters -- optimizes values in parameter dict for inference
        """
        if layer_size is not None:
            self.layer_size = layer_size
        
        self._initialize_parameters(X.shape[0], Y.shape[0])
        
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        
        # optimization loop
        # forward prop for probabilities, compute cost on predictions,
        # backpropagate cost across parameters (i.e. compute gradients),
        # and update parameters
        for i in range(self.num_iterations):
            A2, cache = self._forward_propagate(X)
            cost = self._compute_cost(A2, Y)
            self._backward_propagate(X, Y)
            self._update_parameters()
            
            if self.verbose and i % 1000 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))

        self.is_fit = True


    def predict(self, X):
        """
        Predicts a class on {0, 1} for each input array in X.
        
        Arguments:
            X -- ndarray of input data
        
        Returns
            predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        assert(self.is_fit)
        A2, _ = self._forward_propagate(X)
        predictions = np.round(A2)  # simple 0.5 cutoff
        
        return predictions


    def _initialize_parameters(self, n_x, n_y):
        """
        Arguments:
            n_x -- size of the input layer
            n_y -- size of the output layer
        
        Modifies:
            parameters -- sets dict member containing:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        W1 = np.random.randn(self.layer_size, n_x) * 0.01
        b1 = np.zeros((self.layer_size, 1))
        W2 = np.random.randn(n_y, self.layer_size) * 0.01
        b2 = np.zeros((n_y, 1))
        
        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}


    def _forward_propagate(self, X):
        """
        Argument:
            X -- input data of size (n_x, m)
        
        Modifies:
            cache

        Returns:
            A2 -- The sigmoid output of the second activation
            cache -- sets dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b1']
        
        # forward propagate to calculate class probs
        Z1 = np.dot(W1, X)
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1)
        A2 = sigmoid(Z2)
        
        assert(A2.shape == (1, X.shape[1]))
        
        self.cache = {"Z1": Z1,
                      "A1": A1,
                      "Z2": Z2,
                      "A2": A2}

        return A2, self.cache


    def _backward_propagate(self, X, Y):
        """
        Arguments:
            X -- input data of shape (2, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)
        
        Modifies:
            grads -- sets dict member with gradients for different parameters
        """
        m = X.shape[1]
        
        W1 = self.parameters['W1']
        W2 = self.parameters['W2']

        A1 = self.cache['A1']
        A2 = self.cache['A2']
        
        # backprop to calculate dW1, db1, dW2, db2
        dZ2 = A2 - Y
        dW2 = (1. / m) * np.dot(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1. / m) * np.dot(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        self.grads = {"dW1": dW1,
                      "db1": db1,
                      "dW2": dW2,
                      "db2": db2}
        return self.grads


    def _compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost: https://en.wikipedia.org/wiki/Cross_entropy
        
        Arguments:
            A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
            cost -- cross-entropy cost evaluation.
        """
        m = Y.shape[1]  # number of examples

        logprobs = (-1. / m) * (np.multiply(np.log(A2), Y) + np.multiply(np.log(1. - A2), (1. - Y)))
        cost = np.sum(logprobs)
        cost = np.squeeze(cost)  # makes sure cost is proper dimension

        assert(isinstance(cost, float))
        
        return cost


    def _update_parameters(self):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
            None
        
        Modifies:
            parameters -- member dict updated with new values
        """
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        
        # fetch computed gradients for each param
        dW1 = self.grads['dW1']
        db1 = self.grads['db1']
        dW2 = self.grads['dW2']
        db2 = self.grads['db2']
        
        # descent update on each parameter
        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2
        
        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}
