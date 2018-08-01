"""
Basic logistic regression implementation with a neural network style representation.
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


class LogisticNN(object):

    def __init__(self, num_iterations, learning_rate, verbose=False):
        """
        Arguments:
            num_iterations -- number of iterations of the optimization loop
            learning_rate -- learning rate of the gradient descent update rule
            verbose -- print some innards of the training process?
        """
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.is_fit = False

        # set via call to .fit()
        self._w = None
        self._b = None
        self.grads = None
        self.costs = None


    def fit(self, X, y):
        """
        Fits model weights to input data using gradient descent optimization.
    
        Arguments:
            X -- data of shape (number of features, number of examples)
            Y -- true class vector consisting of 0s and 1s
        
        Returns:
            None; updates member variables via _initialize_weights and _optimize.

        """
        self.is_fit = True

        # initialize parameters with zeros
        self._initialize_weights(X.shape[0])
        self.classes = np.unique(y)

        # use gradient descent to fit weights to training data
        self._optimize(X, y)

        # predict on training input
        preds = self.predict(X)

        # calculate training accuracy
        self._train_acc = 100 - np.mean(np.abs(preds - y)) * 100
        if self.verbose:
            print("train accuracy: {} %".format(self._train_acc))


    def predict(self, X):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
        Arguments:
            X -- data of size (number of features, number of examples)
        
        Returns:
            Y_prediction -- a numpy array (vector) containing binary predictions for examples in X
        """
        assert self.is_fit

        m = X.shape[1]  # accept one or many
        Y_prediction = np.zeros((1,m))
        w = self._w.reshape(X.shape[0], 1)
        b = self._b
        
        # compute activation probabilities
        A = sigmoid(np.dot(w.T, X) + b)
        
        for i in range(A.shape[1]):
            
            # convert sigmoid probabilities into a classification
            Y_prediction[0, i] = np.round(A[0, i])
        
        assert Y_prediction.shape == (1, m)
        
        return Y_prediction


    def _initialize_weights(self, dim):
        w = np.zeros((dim, 1))
        b = 0.
    
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
    
        self._w = w
        self._b = b


    def _optimize(self, X, y):
        costs = []
        w = self._w
        b = self._b
    
        for i in range(self.num_iterations):
            
            # calculate partials and cost
            grads, cost = self._propagate(w, b, X, y)
            dw = grads["dw"]
            db = grads["db"]
            
            # update parameters using computed gradients
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            
            # record the costs every 100 iters
            if i % 100 == 0:
                costs.append(cost)
            
            if self.verbose and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        
        # update model state
        self._w = w
        self._b = b
        self._grads = {"dw": dw,
                       "db": db}
        self._costs = costs


    def _propagate(self, w, b, X, y):
        m = X.shape[1]
    
        # forward propagation to compute cost
        # see a definition of logistic regression for
        # where this cost function comes from
        A = sigmoid(np.dot(w.T, X) + b)
        cost = (-1. / m) * np.sum(y * np.log(A) + (1-y) * np.log(1-A))
        
        # 'backpropagate' to compute gradients w.r.t. each weight (and bias)
        dw = (1. / m) * np.dot(X, (A - y).T)
        db = np.sum(A - y) / m

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)  # remove single dimension
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}
        
        return grads, cost
