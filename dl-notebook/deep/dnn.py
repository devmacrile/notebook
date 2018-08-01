"""
Arbitrarily 'deep' neural network implementation.
"""


import numpy as np 


def sigmoid(z):
    """
    Compute the sigmoid of z.

    Arguments:
        x -- A scalar or numpy array of any size.
    """
    return (np.exp(z) / (1 + np.exp(z)))


def relu(z):
    """
    Compute the ReLU (rectified linear unit) of z.

    Arguments:
        z --  scalar value or a numpy array
    """
    return np.maximum(0, z)


def relu_backward(dA, cache):
    """
    Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


class DeepNeuralNetwork(object):

    def __init__(self, num_hidden_layers, layer_sizes, num_iterations=10000, 
                       learning_rate=0.01, seed=False, verbose=False):
        if not isinstance(layer_sizes, list):
            try:
                layer_sizes = list(layer_sizes)
            except:
                raise ValueError("'layer_sizes' must be a sequence (received %s)" % type(layer_sizes))
        
        assert(num_hidden_layers == len(layer_sizes))

        self.nh = num_hidden_layers
        self.layer_sizes = layer_sizes
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

        # set via call to fit
        self.X = None
        self.y = None
        self.is_fit = False

    
    def fit(self, X, y):
        """
        Arguments:
            X -- ndarray of input data, intended shape should be (# examples, size of vector)
            Y -- ndarray of actuals, each value must be one of {0, 1}

        Modifies:
            costs
            parameters

        """
        assert(X.shape[0] == y.shape[0])

        self.X = X.T
        self.y = y

        costs = []

        nx, ny = X.shape[1], 1
        self._initialize_parameters(nx, ny)

        for i in range(self.num_iterations):
            # forward propagate, and calculate cost
            AL, caches = self._forward_propagate(X)
            cost = self._compute_cost(AL)

            # backward propagate error and update weights
            grads = self._backward_propagate(AL, caches)
            self._update_parameters(grads)

            if self.verbose and (i % 100) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

            if i % 100 == 0:
                costs.append(cost)

        self.costs = costs
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
        A, _ = self._forward_propagate(X)
        predictions = np.round(A)  # simple 0.5 cutoff
        return np.squeeze(predictions)


    def _initialize_parameters(self, nx, ny, seed=np.random.seed(42)):
        """
        Modifies:
            parameters
        """
        parameters = {}

        dims = [nx] + self.layer_sizes + [ny]
        L = len(dims)

        for l in range(1, L):
            # TODO move these keys to accessors
            wkey = 'W' + str(l)  # weights matrix
            bkey = 'b' + str(l)  # bias vector
            parameters[wkey] = np.random.randn(dims[l], dims[l-1]) / np.sqrt(dims[l-1]) #* 0.01
            parameters[bkey] = np.zeros((dims[l], 1), dtype=float)
            
        self.parameters = parameters


    def _update_parameters(self, grads):
        """
        Modifies:
            parameters
        """
        L = len(self.parameters) // 2
        for l in range(L):
            # TODO move these keys to accessors
            wkey = 'W' + str(l + 1)
            bkey = 'b' + str(l + 1)
            dwkey = 'dW' + str(l + 1)
            dbkey = 'db' + str(l + 1)

            self.parameters[wkey] = self.parameters[wkey] - self.learning_rate * grads[dwkey]
            self.parameters[bkey] = self.parameters[bkey] - self.learning_rate * grads[dbkey]


    def _forward_propagate(self, X):

        A_prev = X.T
        caches = []

        L = len(self.parameters) // 2
        for l in range(1, L):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A, cache = self._linear_activation_forward(A_prev, W, b)

            A_prev = A
            caches.append(cache)

        # final output activation; sigmoid instead of relu
        AL, cache = self._linear_activation_forward(A_prev, 
                                                    self.parameters['W' + str(L)], 
                                                    self.parameters['b' + str(L)], 
                                                    activation="sigmoid")
        caches.append(cache)

        return AL, caches


    def _linear_activation_forward(self, A_prev, W, b, activation="relu"):

        Z = np.dot(W, A_prev)

        if activation == "sigmoid":
            A = sigmoid(Z + b)
        elif activation == "relu":
            A = relu(Z + b)

        linear_cache = (A_prev, W, b)
        activation_cache = (Z + b)
        cache = (linear_cache, activation_cache)

        return A, cache


    def _compute_cost(self, AL):
        actuals = np.reshape(self.y, AL.shape)
        m = actuals.shape[1]
        cost = - (1. / m) * np.sum(actuals * np.log(AL) + (1. - actuals) * np.log(1. - AL))
        cost = np.squeeze(cost)  # remove empty dimensions, e.g. [[x]] -> x
        assert(cost.shape == ())
        return cost


    def _linear_backward(self, dA, cache, activation="relu"):

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1. / m) * np.dot(dZ, A_prev.T)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    def _backward_propagate(self, AL, caches):
        grads = {}
        L = len(caches)
        actuals = self.y.reshape(AL.shape)
        m = AL.shape[1]

        # initial gradient computation
        dAL = - (np.divide(actuals, AL) - np.divide(1 - actuals, 1 - AL))

        current_cache = caches[L - 1]
        appl = lambda x: x + str(L)
        grads[appl("dA")], grads[appl("dW")], grads[appl("db")] = self._linear_backward(dAL, current_cache, activation="sigmoid")

        for l in reversed(range(1, L)):
            current_cache = caches[l - 1]
            dakey = 'dA' + str(l + 1)
            dA_prev_temp, dW_temp, db_temp = self._linear_backward(grads[dakey], current_cache, activation="relu")

            # store gradient computations using layer index
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp

        return grads

