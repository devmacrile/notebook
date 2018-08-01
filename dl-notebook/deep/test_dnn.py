"""
import numpy as np

from dnn import DeepNeuralNetwork

nn = DeepNeuralNetwork(num_hidden_layers=3, layer_sizes=(20, 10, 5), num_iterations=1000, verbose=True)

X = np.random.randn(100, 50)
y = np.array(np.random.randint(2, size=100), dtype=float)
nn.fit(X, y.T)
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split

from dnn import DeepNeuralNetwork


def plot_decision_boundary(predict, X, y):
    # set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them,
    # and predict the value for each point in the grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # plot the contour and training points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def create_dataset():
    X, Y = sklearn.datasets.make_blobs(n_samples=200, random_state=5, n_features=2, centers=6)
    Y = Y % 2  # map the centers onto two classes
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    return X, Y


def compare_logistic_to_nn():
    X, Y = create_dataset()
    print(type(Y), Y.shape)

    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    plt.savefig('figs/test_dataset.png')

    # fit a logistic regression using sklearn,
    # compute accuracy and plot decision boundary
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)

    lr_preds = clf.predict(X.T)
    lr_accuracy = float((np.dot(Y, lr_preds) + np.dot(1-Y, 1-lr_preds)) / float(Y.size) * 100)
    print('Accuracy of logistic regression: %d ' % lr_accuracy +
          '% ' + "(percentage of correctly labelled datapoints)")

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression (%d" % lr_accuracy + "%" + " accuracy)")

    # fit our hand-rolled neural network and plot decision boundary
    layer_sizes = (25, 10, 5)
    nn = DeepNeuralNetwork(num_hidden_layers=3, 
                           layer_sizes=layer_sizes, 
                           num_iterations=2500, 
                           learning_rate=0.01,
                           verbose=True)

    X = X.T
    #Y = np.squeeze(Y)
    print(X.shape, Y.shape)
    print(X)
    nn.fit(X, Y.T)
    nn_preds = nn.predict(X)
    print(nn_preds.shape, Y.shape)
    nn_accuracy = float((np.dot(Y, nn_preds.T) + np.dot(1-Y, 1-nn_preds.T)) / float(Y.size) * 100)
    print ('Accuracy of neural network: %d ' % nn_accuracy +
           '% ' + "(percentage of correctly labelled datapoints)")

    plt.subplot(122)
    plot_decision_boundary(lambda x: nn.predict(x), X.T, Y)
    plt.title("Deep Network (Hidden-layer sizes = {%s})" % ', '.join(map(str, layer_sizes)))
    plt.savefig('figs/logistic_vs_nn_performance.png')


def classify_digits():
    X, y = sklearn.datasets.load_digits(return_X_y=True)

    # filter digits to be in {0, 1} as our implementation is binary
    digit_one = 2
    digit_two = 5
    mask = np.logical_or(y == digit_one, y == digit_two)
    X = X[mask]
    y = y[mask]
    y[y == digit_one] = 0
    y[y == digit_two] = 1

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train_flat = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flat = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    X_train = X_train_flat / 255.
    X_test = X_test_flat / 255.

    layer_sizes = (25, 10, 5)
    nn = DeepNeuralNetwork(num_hidden_layers=3, 
                           layer_sizes=layer_sizes, 
                           # num_iterations=2500,
                           learning_rate=0.01,
                           verbose=True)

    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))
    print(X_train.shape, y_train.shape)
    print(X_train)
    nn.fit(X_train, y_train.T)
    nn_preds = nn.predict(X_train)
    print(nn_preds.shape, y_train.shape)
    nn_accuracy = float((np.dot(y_train, nn_preds.T) + np.dot(1-y_train, 1-nn_preds.T)) / float(y_train.size) * 100)
    print ('Accuracy of neural network: %d ' % nn_accuracy +
           '% ' + "(percentage of correctly labelled datapoints)")



if __name__ == '__main__':
    # compare_logistic_to_nn()
    classify_digits()
    print('See figs/ directory for output.')
