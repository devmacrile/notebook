import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model

from nn import ShallowNeuralNetwork


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
    num_neurons = 25
    nn = ShallowNeuralNetwork(layer_size=num_neurons, 
                              num_iterations=5000, 
                              learning_rate=0.01, 
                              verbose=True)
    nn.fit(X, Y)
    nn_preds = nn.predict(X)
    nn_accuracy = float((np.dot(Y, nn_preds.T) + np.dot(1-Y, 1-nn_preds.T)) / float(Y.size) * 100)
    print ('Accuracy of neural network: %d ' % nn_accuracy +
           '% ' + "(percentage of correctly labelled datapoints)")

    plt.subplot(122)
    plot_decision_boundary(lambda x: nn.predict(x.T), X, Y)
    plt.title("Single Hidden-Layer Network (Neurons = %i)" % (num_neurons))
    plt.savefig('figs/logistic_vs_nn_performance.png')


def compare_layer_sizes(hidden_layer_sizes):
    X, Y = create_dataset()
    plt.figure(figsize=(16, 32))
    for i, n_h in enumerate(hidden_layer_sizes):
        # fit nn with neuron count
        nn = ShallowNeuralNetwork(layer_size=n_h, num_iterations=5000, learning_rate=0.01)
        nn.fit(X, Y)

        predictions = nn.predict(X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)) / float(Y.size) * 100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        
        # plot decision boundary
        plt.subplot(3, 2, i+1)
        plt.title('Hidden Layer of size %d (Accuracy = %d)' % (n_h, accuracy / 100))
        plot_decision_boundary(lambda x: nn.predict(x.T), X, Y)

    plt.savefig('figs/decision_boundary_by_layer_size.png')

if __name__ == '__main__':
    compare_logistic_to_nn()
    compare_layer_sizes([1, 2, 5, 10, 25, 50])
    print('See figs/ directory for output.')
