"""
Example model utilizing the logistic regression implementation
in logistic_nn.py using a dataset from sklearn.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from logistic_nn import LogisticNN


# load the digits dataset, and split into 
# training and testing chunks using sklearn utilities
digits = load_digits()
X = digits.data
y = digits.target

# filter digits to be in {0, 1} as our implementation is binary
digit_one = 3
digit_two = 8
mask = np.logical_or(y == digit_one, y == digit_two)
X = X[mask]
y = y[mask]
y[y == digit_one] = 0
y[y == digit_two] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# fit our model to training data
clf = LogisticNN(num_iterations=1000, learning_rate=0.001, verbose=True)
clf.fit(X_train.T, y_train.T)

# compute test accuracy
preds = clf.predict(X_test.T)
test_accuracy = 100 - np.mean(np.abs(preds - y_test)) * 100
#print("train accuracy: {} %".format(clf._train_acc))
print("test accuracy: {} %".format(test_accuracy))

# graph "cost" calculation by iteration
plt.plot(range(len(clf._costs)), clf._costs)
plt.xlabel('Iteration (100s)')
plt.ylabel('Cost')
plt.savefig('figs/lr_cost_optimization.png')
plt.show()
