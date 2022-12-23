import numpy as np
from keras.datasets import mnist

from core.Network import network

# loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X[:500]
train_y = train_y[:500]

# printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

data = np.ndarray(shape=(len(train_X), 784))
for i in range(len(train_X)):
    t_data = []
    for y_cord in train_X[i]:
        t_data.extend(y_cord/255)
    data[i] = t_data

correct_results = []
for i in range(len(train_y)):
    t_data = np.zeros(shape=(10, 1))
    t_data[train_y[i]] = 1
    correct_results.append(t_data)

test = network(784, [15], 10)
# inputs = (np.zeros(shape=(2, 1)), 0), ([1, 1], 2)
inputs = data[400]
# print('feedforward test:\n', test.feedforward(inputs), '\n\n')
print('cost before training', test.funcs.cost_function(correct_results[0], test.feedforward(inputs)))

inputs = [i for i in zip(data[:300], correct_results[:300])]
test.train(inputs, learning_rate=3)

inputs = data[400]
print('cost after training', test.funcs.cost_function(correct_results[0], test.feedforward(inputs)))


"""

from core.custom_Functions.StandardFunctions import StandardFunctions

tete = StandardFunctions()
print(tete.paras_influence_on_weighted_input)
print(tete.paras_influence_on_weighted_input[0])
"""
