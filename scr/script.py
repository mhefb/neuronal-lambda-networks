"""from Controler import Controller

control = Controller(3, [2], 1)

control.feedforward([1, 2, 3])
print(control.neural_network.activations)
control.train(1, [[[1, 1, 1], [3]]])"""
import numpy as np

"""import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

import Coppied_Code
net = Coppied_Code.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""


from take2.Network import network

test = network(2, [4, 2], 6)
#inputs = (np.zeros(shape=(2, 1)), 0), ([1, 1], 2)
inputs = np.ones(shape=(2, 1))
print('feedforward test:\n', test.feedforward(inputs), '\n\n')
inputs = [[np.ones(shape=(2, 1)), [2]]]
print('train test:\n', test.train(inputs), '\n\n')
