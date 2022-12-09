import numpy as np

from core.Network import network

test = network(2, [4, 2], 6)
# inputs = (np.zeros(shape=(2, 1)), 0), ([1, 1], 2)
inputs = np.ones(shape=(2, 1))
# print('feedforward test:\n', test.feedforward(inputs), '\n\n')
print('cost before training', test.funcs.cost_function([2], test.feedforward(inputs)))

inputs = [[np.ones(shape=(2, 1)), [2]]]
print('train test:\n', test.train(inputs), '\n\n')

inputs = np.ones(shape=(2, 1))
print('cost after training', test.funcs.cost_function([2], test.feedforward(inputs)))


"""

from core.custom_Functions.StandardFunctions import StandardFunctions

tete = StandardFunctions()
print(tete.paras_influence_on_weighted_input)
print(tete.paras_influence_on_weighted_input[0])
"""
