import scr.Config as Cfg
from scr.Equation import Equation
from scr.Structure.Layer import Layer
import numpy as np


class Network:
    sizes: [int]
    layers: [Layer]
    weighted_inputs: np.ndarray
    activations: np.ndarray

    def __init__(self, sizes):
        self.sizes = sizes

        self.layers = []
        for i in range(1, len(sizes)):
            self.layers.append(Layer(sizes[i], sizes[i - 1]))

    def feedforward(self, data_input: [float]):
        # check for matching sizes
        if self.sizes[0] != len(data_input):
            raise Exception("layer.size = " + str(self.sizes) + ", inputs = " + str(data_input))

        # calculates output for each layer
        outputs = [data_input, []]
        for i in range(len(self.layers)):
            outputs[-1].append(self.layers[i].feedforward(outputs[0]))
        return outputs


