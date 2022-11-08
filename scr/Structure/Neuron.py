import scr.Config as Cfg
from scr.Equation import Equation
import numpy as np


class Neuron:
    weights: np.ndarray
    connection_function: Equation
    activation_function: Equation
    weighted_inputs: [float]
    activation: float

    def __init__(self, no_inputs: int):
        self.weights = np.ndarray(shape=(no_inputs, Cfg.wpf))
        for i in range(no_inputs):
            for j in range(Cfg.wpf):
                self.weights[i][j] = Cfg.initial_weight()

        self.connection_function = Cfg.connection_function
        self.activation_function = Cfg.sigmoid
        self.weighted_inputs = []

    def feedforward(self, inputs: [float]):
        summed = 0
        for inp in inputs:
            self.weighted_inputs.append(self.connection_function(self.weights, inp))
            summed += self.weighted_inputs[-1]
        self.activation = self.activation_function(summed)
        return self.activation
