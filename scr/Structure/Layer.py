from scr.Structure.Neuron import Neuron
import numpy as np


class Layer:
    size: int
    neurons: [Neuron]

    def __init__(self, no_neurons, no_inputs):
        self.size = no_neurons

        self.neurons = []
        for i in range(no_neurons):
            self.neurons.append(Neuron(no_inputs))

    def feedforward(self, inputs: [float]):
        # check for matching sizes
        if self.size != len(inputs):
            raise Exception("layer.size = " + str(self.size) + ", inputs = " + str(inputs))

        # calculates output for each neuron
        outputs = [inputs, []]
        for i in range(len(self.neurons)):
            outputs[-1].append(self.neurons[i].feedforward(outputs[0]))
        return outputs[-1]

    def get_weights(self):
        t_array = np.ndarray(shape=self.size)
        for i in range(len(self.neurons)):
            t_array[i] = self.neurons[i].weights
        return t_array
