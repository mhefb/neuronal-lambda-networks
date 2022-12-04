import numpy as np
from scr.core.custom_Functions import FunctionBase
from scr.core.custom_Functions.StandardFunctions import StandardFunctions


class network:
    structure: list[int]
    paras: list
    prime_activation_function: callable
    prev_layer_function: callable
    paras_influence_on_weighted_input: list[callable]
    funcs: FunctionBase

    def __init__(self, inputs: int, hidden_layers: [int], outputs: int, functions=StandardFunctions()):
        self.structure = hidden_layers
        self.structure.insert(0, inputs)
        self.structure.append(outputs)

        self.funcs = functions
        self.paras = self.funcs.generate_weights(self.structure)

    def feedforward(self, activation: np.array):
        for layer in self.paras:
            activation = self.funcs.activation_function(self.funcs.weighting(layer, activation))
        return activation

    def train(self, training_data: list[np.array, int]):  # , validation_data, learning_rate):
        # activations = list
        weighted_inputs = []
        for (inp, exp_out) in training_data:
            activations = [inp]

            # forward pass
            for layer in self.paras:
                weighted_inp = self.funcs.weighting(layer, activations[-1])
                weighted_inputs.append(weighted_inp)
                activations.append(self.funcs.activation_function(weighted_inp))

            print('temp end')

            # list of error produces by each layer of the network
            # initialised with the values for the fist layer
            errors = [np.dot(self.funcs.prime_cost_function(exp_out, activations[-1]),
                             self.funcs.prime_activation_function(weighted_inputs[-1]))]

            # list for gradient of each parameter of the network
            # initialised with the values for the fist layer
            params_gradient = [[p_inf(errors[-1], activations[-1])
                                for p_inf in self.funcs.paras_influence_on_weighted_input]]

            # backward pass
            for i in range(len(self.paras), 0, -1):

                errors.append(self.funcs.prev_layer_function(self.paras[i], weighted_inputs[i], errors[-1]))

                params_gradient.append(
                    [p_inf(errors[-1], activations[i]) for p_inf in self.paras_influence_on_weighted_input])
