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
        # Set functions
        self.funcs = functions

        # Generate structure
        self.structure = hidden_layers
        self.structure.insert(0, inputs)
        self.structure.append(outputs)

        self.paras = self.funcs.generate_weights(self.structure)

    def feedforward(self, activation: np.array):
        # For each layer:
        for layer in self.paras:
            # activations_layer = activation_function(
            #   weighting_function(parameters_layer, activations_layer-1) )
            activation = self.funcs.activation_function(self.funcs.weighting(layer, activation))
        return activation

    def train(self, training_data: list[(np.array, int)], learning_rate=0.01):  # , validation_data):
        # list of values before the activation_function
        weighted_inputs = []

        # TODO: Training in batches
        for (inp, exp_out) in training_data:
            # 2D-list of the activation of neurons for each layer
            activations = [inp]

            # forward pass
            for layer in self.paras:
                # weighted_input = weighting_function(parameters_layer, activations_layer-1)
                weighted_inp = self.funcs.weighting(layer, activations[-1])
                weighted_inputs.append(weighted_inp)

                # activations_layer = activation_function(weighted_input)
                activations.append(self.funcs.activation_function(weighted_inp))

            # 2D-list of error produces by each neuron of the network
            # initialised with the values for the fist layer
            errors = [self.funcs.prime_cost_function(exp_out, activations[-1])
                      * self.funcs.prime_activation_function(weighted_inputs[-1])]

            # 2D-list for gradient of each parameter of the network
            # initialised with the values for the fist layer
            params_gradient = [[[p_inf(errors[0], activations[-1])
                                for p_inf in self.funcs.paras_influence_on_weighted_input]]]

            # backward pass (from last to first layer)
            for i in range(len(self.paras) - 1, 1, -1):
                # calculating the error produced from the neurons of layer i
                errors.append(self.funcs.prev_layer_function(self.paras[i], weighted_inputs[i - 1], errors[-1]))

                # temporary list of parameter gradients of layer i
                t_params_gradient = []

                # new_i counts upwards (from 0 to last layer)
                new_i = (i - len(self.paras) + 1) * -1

                # calculating the gradient for each parameter
                for p_inf in self.funcs.paras_influence_on_weighted_input:
                    t_params_gradient.append(p_inf(errors[new_i], activations[i]))
                params_gradient.append(t_params_gradient)

                # params_gradient.append([p_inf(errors[new_i], activations[i])
                #                        for p_inf in self.funcs.paras_influence_on_weighted_input])

                # instantly applies a change to the parameters to minimise the cost
                for j in range(self.funcs.no_params_needed):
                    # Δ_cost = - learning_rate * (Δ_cost/Δ_parameters)²
                    # ⇒ Δ_parameters = - learning_rate * Δ_cost/Δ_parameters
                    d_paras = -learning_rate * params_gradient[-1][j]
                    self.paras[i][j] += d_paras
