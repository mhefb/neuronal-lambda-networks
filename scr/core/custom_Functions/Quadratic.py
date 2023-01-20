import numpy as np
from scr.core.custom_Functions.FunctionBase import FunctionBase


"""
z = ∑ (a * x² + b * x + c)
"""


class QuadraticFunction(FunctionBase):
    no_params = 3

    def generate_weights(self, structure: list[int]):
        paras = []
        for k in range(len(structure)):
            if k != 0:
                t_paras = [np.random.randn(structure[k], structure[k - 1] if k > 0 else 1),
                           np.random.randn(structure[k], structure[k - 1] if k > 0 else 1),
                           np.random.randn(structure[k], 1)]
                paras.append(t_paras)
        return paras

    def weighting(self, layer_paras: list[np.ndarray], prev_activations: np.ndarray):
        return_val = layer_paras[0].T @ (prev_activations ** 2) + layer_paras[1].T @ prev_activations + layer_paras[2]
        return return_val

    def prev_layer_function(self, parameters, weighted_inputs, error_of_next_layer):
        # TODO: maybe not transpose parameters[0]
        return (parameters[0].T * weighted_inputs * 2 + parameters[1]) @ error_of_next_layer

    def prime_weighting_0(self, error_of_layer, activation_of_previous_layer, current_parameters):
        activation_of_previous_layer.shape = (len(activation_of_previous_layer), 1)
        activation_of_previous_layer = activation_of_previous_layer ** 2

        gradient = np.dot(error_of_layer, activation_of_previous_layer.transpose())
        return gradient

    def prime_weighting_1(self, error_of_layer, activation_of_previous_layer, current_parameters):
        activation_of_previous_layer.shape = (len(activation_of_previous_layer), 1)

        gradient = np.dot(error_of_layer, activation_of_previous_layer.transpose())
        return gradient

    def prime_weighting_2(self, error_of_layer, activation_of_previous_layer, current_parameters):
        return error_of_layer

    def activation_function(self, z: np.ndarray):
        """Sigmoid function 1 / (1 + e^-z)"""
        return 1.0 / (1.0 + np.exp(-z))

    def prime_activation_function(self, z: np.ndarray):
        """Derivative of the sigmoid function"""
        return np.exp(z) / (1 + np.exp(z)) ** 2

    def cost_function(self, expected_output, last_activation):
        """The standard quadratic cost function
        f(x) = 1/2*n*∑(|y - a|²)"""
        return_val = 0
        for (i, j) in zip(expected_output, last_activation):  # Sums the squared difference
            return_val = (j - i) ** 2
        return_val /= 2 * len(expected_output)  # makes it to half the average cost
        return return_val

    def prime_cost_function(self, expected_output, network_output):
        """'Derivative' of the cost function
        f'(x) = a - y;  needs to be divided by the number of training samples"""
        return network_output - expected_output
