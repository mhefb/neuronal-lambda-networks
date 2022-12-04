import numpy as np
from scr.core.custom_Functions.FunctionBase import FunctionsBase


class StandardFunctions(FunctionsBase):

    def weighting(self, layer_paras: list[np.ndarray], prev_activations):
        """
        The standard: f(x)= weight * inp + bias

        :param layer_paras: weights and biases
        :param prev_activations: activations of the layer before
        :return: z - the weighted inputs
        """
        if layer_paras[0].shape[1] != prev_activations.shape[0]:  # check if the inputs are in the correct shapes
            print('not able to do the weighting, prev_activations and layer_paras do not have matching shapes')
            raise Exception('prev_activations ', layer_paras[0].shape, ' and layer_paras', prev_activations.shape,
                            ' do not have matching shapes')

        result = np.dot(layer_paras[0], prev_activations)  # multiplication by the weights
        result = result + layer_paras[1]  # addition of the biases
        return result

    def generate_weights(self, structure: list[int]):
        """
        Generates the all weights for the neural network

        :param structure: list of the desired number of neurons for each layer
        :return: All weights for the neural network
        """
        paras = []
        for k in range(len(structure)):
            if k != 0:
                t_paras = [np.random.randn(structure[k], structure[k - 1] if k > 0 else 1),
                           np.random.randn(structure[k], 1)]
                paras.append(t_paras)
        return paras

    def activation_function(self, z: np.ndarray):
        """Sigmoid function"""
        return 1.0 / (1.0 + np.exp(-z))

    def prime_activation_function(self, z: np.ndarray):
        """Derivative of the sigmoid function"""
        return np.exp(z) / (1 + np.exp(z)) ** 2

    def cost_function(self, expected_output, last_activation):
        """The standard quadratic cost function
        f(x) = 1/2*n*∑(|y - a|²)"""
        return_val = 0
        for (i, j) in zip(expected_output, last_activation):  # Sums the squared difference
            return_val = (j - i) ** (j - i)
        return_val /= 2 * len(expected_output)  # makes it to half the average cost
        return return_val

    def prime_cost_function(self, expected_output, last_activation):
        """'Derivative' of the cost function
        f'(x) = a - y;  needs to be divided by the number of training samples"""
        return last_activation - expected_output
