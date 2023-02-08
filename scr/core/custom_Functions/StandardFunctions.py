import numpy as np
from core.custom_Functions.FunctionBase import FunctionBase


class StandardFunction(FunctionBase):
    no_params = 2

    def weighting(self, layer_paras: list[np.ndarray], prev_activations):
        """
        The standard: f(x)= weight * inp + bias

        :param layer_paras: weights and biases
        :param prev_activations: activations of the layer before
        :return: z - the weighted inputs
        """

        result = np.dot(layer_paras[0], prev_activations)  # multiplication by the weights
        result.shape = (result.shape[0], 1)
        result = result + layer_paras[1]  # addition of the biases
        return result

    def prime_weighting_0(self, error_of_layer, activation_of_previous_layer, current_parameters):
        """
        gradient for first parameter
        ∂C/∂w_ljk = a_l−1k δ_lj

        :param current_parameters: not used here
        :param error_of_layer: δ_lj
        :param activation_of_previous_layer: a_l−1k
        :return: ∂C/∂w_ljk
        """

        activation_of_previous_layer.shape = (len(activation_of_previous_layer), 1)

        gradient = np.dot(error_of_layer, activation_of_previous_layer.transpose())
        return gradient

    def prime_weighting_1(self, error_of_layer, activation_of_previous_layer, current_parameters):
        """
        gradient for the second parameter
        ∂C/∂b=δ

        :param current_parameters: not used here
        :param error_of_layer: δ
        :param activation_of_previous_layer: not used here
        :return: ∂C/∂b
        """
        return error_of_layer

    def generate_weights(self, structure: list[int]):
        """
        Generates the all weights for the neural network
        Here it is a list per layer for each parameter

        list(list(np.ndarray))

        :param structure: list of the desired number of neurons for each layer
        :return: All parameters for the neural network
        """
        paras = []
        for k in range(len(structure)):
            if k != 0:
                t_paras = [np.random.randn(structure[k], structure[k - 1] if k > 0 else 1),
                           np.random.randn(structure[k], 1)]
                paras.append(t_paras)
        return paras

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

    def prev_layer_function(self, parameters_of_next_layer, weighted_inputs_of_current_layer, error_of_next_layer):
        """
        δ_l = ((w_l+1)^T δ_l+1) ⊙ σ′(z_l)

        :param parameters_of_next_layer: w_l+1
        :param weighted_inputs_of_current_layer: z_l
        :param error_of_next_layer: δ_l+1
        :return: δ_l
        """

        # (w_l+1)^T δ_l+1
        return_val = np.dot(parameters_of_next_layer[0].transpose(), error_of_next_layer)
        # ⊙ σ′(z_l)
        return_val = np.multiply(return_val, self.prime_activation_function(weighted_inputs_of_current_layer))
        return return_val
