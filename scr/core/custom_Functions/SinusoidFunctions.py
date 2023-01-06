import numpy as np
from scr.core.custom_Functions.FunctionBase import FunctionsBase

"""
Function for a connection:
z = sin(x * f) * A + b
z; weighted input
f ≈ frequency;  0. Weight
A ≈ Amplitude;  1. Weight
b ≈ bias;       2. Weight
"""


class SinusoidFunctions(FunctionsBase):
    no_params = 3

    def weighting(self, layer_paras: list[np.ndarray], prev_activations):
        """
        neuron-wise: z = sin(x * f) * A + b
        => z_l_j = sum (sin(a_l-1 * w[0]) * w[1]) + w[2]

        figured out with Erik   (´▽`ʃ♡ƪ)

        :param layer_paras: parameters of the current layer
        :param prev_activations: activations of the layer before
        :return: z - the weighted inputs
        """
        return_val = np.ndarray(shape=(len(layer_paras[2]), 1))

        for i in range(len(layer_paras[2])):
            return_val[i] = layer_paras[2][i] + np.sum(
                np.multiply(
                    layer_paras[1][i],
                    np.sin(
                        np.multiply(
                            layer_paras[0][i],
                            prev_activations
                        )
                    )
                )
            )
        return return_val

    # TODO: test prime_weighting functions
    def prime_weighting_0(self, error_of_layer, activations_of_previous_layer, current_parameters):
        """
        ∂z_l/∂f_l_j_k = A_l_j_k * a_l-1_k * cos (f_l_j_k * a_l-1_k)
        => ∂z_l/∂f_l_j = A_l_j * a_l-1 * cos (f_l_j * a_l-1)

        :param error_of_layer: δ_lj
        :param activations_of_previous_layer: a_l−1
        :param current_parameters:
        :return: ∂C/∂w_ljk
        """
        return_val = np.ndarray(shape=current_parameters[0].shape)

        for j in range(current_parameters[0].shape[0]):

            return_val[j] = np.multiply(current_parameters[0][j], activations_of_previous_layer)
            return_val[j] = np.sin(return_val[j])
            return_val[j] = np.multiply(current_parameters[1][j], return_val[j])
            return_val[j] = np.multiply(activations_of_previous_layer, return_val[j])
            return_val[j] = np.multiply(error_of_layer[j], return_val[j])

        return return_val

    def prime_weighting_1(self, error_of_layer, activation_of_previous_layer, current_parameters):
        """
        ∂z_l/∂A_l_j_k = δ_l_j * sin(a_l-1_k * f_l_j_k)

        :param error_of_layer: δ_l
        :param activation_of_previous_layer: a_l−1_k
        :param current_parameters:
        :return: ∂C/∂w_l_j_k
        """
        return_val = np.ndarray(shape=current_parameters[1].shape)

        for j in range(current_parameters[1].shape[0]):
            for k in range(current_parameters[1].shape[1]):

                return_val[j][k] = np.multiply(activation_of_previous_layer[k], current_parameters[0][j][k])
                return_val[j][k] = np.sin(return_val[j][k])
                return_val[j][k] = np.multiply(error_of_layer[j], return_val[j][k])

        return return_val

    def prime_weighting_2(self, error_of_layer, activation_of_previous_layer, current_parameters):
        """
        ∂C/∂b=δ_l

        :param error_of_layer: δ_l
        :param activation_of_previous_layer: not used here
        :param current_parameters: not used here
        :return: ∂C/∂b
        """
        return error_of_layer

    def generate_weights(self, structure: list[int]):
        paras = []
        for k in range(len(structure)):
            if k != 0:
                t_paras = [np.random.randn(structure[k], structure[k - 1]),
                           np.random.randn(structure[k], structure[k - 1]),
                           np.random.randn(structure[k], 1)]
                paras.append(t_paras)
        return paras

    # TODO: setup prev_layer_function
    def prev_layer_function(self, parameters, weighted_inputs, error_of_next_layer):
        """
        ∂z_l+1/∂z_l = ?

        :param parameters:
        :param weighted_inputs:
        :param error_of_next_layer:
        :return:
        """
        pass

    def activation_function(self, z: np.ndarray):
        """Sigmoid function 1 / (1 + e^-z)"""
        return 1.0 / (1.0 + np.exp(-z))

    def prime_activation_function(self, z: np.ndarray):
        """Derivative of the sigmoid function"""
        return np.exp(z) / (1 + np.exp(z)) ** 2

    def cost_function(self, expected_output, last_activation):
        """The quadratic cost function
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
