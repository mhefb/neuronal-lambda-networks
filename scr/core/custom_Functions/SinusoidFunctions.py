import numpy as np
from core.custom_Functions.FunctionBase import FunctionBase

"""
Function for a connection:
z = sin(x * f) * A + b
z; weighted input
f ≈ frequency;  0. Weight
A ≈ Amplitude;  1. Weight
b ≈ bias;       2. Weight
"""


class SinusoidFunction(FunctionBase):
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
        => ∂z_l/∂f_l_j = A_l_j * a_l-1 * cos (f_l_j * a_l-1) (?)

        :param error_of_layer: δ_l
        :param activations_of_previous_layer: a_l−1
        :param current_parameters:
        :return: ∂C/∂w_l
        """
        '''#return_val = np.ndarray(shape=current_parameters[0].shape)
        return_val = []

        for j in range(current_parameters[0].shape[0]):

            # ∂z_l/∂f_l_j = cos(f_l_j * a_l-1) * A_l_j * a_l-1
            #return_val[j] = np.multiply(current_parameters[0][j], activations_of_previous_layer)
            return_val.append(np.multiply(current_parameters[0][j], activations_of_previous_layer))
            return_val[j] = np.cos(return_val[j])
            return_val[j] = np.multiply(current_parameters[1][j], return_val[j])
            return_val[j] = np.multiply(activations_of_previous_layer, return_val[j])
            return_val[j] = np.multiply(error_of_layer[j], return_val[j])

        return_val = np.asarray(return_val)'''
        # ∂z_l/∂f_l_j = A_l * cos(f_l * a_l-1.T) * a_l-1.T
        return_val = current_parameters[1] * np.cos(current_parameters[0] * activations_of_previous_layer.T) * \
                     activations_of_previous_layer.T

        return_val = error_of_layer * return_val

        if return_val.shape != current_parameters[0].shape:
            raise Exception
        return return_val

    def prime_weighting_1(self, error_of_layer, activation_of_previous_layer, current_parameters):
        """
        ∂z_l/∂A_l_j_k = sin(a_l-1_k * f_l_j_k)
        => ∂z_l/∂A_l_j = sin(a_l-1 * f_l_j) (?)

        ∂C/∂A_l_j = δ_l_j * sin(a_l-1 * f_l_j) (?)

        :param error_of_layer: δ_l
        :param activation_of_previous_layer: a_l−1
        :param current_parameters:
        :return: ∂C/∂w_l
        """
        return_val = np.ndarray(shape=current_parameters[1].shape)

        for j in range(current_parameters[1].shape[0]):
            for k in range(current_parameters[1].shape[1]):
                return_val[j][k] = np.multiply(activation_of_previous_layer[k], current_parameters[0][j][k])
                return_val[j][k] = np.sin(return_val[j][k])
                return_val[j][k] = np.multiply(error_of_layer[j], return_val[j][k])

        if return_val.shape != current_parameters[1].shape:
            raise Exception
        return return_val

    def prime_weighting_2(self, error_of_layer, activation_of_previous_layer, current_parameters):
        """
        ∂z_l/∂b_l_j_k = 1
        => ∂z_l/∂b = 1

        ∂C/∂b = δ_l * 1

        :param error_of_layer: δ_l
        :param activation_of_previous_layer: not used here
        :param current_parameters: not used here
        :return: ∂C/∂b
        """
        if error_of_layer.shape != current_parameters[2].shape:
            raise Exception
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
        δ_l   = ∂C/∂z_l
        δ_l   = ∂C/∂z_l * ∂z_l/∂z_l-1

        δ_l-1 = δ_l * ∂z_l/∂a_l-1 * ∂a_l-1/∂z_l-1

        TODO: ∂z_l_j_k?
        ∂z_l/∂a_l-1       = A_l_j_k * f_l_j_k * cos(f_l_j_k * a_l-1_k)
        ∂z_l/∂z_l-1_j     = SUM ( A_l_j_k * f_l_j_k * σ'(z_l-1_k) * cos(f_l_j_k * σ(z_l-1_k)) )
        => ∂z_l/∂z_l-1_j  = A_l_j * f_l_j * σ'(z_l-1) *cos(f_l_j * σ(z_l-1))

        => ∂z_l/∂z_l-1    = A_l * f_l * cos(f_l * σ(z_l-1))

        :param parameters:
        :param weighted_inputs:
        :param error_of_next_layer:
        :return: ∂z_l/∂z_l-1
        """
        '''return_val = []

        for i in range(parameters[2].shape[0]):
            return_val.append(np.multiply(parameters[0][i], self.activation_function(weighted_inputs)))
            return_val[i] = np.cos(return_val[i])
            return_val[i] = np.multiply(return_val[i], parameters[0][i])
            return_val[i] = np.multiply(return_val[i], parameters[1][i])
            return_val[i] = np.multiply(return_val[i], self.prime_activation_function(weighted_inputs))

        return_val = np.asarray(return_val)'''

        # δ_l = δ_l+1 ⊙ A_l+1.T ⊙ f_l+1.T ⊙ cos(f_l+1.T ⊙ σ(z_l)) * (σ'(z_l))²
        return_val = parameters[0].T * self.activation_function(weighted_inputs)
        return_val = np.cos(return_val)
        return_val = return_val * parameters[0].T
        return_val = return_val * parameters[1].T
        return_val = return_val @ error_of_next_layer
        return_val = return_val * (self.prime_activation_function(weighted_inputs)) ** 2

        return return_val

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
