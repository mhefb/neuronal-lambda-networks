import numpy as np
from core.customWeightingFunctions.FunctionBase import FunctionBase


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
        prev_activations.shape = (prev_activations.shape[0], 1)

        # (784, 10) @ (784, 1) -> (10, 1)
        return_val = layer_paras[0] @ (prev_activations ** 2)
        return_val += layer_paras[1] @ prev_activations
        return_val += layer_paras[2]
        return return_val

    def prev_layer_function(self, parameters, weighted_inputs, error_of_next_layer):
        """
        δ_l   = ∂C/∂z_l
        δ_l-1 = ∂C/∂z_l * ∂z_l/∂z_l-1
        δ_l-1 = δ_l     * ∂z_l/∂a_l-1 * ∂a_l-1/∂z_l-1

        ∂z_l/∂a_l-1 = σ′(z_l)
        ∂a_l-1/∂z_l-1 = ?

        z = b * a² + c * a + d
        z = ∑_k (b_k * a_k² + c_k * a_k) + k
        z_l-1_j/∂a_l-1_k = 2 * a_j_k * b_j_k + c_j_k
        z_l-1_j/∂a_l-1 = ∑_k (2 * a_j_k * b_j_k + c_j_k)
        z_l-1_j/∂a_l-1 = ∑ (2 ⊙ a_j ⊙ b_j + c_j)            # a_j, b_j, c_j = vector over k

        z_l-1_j/∂a_l-1 = ∑_k (2 * a_j_k * b_j_k + c_j_k)
        z_l-1_j/∂a_l-1 = ∑_k (2 ⊙ a_j ⊙ b_j) + ∑_k (c_j)    # a_j, b_j, c_j = vector over k
        z_l-1  /∂a_l-1 = ∑_k (2 ⊙ a   ⊙ b  ) + ∑_k (c)      # a, b, c = matrix over j and k

        => δ_l-1 = δ_l * z_l-1/∂a_l-1 ⊙ σ′(z_l)
        """
        # start: error_of_next_layer
        # goal: weighted_inputs
        return_val = parameters[0].T @ error_of_next_layer
        return_val *= 2
        return_val += np.sum(parameters[1], axis=1).reshape(parameters[1].shape[0], 1)

        return_val = return_val * weighted_inputs
        return return_val

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
