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

    def prime_weighting_0(self, error_of_layer, activations_of_previous_layer, parameters):
        """
        Linear Weighting-Function:
        ∂C/∂w_l_j_k = a_l-1_k * δ_l_j
            = a_l-1_k * ∂C/∂z_l_j
            = ∂z_l_j/∂w_l_j_k * ∂C/∂z_l_j

        Sinusoid Weighting-Function:
        ∂C/∂f_l_j_k = ∂z_l_j/∂f_l_j_k * ∂C/∂z_l_j

        ∂z_l_j/∂f_l_j_k = cos(a_l-1_k * f_l_j_k) * a_l-1_k * A_l_j_k
        ∂z_l_j/∂f_l_j = cos(a_l-1 ⨀ f_l_j) ⨀ a_l-1 ⨀ A_l_j
            shapes: ∂z_l_j/∂f_l_j -> (k); a_l-1 -> (k); f_l_j -> (k); A_l_j
        ∂z_l/∂f_l = cos(a_l-1 ⨀k f_l) ⨀k a_l-1 ⨀k A_l
            shapes: ∂z_l/∂f_l -> (j,k); a_l-1 -> (k); f_l -> (j,k); A_l -> (j,k)

        => ∂C/∂f_l = ∂z_l/∂f_l * ∂C/∂z_l
           ∂C/∂f_l = (cos(a_l-1 ⨀k f_l) ⨀k a_l-1 ⨀k A_l) ⨉ δ_l

        :param error_of_layer:
        :param activations_of_previous_layer:
        :param parameters:
        :return:
        """
        # parameters.shape = (len(this layer), len(previous layer))
        activations_of_previous_layer = activations_of_previous_layer.reshape(len(activations_of_previous_layer))
        error_of_layer = error_of_layer.reshape(len(error_of_layer))

        return_val = []

        for j in range(len(parameters[0])):
            # return_val = a_l-1
            # reshapes from (10, 1) to (10,) shape, to make properly multiplications
            return_val.append(activations_of_previous_layer)

            # return_val = a_l-1 ⨀k f_l
            return_val[-1] = np.multiply(return_val[-1], parameters[0][j])

            # return_val = cos(a_l-1 ⨀k f_l+1)
            return_val[-1] = np.cos(return_val[-1])

            # return_val = cos(a_l-1 ⨀k f_l+1) ⨀k a_l-1
            return_val[-1] = return_val[-1] * activations_of_previous_layer

            # return_val = cos(a_l-1 ⨀k f_l+1) ⨀k a_l-1 ⨀k A_l = ∂z_l/∂f_l
            return_val[-1] = np.multiply(return_val[-1], parameters[1][j])

        return_val = np.asarray(return_val)
        return_val = np.dot(return_val.T, error_of_layer)
        return return_val
    """# TODO: make for loop over j-len(this layer) instead of faulty matrix form
        # return_val = a_l-1
        return_val = activations_of_previous_layer

        # return_val = ⨀k f_l
        for j in parameters[0]:
            j.shape = (len(parameters[0].T), 1)
            return_val = return_val * j

        # return_val = cos(a_l-1 ⨀k f_l+1)
        return_val = np.cos(return_val)

        # return_val = cos(a_l-1 ⨀k f_l+1) ⨀k a_l-1
        return_val = return_val * activations_of_previous_layer

        # return_val = cos(a_l-1 ⨀k f_l+1) ⨀k a_l-1 ⨀k A_l = ∂z_l/∂f_l
        for j in parameters[1]:
            j.shape = (len(parameters[0].T), 1)
            return_val = return_val * j

        # return_val = ∂∂z_l/∂f_l ⨉ δ_l+1
        return np.dot(return_val, error_of_layer)"""

    '''# TODO: test prime_weighting functions
    def prime_weighting_0(self, error_of_layer, activations_of_previous_layer, current_parameters):
        """
        ∂z_l/∂f_l_j_k = A_l_j_k * a_l-1_k * cos (f_l_j_k * a_l-1_k)
        => ∂z_l/∂f_l_j = A_l_j * a_l-1 * cos (f_l_j * a_l-1) (?)

        :param error_of_layer: δ_l
        :param activations_of_previous_layer: a_l−1
        :param current_parameters:
        :return: ∂C/∂w_l
        """
        """#return_val = np.ndarray(shape=current_parameters[0].shape)
        return_val = []

        for j in range(current_parameters[0].shape[0]):

            # ∂z_l/∂f_l_j = cos(f_l_j * a_l-1) * A_l_j * a_l-1
            #return_val[j] = np.multiply(current_parameters[0][j], activations_of_previous_layer)
            return_val.append(np.multiply(current_parameters[0][j], activations_of_previous_layer))
            return_val[j] = np.cos(return_val[j])
            return_val[j] = np.multiply(current_parameters[1][j], return_val[j])
            return_val[j] = np.multiply(activations_of_previous_layer, return_val[j])
            return_val[j] = np.multiply(error_of_layer[j], return_val[j])

        return_val = np.asarray(return_val)"""
        # ∂z_l/∂f_l_j = A_l * cos(f_l * a_l-1.T) * a_l-1.T
        return_val = current_parameters[1] * np.cos(current_parameters[0] * activations_of_previous_layer.T) * \
                     activations_of_previous_layer.T

        return_val = error_of_layer * return_val

        if return_val.shape != current_parameters[0].shape:
            raise Exception
        return return_val'''

    def prime_weighting_1(self, error_of_layer, activations_of_previous_layer, parameters):
        """
        ∂C/∂A_l_j_k = ∂z_l_j/∂A_l_j_k * δ_l_j

        ∂z_l_j/∂A_l_j_k = sin(a_l-1_k * f_l_j_k)
        ∂z_l_j/∂A_l_j = sin(a_l-1 ⨀ f_l_j)
            shapes: ∂z_l_j/∂A_l_j -> (k); a_l-1 -> (k); f_l_j -> (k);
        ∂z_l/∂A_l = sin(a_l-1 ⨀k f_l)
            shapes: ∂z_l/∂A_l -> (j,k); a_l-1 -> (k); f_l -> (j,k);
        => ∂C/∂A_l_j_k = ∂z_l/∂A_l_j_k * ∂C/∂z_l
           ∂C/∂A_l = (sin(a_l-1 ⨀k f_l)) ⨉ δ_l

        :return:
        """
        # parameters.shape = (len(this layer), len(previous layer))
        activations_of_previous_layer = activations_of_previous_layer.reshape(len(activations_of_previous_layer))
        error_of_layer = error_of_layer.reshape(len(error_of_layer))

        return_val = []

        # return_val[j] = a_l-1 ⨀ f_l_j
        for j in parameters[0]:
            return_val.append(activations_of_previous_layer * j)

        return_val = np.asarray(return_val)

        # return_val = sin(a_l-1 ⨀k f_l+1) = ∂z_l/∂A_l
        return_val = np.sin(return_val)

        return_val = np.dot(return_val.T, error_of_layer)
        # return_val = ∂z_l/∂A_l ⨉ δ_l+1
        return return_val

    '''def prime_weighting_1(self, error_of_layer, activation_of_previous_layer, current_parameters):
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
        return return_val'''

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
        return error_of_layer

    def generate_weights(self, structure: list[int]):
        paras = []
        for k in range(len(structure)):
            if k != 0:
                t_paras = [np.random.randn(structure[k], structure[k - 1]),
                           np.random.randn(structure[k], structure[k - 1]),
                           np.random.randn(structure[k], 1)]
                paras.append(t_paras)
                """
                t_paras = [np.ones(shape=(structure[k], structure[k - 1])),
                           np.ones(shape=(structure[k], structure[k - 1])),
                           np.ones(shape=(structure[k], 1))]
                paras.append(t_paras)"""
        return paras

    def prev_layer_function(self, parameters, weighted_inputs, error_of_next_layer):
        """
        MATHS:
        δ_l_j = ∂C/∂z_l_j
        δ_l_j = ∑k ∂C/∂z_l+1_k * ∂∂z_l+1_k/∂z_l_j
        δ_l_j = ∑k δ_l+1_k * ∂z_l+1_k/∂z_l_j

            z_l+1_k = ∑j(sin(σ(z_l_j) * f_l+1_k_j) * A_l+1_k_j) + b_l+1_k

            ∂z_l+1_k/∂z_l_j = cos(σ'(z_l_j) * f_l+1_k_j) * f_l+1_k_j * A_l+1_k_j

        δ_l = ∂z_l+1/∂z_l ⨉ δ_l+1
        shapes: δ_l -> (j, 1); ∂z_l+1/∂z_l -> (j,k); δ_l+1 -> (k,1);

            ∂z_l+1_k/∂z_l_j = cos(σ'(z_l_j) * f_l+1_k_j) * f_l+1_k_j * A_l+1_k_j

            ∂z_l+1/∂z_l_j = ∑k(cos(σ'(z_l_j) * f_l+1_j) * f_l+1_j * A_l+1_j)

            ∂z_l+1/∂z_l = ∑k(cos(σ'(z_l) *j f_l+1) *j f_l+1 *j A_l+1)"""
        # parameters.shape = (len(this layer), len(previous layer))
        weighted_inputs = weighted_inputs.reshape(len(weighted_inputs))
        error_of_next_layer = error_of_next_layer.reshape(len(error_of_next_layer))

        # return_val = σ'(z_l_j)
        return_val = []
        prime_weights = self.prime_activation_function(weighted_inputs)

        for j in range(len(parameters[0])):
            # return_val = σ'(z_l_k) *j f_l+1_k
            return_val.append(prime_weights * parameters[0][j])

            # return_val = cos(σ'(z_l_j) *j f_l+1)
            return_val[-1] = np.cos(return_val[-1])

            # return_val = cos(σ'(z_l_j) *j f_l+1) *j f_l+1
            return_val[-1] = return_val[-1] * parameters[0][j]

            # return_val = cos(σ'(z_l_j) *j f_l+1) *j f_l+1 *j A_l+1 = ∂z_l+1/∂z_l
            return_val[-1] = return_val[-1] * parameters[1][j]

        return_val = np.asarray(return_val)

        # return_val = ∂z_l+1/∂z_l ⨉ δ_l+1
        return np.dot(return_val.T, error_of_next_layer)

    '''# TODO: setup prev_layer_function
    def prev_layer_function(self, parameters, weighted_inputs, error_of_next_layer):
        """
        δ_l   = ∂C/∂z_l
        δ_l-1 = ∂C/∂z_l * ∂z_l/∂z_l-1
        δ_l-1 = δ_l     * ∂z_l/∂a_l-1 * ∂a_l-1/∂z_l-1

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
        """return_val = []

        for i in range(parameters[2].shape[0]):
            return_val.append(np.multiply(parameters[0][i], self.activation_function(weighted_inputs)))
            return_val[i] = np.cos(return_val[i])
            return_val[i] = np.multiply(return_val[i], parameters[0][i])
            return_val[i] = np.multiply(return_val[i], parameters[1][i])
            return_val[i] = np.multiply(return_val[i], self.prime_activation_function(weighted_inputs))

        return_val = np.asarray(return_val)"""

        # δ_l = δ_l+1 * A_l+1.T ⊙ f_l+1.T ⊙ cos(f_l+1.T ⊙ σ(z_l)) ⊙ (σ'(z_l))²
        return_val = np.cos(parameters[0].T * self.activation_function(weighted_inputs)) * \
                     parameters[0].T * parameters[1].T
        return_val = return_val @ error_of_next_layer * (self.prime_activation_function(weighted_inputs)) ** 2

        return return_val'''

    def activation_function(self, z: np.ndarray):
        """Sigmoid function 1 / (1 + e^-z)"""
        return 1.0 / (1.0 + np.exp(-z))

    def prime_activation_function(self, z: np.ndarray):
        """Derivative of the sigmoid function"""
        return -1/(1 + np.exp(z)) ** 2 + 1/(1 + np.exp(z))
        #return np.exp(z) / (1 + np.exp(z)) ** 2

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
