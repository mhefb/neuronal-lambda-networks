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

    def feedforward(self, activation: np.ndarray):
        # For each layer:
        for layer in self.paras:
            # activations_layer = activation_function(
            #   weighting_function(parameters_layer, activations_layer-1) )
            activation = self.funcs.activation_function(self.funcs.weighting(layer, activation))
        return activation

    def train(self, training_data: list[(np.array, np.array)], learning_rate=0.01, batch_size=1):  # , validation_data):
        # splitting training data into batches
        training_batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]

        for sample_batch in training_batches:
            # list of ∂C/∂w for each weight, for each sample batch
            params_gradient = [[]]

            for entry in range(len(sample_batch)):

                # takes care of the forward-pass
                activations, weighted_inputs = self.forward_pass(sample_batch[entry])

                exp_out = sample_batch[entry][1]

                # δ_L = ∇aC ⊙ σ′(z_L); L ≙ last layer, ∇aC ≙ list of ∂C/∂a_L_j for every j, σ ≙ activation function
                error = np.multiply(
                    self.funcs.prime_cost_function(exp_out, activations[-1]),   # ∇aC
                    self.funcs.prime_activation_function(weighted_inputs[-1]))  # σ′(z_L)

                # backward pass
                params_gradient[entry] = self.backward_pass(activations, weighted_inputs, error)

                # calculates ∂C/∂w for the first layer
                t_params_gradient = []
                for p_inf in self.funcs.paras_influence_on_weighted_input:
                    t_params_gradient.append(p_inf(error, activations[-2]))
                params_gradient[entry].insert(0, t_params_gradient)

            for layer in range(len(self.structure[1:])):
                for param_type in range(self.funcs.no_params_needed):
                    t_params_gradient = np.zeros(shape=self.paras[-(layer + 1)][param_type].shape)
                    for entry in range(batch_size):
                        if self.paras[-(layer + 1)][param_type].shape != params_gradient[entry][layer][param_type].shape:
                            raise Exception("self.paras[-(layer+1)][param_type].shape {0} != params_gradient[entry]"
                                            "[layer][param_type].shape {1}".
                                            format(self.paras[-(layer + 1)][param_type].shape, params_gradient[entry]
                                                   [layer][param_type].shape))
                        t_params_gradient += params_gradient[entry][layer][param_type] / batch_size
                    t_params_gradient *= -1 * learning_rate
                    self.paras[-(layer + 1)][param_type] += t_params_gradient

        return

    def backward_pass(self, activations, weighted_inputs, error_last_layer):
        weights_gradient = []
        error = error_last_layer

        for layer in range(2, len(self.paras)+1):
            # calculating the error produced from the neurons of layer
            error = self.funcs.prev_layer_function(self.paras[-layer+1], weighted_inputs[-layer], error)

            # calculating the gradient for each parameter
            t_params_gradient = []
            for p_inf in self.funcs.paras_influence_on_weighted_input:
                t_params_gradient.append(p_inf(error, activations[-layer-1]))
            weights_gradient.append(t_params_gradient)

        return weights_gradient

    def forward_pass(self, sample):
        """
        For activations and weighted_inputs:
            lists on layer basis;
            np.ndarray below that

        :param sample: a single element of the training data
        :returns: activations and weighted_inputs of the network for the specified training data
        """
        activations = [sample[0]]   # initialization with input as first entry
        weighted_inputs = []

        for layer in self.paras:
            # calculating weighted inputs of the layer
            weighted_inp = self.funcs.weighting(layer, activations[-1])
            weighted_inputs.append(weighted_inp)
            # calculating activations of the layer
            activations.append(self.funcs.activation_function(weighted_inp))

        return activations, weighted_inputs

    """
            for layer in range(len(self.paras)):
                for param_type in range(self.funcs.no_params_needed):
                    # instantly applies a change to the parameters to minimise the cost
                    d_paras = 0
                    for entry in range(batch_size):
                        # Δ_cost = - learning_rate * (Δ_cost/Δ_parameters)²
                        # ⇒ Δ_parameters = - learning_rate * Δ_cost/Δ_parameters
                        print('params_gradient[entry][layer][j]', params_gradient[entry][layer][param_type])
                        d_paras += -learning_rate * params_gradient[entry][layer][param_type]
                    d_paras /= batch_size
                    self.paras[layer][param_type] += d_paras"""
