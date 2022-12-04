import numpy as np


class FunctionsBase:

    paras_influence_on_weighted_input: list

    def weighting(self, layer_paras: list[np.ndarray], prev_activations):
        pass

    def generate_weights(self, structure: list[int]):
        pass

    def activation_function(self, z: np.ndarray):
        pass

    def prime_activation_function(self, z: np.ndarray):
        pass

    def cost_function(self, expected_output, last_activation):
        pass

    def prime_cost_function(self, expected_output, last_activation):
        pass

    def prev_layer_function(self, param, param1, param2):
        pass
