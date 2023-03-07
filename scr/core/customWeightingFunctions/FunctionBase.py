import numpy as np


class FunctionBase:
    no_params: int
    paras_influence_on_weighted_input: list

    def __init__(self):
        self.paras_influence_on_weighted_input = []
        try:
            i = 0
            while True:
                self.paras_influence_on_weighted_input.append(
                    self.__getattribute__('prime_weighting_' + str(i)))
                i += 1
        except:
            return

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

    def prev_layer_function(self, parameters, weighted_inputs, error_of_next_layer):
        pass
