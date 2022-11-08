import scr.Config as Cfg
from scr.Equation import Equation
from scr.Structure.Network import Network
import numpy as np


class Controller:
    neural_network: Network

    def __init__(self, inputs_size: int, hidden_layers: [int], output_size: int):
        sizes = hidden_layers
        sizes.insert(0, inputs_size)
        sizes.append(output_size)
        self.neural_network = Network(sizes)

    def feedforward(self, data: []):
        activations = [[[] for neuron in range(self.neural_network.sizes[layer if layer != len(self.neural_network.sizes)
                       else 0])] for layer in range(len(self.neural_network.sizes) + 1)]

        # here: excluding input layer
        weighted_inputs = [[[] for neuron in range(self.neural_network.sizes[layer])]
                           for layer in range(len(self.neural_network.sizes))]

        # enters input data
        activations[0] = data

        # for each Layer
        for layer_idx in range(len(self.neural_network.layers)):
            layer = self.neural_network.layers[layer_idx]
            # for each Neuron
            for neuron_idx in range(len(layer.neurons)):
                neuron = layer.neurons[neuron_idx]

                # initializes with a value
                weighted_inputs[layer_idx][neuron_idx] = 0
                activations[layer_idx + 1][neuron_idx] = 0

                for pre_neuron_idx in range(len(self.neural_network.layers[layer_idx - 1].neurons)):
                    # Calculates the weighted input
                    weighted_inputs[layer_idx][neuron_idx] += neuron.connection_function(
                        neuron.weights[pre_neuron_idx], activations[layer_idx][pre_neuron_idx])

                # Calculates the activation
                activations[layer_idx + 1][neuron_idx] = \
                    neuron.activation_function(weighted_inputs[layer_idx][neuron_idx])

        self.neural_network.weighted_inputs = weighted_inputs
        self.neural_network.activations = activations

    def train(self, epochs: int, training_data, batch_size=1, learning_rate=0.1):
        # Training data = tuple(input, expected outputs)
        # divides the data into batches
        batches = []
        for i in range(len(training_data) // batch_size):
            batches.append(training_data[i * batch_size:(i + 1) * batch_size])

        # makes a new batch out of the last elements, if the data does not split perfectly into batches
        if len(training_data) % batch_size != 0:
            batches.append(training_data[len(training_data) - 1 - batch_size:-1])

        # TODO: Test if it works
        # TODO: Adapt to a batch size > 1
        for epoch in range(epochs):
            for sample_batch in batches:

                # Initialises variables / fields
                weights = np.ndarray(shape=(len(self.neural_network.sizes)))
                weighted_inputs = np.ndarray(shape=len(sample_batch))
                activations = np.ndarray(shape=len(sample_batch))
                outputs = np.ndarray(shape=len(sample_batch))
                ext_outputs = np.ndarray(shape=len(sample_batch))

                # Collects data from the neural network
                for i in range(batch_size):
                    self.feedforward(sample_batch[i][0])

                    activations[i] = self.neural_network.activations
                    weighted_inputs[i] = self.neural_network.weighted_inputs

                    outputs[i] = activations[i][-1]
                    ext_outputs[i] = sample_batch[i][1]

                    for idx_layer in range(len(self.neural_network.layers)):
                        weights[idx_layer] = self.neural_network.layers[idx_layer].get_weights()

                # Calculating the derivative
                """# func = a_cost(m, input, a) = (a - input)/m
                func = Cfg.a_cost_function.formula

                # func(m, input, a) = (a - (1 / 1 + e ^ - (weight.layer_idx[0] * input + weight.layer_idx[1])))/m
                func = func[:func.find('=') + 1] + func[func.find('=') + 1:]. \
                    replace("input", "(" + Cfg.sigmoid.term.
                            replace("input", "(" + Cfg.connection_function.term + ")") + ")"). \
                    replace("weight[", "weight" + str(layer_idx) + "[")"""

                # δLj = ∂C/∂aLj * σ′(zLj)
                # a_sigmoid(input) = e^input / (e^input + 1)^2
                # a_cost_function(m, input[], y[]) = ∑i((y[i] - input[i]))/m
                # needs to be executed for every entry
                cost_first_layer = Equation("c(" + Cfg.a_cost_function.variables + ')= ' +
                                            Cfg.a_cost_function.term + '*' + Cfg.a_sigmoid.term)

                # Scales down to every Neuron
                costs = []
                for output, ext_output in zip(outputs, ext_outputs):
                    costs.append(cost_first_layer(output, ext_output))

                for layer_idx in range(len(self.neural_network.layers), 0, -1):
                    idx_layer = self.neural_network.layers[layer_idx]
                    """# func(m, input, a) = (a - 1 / (1 + e ^ - weight.layer_idx[0] * input + weight.layer_idx[1]))/m
                    func = func[:func.find('=') + 1] + func[func.find('=') + 1:]. \
                        replace("input", "(" + Cfg.sigmoid.term.
                                replace("input", "(" + Cfg.connection_function.term + ")") + ")"). \
                        replace("weight[", "weight" + str(layer_idx) + "[")
                    func.replace("m,", "m, weight" + str(layer_idx) + "[]")
                    derivatives = func()"""
                    for neuron_idx in range(len(idx_layer.neurons)):
                        neuron = idx_layer.neurons[neuron_idx]
                        for weight_idx in range(len(neuron.weights)):
                            # Calculates the cost of a single weight
                            # Normally: ∂C/∂w = a_in * δ_out and ∂C/∂b[l, j] = δ[l, j]
                            derivative_weight = Cfg.a_connection_function()[weight_idx] \
                                .replace('input', activations[layer_idx][neuron_idx])
                            derivative_weight *= costs[neuron_idx]

                            # Applies the changes to the weight
                            neuron.weights[weight_idx] -= learning_rate * derivative_weight

                    # Calculates the cost of the next layer
                    # δ[l, j]=((w[l+1, j])T δ[l+1, j]) * σ′(z[l, j])
                    t_costs = np.array()
                    for i in range(len(idx_layer.neurons)):
                        # weights[layer_idx]: 1. idx := to_neuron;    2. idx := from_neuron
                        # transpose hopefully switches the next two axis
                        t_costs = weights[layer_idx].transpose() @ costs
                        t_costs = t_costs * Cfg.a_sigmoid(weighted_inputs[layer_idx - 1])
                    costs = t_costs
                print(Cfg.cost_function(batch_size, sample_batch[1], self.neural_network.feedforward(sample_batch[0])))

    """def __get_function__(self, no_layer: int):
        func = self.a_cost_function.formula.replace("y", "input")
        for layer_idx in range(len(self.layers), 0, -1):
            for neuron_idx in range(len(self.layers[layer_idx].neurons)):
                neuron = self.layers[layer_idx].neurons[neuron_idx]
                t_str = neuron.activation_function.formula.replace("weight" + str(layer_idx) + str(neuron_idx))
                func = func.replace("input", t_str)"""
