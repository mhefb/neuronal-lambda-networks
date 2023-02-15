import csv
import multiprocessing as mp
import os
import numpy as np

from core.custom_Functions import FunctionBase
from core.custom_Functions.StandardFunctions import StandardFunction


class Network:
    structure: list[int]
    paras: list[list[np.ndarray]]
    funcs: FunctionBase
    avg_error: list
    given_commands: list[dict]

    @staticmethod
    def createFrom(structure: list[int], paras: list[list[np.ndarray]], funcs: FunctionBase):
        net = Network()

        net.funcs = funcs

        net.structure = structure
        net.avg_error = []
        net.given_commands = []

        net.paras = paras

        return net

    @staticmethod
    def createBasic(inputs: int, hidden_layers: [int], outputs: int, functions=StandardFunction()):
        net = Network()

        net.funcs = functions

        # Generate structure
        net.structure = hidden_layers
        net.structure.insert(0, inputs)
        net.structure.append(outputs)
        net.avg_error = []
        net.given_commands = []

        net.paras = net.funcs.generate_weights(net.structure)

        return net

    def feedforward(self, activation: np.ndarray):
        # For each layer:
        for layer in self.paras:
            # activations_layer = activation_function(
            #   weighting_function(parameters_layer, activations_layer-1) )
            activation = self.funcs.activation_function(self.funcs.weighting(layer, activation))
        return activation

    def train_epoch_wise(self, training_data: list[(np.array, np.array)], validation_data: list[(np.array, np.array)],
                         no_epochs=1, learning_rate=0.01, batch_size=1, tolerance=0.95, log_file_path=None):
        t_command = {'len_training_data': len(training_data), 'len_validation_data': len(validation_data), 'no_epochs': no_epochs,
                     'learning_rate': learning_rate, 'batch_size': batch_size, 'tolerance': tolerance,
                     'log_file_path': log_file_path}
        self.given_commands.append(t_command)
        for epoch in range(no_epochs):
            print('\nEpoch:', epoch + 1)
            self.train_batch_wise(training_data, learning_rate=learning_rate, batch_size=batch_size)
            self.log(validation_data, tolerance=tolerance, log_file_path=log_file_path)

    def train_batch_wise(self, training_data: list[(np.array, np.array)], learning_rate=0.01, batch_size=1):
        # mp.set_start_method(method='fork')

        # splitting training data into batches
        training_batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]

        training_batches[-1] = training_data[-batch_size - 1:-1]
        for sample_batch in training_batches:
            # list of ∂C/∂w for each weight, for each sample batch
            params_gradient = []

            batch_pool = mp.pool.Pool()
            results = batch_pool.map(self.train_single_entry, sample_batch)
            params_gradient = results

            # apply changes to parameters
            self.adjust_parameters(params_gradient, learning_rate, batch_size)

    def train_single_entry(self, entry: (np.array, np.array)):
        # takes care of the forward-pass
        activations, weighted_inputs = self.forward_pass(entry)

        exp_out = entry[1]

        # δ_L = ∇aC ⊙ σ′(z_L); L ≙ last layer, ∇aC ≙ list of ∂C/∂a_L_j for every j, σ ≙ activation function
        error = np.multiply(
            self.funcs.prime_cost_function(exp_out, activations[-1]),  # ∇aC
            self.funcs.prime_activation_function(weighted_inputs[-1]))  # σ′(z_L)

        # backward pass
        params_gradient = self.backward_pass(activations, weighted_inputs, error)

        # calculates ∂C/∂w for the first layer
        t_params_gradient = []
        for p_inf in self.funcs.paras_influence_on_weighted_input:
            t_params_gradient.append(p_inf(error, activations[-2], self.paras[-1]))
        params_gradient.insert(0, t_params_gradient)

        return params_gradient

    def adjust_parameters(self, params_gradient, learning_rate, batch_size):
        for layer in range(len(self.structure[1:])):
            for param_type in range(self.funcs.no_params):
                t_params_gradient = np.zeros(shape=self.paras[-(layer + 1)][param_type].shape)
                for entry in range(len(params_gradient)):
                    # print('entry:', entry, 'layer:', layer, 'param_type:', param_type)
                    # print(params_gradient[entry][layer][param_type].shape)

                    if self.paras[-(layer + 1)][param_type].shape != params_gradient[entry][layer][param_type].shape:
                        raise Exception("self.paras[-(layer+1)][param_type].shape {0} != params_gradient[entry]"
                                        "[layer][param_type].shape {1}".
                                        format(self.paras[-(layer + 1)][param_type].shape, params_gradient[entry]
                        [layer][param_type].shape))
                    t_params_gradient += params_gradient[entry][layer][param_type] / batch_size
                t_params_gradient *= -1 * learning_rate
                self.paras[-(layer + 1)][param_type] += t_params_gradient

    def backward_pass(self, activations, weighted_inputs, error_last_layer):
        weights_gradient = []
        error = error_last_layer

        for layer in range(2, len(self.paras) + 1):
            # calculating the error produced from the neurons of layer
            error = self.funcs.prev_layer_function(self.paras[-layer + 1], weighted_inputs[-layer], error)

            # calculating the gradient for each parameter
            t_params_gradient = []
            for p_inf in self.funcs.paras_influence_on_weighted_input:
                t_params_gradient.append(p_inf(error, activations[-layer - 1], self.paras[-layer]))
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
        activations = [sample[0]]  # initialization with input as first entry
        weighted_inputs = []

        for layer in self.paras:
            # calculating weighted inputs of the layer
            weighted_inp = self.funcs.weighting(layer, activations[-1])
            weighted_inputs.append(weighted_inp)
            # calculating activations of the layer
            activations.append(self.funcs.activation_function(weighted_inp))

        return activations, weighted_inputs

    def log(self, validation_data, tolerance=0.95, log_file_path=None):
        got_correct = 0
        avg_error = 0

        variation = np.ones(shape=(self.structure[-1], 1))
        variation *= (1 - tolerance)

        for sample in validation_data:
            result = self.feedforward(sample[0])
            if (np.greater_equal(result, sample[1] - variation).all() and
                np.less_equal(result, sample[1] + variation).all()) or \
                    np.array_equal(result, sample[1]):
                got_correct += 1
            avg_error += np.sum(np.abs(sample[1] - result))

        print(got_correct, '/', len(validation_data), 'were correct within a', tolerance, 'tolerance range')

        for sample in validation_data:
            result = self.feedforward(sample[0])
            if np.argmax(result) == np.argmax(sample[1]):
                got_correct += 1

        print(got_correct, '/', len(validation_data), 'were correctly guessed')

        avg_error *= 1 / len(validation_data)
        self.avg_error.append(np.sum(avg_error))

        existed_before = os.path.exists(log_file_path)
        if log_file_path:
            log_file = open(str(log_file_path + ' log.csv'), mode='a')
            log = csv.writer(log_file, delimiter=';')
            if not existed_before:
                log.writerow(['no validation datapoints', 'got correct', '% got correct', 'avg_error', '']
                             + list(self.given_commands[-1].keys()) if self.given_commands[-1] else [])
            log.writerow([len(validation_data), got_correct, got_correct / len(validation_data), str(avg_error).replace('.', ','),
                          ''] + [i[1] for i in self.given_commands[-1].items()])
            log_file.close()

    def evaluate(self, validation_data, log_file_path=None):
        got_correct = 0
        all_results = []

        for sample in validation_data:
            result = self.feedforward(sample[0])
            if np.argmax(result) == np.argmax(sample[1]):
                got_correct += 1
            all_results.append(result)

        if True:
            if log_file_path:
                # create csv.writer
                log_file = open(log_file_path + '.csv', mode='w')
                log = csv.writer(log_file, delimiter=';', lineterminator='\n')

                # overall information
                log.writerow(['structure:', self.structure])
                log.writerows([])
                if self.given_commands:
                    log.writerow(self.given_commands[0].keys())
                    for command in self.given_commands:
                        log.writerow(i[1] for i in command.items())
                log.writerow([])
                log.writerow(
                    ['tested on data points', len(validation_data), 'correct results:', got_correct,
                     '% correct results:', got_correct / len(validation_data)])
                log.writerow([])

                # results headers
                log.writerow(['result on data points'])
                log.writerow(['right answer', '', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

                # all data points
                change = np.zeros(shape=(len(all_results) - 1, len(all_results[0]), 1))
                for i in range(len(all_results)):
                    t_row = [np.argmax(validation_data[i][1]), '']
                    for j in all_results[i]:
                        t_row.append(str(float(j)).replace('.', ','))
                    t_row.append('')

                    """# total change in the results
                    if i == 0:
                        for j in range(1, len(all_results)):
                            change[j-1] = np.abs(all_results[j] - all_results[j - 1])
                        t_row.append('total')
                        for k in change.T[0]:
                            t_row.append(str(float(np.sum(k))).replace('.', ','))

                    # average change in the results
                    if i == 1:
                        t_row.append('average')
                        for k in change.T[0]:
                            t_row.append(str(float(np.sum(k)/len(all_results))).replace('.', ','))"""

                    log.writerow(t_row)
                log_file.close()
                print('Created evaluation successful')
