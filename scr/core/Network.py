import csv
import os
import numpy as np

from core.customWeightingFunctions import FunctionBase
from core.customWeightingFunctions.StandardFunctions import StandardFunction


class Network:
    structure: list[int]
    paras: list[list[np.ndarray]]
    funcs: FunctionBase
    given_commands: list[dict]

    def __init__(self, structure: list[int], functions=StandardFunction(), paras=None, given_commands=None):
        self.funcs = functions

        # Generate structure
        self.structure = structure
        self.given_commands = given_commands if given_commands else []
        self.paras = paras if paras else self.funcs.generate_weights(self.structure)

    def feedforward(self, activation: np.ndarray):
        for layer in self.paras:
            activation = self.funcs.activation_function(self.funcs.weighting(layer, activation))
        return activation

    def forward_pass(self, given_inputs):
        """
        For activations and weighted_inputs:
            lists on layer basis;\n
            np.ndarray below that

        :param given_inputs: inputs for the network of a data point in the training data
        :returns: activations and weighted_inputs of the network for the specified training data
        """
        activations = [given_inputs]  # initialization with input as first entry
        weighted_inputs = []

        for layer in self.paras:
            # calculating weighted inputs of the layer
            weighted_inp = self.funcs.weighting(layer, activations[-1])
            weighted_inputs.append(weighted_inp)
            # calculating activations of the layer
            activations.append(self.funcs.activation_function(weighted_inp))

        return activations, weighted_inputs

    def evaluate_network(self, data):
        got_correct = 0
        for sample in data:
            result = self.feedforward(sample[0])
            if np.argmax(result) == np.argmax(sample[1]):
                got_correct += 1

        return got_correct

    """def evaluate_network(self, data, tolerance=0.70):
        got_correct = 0
        avg_error = 0

        variation = np.ones(shape=(self.structure[-1], 1))
        variation *= (1 - tolerance)

        for sample in data:
            result = self.feedforward(sample[0])
            if (np.greater_equal(result, sample[1] - variation).all() and
                np.less_equal(result, sample[1] + variation).all()) or \
                    np.array_equal(result, sample[1]):
                got_correct += 1
            avg_error += np.sum(np.abs(sample[1] - result))

        #print(got_correct, '/', len(validation_data), 'were within a', tolerance, 'tolerance range of the ideal result')
        return got_correct, avg_error"""

    def log(self, validation_data, log_file_path=None):
        got_correct = self.evaluate_network(validation_data)
        print(got_correct, '/', len(validation_data), 'were correctly guessed')

        existed_before = os.path.exists(log_file_path)
        if log_file_path:
            log_file = open(str(log_file_path + ' log.csv'), mode='a')
            log = csv.writer(log_file, delimiter=';')
            if not existed_before:
                log.writerow(['no validation datapoints', 'got correct', '% got correct', 'avg_error', '']
                             + list(self.given_commands[-1].keys()) if self.given_commands[-1] else [])
            log.writerow([len(validation_data), got_correct, got_correct / len(validation_data),
                          ''] + [i[1] for i in self.given_commands[-1].items()])
            log_file.close()

    def create_evaluation(self, validation_data, log_file_path=None):
        got_correct = self.evaluate_network(validation_data)
        all_results = []

        for sample in validation_data:
            result = self.feedforward(sample[0])
            all_results.append(result)

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

    def __sub__(self, other):
        """
        Compares two neural networks, if they have the same structure and weighting function.
        This is based on a comparison of each parameter of the networks.

        :param other: Network to be compared with
        :return: np.ndarray, containing the change in value from the older to the newer model
         for parameter in each layer.
        """
        if not self.structure == other.structure:
            raise Exception("can not compare networks with a different structure")
        if not type(self.funcs) == type(other.funcs):
            raise Exception("can not compare networks with a different function")

        is_self_elder = len(self.given_commands) >= len(other.given_commands)
        elder = self.paras if is_self_elder else other.paras
        younger = other.paras if is_self_elder else self.paras

        return_val = []
        for layer in range(len(self.paras)):
            return_val.append(
                [elder[layer][i] - younger[layer][i] for i in range(0, self.funcs.no_params)])
        return np.asarray(return_val, dtype=object)
