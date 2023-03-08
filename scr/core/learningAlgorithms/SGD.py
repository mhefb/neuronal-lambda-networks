import multiprocessing as mp
import numpy as np

from core.Network import Network


def __apply_gradients(net, params_gradient, learning_rate: float, batch_size: int):
    """
    calculates a good enough change to the parameters of the network, based on the previously calculated gradients

    :param net: Network to apply the change to
    :param params_gradient: calculated gradients
    :param learning_rate: how much the parameters should change, typically a small positive number
    :param batch_size: over how many data samples were the gradients calculated
    """
    for layer in range(len(net.structure[1:])):
        for param_type in range(net.funcs.no_params):

            t_params_gradient = np.zeros(shape=net.paras[-(layer + 1)][param_type].shape)

            for entry in range(len(params_gradient)):
                if net.paras[-(layer + 1)][param_type].shape != params_gradient[entry][layer][param_type].shape:
                    raise Exception("self.paras[-(layer+1)][param_type].shape {0} != params_gradient[entry]"
                                    "[layer][param_type].shape {1}".
                                    format(net.paras[-(layer + 1)][param_type].shape, params_gradient[entry]
                                           [layer][param_type].shape))

                t_params_gradient += params_gradient[entry][layer][param_type] / batch_size
            t_params_gradient *= -1 * learning_rate
            net.paras[-(layer + 1)][param_type] += t_params_gradient


def train_epoch(net: Network, training_data: list[(np.array, np.array)], validation_data: list[(np.array, np.array)],
                no_epochs=1, learning_rate=0.01, batch_size=1, log_file_path=None):
    t_command = {'len_training_data': len(training_data), 'len_validation_data': len(validation_data),
                 'no_epochs': no_epochs, 'learning_rate': learning_rate, 'batch_size': batch_size,
                 'log_file_path': log_file_path}
    net.given_commands.append(t_command)
    for epoch in range(no_epochs):
        print('\nEpoch:', epoch + 1)
        train_batch(net, training_data, learning_rate=learning_rate, batch_size=batch_size)
        net.log(validation_data, log_file_path=log_file_path)


def train_batch(net: Network, training_data: list[(np.array, np.array)], learning_rate=0.01, batch_size=1):
    # mp.set_start_method(method='fork')

    # splitting training data into batches
    training_batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]

    training_batches[-1] = training_data[-batch_size - 1:-1]
    for sample_batch in training_batches:

        # list of ∂C/∂w for each weight, for each sample batch
        batch_pool = mp.pool.Pool()
        results = batch_pool.starmap(__calc_single_entry, [(net, smpl) for smpl in sample_batch])
        params_gradient = results

        # apply changes to parameters
        __apply_gradients(net, params_gradient, learning_rate, batch_size)


def __calc_single_entry(net: Network, entry: (np.array, np.array)):
    # takes care of the forward-pass
    activations, weighted_inputs = net.forward_pass(entry[0])

    # δ_L = ∇aC ⊙ σ′(z_L); L ≙ last layer, ∇aC ≙ list of ∂C/∂a_L_j for every j, σ ≙ activation function
    error = np.multiply(
        net.funcs.prime_cost_function(entry[1], activations[-1]),  # ∇aC
        net.funcs.prime_activation_function(weighted_inputs[-1]))  # σ′(z_L)

    # backward pass
    params_gradient = __backward_pass(net, activations, weighted_inputs, error)

    # calculates ∂C/∂w for the first layer
    t_params_gradient = []
    for p_inf in net.funcs.paras_influence_on_weighted_input:
        t_params_gradient.append(p_inf(error, activations[-2], net.paras[-1]))
    params_gradient.insert(0, t_params_gradient)

    return params_gradient


def __backward_pass(net: Network, activations, weighted_inputs, error_last_layer):
    weights_gradient = []
    error = error_last_layer

    for layer in range(2, len(net.paras) + 1):
        # calculating the error produced from the neurons of layer
        error = net.funcs.prev_layer_function(net.paras[-layer + 1], weighted_inputs[-layer], error)

        # calculating the gradient for each parameter
        t_params_gradient = []
        for p_inf in net.funcs.paras_influence_on_weighted_input:
            t_params_gradient.append(p_inf(error, activations[-layer - 1], net.paras[-layer]))
        weights_gradient.append(t_params_gradient)

    return weights_gradient
