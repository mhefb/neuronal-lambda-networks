import multiprocessing as mp
import numpy as np
import copy

from core.Network import Network

accepted_changes = []


def train_multiple(net: Network, training_data: list[(np.array, np.array)], validation_data: list[(np.array, np.array)],
                   no_changes: int, change_mul=0.05, no_tests=20, log_file_path=None):
    global accepted_changes

    t_command = {'learning algorithm': 'Bogo Train', 'len_training_data': len(training_data), 'len_validation_data': len(validation_data),
                 'no_changes': no_changes, 'change_mul': change_mul, 'log_file_path': log_file_path}
    net.given_commands.append(t_command)

    start_idx = 0
    for cng in range(1, no_changes+1):
        if cng % (no_changes / 10) == 0:
            net.log(validation_data, log_file_path=log_file_path)
            print('Change', cng, ':', 'accuracy', net.accuracy[-int(no_changes/10):], 'last cost', np.average(net.last_costs))
        try_single(net, training_data, change_mul=change_mul, no_tests=no_tests, start_idx=start_idx, enable_message=False)
        start_idx += no_tests
        if start_idx >= len(training_data):
            start_idx = 0

    print('accepted/rejected changes: ', np.average(np.asarray(accepted_changes)))


def try_single(net: Network, training_data: list[(np.array, np.array)], change_mul=0.05, no_tests=20, start_idx=None,
               enable_message=True):
    # mp.set_start_method(method='fork')

    global accepted_changes

    if len(training_data) < no_tests:
        raise Exception('Bogo Train cannot test {0} times with only {1} data points'.format(no_tests, len(training_data)))
    if start_idx is None:
        start_idx = np.random.randint(0, len(training_data) - no_tests, size=1)
    if len(training_data) < start_idx + no_tests:
        print('Training denied: Bogo Train has only {0} data points and could never test on data point {1}'.format(len(training_data), start_idx + no_tests))
        return
    if type(net.paras).__name__ != 'ndarray':
        net.paras = np.asarray(net.paras, dtype=object)
    """if len(net.accuracy) == 1:
        net.evaluate_network(training_data[start_idx:start_idx+no_tests])"""
    if len(net.last_costs) == 1:
        batch_pool = mp.pool.Pool()
        results = batch_pool.map(net.feedforward, [i[0] for i in training_data[start_idx:start_idx+no_tests]])
        batch_pool = mp.pool.Pool()
        costs = batch_pool.starmap(net.funcs.cost_function,
                                   [(i[1], res) for i, res in zip(training_data[start_idx:start_idx+no_tests], results)])
        net.last_costs = costs

    old_paras = copy.deepcopy(net.paras)

    d_paras = net.funcs.generate_weights(structure=net.structure)
    d_paras = np.asarray(d_paras, dtype=object)
    d_paras *= change_mul

    net.paras += d_paras

    batch_pool = mp.pool.Pool()
    results = batch_pool.map(net.feedforward, [i[0] for i in training_data[start_idx:start_idx+no_tests]])
    batch_pool = mp.pool.Pool()
    costs = batch_pool.starmap(net.funcs.cost_function,
                               [(i[1], res) for i, res in zip(training_data[start_idx:start_idx+no_tests], results)])

    """# Comparison based on the networks accuracy
    if net.accuracy[-2] > net.accuracy[-1]:
        print('change got rejected: old', net.accuracy[-2], 'vs. new', net.accuracy[-1])
        net.paras = old_paras
    else:
        print('change got accepted: old', net.accuracy[-2], 'vs. new', net.accuracy[-1])"""

    # Comparison based on the networks last costs
    if np.average(net.last_costs) < np.average(costs):
        if enable_message:
            print('change got rejected: old', np.average(net.last_costs), 'vs. new', np.average(costs))
        net.paras = old_paras
        accepted_changes.append(0)
    else:
        if enable_message:
            print('change got accepted: old', np.average(net.last_costs), 'vs. new', np.average(costs))
        accepted_changes.append(1)
        net.last_costs = costs
