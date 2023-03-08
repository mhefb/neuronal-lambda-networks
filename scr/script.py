def transform_data(data_x, daty_y):
    data = np.ndarray(shape=(len(data_x), 784))
    for i in range(len(data_x)):
        t_data = []
        for y_cord in data_x[i]:
            t_data.extend((y_cord / 255))
        data[i] = t_data

    correct_results = []
    for i in range(len(daty_y)):
        t_data = np.zeros(shape=(10, 1))
        t_data[daty_y[i]] = 1
        correct_results.append(t_data)

    return zip(data, correct_results)


if __name__ == '__main__':
    import math
    import numpy as np
    from keras.datasets import mnist
    from core.Network import Network
    import core.learningAlgorithms.SGD as SGD
    import core.learningAlgorithms.bogoTrain as BOGO
    from core.customWeightingFunctions.StandardFunctions import StandardFunction
    from core.customWeightingFunctions.SinusoidFunctions import SinusoidFunction
    from core.customWeightingFunctions.Quadratic import QuadraticFunction
    from core.exportsystem.Exporter import load, save

    # loading the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    path = 'C:\\Users\\Besitzer\\Documents\\Projekte\\Neuronale Lambda Netze\\'
    file_name = 'ext_BOGO_teeeet'

    print('loading data ...')
    val_data = list(transform_data(test_X, test_y))
    inputs = list(transform_data(train_X, train_y))
    print('... prepared data')

    net = Network([784, 15, 10])#, functions=SinusoidFunction())
    # starting evaluation
    net.create_evaluation(val_data, log_file_path=path + file_name + '_basic')
    save(path + file_name + '_basic' + '.lul', net)
    print('basic evaluation completed successfully')

    #SGD.train_epoch(net, inputs, val_data, no_epochs=1, learning_rate=10, batch_size=5, log_file_path=path + file_name)
    BOGO.train_multiple(net, inputs, val_data, no_changes=2000, change_mul=0.1, no_tests=500,
                        log_file_path=path + file_name)
    BOGO.train_multiple(net, inputs, val_data, no_changes=200, change_mul=0.01, no_tests=500,
                        log_file_path=path + file_name)

    save(path + file_name + '_second.lul', net)
    net.create_evaluation(val_data, log_file_path=path + file_name + '_second')

    basic_net = load(path + file_name + '_basic' + '.lul')
