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
    from core.custom_Functions.StandardFunctions import StandardFunction
    from core.custom_Functions.SinusoidFunctions import SinusoidFunction
    from core.custom_Functions.Quadratic import QuadraticFunction
    from core.exportsystem.Exporter import load, save

    # loading the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    path = 'C:\\Users\\Besitzer\\Documents\\Projekte\\Neuronale Lambda Netze\\'
    file_name = 'teet'

    print('loading data ...')
    val_data = list(transform_data(test_X[:500], test_y[:500]))
    inputs = list(transform_data(train_X[:20], train_y[:20]))
    print('... prepared data')

    net = Network([784, 15, 10], functions=SinusoidFunction())
    # starting evaluation
    net.create_evaluation(val_data, log_file_path=path + file_name + '_basic')
    save(path + file_name + '_basic' + '.lul', net)
    print('basic evaluation completed successfully')

    net.train_epoch_wise(inputs, val_data, no_epochs=1, learning_rate=10, batch_size=5, log_file_path=path + file_name)
    #net = Network([784, 15, 10], functions=SinusoidFunction())
    save(path + file_name + '_second.lul', net)
    net.create_evaluation(val_data, log_file_path=path + file_name + '_second')

    basic_net = load(path + file_name + '_basic' + '.lul')

    print(net - basic_net)
