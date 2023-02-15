def transform_data(data_x, daty_y):
    data = np.ndarray(shape=(len(data_x), 784))
    for i in range(len(data_x)):
        t_data = []
        for y_cord in data_x[i]:
            t_data.extend(y_cord / 255)
        data[i] = t_data

    correct_results = []
    for i in range(len(daty_y)):
        t_data = np.zeros(shape=(10, 1))
        t_data[daty_y[i]] = 1
        correct_results.append(t_data)

    return zip(data, correct_results)


if __name__ == '__main__':
    import numpy as np
    from keras.datasets import mnist
    from core.Network import Network
    from core.custom_Functions.StandardFunctions import StandardFunction
    from core.custom_Functions.SinusoidFunctions import SinusoidFunction
    from core.exportsystem.Exporter import load, save

    # loading the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    """
    net = Network.createBasic(784, [30, 15], 10, functions=SinusoidFunction())
    #net = load('/root/SinusoidTest_5_2.lul')

    inputs = list(transform_data(train_X, train_y))
    val_data = list(transform_data(test_X, test_y))

    name = "SinusoidTest_T2_1"

    print('flags', net.paras[0][0].flags)

    net.train_epoch_wise(inputs, val_data, no_epochs=5, learning_rate=2, batch_size=10, log_file_path=name)
    save(name + '_1' + '.lul', net)

    net.train_epoch_wise(inputs, val_data, no_epochs=10, learning_rate=.5, batch_size=10, log_file_path=name)
    save(name + '_2' + '.lul', net)

    net.train_epoch_wise(inputs, val_data, no_epochs=10, learning_rate=0.1, batch_size=20, log_file_path=name)
    save(name + '_3' + '.lul', net)"""

    net = Network.createBasic(784, [10], 10, functions=SinusoidFunction())
    # net = load('C:\\Users\\Besitzer\\Documents\\Projekte\\Neuronale Lambda Netze\\Tests\\SinusoidTest_Noch_Weniger_Learing_Rate_1.lul')

    inputs = list(transform_data(train_X[:10], train_y[:10]))
    val_data = list(transform_data(test_X, test_y))

    name = 'C:\\Users\\Besitzer\\Documents\\Projekte\\Neuronale Lambda Netze\\temp_test'

    #net.evaluate(val_data,
                 # log_file_path='C:\\Users\\Besitzer\\Documents\\Projekte\\Neuronale Lambda Netze\\temp_test_before')

    net.train_epoch_wise(inputs, val_data, no_epochs=1, learning_rate=0.02, batch_size=10, log_file_path=name)

    #net.evaluate(val_data, log_file_path='C:\\Users\\Besitzer\\Documents\\Projekte\\Neuronale Lambda Netze\\temp_test_after')
