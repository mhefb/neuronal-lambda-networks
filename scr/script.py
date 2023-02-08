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

    # printing the shapes of the vectors
    print('X_train: ' + str(train_X.shape))
    #print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    #print('Y_test:  ' + str(test_y.shape))

    #net = Network.createBasic(784, [15], 10, functions=SinusoidFunction())
    net = load('/root/SinusoidTest_5_2.lul')

    inputs = list(transform_data(train_X, train_y))
    val_data = list(transform_data(test_X, test_y))

    name = "SinusoidTest_6"

    print('flags', net.paras[0][0].flags)

    net.train_epoch_wise(inputs, val_data, no_epochs=5, learning_rate=5, batch_size=20, log_file_path=name)
    save(name + '_1' + '.lul', net)

    net.train_epoch_wise(inputs, val_data, no_epochs=10, learning_rate=1, batch_size=20, log_file_path=name)
    save(name + '_2' + '.lul', net)

    net.train_epoch_wise(inputs, val_data, no_epochs=10, learning_rate=0.1, batch_size=20, log_file_path=name)
    save(name + '_3' + '.lul', net)
