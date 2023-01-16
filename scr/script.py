import numpy as np
from keras.datasets import mnist
from core.Network import Network
from core.custom_Functions.StandardFunctions import StandardFunctions
from core.custom_Functions.SinusoidFunctions import SinusoidFunctions
from core.exportsystem.Exporter import load, save


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


# loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# printing the shapes of the vectors
#print('X_train: ' + str(train_X.shape))
#print('Y_train: ' + str(train_y.shape))
#print('X_test:  ' + str(test_X.shape))
#print('Y_test:  ' + str(test_y.shape))

test = Network.createBasic(784, [15], 10, functions=SinusoidFunctions())

#save("MyNetwork.lul", net1)
#net2 = load("MyNetwork.lul")

inputs = list(transform_data(train_X, train_y))
val_data = list(transform_data(test_X, test_y))

test.train_epoch_wise(inputs, val_data, no_epochs=30, learning_rate=3, batch_size=10)
print('avg error', test.avg_error)

test.avg_error = []
test.train_epoch_wise(inputs, val_data, no_epochs=10, learning_rate=1, batch_size=20)
print('avg error', test.avg_error)

test.avg_error = []
test.train_epoch_wise(inputs, val_data, no_epochs=10, learning_rate=0.1, batch_size=40)
print('avg error', test.avg_error)

for t in range(80, 101, 1):
    test.evaluate(val_data, tolerance=(t / 100))

save("MyNetwork.lul", test)

"""
test = Network(1, [2], 1)
print(test.feedforward([0]))
fh.save(test, path="")

test = None

test2 = fh.read(path="")
print(test2.feedforward([0]))

from core.custom_Functions.StandardFunctions import StandardFunctions

tete = StandardFunctions()
print(tete.paras_influence_on_weighted_input)
print(tete.paras_influence_on_weighted_input[0])
"""
