from Controler import Controller

control = Controller(3, [2], 1)

control.feedforward([1, 2, 3])
print(control.neural_network.activations)
control.train(1, [[[1, 1, 1], [3]]])
