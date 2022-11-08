from math import e
from Equation import Equation


def initial_weight():
    return 0.5


sigmoid = Equation("f(input) = 1 / (1 + math.e ^ input)")
a_sigmoid = Equation("f(input) = math.e^input / (math.e^input + 1)^2")

wpf = 2
connection_function = Equation("f(weight[], input) = weight[0] * input + weight[1]")
a_connection_function = ["input", "1"]

cost_function = Equation("f(m, y[], input[]) = 1/(2*m) * ∑i((y[i] - input[i])^2)")
a_cost_function = Equation("f(m, input[], y[]) = ∑i((y[i] - input[i]))/m")
