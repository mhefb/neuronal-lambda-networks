import math

import numpy as np

# must be sorted by priority
DEFAULT_OPERATORS = ('(', ')', '^', '*', '/', '+', '-')


class Equation:
    formula: str
    term: str
    variables: [str]
    calculation: callable
    numbers: [float]
    operators: [str]

    def __init__(self, formula: str):
        if formula.count("(") != formula.count(")"):
            raise Exception("Wrong bracket placement")
        self.formula = formula.replace(" ", "")
        self.term = self.formula[self.formula.find('=')+1:]

        self.variables = self.formula[self.formula.find("(")+1: self.formula.find(")")]
        self.variables = self.variables.split(',')

    def __call__(self, *args, **kwargs):
        # *args (Non-Keyword Arguments)
        # **kwargs (Keyword Arguments)

        if len(args) != len(self.variables):
            print(self.formula)
            print(self.variables)
            raise Exception("len(args) != len(self.variables)")

        for i in range(len(args)):
            argument = args[i]

            if self.variables[i].find('[]') != -1:
                if isinstance(argument, np.ndarray) or isinstance(argument, list):
                    for j in range(len(argument)):
                        varname = self.variables[i].replace('[]', '[' + str(j) + ']')
                        self.term = self.term.replace(varname, str(argument[j]))
            elif isinstance(argument, int) or isinstance(argument, float):
                varname = self.variables[i]
                self.term = self.term.replace(varname, str(argument))

        # TODO make this work
        # TODO use other Equations as inputs
        # TODO How should __call__() be used?
        self.term = self.term.replace('math.e', str(math.e))
        return_val = self.__dissolve_term__()
        self.term = self.formula[self.formula.find('=')+1:]
        return return_val

    def __dissolve_term__(self):
        func = self.term

        # From my old C# program
        # dividing the entry, extracting numbers
        for i in DEFAULT_OPERATORS:
            func = func.replace(i, '#')

        # splits manipulated string, only numbers remain
        while func.find('##') != -1:
            func = func.replace('##', '#')
        if func[-1] == '#':
            func = func[:-1]

        # Converting from String to Integer
        numbers = []
        for t_number in func.split('#'):
            numbers.append(float(t_number))

        # Looks for negative numbers
        for i in range(len(self.term)):
            if self.term[i] == '-' and self.term[i - 1] in DEFAULT_OPERATORS:
                # Number after must be -
                t_term = self.term[i:]
                for j in range(0, len(numbers)-1):
                    if t_term.find(str(numbers[j])) != -1:
                        numbers[j] *= - 1
                        self.term.replace(self.term[i], '')
                        continue

        # reset the entry
        func = self.term

        # extracting operators, deleting the numbers
        for i in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ','):
            func = func.replace(i, '')

        # saves extracted operators
        func_copy = func
        operators = [[] for _ in DEFAULT_OPERATORS]

        # saves operators, based on their type
        for i in range(len(func)):
            j = DEFAULT_OPERATORS.index(func[i])
            operators[j].append(i)

        # TODO: product ('Π')
        # TODO: sum ('∑')

        # isolates brackets
        for i in range(2, len(DEFAULT_OPERATORS)):
            func = func.replace(DEFAULT_OPERATORS[i], "")

        # sorts for bracket pairs
        bracket_pairs = []
        while func.count("(") != 0:
            i = func.find("()")
            idx = [0, 0]
            for j in range(i):
                if func[j] == '(':
                    idx[0] += 1
                if func[j] == ')':
                    idx[1] += 1
            bracket_pairs.append([operators[0][idx[0]] + 1, operators[1][idx[1]]])
            func = func.replace("()", "", 1)

        # Calculate
        calculated_numbers = []
        for bracket in bracket_pairs:
            # limits the frame of operators and numbers to the brackets
            func = func_copy[bracket[0]:bracket[1]]
            calculated_numbers = numbers[bracket[0]-1:bracket[1]+1]

            # sorts the operators, based on their type
            t_operators = [[] for _ in DEFAULT_OPERATORS]
            for i in range(len(func)):
                j = DEFAULT_OPERATORS.index(func[i])
                t_operators[j].append(i)

            # Calculates the bracket
            calculated_numbers = self.__calculate__(calculated_numbers, t_operators)

        func = func_copy.replace('(', '')
        func = func.replace(')', '')
        calculated_numbers = numbers

        # sorts the operators, based on their type
        t_operators = [[] for _ in DEFAULT_OPERATORS]
        for i in range(len(func)):
            j = DEFAULT_OPERATORS.index(func[i])
            t_operators[j].append(i)

        # Calculates the bracket
        calculated_numbers = self.__calculate__(calculated_numbers, t_operators)

        # final check and return
        if len(calculated_numbers) != 1:
            print("calculated_numbers: " + str(calculated_numbers))
            raise Exception("calculated_numbers has " + str(len(calculated_numbers)) + " entries lef")
        return calculated_numbers[0]

    def __calculate__(self, numbers: [float], operators: [str]):
        # checks if sizes fit
        len_operators = 0
        for op in operators:
            len_operators += len(op)
        if len_operators + 1 != len(numbers):
            print("numbers: " + str(numbers))
            print("operators: " + str(operators))
            raise Exception("Equation.__calculate__: got " + str(len(numbers)) +
                            " numbers and " + str(len_operators) + " operators")

        for operator_type in range(len(operators)):
            for i in range(len(operators[operator_type])):
                # performs operations on numbers
                if operator_type == 2:
                    numbers[i] = numbers[i] ** numbers[i + 1]
                elif operator_type == 2:
                    numbers[i] = numbers[i] * numbers[i + 1]
                elif operator_type == 3:
                    numbers[i] = numbers[i] / numbers[i + 1]
                elif operator_type == 4:
                    numbers[i] = numbers[i] + numbers[i + 1]
                elif operator_type == 5:
                    numbers[i] = numbers[i] - numbers[i + 1]

                # refits lists
                numbers.pop(i + 1)
                operators[operator_type].pop(i)
                for j in range(len(operators[operator_type])):
                    if j > 0:
                        operators[operator_type][j] -= 1 if j > i else 0

        return numbers
