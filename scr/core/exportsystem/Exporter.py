from core.Network import Network
from core.customWeightingFunctions import FunctionBase
from core.customWeightingFunctions.StandardFunctions import StandardFunction
from core.customWeightingFunctions.SinusoidFunctions import SinusoidFunction
from core.customWeightingFunctions.Quadratic import QuadraticFunction
import numpy as np
import json
import base64

# Add new function-CLASSES here (Not files)

MAP = {
    "standard": StandardFunction,
    "sinusoid": SinusoidFunction,
    "quadratic": QuadraticFunction
}


# Better version of is_instance, which only compaires the class-names
def __bisinstance(obj: object, clazz: type):
    return type(obj).__name__ == clazz.__name__


def __load_funcs(name: str):
    for key in MAP:
        if key == name:
            return MAP[key]()


def __save_funcs(funcs: FunctionBase):
    for key in MAP:
        if __bisinstance(funcs, MAP[key]):
            return key

    raise "Unknown Function found: " + str(funcs)


def __load_paras(paras: list[list[str]]):
    listSecond = []

    for first in range(len(paras)):

        listFirst = []

        for second in range(len(paras[first])):
            # Gets the array
            (sizeX, encodedStr) = paras[first][second]

            # Saves the ndarray like this:
            # UTF8-STRING >> BASE64 >> Raw-Bytes >> Reshape

            asNumber = np.frombuffer(base64.b64decode(encodedStr.encode("utf-8")))

            asNumber = np.copy(asNumber)

            # Resizes the array
            listFirst.append(asNumber.reshape((sizeX, int(len(asNumber) / sizeX))))

        listSecond.append(listFirst)

    return listSecond


def __save_paras(paras: list[list[np.ndarray]]):
    listSecond = []

    for first in range(len(paras)):

        listFirst = []

        for second in range(len(paras[first])):
            # Gets the array
            arr: np.ndarray = paras[first][second]

            # Saves the ndarray like this:
            # Raw-Bytes >> Base64 >> UTF8-String
            listFirst.append([len(arr), base64.b64encode(arr.tobytes()).decode('utf-8')])

        listSecond.append(listFirst)

    return listSecond


def __save_commands(given_commands: list[dict]):
    return_val = {}

    for i in range(len(given_commands)):
        return_val[str(i)] = given_commands[i]

    return return_val


def __load_commands(inp: dict[str]):
    return_val = []

    for i in range(len(inp)):
        return_val.append(inp[str(i)])

    return return_val


def exportString(net: Network):
    return json.dumps({
        "func": __save_funcs(net.funcs),
        "paras": __save_paras(net.paras),
        "structure": net.structure,
        "commands": __save_commands(net.given_commands)
    })


def importString(jsonNet: str):
    obj = json.loads(jsonNet)
    # print(obj)
    return Network(
        obj["structure"],
        functions=__load_funcs(obj["func"]),
        paras=__load_paras(obj["paras"]),
        given_commands=__load_commands(obj["commands"])
    )


def save(file: str, net: Network):
    with open(file, mode="w") as fp:
        fp.write(exportString(net))


def load(file: str):
    with open(file, mode='r') as fp:
        return importString(fp.read())
