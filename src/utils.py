import os
import numpy as np


def read_data(data_name):
    if os.path.exists(f"data/{data_name}"):
        data = np.loadtxt(f"data/{data_name}", delimiter=",")
    elif os.path.exists(f"data/{data_name}.csv"):
        data = np.loadtxt(f"data/{data_name}.csv", delimiter=",")
    x, y = data[:, :-1], data[:, -1].astype(int)
    return x, y
