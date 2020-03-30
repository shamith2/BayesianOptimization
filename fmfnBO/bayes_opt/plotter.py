import os
import sys
import json
from matplotlib import pyplot as my_plt
from matplotlib import patches as my_patch
import pandas as pd
import numpy as np

class JSONPlotter():
    def __init__(self, filename, rename, f):
        try:
            os.remove(rename)
        except OSError:
            pass
        self.change(filename, rename)
        self.plot(rename, f)
        self.lines = []

    def change(self, filename, rename):
        with open(filename, "r") as f:
            file_data = f.read()

        with open(filename, "r") as f:
            self.lines = f.readlines()
            first_line = self.lines[0]
            file_lines = [''.join([',', line.strip(), '\n']) for line in self.lines[1:]]

        with open(rename, "w+") as f:
            f.write('{' + '\n' + '  ' + '"logs"' + ':' + ' ' + '[' + '\n')
            f.writelines(first_line)
            f.writelines(file_lines)
            f.write("    " + "]" + "\n" + "}")
 
    def plot(self, rename, f):
        with open(rename) as json_file:
            data = pd.DataFrame(json.load(json_file)['logs'])

        _x = data['params'].to_list()
        dim = len(_x[0].values())
        x_list = [[] for i in range(dim)]
        for x_ in _x:
            for i in range(dim):
                x_list[i].append(tuple(x_.values())[i])

        x_eval = np.arange(1, len(self.lines)+1, 1)
        y = data['target'].to_numpy()

        my_plt.figure('FmFnBO')
        my_plt.title("function = {} :: evaluations = {}".format(f, len(self.lines)))
        blue = my_patch.Patch(color='blue', label='fmfn')
        my_plt.legend(handles=[blue])

        my_plt.plot(x_list[0], y, 'b')

        my_plt.show()