import os
import json
from matplotlib import pyplot as my_plt
from matplotlib import patches as my_patch
import pandas as pd
import numpy as np

class JSONPlotter():
    def __init__(self, log_name, plot_name, func, x_eval=True):
        try:
            os.remove(plot_name)
        except OSError:
            pass
        self.change(log_name, plot_name)
        self.plot(plot_name, func, x_eval)
        self.lines = []

    def change(self, log_name, plot_name):
        with open(log_name, "r") as f:
            file_data = f.read()

        with open(log_name, "r") as f:
            self.lines = f.readlines()
            first_line = self.lines[0]
            file_lines = [''.join([',', line.strip(), '\n']) for line in self.lines[1:]]

        with open(plot_name, "w+") as f:
            f.write('{' + '\n' + '  ' + '"logs"' + ':' + ' ' + '[' + '\n')
            f.writelines(first_line)
            f.writelines(file_lines)
            f.write("    " + "]" + "\n" + "}")
 
    def plot(self, plot_name, func, x_eval=True):
        with open(plot_name) as json_file:
            data = pd.DataFrame(json.load(json_file)['logs'])

        _x = data['params'].to_list()
        dim = len(_x[0].values())
        x_list = [[] for i in range(dim)]
        for x_ in _x:
            for i in range(dim):
                x_list[i].append(tuple(x_.values())[i])

        n_eval = np.arange(1, len(self.lines)+1, 1)
        y = data['target'].to_numpy()
        
        if x_eval:
            x = n_eval
        else:
            x = x_list[0]

        my_plt.figure('FmFnBO')
        my_plt.title("function = {} :: evaluations = {}".format(func, len(self.lines)))
        blue = my_patch.Patch(color='blue', label='fmfn')
        my_plt.legend(handles=[blue])

        my_plt.plot(x, y, 'b')

        my_plt.show()
