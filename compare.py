import sys
from matplotlib import pyplot as my_plt
from matplotlib import patches as my_patch
import numpy as np
import pickle
import json
import pandas as pd
import subprocess

from HyperSphere.BO.run_BO import BO
from fmfnBO.runBO import fmfnBO
from HyperSphere.test_functions.benchmarks import *
from target import *

def compare_plots(hs_files, fmfn_files):
    # data file
    data_config_filename = hs_files + '/data_config.pkl'
    f_name, log_file, plot_file = fmfn_files

    # HyperSphere
    with open(data_config_filename, 'rb') as data_config_file:
        for key, val in pickle.load(data_config_file).items():
            if key == 'output':
                y1 = val.data.numpy()
            
            if key == 'x_input':
                x1 = val.data.numpy()
                x1 = x1[:,0]

    # fmfn/BayesianOptimization
    # read from file
    with open(log_file, "r") as f:
        file_data = f.read()

    # add trailing commas for each line
    with open(log_file, "r") as f:
        lines = f.readlines()
        first_line = lines[0]
        file_lines = [''.join([',', line.strip(), '\n']) for line in lines[1:]]

    # encaspulate the data
    with open(plot_file, "w+") as f:
        f.write('{' + '\n' + '  ' + '"logs"' + ':' + ' ' + '[' + '\n')
        f.writelines(first_line)
        f.writelines(file_lines)
        f.write("    " + "]" + "\n" + "}")

    # load data for visualization
    with open(plot_file) as json_file:
        data = pd.DataFrame(json.load(json_file)['logs'])

    _x = data['params'].to_list()
    dim = len(_x[0].values())
    x_list = [[] for i in range(dim)]
    for x_ in _x:
        for i in range(dim):
            x_list[i].append(tuple(x_.values())[i])

    x_n_eval = np.arange(1, len(lines)+1, 1)

    # x_input
    x2 = x_list[0]
    # y = f(x)
    y2 = data['target'].to_numpy()

    # visualization
    my_plt.figure('HyperSphere vs fmfn')
    my_plt.title(f"function = {f_name} :: evaluations = 50")
    red = my_patch.Patch(color='red', label='HyperSphere')
    blue = my_patch.Patch(color='blue', label='fmfn')
    my_plt.legend(handles=[red, blue])

    my_plt.plot(x1, y1, 'r')
    my_plt.plot(x2, y2, 'b')

    my_plt.axhline(y=-106.76)

    my_plt.show()

if __name__ == '__main__':
    
    # parameters
    geometry = 'sphere' 
    func = 'birdy' 
    fmfn_func = 'neg_birdy'
    d = '2' 
    e = '50'

    # subprocess
    hs = subprocess.Popen(args=['python', 'HyperSphere/BO/run_BO.py', '-g', geometry,
                                '--parallel', '-f', func, '-d', d, '-e', e], stdout=subprocess.PIPE, universal_newlines=True)

    while True:
        output = hs.stdout.readline()
        if output not in [' ', '\n', '']:
            hs_files = output
        print(output.strip())
        return_code = hs.poll()
        if return_code is not None: 
            break

    fmfn = subprocess.Popen(args=['python', 'fmfnBO/runBO.py', '-f', str(fmfn_func), '-d', str(d),
                                  '-e', str(e)], stdout=subprocess.PIPE, universal_newlines=True)

    while True:
        output = fmfn.stdout.readline()
        if output not in [' ', '\n', '']:
            fmfn_files = output
        print(output.strip())
        return_code = fmfn.poll()
        if return_code is not None: 
            break

    # clean up hs_files
    hs_files = hs_files.strip('\n')

    # clean up fmfn_files
    fmfn_files = [line.strip().strip("(").strip(")").strip("'") for line in fmfn_files.strip("\n").split(",")]
    
    # compare
    compare_plots(hs_files, fmfn_files)
