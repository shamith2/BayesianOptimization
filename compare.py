from matplotlib import pyplot as my_plt
from matplotlib import patches as my_patch
import numpy as np
import pickle
import json
import pandas as pd


data_config_filename = r'/::host::/BayesianOptimization/HyperSphere/experiments/::func_name::/data_config.pkl'
log_file = r'/::host::/BayesianOptimization/fmfnBO/logs/::log_file::.json'
plot_file = r'/::host::/BayesianOptimization/fmfnBO/plots/::plot_file::.json'

# HyperSphere
data_config_file = open(data_config_filename, 'rb')
for key, val in pickle.load(data_config_file).items():
    if key == 'output':
        y1 = val.data.numpy()
    
    if key == 'x_input':
        x1 = val.data.numpy()

data_config_file.close()

# fmfn/BayesianOptimization
with open(log_file, "r") as f:
    file_data = f.read()

with open(log_file, "r") as f:
    lines = f.readlines()
    first_line = lines[0]
    file_lines = [''.join([',', line.strip(), '\n']) for line in lines[1:]]

with open(plot_file, "w+") as f:
    f.write('{' + '\n' + '  ' + '"logs"' + ':' + ' ' + '[' + '\n')
    f.writelines(first_line)
    f.writelines(file_lines)
    f.write("    " + "]" + "\n" + "}")

with open(plot_file) as json_file:
    data = json.load(json_file)

x2 = np.arange(1, len(lines)+1, 1)
y2 = pd.DataFrame(data['logs']).to_numpy()[:,0]

my_plt.figure('HyperSphere vs fmfn')
my_plt.title("function = 'bird Function' :: evaluations = 50")
red = my_patch.Patch(color='red', label='HyperSphere')
blue = my_patch.Patch(color='blue', label='fmfn')
my_plt.legend(handles=[red, blue])

my_plt.plot(x1[:,0], y1, 'r')
my_plt.plot(x2, y2, 'b')

my_plt.show()
