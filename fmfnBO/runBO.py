import os
import sys
import argparse
import pathlib

from bayes_opt import BayesianOptimization, Events, JSONLogger

# current working directory
cwd = pathlib.Path(__file__).parent.absolute()
parent = os.path.split(cwd)[0]
if str(parent) not in sys.path:
	sys.path.append(parent)

from target import *

def fmfnBO(n_eval, ndim, func=None):
	assert func is not None

	logs = os.path.join(cwd, 'logs/')
	plots = os.path.join(cwd, 'plots/')

	if not os.path.isdir(logs):
		os.mkdir(logs)

	if not os.path.isdir(plots):
		os.mkdir(plots)

	log_name = os.path.join(logs + str(func) + ".json")
	plot_name = os.path.join(plots + str(func) + ".json")

	optimizer = BayesianOptimization(func, func.pbounds, random_state=1)

	logger = JSONLogger(log_name)
	optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

	steps = n_eval
	optimizer.maximize(init_points=2, n_iter=0, acq='ei', kappa=5)
	optimizer.maximize(init_points=0, n_iter=steps, acq='ei', kappa=5)
	
	print("Output Files:")
	return (func, log_name, plot_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FmFn Bayesian Optimization runner')
	parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=1)
	parser.add_argument('-d', '--dim', dest='ndim', type=int, default=1)
	parser.add_argument('-f', '--func', dest='func_name')

	args = parser.parse_args()
	fmfn_args = vars(args)
	if args.func_name is not None:
		exec('func=' + 'fmfn_' + args.func_name)
		fmfn_args['func'] = func
	del fmfn_args['func_name']

	fmfn_files = fmfnBO(**fmfn_args)
	print(fmfn_files)