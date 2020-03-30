import os
import sys

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger
from bayes_opt.plotter import JSONPlotter
from bayes_opt.event import Events

sys.path.append('/::host::/')
from target import *

func = birdy

logs = r'/::host::/BayesianOptimization/fmfnBO/logs/'
plots = r'/::host::/BayesianOptimization/fmfnBO/plots/'

if not os.path.isdir(logs):
	os.mkdir(logs)

if not os.path.isdir(plots):
	os.mkdir(plots)

path = os.path.join(logs + str(func) + ".json")
rename = os.path.join(plots + str(func) + ".json")

optimizer = BayesianOptimization(func, func.pbounds, random_state=1)

logger = JSONLogger(path)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

steps = 50
optimizer.maximize(init_points=2, n_iter=0, acq='ei', kappa=5)
optimizer.maximize(init_points=0, n_iter=steps, acq='ei', kappa=5)

JSONPlotter(filename=path, rename=rename, f=str(func))
