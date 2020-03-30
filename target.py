import torch
import math
import numpy as np

# HyperSphere Functions

def neg_birdy(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	n_repeat = ndim // 2
	x = x * 2 * math.pi
	output = 0
	for i in range(n_repeat):
		output += (x[:, 2 * i] - x[:, 2 * i + 1]) ** 2 + torch.exp((1 - torch.sin(x[:, 2 * i])) ** 2) * torch.cos(x[:, 2 * i + 1]) + torch.exp((1 - torch.cos(x[:, 2 * i + 1])) ** 2) * torch.sin(x[:, 2 * i])
	output /= float(n_repeat)
	if flat:
		return -1.0*output.squeeze(0)
	else:
		return -1.0*output

neg_birdy.dim = 0

def bf(x):   # bohachevsky_function
    return x[0]**2 + 2*(x[1]**2) - 0.3*torch.cos(3*math.pi*x[0]) - 0.4*torch.cos(4*math.pi*x[1]) + 0.7

bf.dim = 0

def nbf(x):   # bohachevsky_function
    return -1.0*(x[0]**2 + 2*(x[1]**2) - 0.3*torch.cos(3*math.pi*x[0]) - 0.4*torch.cos(4*math.pi*x[1]) + 0.7)

nbf.dim = 0

def sphere(x):
	return x[0]**2 + x[1]**2 + x[2]**2

sphere.dim = 0

def neg_sphere(x):
	return -1.0*(x[0]**2 + x[1]**2 + x[2]**2)

neg_sphere.dim = 0

def cit(x):
	return -0.0001*(torch.pow((torch.sin(x[0]) * torch.sin(x[1]) * torch.exp(100 - (torch.sqrt(x[0]**2 + x[1]**2)/math.pi)) + 1), 0.1))

cit.dim = 0

def neg_cit(x):
	return 0.0001*(torch.pow((torch.sin(x[0]) * torch.sin(x[1]) * torch.exp(100 - (torch.sqrt(x[0]**2 + x[1]**2)/math.pi)) + 1), 0.1))

neg_cit.dim = 0

def neg_htf(x):
	return torch.abs(torch.sin(x[0]) * torch.sin(x[1]) * torch.exp(torch.abs(1 - (torch.sqrt(x[0]**2 + x[1]**2)/math.pi))))

neg_htf.dim = 0

def htf(x):
	return -1.0*torch.abs(torch.sin(x[0]) * torch.sin(x[1]) * torch.exp(torch.abs(1 - (torch.sqrt(x[0]**2 + x[1]**2)/math.pi))))

htf.dim = 0

def pf(x):
	return -1.0*((x[0]**2)/2 + (x[1]**2/2))

pf.dim = 0

def neg_pf(x):
	return ((x[0]**2)/2 + (x[1]**2/2))

neg_pf.dim = 0

def neg_branin(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	n_repeat = ndim // 2  # changed from float to int
	n_dummy = ndim % 2

	shift = torch.cat([torch.FloatTensor([2.5, 7.5]).repeat(n_repeat), torch.zeros(n_dummy)])

	if hasattr(x, 'data'):
		x.data = x.data * 7.5 + shift.type_as(x.data)
	else:
		x = x * 7.5 + shift.type_as(x)
	a = 1
	b = 5.1 / (4 * math.pi ** 2)
	c = 5.0 / math.pi
	r = 6
	s = 10
	t = 1.0 / (8 * math.pi)
	output = 0
	for i in range(n_repeat):
		output += a * (x[:, 2 * i + 1] - b * x[:, 2 * i] ** 2 + c * x[:, 2 * i] - r) ** 2 + s * (1 - t) * torch.cos(x[:, 2 * i]) + s
	output /= float(n_repeat)
	if flat:
		return -1.0*output.squeeze(0)
	else:
		return -1.0*output

neg_branin.dim = 0

def mcf(x):
	return (torch.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1)

mcf.dim = 0

def neg_mcf(x):
	return -1.0*(torch.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1)

neg_mcf.dim = 0

def sq(x):
	return torch.sqrt(1 - x[0]**2)

sq.dim = 0

# FmFn Functions

def sphere(x1, x2):
	return (x1**2 + x2**2)

sphere.pbounds = {'x1': (-5.12, 5.12), 'x2': (-5.12, 5.12)}

def branin(x1, x2):
	a = 1
	b = 5.1 / (4 * math.pi ** 2)
	c = 5.0 / math.pi
	r = 6
	s = 10
	t = 1.0 / (8 * math.pi)

	return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

branin.pbounds = {'x1': (-5, 10), 'x2': (0, 15)}

def fmfn_neg_branin(x1, x2):
	a = 1
	b = 5.1 / (4 * math.pi ** 2)
	c = 5.0 / math.pi
	r = 6
	s = 10
	t = 1.0 / (8 * math.pi)

	return -1.0 * (a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s)

neg_branin.pbounds = {'x1': (-5, 10), 'x2': (0, 15)}

def birdy(x1, x2):
	return (np.sin(x1) * np.exp((1 - np.cos(x2))**2) + np.cos(x2) * np.exp((1 - np.sin(x1)**2)) + (x1 - x2)**2) 

birdy.pbounds = {'x1': (-2*np.pi, 2*np.pi), 'x2': (-2*np.pi, 2*np.pi)}

def fmfn_neg_birdy(x1, x2):
	return -1.0 * (np.sin(x1) * np.exp((1 - np.cos(x2))**2) + np.cos(x2) * np.exp((1 - np.sin(x1)**2)) + (x1 - x2)**2) 

neg_birdy.pbounds = {'x1': (-2*np.pi, 2*np.pi), 'x2': (-2*np.pi, 2*np.pi)}
