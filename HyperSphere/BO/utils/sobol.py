import os, sys, stat
import tempfile
import numpy as np
import torch
import pathlib

def sobol_generate(n_dim, n_point, n_skip=0):
	if n_dim > 1111:
		raise Exception('This program supports sobol sequence of dimension up to 1111')
	while True:
		try:
			sequence_file = tempfile.NamedTemporaryFile('r')
			filename = sequence_file.name
			cmd_file = os.path.join(pathlib.Path(__file__).parent.absolute(), 'sobol_c/sobol')
			cmd = cmd_file + ' ' + str(int(n_dim)) + ' ' + str(int(n_point)) + ' ' + str(int(n_skip)) + ' ' + filename
			if (os.error(os.system(cmd))):	
				os.chmod(cmd_file, stat.S_IRWXU)   # stat.S_IRWXU : read, write and execute permission for owner
				print(f"\n{cmd_file}: Changed File Permissions to stat.S_IRWXU")
			os.system(cmd)
			sequence = np.fromfile(sequence_file.file, dtype=np.float64).astype(np.float32)
			sequence = sequence.reshape([n_point, n_dim])
			sequence_file.close()
			return torch.from_numpy(sequence)
		except ValueError:
			print('%d data in file, but reshape is in (%d, %d)' % (sequence.size, n_point, n_dim))
			continue


if __name__ == '__main__':
	print(sobol_generate(1111, 10001, 0)[-1]) # added parentheses