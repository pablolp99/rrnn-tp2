import numpy as np
import logging
import setup_logger

logger = logging.getLogger("UnsupervisedModelLogger")

class UnsupervisedModel:
	VALID_ALGORITHMS = [
		'hebb',
		'oja_sim',
		'oja_gen',
		'sanger'
	]
	
	def __init__(self, dataset, input_size, output_size, lr=0.001, algorithm='hebb', normalize=True, log="NOTSET"):
		if algorithm not in self.VALID_ALGORITHMS:
			raise ValueError("Algorithm '{}' does not exist as a valid algorithm".format(algorithm))
		
		if normalize:
			self.data = self.__normalize_dataset(dataset)
		else:
			self.data = dataset
		
		self.__set_log_level(log)   
		self.lr = lr
		self.algorithm = algorithm
		self.n = input_size
		self.m = output_size
		self.w = self.__init_w()
		self.trained = False
		
	@staticmethod
	def __normalize_dataset(dataset):
		# Normalizes data in the range (0, 1)
		ret = []
		for d in dataset:
			ret.append((d - d.min()) / (d.max() - d.min()))
		return np.array(ret)
		
	def __init_w(self):
		# Random matrix of weights, with shape (N, M)
		# and values come from a N(0, 1) distribution
		return np.random.normal(0, 1, (self.n, self.m))
	
	def __set_log_level(self, log):
		# Sets log level
		if log in ["INFO", "DEBUG", "NOTSET"]:
			logging.basicConfig(level=eval("logging.{}".format(log)))
		else:
			raise ValueError("No log level named '{}'".format(log))
		return

	def __str__(self):
		# Printable representation
		s = "Algorithm: '{}' - W shape: {} - Trained: {}\n".format(self.algorithm, self.w.shape, self.trained)
		s += self.w.__repr__()
		return s
	
	def __optimizer(self, xit, y, i, j):
		# Selects the optimizer, setted previously
		# with the `algorithm` parameter
		if self.algorithm == 'hebb':
			return self.__hebb(xit, y)
		elif self.algorithm == 'oja_sim':
			return self.__oja_sim(xit, y, i)
		elif self.algorithm == 'oja_gen':
			return self.__oja_gen(xit, y, i)
		elif self.algorithm == 'sanger':
			return self.__sanger(xit, y, i, j)

	def __hebb(self, xit, y):
		# This does never cycle, thus
		# Xit = 0
		return
	
	def __oja_sim(self, xit, y, i):
		# This does just iterate 1 time, thus
		# Xit += Wij * Yj
		xit += y[0] * self.w[i][0]
		return xit
	
	def __oja_gen(self, xit, y, i):
		# This iterates `M` times, thus
		# Xit = Sum_0^M Yk * Wik
		for k in range(self.m):
			xit += y[k] * self.w[i][k]
		return xit
	
	def __sanger(self, xit, y, i, j):
		# This iterates `j` times
		# Xit = Sum_0^j Yk * Wik
		for k in range(j):
			xit += y[k] * self.w[i][k]
		return xit
	
	def train(self):
		# This method trains the model with
		# the dataset specified in the init method
		with tqdm(total=len(self.data)) as pbar:
			for idx, x in enumerate(self.data):
				logging.info("Data Iteration Nº {}".format(idx))
				y = x.dot(self.w)
				dw = np.zeros(self.w.shape)
				for i in range(self.n):
					for j in range(self.m):
						xit = 0
						self.__optimizer(xit, y, i, j)
						dw[i][j] = (self.lr * (x[i] - xit) * y[j])
				logging.debug("Updating weight matrix values")
				self.w += dw
				pbar.update(1)
		self.trained = True
				
	def predict(self, v):
		# This method
		if !self.trained:
			logging.warn("Model not trained")
		return v.dot(self.w)
