try:
	import numpy as np
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import seaborn as sns
except:
	raise ModuleNotFoundError("Some modules could not be found. Try installing the `requirements.txt`")

class UnsupervisedModel:
	
	VALID_ALGORITHMS = [
		'oja_gen',
		'sanger'
	]
	
	def __init__(self, dataset, input_size, output_size, error=0.1, max_epochs=10, normal_params=(0, 0.1), lr=0.001, algorithm='sanger', normalize=True):
		if algorithm not in self.VALID_ALGORITHMS:
			raise ValueError("Algorithm '{}' does not exist as a valid algorithm".format(algorithm))
		
		self.data = self.__normalize_dataset(dataset)
		self.error = error
		self.max_epochs = max_epochs
		self.lr = lr
		self.algorithm = algorithm
		self.n = input_size
		self.m = output_size
		self.w = self.__init_w(normal_params)
		self.w_mean, self.w_var = normal_params
		self.trained = False
		self.error_convergence = list()
		self.epoch = 1
		
	@staticmethod
	def __normalize_dataset(dataset):
		# Normalizes data in the range (0, 1)
		ret = []
		for d in dataset.T:
			ret.append((d - d.min()) / (d.max() - d.min()))
		return np.array(ret).T
		
	def __init_w(self, normal_params):
		# Random matrix of weights, with shape (N, M)
		# and values come from a N(0, 1) distribution
		return np.random.normal(normal_params[0], normal_params[1], (self.n, self.m))

	def __str__(self):
		# Printable representation
		s = "Algorithm: '{}' - W shape: {} - Trained: {} - LR: {}\n".format(self.algorithm, self.w.shape, self.trained, self.lr)
		s += "Normal Params: mean: {} - var: {}\n".format(self.w_mean, self.w_var)
		s += "Data mean: {} - Data var: {}".format(round(self.data.mean(), 3), round(self.data.var(), 3))
		return s
	
	def __optimizer(self, x):
		# Selects the optimizer, setted previously
		# with the `algorithm` parameter
		if self.algorithm == 'oja_gen':
			return self.__oja_gen(x)
		elif self.algorithm == 'sanger':
			return self.__sanger(x)

	def __oja_gen(self, x):
		y = x.dot(self.w)
		z = y.dot(self.w.T)
		return self.lr * np.outer(x-z, y)

	def __sanger(self, x):
		y = x.dot(self.w)
		d = np.triu(np.ones((self.m, self.m)))
		z = self.w.dot(y.T*d)
		return self.lr * np.outer(x - z.sum(axis=1), y.T) * self.mult
		# z = self.w.dot(y.T.dot(d))
		# return self.lr * np.outer(x - z, y.T) * self.mult

	def train(self):
		# This method trains the model with
		# the dataset specified in the init method
		o = self.error+1
		self.mult = 1
		self.epoch = 1

		pbar = tqdm(total=len(self.data))

		while abs(o) >= self.error and self.epoch <= self.max_epochs-1:
			pbar.set_description("Epoch {} - Orthogonality: Inf".format(self.epoch, round(o,2)))
			pbar.reset()
			for idx, x in enumerate(self.data):
				self.w += self.__optimizer(x)

				o = np.sum(self.w.T.dot(self.w)) - self.m

				self.mult = 1 if o >= 0 else -1

				pbar.set_description("Epoch {} - Orthogonality: {}".format(self.epoch, round(o,4)))
				pbar.update(1)
			self.epoch+=1
			self.error_convergence.append(o)
		pbar.close()
		msg = "Training concluded with an Orthogonality value of {} in {} epochs using a LR={}".format(round(o, 3), self.epoch, self.lr)
		print(msg)
		return o
				
	def predict(self, v):
		# This method predicts a vector
		return v.dot(self.w)

	def save_model(self, model_name):
		np.save(model_name, self.w)
		return

	def load_model(self, model_name):
		self.w = np.load(model_name + '.npy', allow_pickle=False)
		self.trained = True
		return

	def plot_convergence(self, save=False, filename=None):
		sns.set_style("darkgrid")
		sns.lineplot(x=list(range(1, self.epoch)), y=self.error_convergence)
		plt.xlabel('Epoch')
		plt.ylabel('Orthogonality')
		if save:
			if filename is None:
				raise ValueError("Filename not indicated")
			plt.savefig(filename)

		plt.show()
