import numpy

class KNN:

	# assumes one-hot encoding is used for categorical variables
	# X is a 2d numpy array
	# y 1d np array(vector)
	def fit(self, X, y, k=1, standardize=True, 
			 reg_clas=False, missing_vals=False,
			 k_func=None, measure=None):
		self.X = X.astype(float)
		self.k = k
		self.y = y.astype(float)
		self.N, self.d = X.shape
		self.std = standardize
		if self.std:
			self.mu = np.mean(X, axis=0, keepdims=True)
			X -= self.mu
			self.s = np.sqrt(np.sum(X**2, axis=0, keepdims=True))
			X /= self.s
		if k_func in None:
			if reg_clas:
				self.f = _mode
			else:
				self.f = _mean
		else:
			self.f = k_func
		if measure is None:
			self.m = _euclidean

	"""
	Functions of k-nearest neighbours
	
	Expects y to be a np vector.
	"""
	def _mean(y):
		return np.mean(y)
	
	def _mode(y):
		unique, counts = np.unique(y, return_counts=True)
		return unique[np.argmax(counts)]
	
	"""
	Distance measures

	Expects x and y to be np vectors.
	"""
	def _euclidean(x, y):
		d = y - x
		np.sqrt(np.sum(d**2))
	
	"""
	Predicts output for a single observation.
	"""
	def predict(self, x):
		x = x.astype(float)
		if self.std:
			x -= self.mu
			x /= self.s
		dists = np.empty((self.N,), dtype=float)
		for i in range(self.N):
			dists[i] = self.m(X[i,], x)
		# TODO: potentially use nlogn speed sort?
		# question is what will be faster, O(kn) or O(nlogn)
		# will the constant factor be too great in O(nlogn)?
		order = np.arange(0, self.N)
		for i in range(self.k):
			for j in range(i+1, self.N):
				if dists[j] < dists[i]:
					temp = dists[i]
					dists[i] = dists[j]
					dists[j] = temp
					temp = order[i]
					order[i] = order[j]
					order[j] = temp
		return self.f(dists[order[:self.k]])
