from knn import KNN
import numpy as np
import matplotlib.pyplot as plt

# Fit knn on sample of size 30
# Consider plot the way future samples get labelled for varying k

N = 50
K = 4
true_labels = np.random.randint(0, K, N)
true_means = np.random.rand(K, 2)*10
covs = np.empty((K, 2, 2))
for i in range(K):
	cov_ = np.random.rand(2)
	covs[i] = np.outer(cov_, cov_)
	
X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(5, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.title('Training sample')
plt.xticks([])
plt.yticks([])

k1 = KNN()
k3 = KNN()
k10 = KNN()
k1.fit(X, true_labels, k=1, reg_clas=True)
k3.fit(X, true_labels, k=3, reg_clas=True)
k10.fit(X, true_labels, k=10, reg_clas=True)

N = 10000
true_labels = np.random.randint(0, K, N)

X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(5, 1, 2)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.title('True large sample')
plt.xticks([])
plt.yticks([])

classified_labels1 = np.empty(N, int)
for i in range(N):
	classified_labels1[i] = k1.predict(X[i])

plt.subplot(5, 1, 3)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels1)
plt.title('Classified large sample k = 1')
plt.xticks([])
plt.yticks([])

classified_labels3 = np.empty(N, int)
for i in range(N):
	classified_labels3[i] = k3.predict(X[i])

plt.subplot(5, 1, 4)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels3)
plt.title('Classified large sample k = 3')
plt.xticks([])
plt.yticks([])

classified_labels10 = np.empty(N, int)
for i in range(N):
	classified_labels10[i] = k10.predict(X[i])

plt.subplot(5, 1, 5)
plt.scatter(X[:, 0], X[:, 1], c=classified_labels10)
plt.title('Classified large sample k = 10')
plt.xticks([])
plt.yticks([])

plt.show()
