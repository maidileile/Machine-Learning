# k-means-starter.py
# parsons/28-feb-2017
#
# Running k-means on the iris dataset.
#
# Code draws from:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
print(X)
print(y)

k =3

#Cx = np.random.randint(0, np.max(X[:,0]) , size = k)
#Cy = np.random.randint(0, np.max(X[0,]) , size = k)
#C = np.array(list(zip(Cx,Cy)), dtype = np.float32)
#print(C)

#
# Put your K-means code here.
#
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#
# Plot everything
#
plt.subplot( 1, 2, 1 )
# Plot the original data 
plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
plt.scatter(X[:, 0], X[:, 1], , c = y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker = '*', c = 'black')
plt.show()

