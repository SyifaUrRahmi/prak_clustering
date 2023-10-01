#Generate Data
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=750, factor=0.3, noise=0.1)
X = StandardScaler().fit_transform(X)

# Membuat model DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)

# Melakukan clustering pada data
cluster_labels = dbscan.fit_predict(X)

# Menampilkan hasil clustering
plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Hasil DBSCAN Clustering')
plt.show()