from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

#kendi veri setimizi oluşturalım
x,_ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.title("örnek veri")

#kümeleme de kullanılan dört farkli linkage methodunu oluşturalım
linkage_methods = ["ward", "single", "average", "complete" ]

plt.figure()

for i, linkage_method in enumerate(linkage_methods, 1):
    
    model =AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    cluster_labels = model.fit_predict(x)
    #Dendrogram ile görselleştirelim.
    plt.subplot(2,4, i)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    dendrogram(linkage(x, method = linkage_method), no_labels = True)
    plt.xlabel("veri noktaları")
    plt.ylabel("uzaklık")
    
    #scatter point ile görselleştirelim.
    
    plt.subplot(2,4, i+4)
    plt.scatter(x[:,0], x[:,1], c=cluster_labels, cmap="viridis")
    plt.title(f"{linkage_method.capitalize()} linkage Clustering")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
