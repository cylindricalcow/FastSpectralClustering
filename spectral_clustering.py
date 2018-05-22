import numpy as np
import sklearn.cluster.Kmeans as Kmeans

#Landmark based representation algorithm from https://int8.io/large-scale-spectral-clustering-with-landmark-based-representation/
#Landmarks are vectors

def find_p(X, start=2, end=15):
    #make an elbow graph that allows user to set min_cluster_size. Min_cluster_size should
    #be the first one with a dramatic drop from ~10^2 to ~10^1.
    min_size=[]
    number_of_clusters=[]
    for i in range(start,end+1):
        min_size.append(i)   
        number_of_clusters.append(cluster(X,method,min_cluster_size=i)[0])
    _, ax=plt.subplots()
    
    ax.set(ylabel='Inertia', xlabel='Number of clusters', title='The elbow method')
    plt.xticks(np.arange(start,end, 1))    
    plt.plot(min_size,number_of_clusters)
    plt.show()    

def getLandmarks(X, p, method="random"):
    if method=="random":
        N = len(X)
        landmarks = X[:, np.random.permutation(np.arange(N))[:p]]
        return landmarks
    else:
        kmeans_model=KMeans(n_clusters=p).fit(X)
        return kmeans_model.cluster_centers_
