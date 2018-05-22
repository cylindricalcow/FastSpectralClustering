import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
#Landmark based representation algorithm from https://int8.io/large-scale-spectral-clustering-with-landmark-based-representation/
#Landmarks are vectors

def find_p(X, start=1, end=10):
    #make an elbow graph that allows user to set min_cluster_size. Min_cluster_size should
    #be the first one with a dramatic drop from ~10^2 to ~10^1.
    min_size=[]
    number_of_clusters=[]
    for i in range(start,end+1):
        min_size.append(i)   
        number_of_clusters.append(KMeans(n_clusters=i).fit(X).inertia_)
    _, ax=plt.subplots()
    
    ax.set(ylabel='Inertia', xlabel='Number of clusters', title='The elbow method')
    plt.xticks(np.arange(start,end, 1))    
    plt.plot(min_size,number_of_clusters)
    plt.show()    

def get_Landmarks(X, p, method="random"):
    if method=="random":
        N = len(X)
        perm= np.random.permutation(np.arange(N))
        print(perm)
        landmarks = X[perm[:p],:]
        return landmarks
    else:
        kmeans_model=KMeans(n_clusters=p).fit(X)
        return kmeans_model.cluster_centers_

#Test get_Landmarks
test1=False
if test1:
    X,y=make_blobs(centers=2, random_state=42)

    landmarks_rand=get_Landmarks(X,2)
    print(landmarks_rand)

    landmarks_kmeans=get_Landmarks(X,2,"KMeans")
    print(landmarks_kmeans)
    find_p(X)

#Compute Zhat
    
