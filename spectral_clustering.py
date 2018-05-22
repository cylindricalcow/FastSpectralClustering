import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
import scipy

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

#Compute Zhat
def gaussian_kernel(dist_mat, bandwidth):
    return np.exp(-dist_mat / (2*bandwidth**2))

def compose_Sparse_ZHat_Matrix(X, landmarks, bandwidth, r):
    dist_mat=pairwise_distances(X,landmarks)
    sim_mat=gaussian_kernel(dist_mat, bandwidth)
    
    Zhat = np.zeros(sim_mat.shape)

    for i in range(Zhat.shape[0]):
        #may need j.sort.selectperm
        top_Landmarks_indices = np.argsort(-sim_mat[i,:])[:r]
        top_Landmarks_coefs = sim_mat[i,top_Landmarks_indices]
        top_Landmarks_coefs /= np.sum(top_Landmarks_coefs)
        Zhat[i, top_Landmarks_indices] = top_Landmarks_coefs
    #May be wrong    
    diagm=np.sum(Zhat, axis=0)**(-1/2)
    return diagm*Zhat

def LSC_Clustering(X, n_clusters, n_landmarks, method, non_zero_landmark_weights, bandwidth):
    landmarks = get_Landmarks(X, n_landmarks, method)
    Zhat = compose_Sparse_ZHat_Matrix(X, landmarks, bandwidth, non_zero_landmark_weights)
    svd_result = np.linalg.svd(Zhat, full_matrices=False)[0]
    clustering_result = KMeans(n_clusters=n_clusters).fit(svd_result)
    return clustering_result
#Test get_Landmarks
test1=False
if test1:
    X,y=make_blobs(centers=2, random_state=42)

    landmarks_rand=get_Landmarks(X,2)
    print(landmarks_rand)

    landmarks_kmeans=get_Landmarks(X,2,"KMeans")
    print(landmarks_kmeans)
    find_p(X)
    print(compose_Sparse_ZHat_Matrix(X, landmarks_kmeans, 1, 5))

test2=True
if test2:
    X,y=make_blobs(centers=2, random_state=42)
    labels=LSC_Clustering(X, 2, 4, "Kmeans", 4, 0.5).labels_
    
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.show()
