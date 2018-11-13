
# coding: utf-8

# In[1]:


from utils import * 
import sys 
from math import exp
import matplotlib
import numpy as np
from numpy import *
from matplotlib import pyplot as plt


# In[2]:


def loadPoints(filename):#load data
    inputf = open(filename, "r")  
    lines = inputf.readlines()
    # number of data points and dimension
    nData = len(lines) #number of data
    nDim = 2 #dimension
    
    # create data matrix
    data_matrix = [[0]*nDim for i in range(nData)]
    
    i = 0
    for line in lines:
        info = line.strip().split(',')
        for j in range(nDim):
            data_matrix[i][j] = float(info[j]) 
        i = i+1          
    inputf.close()                
    return data_matrix,nData


# In[3]:


def squaredDistance(vec1, vec2):
    sum = 0 
    dim = len(vec1) 
    
    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) 
    
    return sum


# In[4]:


def computeSSE(data, centers, clusterID):
    """
    objective function: calculate Sum of Squared Errors
    :param data: data points: list of list [[a,b],[c,d]....]
    :param centers: center points: list of list [[a,b],[c,d]]
    :param clusterID: list：0 or 1
    :return: SSE
    """
    sse = 0 
    nData = len(data) #
    for i in range(nData):
        c = clusterID[i]
        sse += squaredDistance(data[i], centers[c]) 
        
    return sse 


# In[5]:


def updateClusterID(data, centers):
    """
    assign the closet center to each data point
    :param data: data points: list of list [[a,b],[c,d]....]
    :param centers: center points: list of list [[a,b],[c,d]]
    :return: clusterID:list
    """
    nData = len(data) # number of data
    nCenters = len(centers) # number of centers
    
    clusterID = [0] * nData
    dis_Centers = [0] * nCenters # the distance between one data point to each center, since K=2, list [len1,len2]
    
    # assign the closet center to each data point
    for i in range(nData):
        for c in range(nCenters):
            dis_Centers[c] = squaredDistance(data[i], centers[c])
        clusterID[i] = dis_Centers.index(min(dis_Centers))
    return clusterID


# In[6]:


def updateCenters(data, clusterID, K):
    """
    Recalculate the center point
    :param data: data points: list of list [[a,b],[c,d]....]
    :param clusterID: list:0 or 1
    :param K: number of clusters 
    :return: centers:list of list
    """
    nDim = len(data[0]) # the dimension
    centers = [[0] * nDim for i in range(K)] # list of list [[a,b],[c,d]]

    clusterids = sorted(set(clusterID)) 
    for cid in clusterids:
        # get the index from clusterID where data points belong to the same cluster
        indices = [i for i, j in enumerate(clusterID) if j == cid]
        cluster = [data[i] for i in indices]
        if len(cluster) == 0:
            #If a cluster doesn't have any data points, leave it to ALL 0s
            centers[cid] = [0] * nDim
        else:
            # compute the centroids (i.e., mean point) of each cluster
            centers[cid] = [float(sum(col))/len(col) for col in list(zip(*cluster))]
    return centers 


# In[7]:


def kmeans(data, centers, maxIter = 100, tol = 1e-8):
    """
    :param data: data points: list of list [[a,b],[c,d]....]
    :param centers: center points: list of list [[a,b],[c,d]]
    :param maxIter: max number of iterations
    :param tol: change rate
    :return: clusterID: list
    """
    nData = len(data) 

    K = len(centers) 
    
    clusterID = [0] * nData
    
    if K >= nData:
        for i in range(nData):
            clusterID[i] = i
        return clusterID

    nDim = len(data[0]) 
    
    lastDistance = 1e100
    
    for niter in range(maxIter):
        clusterID = updateClusterID(data, centers) 
        centers = updateCenters(data, clusterID, K)      
        curDistance = computeSSE(data, centers, clusterID) # objective function
        print ("# of iterations:", niter) 
        print ("SSE = ", curDistance)
        if lastDistance - curDistance < tol or (lastDistance - curDistance)/lastDistance < tol:
            return clusterID   
        lastDistance = curDistance

    return clusterID


# In[8]:


def kernel(data, sigma):
    """
    RBF kernel-k-means
    :param data: data points: list of list [[a,b],[c,d]....]
    :param sigma: Gaussian radial basis function
    :return: Gram:n*n matrix
    """
    nData = len(data)
    Gram = [[0] * nData for i in range(nData)] # nData x nData matrix

    for i in range(nData):
        for j in range(i,nData):
            if i != j: # diagonal element of matrix = 0
                # RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
                square_dist = squaredDistance(data[i],data[j])
                base = 2.0 * sigma**2
                Gram[i][j] = exp(-square_dist/base)
                Gram[j][i] = Gram[i][j]
    return Gram


# In[9]:


def calLaplacianMatrix(WMatrix): 
    """
    :param WMatrix: Weight matrix:n*n matrix
    :return: norm_L : Normalized Laplacian matrix
    """
    # compute the Degree Matrix: D=sum(W) 
    DMatrix = np.diag(np.sum(WMatrix, axis=0))
    
    # compute the Laplacian Matrix: L=D-W 
    LMatrix = DMatrix - WMatrix
    
    # normalize
    D_sqrt = np.diag(1.0/(np.sum(WMatrix, axis=0))**0.5)
    norm_L = np.dot(np.dot(D_sqrt,LMatrix),D_sqrt)

    return norm_L


# In[10]:


def getKSmallestEigVec(LMatrix,k):
    """
    :param LMatrix: Normalized Laplacian matrix
    :param k: Select k eigenvectors
    :return: eigval: list of eigenvalues
    :return: H: a matrix of minimum k eigenvectors
    """
    eigval,eigvec = np.linalg.eig(LMatrix)
    dim = len(eigval)
    
    Eigval = zip(eigval, range(dim))
    Eigval = sorted(Eigval, key=lambda x:x[0])
    H = np.vstack([eigvec[:,i] for (r0,i) in Eigval[:k]]).T

    return eigval, H


# In[11]:


def plot_cluster(results):
    fig = plt.figure()

    plt.figure(figsize=(8, 5), dpi=80)
    ax = plt.subplot(111)

    indices = [i for i, j in enumerate(results) if j == 0]
    cluster = [raw_data[i] for i in indices]
    zip_cluster = list(zip(*cluster))
    type0_x = list(zip_cluster[0])
    type0_y = list(zip_cluster[1])
    p1 = ax.scatter(type0_x,type0_y,marker = '*',color = 'r',label='1')

    indices = [i for i, j in enumerate(results) if j == 1]
    cluster = [raw_data[i] for i in indices]
    zip_cluster = list(zip(*cluster))
    type1_x = list(zip_cluster[0])
    type1_y = list(zip_cluster[1])
    p2 = ax.scatter(type1_x,type1_y,marker = 'o',color ='b',label='2')

    ax.legend((p1, p2), ('cluster1', 'cluster2',), loc=2)
    plt.show()


# In[12]:


dataFilename = "moons.txt"
raw_data,nData= loadPoints(dataFilename) 

sigma = 0.1
K = 2  
print ('K=',K)

Gdata = kernel(raw_data, sigma)  
Laplacian = calLaplacianMatrix(Gdata)
eigval,eigvec = getKSmallestEigVec(Laplacian,2)

centers = []
random_array = np.random.randint(0,999,2)
for i in random_array:
    centers.append(eigvec[i])
print("The index of initial cluster center point:",random_array)

results = kmeans(eigvec, centers)
plot_cluster(results)

