# Date: March 01, 2019
# Author: Vatsal Shah
# Google review rating with different attraction

# Dependencies
import pandas as pd 
import numpy as np 
import random as rd

# for plot styling
import matplotlib.pyplot as plt
import matplotlib.cm as cm
print("libraries imported")
print("\n")


dataset_url = "C:/train.csv"
dataset = pd.read_csv(dataset_url)
dataset.describe()
print(dataset.head())
print(dataset.describe())

# Drop last column from dataset
X = dataset.iloc[:, :-1].values
print(X)

m=X.shape[0] #number of training examples
n=X.shape[1] #number of features. Here n=25
n_iter=100

K=5 # number of clusters

print(m)
print(n)

# Step 1: Initialize the centroids randomly from the data points:
    # Centroids is a n x K dimentional matrix, where each column will be a centroid for one cluster.
# Step 2.a: For each training example compute the euclidian distance from the centroid and assign the cluster based on the minimal distance
# Step 2.b: We need to regroup the data points based on the cluster index C and store in the Output dictionary and also compute the mean of separated clusters and assign it as new centroids. Y is a temporary dictionary which stores the solution for one particular iteration.

#step 1
Centroids=np.array([]).reshape(n,0)
for i in range(K):
    rand=rd.randint(0,m-1)
    Centroids=np.c_[Centroids,X[rand]]

for i in range(n_iter):
     
     #step 2.a
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X - Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     
     #step 2.b
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
      for k in range(K):
          Y[k+1]=Y[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y

# the original unclustered data
plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.show()

# the clustered data
color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()

