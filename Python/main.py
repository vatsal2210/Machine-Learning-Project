#!/usr/bin/env python
# coding: utf-8

# Import Library

import time, os, sys
from copy import deepcopy
import numpy as np
import pandas as pd

# for plot styling
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

sys.stdout = open("result.txt", "w")
print("Group 16 - Machine Learning")
print("Result - Python \n")
start_time_script = time.time()

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
print("Libraries imported\n")

# Importing the dataset
print("Importing the dataset")
data = pd.read_csv('C:/train.csv', index_col = ["User"])
# Drop last column from dataset
data = data.iloc[:, :-1]

# Rename category names 
print("- Rename category names\n")
data.rename(columns = {'Category 1':'Churches', 'Category 2':'Resorts', 'Category 3':'Beaches', 'Category 4':'Parks', 'Category 5':'Theatres', 'Category 6':'Museums', 'Category 7' :'Malls', 'Category 8':'Zoo', 'Category 9':'Restaurants', 'Category 10':'Pubs/Bars', 'Category 11':'Local Services', 'Category 12':'Burger/Pizza Shops', 'Category 13':'Hotels/Other Lodgings', 'Category 14':'Juice Bars', 'Category 15':'Art Galeries', 'Category 16':'Dance Clubs', 'Category 17':'Swimming Pools', 'Category 18':'Gyms', 'Category 19':'Bakeries', 'Category 20':'Beauty & Spas', 'Category 21':'Cafes', 'Category 22':'View Points', 'Category 23':'Monuments', 'Category 24':'Gardens'}, inplace=True)

print('Dataset', data.shape, '\n')
print('Top 5 Records') 
print(data.head())
print('\n')

# Data Preparation
print('Check Null values')
print(data.isna().sum())
data.fillna(data.mean(), inplace=True)
print('\n')

# Fill missing values with mean column values in the train set
print('Replaced with mean value')
data.fillna(data.mean(), inplace=True)
print(data.isna().sum())
print('\n')

# Print visualize graph
data.plot(kind='hist', subplots=True, layout=(5,5), sharex=False)
plt.title("Group - 16 - All category frequency with rating - hist")

data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
plt.title("Group - 16 - All category frequency with rating - density")

data.plot(kind='box', subplots=True, layout=(5,5), sharex=False)
plt.title("Group - 16 - All category frequency with rating - box")

# generating correlation heatmap 
sns.heatmap(data.corr(), annot = True) 
plt.show() 

# PCA for selection features
from sklearn.decomposition import PCA 
  
pca = PCA(3) 
pca.fit(data) 
  
pca_data = pd.DataFrame(pca.transform(data)) 

print("PCA Top records")
print(pca_data.head())
print('\n')

f1 = data['Churches'].values
f2 = data['Resorts'].values
f3 = data['Beaches'].values
f4 = data['Parks'].values
f5 = data['Theatres'].values
f6 = data['Museums'].values
f7 = data['Malls'].values
f8 = data['Zoo'].values
f9 = data['Restaurants'].values
f10 = data['Pubs/Bars'].values
f11 = data['Local Services'].values
f12 = data['Burger/Pizza Shops'].values
f13 = data['Hotels/Other Lodgings'].values
f14 = data['Juice Bars'].values
f15 = data['Art Galeries'].values
f16 = data['Dance Clubs'].values
f17 = data['Swimming Pools'].values
f18 = data['Gyms'].values
f19 = data['Bakeries'].values
f20 = data['Beauty & Spas'].values
f21 = data['Cafes'].values
f22 = data['View Points'].values
f23 = data['Monuments'].values

X = np.array(list(zip(f1, f2)))

# Custom Model - Python
print("--------------------------------")
print("Custom Model in Python \n")
# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X), size=k)

# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X), size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print('Centroid')
print(C)
print('\n')

# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)

# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))

# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)

stime = time.time()
# Loop will run till the error becomes zero
while error != 0:
    
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
        
    # Storing the old centroid values
    C_old = deepcopy(C)
    
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

print("Time for custom model - Assigning each value to its closest cluster: %.3fs" % (time.time() - stime))
print('\n')

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

plt.rcParams['figure.figsize'] = (16, 9)
plt.show()

# ## Prebuild Library - Python
print("--------------------------------")
print("KMeans prebuild library in Python \n")

from sklearn.cluster import KMeans

# Initializing KMeans
kmeans = KMeans(n_clusters=3)

# Fitting with inputs
stime = time.time()
kmeans = kmeans.fit(X)
print("Time for fitting KMeans: %.3fs" % (time.time() - stime))
print('\n')

# Predicting the clusters
stime = time.time()
labels = kmeans.predict(X)
print("Time for prediction KMeans: %.3fs" % (time.time() - stime))
print('\n')

# Getting the cluster centers
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print('Comparing with scikit-learn centroids')
print('From Scratch')
print(C) # From Scratch
print('From sci-kit learn')
print(centroids) # From sci-kit learn

# ## Evaluation
Sum_of_squared_distances = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], c=labels)
ax.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=1000)

print('\n------------------------')
stop_time_script = time.time()
print("Code run in %.2fs" % (stop_time_script - start_time_script))
sys.stdout.close()