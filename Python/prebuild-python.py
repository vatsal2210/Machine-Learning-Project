# Date: March 01, 2019
# Google review rating with different attraction

# Dependencies
import pandas as pd 
import numpy as np 

# for plot styling
import seaborn as sns 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

print("libraries imported")
print("\n")

# Importing the dataset
data = pd.read_csv('../train.csv', index_col = ["User"])

dataFrame = pd.DataFrame(data)
print('DataFrame', dataFrame.shape)

# Rename category names 
dataFrame.rename(columns = {'Category 1':'Churches', 'Category 2':'Resorts', 'Category 3':'Beaches', 'Category 4':'Parks', 'Category 5':'Theatres', 'Category 6':'Museums', 'Category 7' :'Malls', 'Category 8':'Zoo', 'Category 9':'Restaurants', 'Category 10':'Pubs/Bars', 'Category 11':'Local Services', 'Category 12':'Burger/Pizza Shops', 'Category 13':'Hotels/Other Lodgings', 'Category 14':'Juice Bars', 'Category 15':'Art Galeries', 'Category 16':'Dance Clubs', 'Category 17':'Swimming Pools', 'Category 18':'Gyms', 'Category 19':'Bakeries', 'Category 20':'Beauty & Spas', 'Category 21':'Cafes', 'Category 22':'View Points', 'Category 23':'Monuments', 'Category 24':'Gardens'}, inplace=True)

# Drop last column from dataset
# dataFrame = dataFrame.iloc[:, :-1]
print("***** DataFrame *****")
print(dataFrame.head())
print("\n")

print("***** DataFrame Describe *****")
print(dataFrame.describe())
print("\n")

print(dataFrame.isna().sum())
print("\n")
dataFrame.fillna(dataFrame.mean(), inplace=True)

# Fill missing values with mean column values in the train set
dataFrame.fillna(dataFrame.mean(), inplace=True)
print(dataFrame.isna().sum())
print("\n")

# Print visualize graph
dataFrame.plot(kind='hist', subplots=True, layout=(5,5), sharex=False)
plt.title("Group - 16 - All category frequency with rating - hist")

dataFrame.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
plt.title("Group - 16 - All category frequency with rating - density")

dataFrame.plot(kind='box', subplots=True, layout=(5,5), sharex=False)
plt.title("Group - 16 - All category frequency with rating - box")

# generating correlation heatmap 
sns.heatmap(dataFrame.corr(), annot = True) 
plt.show() 

# Using custom library
from sklearn.cluster import KMeans

# from sklearn.cluster import KMeans 
clusters = 3
kmeans = KMeans(n_clusters = clusters) 
kmeans.fit(dataFrame) 
# Predicting the clusters
labels = kmeans.predict(dataFrame)
# Getting the cluster centers
centroids = kmeans.cluster_centers_


# PCA
from sklearn.decomposition import PCA 
pca = PCA(3) 
pca.fit(dataFrame) 
pca_data = pd.DataFrame(pca.transform(dataFrame)) 
print(pca_data.head())

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig = plt.figure() 
ax = fig.add_subplot(111, projection = '3d') 
ax.scatter(pca_data[0], pca_data[1], pca_data[2],  
           c = list(map(lambda label : colors[label], 
                                            kmeans.labels_))) 
   
str_labels = list(map(lambda label:'% s' % label, kmeans.labels_)) 
   
list(map(lambda data1, data2, data3, str_label: 
        ax.text(data1, data2, data3, s = str_label, size = 16.5, 
        zorder = 20, color = 'k'), pca_data[0], pca_data[1], 
        pca_data[2], str_labels)) 
   
plt.show() 