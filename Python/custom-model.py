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
plt.show()

# Using custom library

# Getting the values and plotting it
from sklearn.decomposition import PCA 
pca = PCA(3) 
pca.fit(dataFrame) 
pca_data = pd.DataFrame(pca.transform(dataFrame)) 
print(pca_data.head())

X = np.array(list(zip(pca_data)))
plt.scatter(pca_data[:,0],pca_data[:,1], c='black', s=200)

# Getting the values and plotting it
f1 = data['Category 1'].values
f2 = data['Category 2'].values
f3 = data['Category 3'].values
f4 = data['Category 4'].values
f5 = data['Category 5'].values
f6 = data['Category 6'].values
f7 = data['Category 7'].values
f8 = data['Category 8'].values
f9 = data['Category 9'].values
f10 = data['Category 10'].values
f11 = data['Category 11'].values
f12 = data['Category 12'].values
f13 = data['Category 13'].values
f14 = data['Category 14'].values
f15 = data['Category 15'].values
f16 = data['Category 16'].values
f17 = data['Category 17'].values
f18 = data['Category 18'].values
f19 = data['Category 19'].values
f20 = data['Category 20'].values
f21 = data['Category 21'].values
f22 = data['Category 22'].values
f23 = data['Category 23'].values
f24 = data['Category 24'].values

X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=200)
# f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24

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
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)

# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))

# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)

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

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')