# Date: March 01, 2019
# Google review rating with different attraction

# Dependencies
import pandas as pd 
import numpy as np 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection

# for plot styling
import seaborn as sns 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
print("libraries imported")
print("\n")


train_url = "C:/train.csv"
train = pd.read_csv(train_url)

# Drop last column from dataset
train = train.iloc[:, :-1]

# print("***** Train_Set *****")
# print(train.head())
# print("\n")

# print("***** Train_Set Describe *****")
# print(train.describe())
# print("\n")

# print(train.columns.values)
# print("\n")

# Check Null values
train.isna().head()

print("*****In the train set*****")
print(train.isna().sum())
print("\n")

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
print(train.isna().sum())
print("\n")

# Print visualize graph
    # train.hist()

    # train.plot(kind='hist', subplots=True, layout=(5,5), sharex=False)
    # plt.title("All category frequency with rating - Hist")
    # plt.show()

    # train.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
    # plt.title("All category frequency with rating - Density")
    # plt.show()

    # train.plot(kind='box', subplots=True, layout=(5,5), sharex=False)
    # plt.title("All category frequency with rating - box")
    # plt.show()

# fig, ax = plt.subplots(figsize=(30,10))
# plt.xticks(fontsize=20) # X Ticks
# plt.yticks(fontsize=20) # Y Ticks
# ax.set_title('Number of users per category', fontweight="bold", size=25) # Title
# ax.set_ylabel('Number of users', fontsize = 25) # Y label
# ax.set_xlabel('Category', fontsize = 25) # X label
# dp.groupby(['category']).count()['users'].plot(ax=ax, kind='bar')

array = train.values
X = array[:, 1:24]

X_train, X_test = model_selection.train_test_split(X, test_size=0.2, random_state=5)

print("*****Size train set*****")
print(X_train.shape)
print(X_train)
print("\n")

print("*****Size test set*****")
print(X_test.shape)
print("\n")

#KMeans
km = KMeans(n_clusters=5)
predictions  = km.fit(X)
y_KMeans = km.predict(X)
labels = km.labels_

print(predictions)
print(labels)

plt.scatter(X[:, 0], X[:, 1], c=y_KMeans, s=50, cmap='rainbow')

centroid = km.cluster_centers_
plt.scatter(centroid[:, 0], centroid[:, 1], c='black', s=200, alpha=0.5)
plt.show()

plt.scatter(X[:,0],X[:,1], c=labels, cmap='rainbow')  
plt.show()