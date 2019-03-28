# Date: March 01, 2019
# Google review rating with different attraction

# Assign rows - Done
# Assign random centroid - Done
# Predict it
# Plot it


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
train_df = pd.DataFrame(train, index=['User', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7' ,'Category 8', 'Category 9', 'Category 10', 'Category 11', 'Category 12', 'Category 13', 'Category 14', 'Category 15', 'Category 16', 'Category 17' ,'Category 18' ,'Category 19', 'Category 20', 'Category 21' ,'Category 22', 'Category 23' , F'Category 24'])

# Drop last column from dataset
train = train.iloc[:, :-1]
train.drop(['User'], axis = 1, inplace = True)

print("***** Train_Set *****")
print(train.head())
print("\n")

print("***** Train_Set Describe *****")
print(train.describe())
print("\n")

print(train.columns.values)
print("\n")

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

# array = train.values
# X = array[1:500]
# Y = array[501:1000]

# # Assign random centroid

# #KMeans
# km = KMeans(n_clusters=3)
# predictions  = km.fit(X)
# y_KMeans = km.predict(Y)
# labels = km.labels_

# print(predictions)
# print(labels)

# plt.scatter(X, Y, c=y_KMeans, s=1000, cmap='rainbow')

# # centroid = km.cluster_centers_
# # plt.scatter(centroid[:, 0], centroid[:, 0], c='black', s=200, alpha=0.5)
# plt.show()

# # plt.scatter(X[:,0],Y[:,0], c=labels, cmap='rainbow')  
# # END

# ---------------------------------------------

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)