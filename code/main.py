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


train_url = "D:/College/Graduate/Term/Winter 2019/ML/Project/code/train.csv"
train = pd.read_csv(train_url)

# Drop last column from dataset
train = train.iloc[:, :-1]

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
km = KMeans(n_clusters=3)
predictions  = km.fit(X_train)
# km.predict(X_test)
print(predictions)
labels = km.labels_
centroid = km.cluster_centers_