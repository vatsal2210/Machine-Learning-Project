
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


# In[3]:


import seaborn as sns 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
print("libraries imported")
print("\n")


# In[18]:


train_url = "F:/Raja/MS/subjects/Winter/Machine Learning/Project/train1.csv"
train = pd.read_csv(train_url)
train_df = pd.DataFrame(train)


# In[19]:


train = train.iloc[:, :-1]


# In[20]:


print("***** Train_Set *****")
print(train.head())
print("\n")

print("***** Train_Set Describe *****")
print(train.describe())
print("\n")

print(train.columns.values)
print("\n")


# In[21]:


# Check Null values
train.isna().head()

print("*****In the train set*****")
print(train.isna().sum())
print("\n")


# In[17]:


# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
print(train.isna().sum())
print("\n")


# In[ ]:


train_url['sum'] = train_url.drop('User', axis=1).sum(axis=1)
print (train_url)

