
# coding: utf-8

# In[6]:


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


# In[7]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


# In[8]:


train_url = "C:/train.csv"
train = pd.read_csv(train_url)
train_df = pd.DataFrame(train, index=['User', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7' ,'Category 8', 'Category 9', 'Category 10', 'Category 11', 'Category 12', 'Category 13', 'Category 14', 'Category 15', 'Category 16', 'Category 17' ,'Category 18' ,'Category 19', 'Category 20', 'Category 21' ,'Category 22', 'Category 23' , F'Category 24'])


# In[9]:


train.drop(['User'], axis = 1, inplace = True)


# In[10]:


print(train.head())
print("\n")


# In[11]:


train.rename(columns = {'Category 1':'Churches', 'Category 2':'Resorts', 'Category 3':'Beaches', 'Category 4':'Parks', 'Category 5':'Theatres', 'Category 6':'Museums', 'Category 7' :'Malls', 'Category 8':'Zoo', 'Category 9':'Restaurants', 'Category 10':'Pubs/Bars', 'Category 11':'Local Services', 'Category 12':'Burger/Pizza Shops', 'Category 13':'Hotels/Other Lodgings', 'Category 14':'Juice Bars', 'Category 15':'Art Galeries', 'Category 16':'Dance Clubs', 'Category 17':'Swimming Pools', 'Category 18':'Gyms', 'Category 19':'Bakeries', 'Category 20':'Beauty & Spas', 'Category 21':'Cafes', 'Category 22':'View Points', 'Category 23':'Monuments', 'Category 24':'Gardens'}, inplace=True)


# In[12]:


train.isna().head()


# In[13]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")


# In[14]:


train.fillna(train.mean(), inplace=True)
print(train.isna().sum())
print("\n")


# In[15]:


train.hist()
train.plot(kind='hist', subplots=True, layout=(5,5), sharex=False)
plt.title("All category frequency with rating - Hist")
plt.show()

train.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
plt.title("All category frequency with rating - Density")
plt.show()

train.plot(kind='box', subplots=True, layout=(5,5), sharex=False)
plt.title("All category frequency with rating - box")
plt.show()


# In[20]:


train.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=16, ylabelsize=16, grid=False)    
plt.tight_layout(rect=(0, 0, 8, 8))

