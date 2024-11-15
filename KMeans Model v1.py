#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import everything we need for modeling
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import itertools
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os
new_directory = "/Users/kritc/Downloads"
os.chdir(new_directory)


# In[6]:


#load data
data = pd.read_csv('Agony - Health And Zillow Population Fixed.csv')

#filtering only on 2021 data for kmeans
data = data[data['Year'] == 2021]

#drop the duplicate rows 
data = data.drop_duplicates()

# Drop rows with NaN values
data = data.dropna()

data.head()


# In[7]:


#get predictors (not State, FIPS, County, Year)
predictors = data.drop(columns=['FIPS', 'State', 'Year', 'County'])

#scale predictors
scaler = StandardScaler()
scaled_predictors = scaler.fit_transform(predictors)


# In[8]:


# create function for kcv to find optimal number of clusters
def kcv_kmeans (kcv_data, clusters, n_splits = 5):
    sse = []
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 123)
    inertia = {}

    for k in clusters:
        fold_inertias = []
        for train, test in kf.split(kcv_data):
            km_kcv = KMeans(n_clusters=k, random_state = 123)
            km_kcv.fit(kcv_data[train])
            #looking for inertia so we an use the elbow method
            fold_inertia = km_kcv.inertia_
            fold_inertias.append(fold_inertia)
            
        avg_inertia = np.mean(fold_inertias)
        inertia[k] = avg_inertia
    return inertia


# In[18]:


# getting optimal_k using kcv function on scaled predictors
clusters = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
inertias = kcv_kmeans(scaled_predictors, clusters = clusters)

plt.plot(list(inertias.keys()), list(inertias.values()), marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Average Inertia")
plt.title("Optimal K")
plt.show()


# In[19]:


# create function for kmeans model to find similar counties - will use optimal k
def kmeans(model_data, optimal_k):
    km = KMeans(n_clusters = optimal_k, random_state = 123)
    assigned_clusters = km.fit_predict(model_data)
    
    return assigned_clusters  


# In[20]:


#run kmeans using optimal k
assigned_clusters = kmeans(scaled_predictors, optimal_k = 200)
assigned_clusters


# In[21]:


#add clusters to data
data['cluster'] = assigned_clusters
#data.head()
data.shape


# In[22]:


#putting data w/ clusters into a csv file
data.to_csv('kmeans_model_data_v1.csv', index=False)

