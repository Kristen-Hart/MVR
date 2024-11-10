#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[4]:


#load data
data = pd.read_csv('final_dataset.csv')

#getting all columns except for outflow data
data = data[['FIPS', 'State', 'Year', 'zillow_value', 'County', 'X..Fair.Poor', 'Physically.Unhealthy.Days', 'Mentally.Unhealthy.Days', 'X..Smokers', 'X..Obese', 'X..Physically.Inactive', 'X..Excessive.Drinking', 'X..Some.College', 'Population.1', 'X..Some.College.1', 'X..Social.Associations', 'Association.Rate', 'X..Severe.Housing.Problems', 'X..Insufficient.Sleep']]

#filtering only on 2021 data for kmeans
data = data[data['Year'] == 2021]

#drop the duplicate rows since we dropped the outflow columns
data = data.drop_duplicates()
data.head()


# In[5]:


#get predictors (not State, FIPS, County)
predictors = data[['zillow_value', 'X..Fair.Poor', 'Physically.Unhealthy.Days', 'Mentally.Unhealthy.Days', 'X..Smokers', 'X..Obese', 'X..Physically.Inactive', 'X..Excessive.Drinking', 'X..Some.College', 'Population.1', 'X..Some.College.1', 'X..Social.Associations', 'Association.Rate', 'X..Severe.Housing.Problems', 'X..Insufficient.Sleep']]

#scale predictors
scaler = StandardScaler()
scaled_predictors = scaler.fit_transform(predictors)


# In[6]:


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


# In[10]:


# getting optimal_k using kcv function on scaled predictors
clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
inertias = kcv_kmeans(scaled_predictors, clusters = clusters)

plt.plot(list(inertias.keys()), list(inertias.values()), marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Average Inertia")
plt.title("Optimal K")
plt.show()


# In[11]:


# create function for kmeans model to find similar counties - will use optimal k
def kmeans(model_data, optimal_k):
    km = KMeans(n_clusters = optimal_k, random_state = 123)
    assigned_clusters = km.fit_predict(model_data)
    
    return assigned_clusters  


# In[12]:


#run kmeans using optimal k
assigned_clusters = kmeans(scaled_predictors, optimal_k = 85)
assigned_clusters


# In[13]:


#add clusters to data
data['cluster'] = assigned_clusters
#data.head()
data.shape


# In[15]:


#putting data w/ clusters into a csv file
data.to_csv('kmeans_model_data.csv', index=False)

