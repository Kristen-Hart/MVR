#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[23]:


#loading data - using only outflow
outflow = pd.read_csv('cleaned_countyoutflow.csv')
#zillow and health data
zh = pd.read_csv('Agony - Health And Zillow Population Fixed.csv')


# # Data Cleaning

# In[24]:


#converting outflow year column into an actual year (If 16-17, converting to 2017)
outflow.loc[outflow['year'].str.endswith('17'), 'year'] = '2017'
outflow.loc[outflow['year'].str.endswith('18'), 'year'] = '2018'
outflow.loc[outflow['year'].str.endswith('19'), 'year'] = '2019'
outflow.loc[outflow['year'].str.endswith('20'), 'year'] = '2020'
outflow.loc[outflow['year'].str.endswith('21'), 'year'] = '2021'

#converting year to int type and renaming for later merging
outflow['year'] = outflow['year'].astype(int)
outflow = outflow.rename(columns={'year': 'Year', 'y2fips': 'FIPS'})

#check
#outflow.head()
#health.head()


# In[25]:


#converting Year to int type
zh['Year'] = zh['Year'].astype(int)

#check
#zillow_melted.head()


# In[26]:


#merging with outflow data
data = pd.merge(zh, outflow, on=['Year', 'FIPS'], how='inner')

#check
data.head()


# In[27]:


#dropping columns that are duplicative
data = data.drop(columns = ['y2_state', 'y2_countyname', 'origin_county_fips', 'destination_county_fips', 'destination_state_fips'])

#check
data.head()


# In[32]:


#write data to csv file - 2 pieces because file too large for github
data[:100000].to_csv('final_dataset_pt1.csv', index = False)
data[100000:].to_csv('final_dataset_pt2.csv', index = False)


# # EDA

# In[29]:


#dimensions
data.shape


# In[30]:


#summary stats
data.describe()


# In[31]:


#info on data types
data.info()


# In[20]:


#boxplots

#dropping non-number columns for boxplots and correlation
num_data = data.drop(columns=['State', 'County'])

for column in num_data.columns:
    plt.figure(figsize=(6, 4))
    sb.boxplot(y=num_data[column])
    plt.title(f"Box Plot of {column}")
    plt.ylabel(column)
    plt.show()


# In[22]:


#correlation

#plot correlations
plt.figure(figsize=(10, 8))
sb.heatmap(num_data.corr(), annot=True, fmt=".2f",annot_kws={"size": 6}, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix", fontsize=16)

plt.show()

