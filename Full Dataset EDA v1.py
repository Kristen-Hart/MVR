#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[64]:


#loading data - using only outflow
outflow = pd.read_csv('cleaned_countyoutflow.csv')
zillow = pd.read_csv('zillow_yearly_clean.csv')
health = pd.read_csv('health_data_clean.csv')


# # Data Cleaning

# In[65]:


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


# In[66]:


#remove columns from unnecesary years from zillow data (not 2017-2021)
zillow = zillow.iloc[:, list(range(0, 4)) + list(range(21, 26))]

#zillow.head()


# In[67]:


#melt zillow data so years become a column
zillow_melted = pd.melt(zillow, id_vars = ['FIPS', 'RegionName', 'State', 'FIPS.1'], value_vars=['X2017', 'X2018', 'X2019', 'X2020', 'X2021'], var_name='Year', value_name='zillow_value')

#check
#zillow_melted.head()


# In[68]:


#remove X in front of year for zillow data
zillow_melted.loc[zillow_melted['Year'].str.startswith('X2017'), 'Year'] = '2017'
zillow_melted.loc[zillow_melted['Year'].str.startswith('X2018'), 'Year'] = '2018'
zillow_melted.loc[zillow_melted['Year'].str.startswith('X2019'), 'Year'] = '2019'
zillow_melted.loc[zillow_melted['Year'].str.startswith('X2020'), 'Year'] = '2020'
zillow_melted.loc[zillow_melted['Year'].str.startswith('X2021'), 'Year'] = '2021'

#converting Year to int type
zillow_melted['Year'] = zillow_melted['Year'].astype(int)

#check
#zillow_melted.head()


# In[69]:


#merging zillow and health data
merged_z_h = pd.merge(zillow_melted, health, on=['Year', 'FIPS'], how='inner')

#check
#merged_z_h.head()


# In[70]:


#merging with outflow data
data = pd.merge(merged_z_h, outflow, on=['Year', 'FIPS'], how='inner')

#check
#data.head()


# In[71]:


#dropping columns that are duplicative
data = data.drop(columns = ['RegionName', 'FIPS.1', 'State_y', 'y2_state', 'y2_countyname', 'origin_county_fips', 'destination_county_fips'])

#renaming for clarity
data = data.rename(columns={'State_x': 'State'})

#check
#data.head()


# In[74]:


#write data to csv file
data.to_csv('final_dataset.csv', index = False)


# # EDA

# In[48]:


#dimensions
data.shape


# In[49]:


#summary stats
data.describe()


# In[50]:


#info on data types
data.info()


# In[61]:


#boxplots

#dropping non-number columns for boxplots and correlation
num_data = data.drop(columns=['State', 'County'])

for column in num_data.columns:
    plt.figure(figsize=(6, 4))
    sb.boxplot(y=num_data[column])
    plt.title(f"Box Plot of {column}")
    plt.ylabel(column)
    plt.show()


# In[62]:


#correlation

#plot correlations
plt.figure(figsize=(10, 8))
sb.heatmap(num_data.corr(), annot=True, fmt=".2f",annot_kws={"size": 6}, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix", fontsize=16)

plt.show()

#moderately strong negative corr between zillow value and smokers/obesity/physically inactive
    #similarly between come college and fair.poor/physically unhealthy/mentally unhealthy/smokers/obesity/physically inactive
#moderately strong positive correlation severe housing problems and some college/population
    #slightly weaker but interesting corr between mentally unhealthy days and year
#moderately to very strong positive corr between fair.poor, physically unhealthy, mentally unhealthy, smokers, obesity, and physically inactive
#extremely strong correlations between num_return, num_indv, and agi
#extremely strong positive correlation between population and some college, population and social assoc, and some college and social assoc

