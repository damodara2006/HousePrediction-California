#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[57]:


df = pd.read_csv('housing.csv')


# In[13]:


sns.lineplot(x='ocean_proximity',y='median_house_value',data=df,)


# In[3]:


df.dropna( inplace=True)
df.columns


# In[32]:


plt.figure(figsize=(20,15))
sns.scatterplot(x='longitude',y='latitude', data=df, hue='median_house_value'),


# In[19]:


# plt.figure(figsize=(20,10))
# sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')


# In[16]:


df['total_rooms'] = np.log(df['total_rooms'] + 1)
df['total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
df['population'] = np.log(df['population'] + 1)
df['households'] = np.log(df['households'] + 1)


# In[17]:


df = df.join(pd.get_dummies(df["ocean_proximity"])).drop('ocean_proximity',axis=1)


# In[22]:


x = df.drop('median_house_value',axis=1) 
y = df['median_house_value']


# In[52]:


df


# In[25]:


from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test = train_test_split(x , y, test_size=0.2 )


# In[30]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()


# In[53]:


x_train


# In[39]:


model.fit(x_train,x_test)


# In[84]:


model.predict(y_train)


# In[85]:


y_test


# In[86]:


model.score(y_train, y_test)


# In[82]:


def predict():
    new_data = pd.DataFrame([{
    'longitude': -122.23,
    'latitude': 27.88,
    'housing_median_age': 1.0,
    'total_rooms': np.log(0 + 1),
    'total_bedrooms': np.log(0 + 1),
    'population': np.log(2 + 1),
    'households': np.log(1 + 1),
    'median_income': 0.3252,
    '<1H OCEAN': 1,
    'INLAND': 0,
    'ISLAND': 0,
    'NEAR BAY': 0,
    'NEAR OCEAN': 0}])
    prediction = model.predict(new_data)
    print("Predicted median house value:", prediction[0])
    
    

# Predict using your trained model


# In[83]:




# In[ ]:




