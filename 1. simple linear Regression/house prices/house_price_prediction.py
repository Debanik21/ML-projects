#!/usr/bin/env python
# coding: utf-8

# # Predicting House Prices Using Multiple Linear Regression
# 
# In this project we are gonna see how machine learning algorithms help us predict house prices. Linear Regression is a model of predicting new future data by using the existing correlation between the old data. Here, machine learning helps us identify this relationship between feature data and output, so we can predict future values.

# In[1]:


import pandas as pd
# we use sklearn library in many machine learning calculations..
from sklearn import linear_model

# we import out dataset: housepricesdataset.csv
df = pd.read_csv("housepricesdataset.csv",sep = ";")


# In[2]:


# lets see and check our data set:
df


# ### The following is our feature set:

# In[3]:


df[['area', 'roomcount', 'buildingage']]


# ### The following is the output(result) data:

# In[4]:


df['price']


# In[5]:


# we define a linear regression model here:

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'roomcount', 'buildingage']], df['price'])


# #### Since our model is ready, we can make predictions now:

# In[6]:


# lets predict a house with 230 square meters, 4 rooms and 10 years old building..

reg.predict([[230,4,10]])


# #### Now lets predict a house with 230 square meters, 6 rooms and 0 years old building - its new building..

# In[7]:


reg.predict([[230,6,0]])


# #### Now lets predict a house with 355 square meters, 3 rooms and 20 years old building

# In[8]:


reg.predict([[355,3,20]])


# In[10]:


# You can make as many prediction as you want..
reg.predict([[230,4,10], [230,6,0], [355,3,20], [275, 5, 17]])


# ### Now we'll see the coefficients of our multilinear regression formula..

# In[11]:


reg.coef_


# In[12]:


reg.intercept_


# In[13]:


# Lets see the coeffients of Multiple Linear regression formula..
# y = a + b1X1 + b2X2 + b3X3 + b4X4 + b5X5 ...etc

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3

y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




