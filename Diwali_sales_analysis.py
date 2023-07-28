#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import python librray :-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# import csv file 

df= pd.read_csv("E:\Diwali Sales Data (1).csv",encoding='unicode_escape')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[8]:


# Drop unrelated/blank column

df.drop(['Status', 'unnamed1'],axis = 1, inplace=True)


# In[9]:


# Check all null values
pd.isnull(df).sum()


# In[10]:


# Drop all null values

df.dropna(inplace=True)


# In[11]:


# Change datatype
df['Amount'] = df['Amount'].astype('int')


# In[12]:


df['Amount'].dtypes


# In[13]:


df.columns


# In[14]:


# Rename columns
df.rename(columns = {'Marital_Status' : 'Shadi'})


# In[15]:


# Describe () method returns description of the data in the dataframe( i.e count, mean , std,etc)

df.describe()


# In[16]:


# Use describe () for specific columns 

df[['Age','Orders', 'Amount']].describe()


# # Exploratory Data Analysis

# In[17]:


# Plotting a bar chart for gender and it's count

ax = sns.countplot(x = 'Gender', data = df)

for bars in ax.containers:
    ax.bar_label(bars)


# In[18]:


# Plotting a bar chart for gender vs total amount

sales_gen = df.groupby (['Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)

sns.barplot(x = 'Gender',y='Amount',data = sales_gen)


# #### From above graphs we can see that most of the buyers are females and even the purchasing power of females are greater than men

# ### Age

# In[20]:


ax = sns.countplot(data = df, x='Age Group', hue ='Gender')

for bars in ax.containers:
    ax.bar_label(bars)


# In[21]:


# Total amount Vs Total Age Group

sales_age = df.groupby(['Age Group'],as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)

sns.barplot(x= 'Age Group', y = 'Amount', data = sales_age)


# ### from the above graphs we can see that most of the buyers are of age group between 26-35 yrs female

# ### STATE

# In[25]:


# Total number of orders from top 10 states

sales_state = df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by = 'Orders', ascending = False).head(10)

sns.set(rc={'figure.figsize':(15,5)})
sns.barplot(data = sales_state, x = 'State', y = 'Orders')


# In[26]:


# Total amount/sales from top 10 states

sales_state = df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by = 'Amount',ascending = False).head(10)

sns.set(rc={'figure.figsize':(15,5)})
sns.barplot(data = sales_state, x = 'State', y = 'Amount')


# ### From the above graph we can see that most of the orders & total sales / amount are from uttar pradesh , Maharashtra and Karnataka respectively.

# ### Marital Status

# In[28]:


ax = sns.countplot(data = df, x = 'Marital_Status')

sns.set(rc={'figure.figsize' :(7,5)})
for bars in ax.containers:
    ax.bar_label(bars)


# In[30]:


sales_state = df.groupby(['Marital_Status', 'Gender'], as_index=False)['Amount'].sum().sort_values(by ='Amount', ascending=False)

sns.set(rc={'figure.figsize':(6,5)})
sns.barplot(data = sales_state,x ='Marital_Status',y = 'Amount',hue = 'Gender')


# ### From the above graph we can see that most of the buyers are married (women) and they have hogh purchasing power.

# ### Occupation 

# In[31]:


sns.set(rc={'figure.figsize':(20,5)})
ax = sns.countplot(data = df, x = 'Occupation')

for bars in ax.containers:
    ax.bar_label(bars)


# In[32]:


sales_state = df.groupby(['Occupation'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)

sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(data = sales_state,x= 'Occupation',y = 'Amount')


# ### From the above graph we can see thar most of the buyers are working in IT , Healthcare and Aviation sector

# # Product Category

# In[33]:


sns.set(rc={'figure.figsize':(20,5)})
ax = sns.countplot(data=df, x= 'Product_Category')

for bars in ax.containers:
    ax.bar_label(bars)


# In[34]:


sales_state = df.groupby(['Product_Category'], as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False).head(10)

sns.set(rc={'figure.figsize' :(20,5)})
sns.barplot(data = sales_state, x = 'Product_Category', y = 'Amount')


# ### From the above graph we can see that most of the sold products are from Food, Clothing,and Electonics category

# In[37]:


sales_state = df.groupby(['Product_ID'], as_index=False)['Orders'].sum().sort_values(by='Orders',ascending=False).head(10)

sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(data = sales_state,x = 'Product_ID',y = 'Orders')


# In[38]:


# Top 10 most sold products (same thing as above)

fig1, ax1 = plt.subplots(figsize=(12,7))
df.groupby('Product_ID')['Orders'].sum().nlargest(10).sort_values(ascending= False).plot(kind='bar')


# ### Conclusion :-

# Married women age group between 26-35 yrs from Up, Maharashtra and Karnataka working in IT, Healrhcare and Aviation are more likely to but products from Food,Clothing and Electronics category
