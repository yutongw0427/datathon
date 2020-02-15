#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


# Import Packages 
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

# Import for plots
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb

# Import for complex plots (like ggplot2)
from plotnine import *


# In[2]:


part1 = pd.read_csv('Part 1.csv')


# In[24]:


# Read Data
part1 = pd.read_csv('Part 1.csv', skiprows=1)
part2 = pd.read_csv('Part 2.csv', skiprows=1)
part3 = pd.read_csv('Part 3.csv', skiprows=1)
part4a = pd.read_csv('Part 4a.csv', skiprows=1)
part4b = pd.read_csv('Part 4b.csv', skiprows=1)
part5 = pd.read_csv('Part 5.csv', skiprows=1)


# In[25]:


# Combine the data
data = pd.concat([part1, part2, part3, part4a, part5, part5])


# In[26]:


data.shape


# In[27]:


data.head()


# In[17]:


data['YEAR.id'].unique()


# In[28]:





# In[ ]:




