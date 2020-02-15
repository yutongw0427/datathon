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

import glob
# i put my csv files with my .py file under the same folder.
file_name = []
for file in glob.glob("Part*.csv"):
    file_name.append(file)
file_name.sort()


import pandas as pd
raw_data = pd.DataFrame()
dfs = [pd.read_csv(f, skiprows=2, header=None) for f in file_name]
raw_data=pd.concat(dfs, ignore_index = True)
raw_data.head(10)
#still need to assign column names


# Change Column names
raw_data.columns = ['GEO_id', 'GEO_id2', 'GEO_display-label', 'NAICS_id', 'NAICS_display-label', 'RCPSZFE_id', 'RCPSZFE_display-label', 'YEAR_id',	'ESTAB']

