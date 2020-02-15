#!/usr/bin/env python
# coding: utf-8

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

# ------------------------------Read Data---------------------------------

import glob
# -----IMPORTANT! CSV FILES ARE UNDER THE SAME FOLDER WITH PY CODE!-------
file_name = []
for file in glob.glob("Part*.csv"):
    file_name.append(file)
file_name.sort()

raw_data = pd.DataFrame()
dfs = [pd.read_csv(f, skiprows=2, header=None) for f in file_name]
raw_data=pd.concat(dfs, ignore_index = True)
raw_data.head(10)

# Change Column names
raw_data.columns = ['GEO_id', 'GEO_id2', 'GEO_display_label', 'NAICS_id', 
                    'NAICS_display_label', 'RCPSZFE_id', 'RCPSZFE_display_label', 'YEAR_id','ESTAB']

# Convert column types to object
# Only 'ESTAB' is int
for col in ['GEO_id2', 'RCPSZFE_id', 'YEAR_id']:
    raw_data[col] = raw_data[col].astype('object')
raw_data.dtypes

#--------------------------pivot table transformation-----------------------
raw_data['GEO_NAICS'] = raw_data['GEO_id2'].astype(str)+'_'+raw_data['NAICS_id'].astype(str)

data=pd.DataFrame(raw_data.pivot(index='GEO_NAICS', columns='RCPSZFE_id',values='ESTAB'))
data = data.fillna(0)
data=data.reset_index()
data[['zipcode','naics']] = data.GEO_NAICS.str.split('_', expand=True)

data['score'] = data[123]*175+data[125]*375 + data[131]*750 + data[132]*1500
zip_sum = data.groupby('zipcode').agg('sum')
zip_sum = zip_sum.reset_index()

