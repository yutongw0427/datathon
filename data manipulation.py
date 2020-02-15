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

# ------------------------ADD COLUMNS NAMES--------------------------------
raw_data.columns = ['GEO_id', 'GEO_id2', 'GEO_display_label', 'NAICS_id', 
                    'NAICS_display_label', 'RCPSZFE_id', 'RCPSZFE_display_label', 'YEAR_id','ESTAB']

# CREATE COLUMN CITY_STATE, CITY, ZIPCODE, AND UNIQUE INDEX COMBINDED WITH ZIPCODE AND NAICS_ID FOR LATER PIVOT USE
raw_data['city_state']=raw_data['GEO_display_label'].str.replace(r'[^(]*\(|\)[^)]*', '')
raw_data[['city','state']]=raw_data.city_state.str.split(', ',expand=True)
raw_data['zipcode'] = raw_data['GEO_display_label'].str.slice(start = 4, stop = 9)
raw_data['zip_naics'] = raw_data['zipcode'].astype(str)+'_'+raw_data['NAICS_id'].astype(str)

# CREATE MAP FOR NAICS_ID TO ITS DISPLAY_LABEL, AND ZIPCODE TO CITY_STATE
naics_dict = pd.Series(raw_data['NAICS_display_label'].values,index=raw_data['NAICS_id']).to_dict()
zip_dict = pd.Series(raw_data['city_state'].values,index=raw_data['zipcode']).to_dict()


#--------------------------pivot table transformation-----------------------
# DATA: PIVOT: ROWS--->COLUMNS, NEW COLUMN WILL BE THE RCPSZFE_id
data=pd.DataFrame(raw_data.pivot(index='zip_naics', columns='RCPSZFE_id',values='ESTAB'))
data=data.fillna(0)

# RESET_INDEX: MAKE PIVOT TABLE TO THE DATAFRAME IN PANDAS
data=data.reset_index()

# SPLIT THE UNIQUE_INDEX BACK TO zipcode AND naics
data[['zipcode','naics']] = data.zip_naics.str.split('_',expand=True)
# CONVERT THE NACIS_ID BACK TO INDUSTRY, TEXT FORM AND ADD COLUMN OF CITY AND STATE
data['industry']=data['naics'].map(naics_dict)
data['city_state'] = data['zipcode'].map(zip_dict)
data[['city','state']] = data['city_state'].str.split(', ', expand = True)

#------------------------compute score ------------------------
data['score'] = data[123]*175+data[125]*375 + data[131]*750 + data[132]*1500

data.head()
