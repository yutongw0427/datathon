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

#-----------------------------
import geopandas as gpd
# import folium as fo
import matplotlib.pyplot as plt
# import contextily
import pandas as pd
# usa = gpd.read_file('tl_2018_us_zcta510.shp')
data = pd.read_csv("sample.csv")
zipcd = pd.read_csv('zipcode.csv', delimiter=";")

new = zipcd["geopoint"].str.split(",", n=1, expand=True)

# making separate first name column from new data frame
data["lat"] = new[0].astype(float)
# making separate last name column from new data frame
data["long"] = new[1].astype(float)

print(data.describe())
point = gpd.GeoDataFrame(data, geometry = gpd.points_from_xy(data.lat, data.long))


fig, ax = plt.subplot(figsize=(15,15))
point.plot(ax=ax)

# # contextily.add_basemap(ax)
# # plt.show()
#
# print(point.head())

# --------------model-----------------------
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X = pd.get_dummies(retail['industry'], prefix='indus')
X = pd.concat([X,retail[['pop_density', 'area', 'wage',  'median_age']], pd.get_dummies(retail['state'], prefix='state')], axis =1)
y = retail[['score']]

data_matrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ="reg:squarederror", colsample_bytree = 0.3, learning_rate = 0.3,
                max_depth = 6, alpha = 5, n_estimators = 10)
xg_reg.fit(X_train,y_train)
xgb.plot_importance(xg_reg)
plt.show()
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(rmse)
