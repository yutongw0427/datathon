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
# Change Column names
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
# SPLIT THE UNIQUE_INDEX BACK TO zipcode AND naics
data[['zipcode','naics']] = data.zip_naics.str.split('_',expand=True)
# CONVERT THE NACIS_ID BACK TO INDUSTRY, TEXT FORM AND ADD COLUMN OF CITY AND STATE
data['industry']=data['naics'].map(naics_dict)
data['city_state'] = data['zipcode'].map(zip_dict)
data[['city','state']] = data['city_state'].str.split(', ', expand = True)

#------------------------compute score ------------------------
data['score'] = data[123]*175+data[125]*375 + data[131]*750 + data[132]*1500

data.head()


# ----------------------RETAIL DATA / LEN(NAICS) == 3: HIGHER CATEGORY-------
retail = data[data['naics'].astype(str).str.len()==3]


# ---------------------- Merge 5 Datasets ----------------------
## US zip information 2020 to get the zipcode area
uszips = pd.read_csv('uszips.csv')
uszips.rename(columns={'zip': 'zipcode'}, inplace=True)
uszips['zipcode'] = uszips['zipcode'].astype('str').str.zfill(5)
uszips.head()
uszips['area'] = uszips['population'] / uszips['density']
area = uszips[['zipcode', 'area']]

## Read population by zipcode
pop_by_zip = pd.read_csv('pop-by-zip-code.csv')
pop_by_zip = pop_by_zip[['zip_code', 'y-2012']]
pop_by_zip.columns = ['zipcode', 'pop_2012']
pop_by_zip['zipcode'] = pop_by_zip['zipcode'].astype('str').str.zfill(5) # Fill 0 before zipcode

## Read Median Age
age = pd.read_csv('Median_age_by_ZIP_code.csv')
age.columns = ['zipcode', 'population', 'median_age']
age['zipcode'] = age.zipcode.astype('str').str.zfill(5)
age = age[['zipcode', 'median_age']]

## Read Tax and Income
income = pd.read_csv('12zpallagi.csv') # We probably only need column zipcode and A00200
income['zipcode'] = income['zipcode'].astype('str').str.zfill(5)
wage = income[['zipcode','A00200']]
wage.columns = ['zipcode', 'wage']
wage = wage.groupby('zipcode').mean()

## sales tax
sales_tax = pd.read_csv('state_sales_tax_rate.csv')
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}
sales_tax['state'] = 'NA'
for i in range(sales_tax.shape[0]):
    sales_tax['state'][i] = us_state_abbrev[sales_tax['State'][i]]
sales_tax = sales_tax[['state', 'Combined Rate']]

unemp = read_csv("unemp_rate.csv")


## Combine the Data
# Merge populatino, area, wage and median age
retail = data.merge(pop_by_zip, how='left', on='zipcode').merge(area, how = 'left', on = 'zipcode').merge(wage, how = 'left', on = 'zipcode').merge(age, how = 'left', on = 'zipcode').merge(unemp, how = 'left', on = 'ZIP')

# Calculate the density
retail['pop_density'] = retail['pop_2012'] / retail['area']

# Merge state sales tax rate
retail = retail.merge(sales_tax, how='left', on='state')

# ----------------------Aggregate and plot-----------------------------

zip_sum = retail.groupby('zipcode').agg('sum')
zip_sum = retail.reset_index()
zip_score = zip_sum.loc[:,['zipcode','naics','score']]

retail_dominant_industry = retail.loc[retail.groupby('zipcode')['score'].idxmax()]

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

#df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   #dtype={"fips": str})

import plotly.express as px

fig = px.choropleth_mapbox(retail_dominant_industry, geojson=counties, locations='zipcode', color='naics',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'industry':'Dominant'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ----------- TOTAL SCORE FOR EACH INDUSTRY TYPE -----------------------
retail.groupby('naics').agg('sum').sort_values('score', ascending=False)

# HUMAN SELECTED TOP 4 SCORE
top_industry = ['447','448','445','441']
# INDUSTRY NAME FOR TOP 4
[naics_dict[i] for i in top_industry]




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
X = pd.concat([X,retail[['pop_2012', 'wage',  'median_age']], pd.get_dummies(retail['state'], prefix='state')], axis =1)
y = retail[['score']]

data_matrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)
xgb.plot_importance(xg_reg)
plt.show()
