#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:38:14 2018

@author: KimSooHyeon
"""

import pandas as pd
import numpy as np
import re

## start wrangling
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]
data = {}
for file in data_files:
    name = file.replace(".csv", "")
    datapath = "schools/" + file
    df = pd.read_csv(datapath)
    data[name] = df
    
all_survey = pd.read_csv("schools/survey_all.txt", delimiter='\t', encoding="latin1")
d75_survey = pd.read_csv("schools/survey_d75.txt", delimiter='\t', encoding="latin1")
survey = pd.concat([all_survey, d75_survey], axis=0, sort=True)

#print(survey.head())

## need to unify common column name
survey["DBN"] = survey["dbn"] 
## Only meaningful columns
cols = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", \
        "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", \
        "eng_t_11", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11", \
        "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11"]
survey = survey.loc[:, cols]
## including in data dict
data["survey"] = survey
#print(data["survey"].shape)
## unify common column name
data['hs_directory']['DBN'] = data['hs_directory']['dbn']

## unify common column name: in "class_size" DBN = padded_csd + SCHOOL CODE
def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return string_representation.zfill(2)

data['class_size']['padded_csd'] = data['class_size']['CSD'].apply(pad_csd)
data['class_size']['DBN'] =  data['class_size']['padded_csd'] + data['class_size']['SCHOOL CODE']
#print(data['class_size'].head())

## Averaging sat_score in "sat_results"
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]
#print(data['sat_results']['sat_score'].head())

## Securing locations (lat, lon) of high schools
def extract_latitude(string):
    loc_str = re.findall("\(.+\)", string)
    loc_list = loc_str[0].split(',')
    lat = loc_list[0].replace('(', '')
    return lat

def extract_longitude(string):
    loc_str = re.findall("\(.+\)", string)
    loc_list = loc_str[0].split(',')
    lon = loc_list[1].replace(')', '')
    return lon

data['hs_directory']['lat'] = data['hs_directory']['Location 1'].apply(extract_latitude)
data['hs_directory']['lon'] = data['hs_directory']['Location 1'].apply(extract_longitude)
## extract
cols = ['lat', 'lon']
for col in cols:
    data['hs_directory'][col] = pd.to_numeric(data['hs_directory'][col], errors='coerce')
#print(data['hs_directory']['lat'].head())    
#print(data['hs_directory']['lon'].head())
    
## Reducing "class_size" so that we do not have duplicate rows 
class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]
#print(class_size.head())    
class_size = class_size.groupby("DBN").agg(np.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size
#print(data["class_size"].head())

## Condensing "demographics" next - by most recent year: 2011-2012
data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]
#print(data["demographics"].head())

## Condensing "Graduation" -  by most recent year: 2006
data["graduation"] =  data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] =  data["graduation"][data["graduation"]["Demographic"] \
                                         == 'Total Cohort']
#print(data["graduation"].head())

## Merging data
combined = data["sat_results"]
combined = combined.merge(data["ap_2010"], how="left", on="DBN")
combined = combined.merge(data["graduation"], how="left", on="DBN")
#print(combined.iloc[1])
#print(combined.shape)
combined = combined.merge(data["class_size"], on = "DBN", how = "inner")
combined = combined.merge(data["demographics"], on = "DBN", how = "inner")
combined = combined.merge(data["survey"], on = "DBN", how = "inner")
combined = combined.merge(data["hs_directory"], on = "DBN", how = "inner")
#print(combined.head(3))
#print(combined.shape)
combined = combined.fillna(combined.mean())
combined = combined.fillna(0)
#print(combined.head())

## extract district
def first_two(string):
    return string[:2]

combined["school_dist"] = combined["DBN"].apply(first_two)
#print(combined["school_dist"].head())

## Study Correlation
correlations = combined.corr() 
correlations = correlations["sat_score"] # our interest is correlation with "sat_score"
#print(correlations)
#print(correlations[correlations > 0])
#print(correlations[correlations < 0])
""" 
    > total_enrollment has a strong positive correlation with sat_score
        -> larger schools tend to do better on the SAT
        > Other columns that are proxies for enrollment correlate similarly. 
             - total_students, N_s, N_p, N_t, AP Test Takers, Total Exams Taken, and NUMBER OF SECTIONS.
    > Both the percentage of females (female_per) and number of females (female_num) 
    at a school correlate positively with SAT score, whereas the percentage of males 
    (male_per) and the number of males (male_num) correlate negatively. 
    This could indicate that women do better on the SAT than men.
    > Teacher and student ratings of school safety (saf_t_11, and saf_s_11) correlate with sat_score.
    > Student ratings of school academic standards (aca_s_11) correlate with sat_score, 
    but this does not hold for ratings from teachers and parents (aca_p_11 and aca_t_11).
    > There is significant racial inequality in SAT scores (white_per, asian_per, black_per, hispanic_per).
    > The percentage of English language learners at the school (ell_percent, frl_percent) 
    has a strong negative correlation with SAT scores.
"""

## scatter plot: total_enrollment vs. sat_score
#%matplotlib inline
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x=combined["total_enrollment"], y=combined["sat_score"])
ax.set_xlabel("total_enrollment")
ax.set_ylabel("sat_score")
plt.show()
"""
    > Judging from the plot we just cretated, not an extremely strong correlation. 
    > There is a big cluster at the left bottom
        -> This cluster may be what's making the r value so high.
"""

## extracting cluster schools: total_enrollment < 1000 & sat_score < 1000
low_enrollment = combined[(combined['total_enrollment'] < 1000) & (combined['sat_score'] < 1000)]
# print(low_enrollment["SCHOOL NAME"])

combined.plot.scatter(x='ell_percent', y='sat_score')
plt.show()

##draw map using Basemap
from mpl_toolkits.basemap import Basemap

m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'  # resolution of boundary: intermediate
)
m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

#Note: We need to convert series to list
longitudes = combined["lon"].tolist()
latitudes = combined["lat"].tolist()
m.scatter(longitudes, latitudes, s=20, zorder=2, latlon=True, \
          c=combined["ell_percent"], cmap="summer") # we use lat,lon in degrees
plt.show()
"""
     it's hard to interpret the map
     -> aggregate by district, which will enable us 
     to plot ell_percent district-by-district instead of school-by-school
"""

## group by district
districts = combined.groupby('school_dist').agg(np.mean)
districts.reset_index(inplace=True)
#print(districts.head())

m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'  # resolution of boundary: intermediate
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = districts["lon"].tolist()
latitudes = districts["lat"].tolist()
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=districts["ell_percent"], cmap="summer")
plt.show()